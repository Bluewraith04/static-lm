use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::model::transformer::{ModelWeights, TransformerWeights};
use crate::quant::blocks::{BlockQ4_0, BlockQ8_0, QK4_0, QK8_0};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" in little-endian
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// GGML tensor element data type codes in GGUF.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Unsupported(u32),
}

impl GgmlType {
    pub const fn from_u32(code: u32) -> Self {
        match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            other => Self::Unsupported(other),
        }
    }

    pub const fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
            Self::Unsupported(code) => code,
        }
    }

    /// Number of elements per quantization block (`1` for unquantized types).
    pub const fn block_elements(self) -> Option<usize> {
        match self {
            Self::F32 | Self::F16 => Some(1),
            Self::Q4_0 => Some(QK4_0),
            Self::Q8_0 => Some(QK8_0),
            Self::Unsupported(_) => None,
        }
    }

    /// Size in bytes of a single block.
    pub const fn block_bytes(self) -> Option<usize> {
        match self {
            Self::F32 => Some(std::mem::size_of::<f32>()),
            Self::F16 => Some(std::mem::size_of::<u16>()),
            Self::Q4_0 => Some(std::mem::size_of::<BlockQ4_0>()),
            Self::Q8_0 => Some(std::mem::size_of::<BlockQ8_0>()),
            Self::Unsupported(_) => None,
        }
    }
}

/// Parsed GGUF metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// Stores `(elem_type_code, count, byte_offset_in_file, byte_len)` without heap-copying large arrays.
    Array {
        elem_type: u32,
        len: usize,
        byte_offset: usize,
        byte_len: usize,
    },
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            Self::U32(v) => Some(v),
            Self::U64(v) => u32::try_from(v).ok(),
            Self::I32(v) if v >= 0 => Some(v as u32),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match *self {
            Self::U32(v) => Some(v as usize),
            Self::U64(v) => usize::try_from(v).ok(),
            Self::I32(v) if v >= 0 => Some(v as usize),
            Self::I64(v) if v >= 0 => usize::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Descriptor for a single tensor in the GGUF tensor directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub ggml_type: GgmlType,
    pub n_dims: u32,
    pub dims: [u64; 4],
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of logical elements across all dimensions.
    pub fn num_elements(&self) -> usize {
        let mut prod = 1usize;
        for i in 0..(self.n_dims as usize) {
            prod = prod
                .checked_mul(self.dims[i] as usize)
                .expect("tensor element count overflow");
        }
        prod
    }

    /// Total byte length of the tensor payload.
    pub fn byte_len(&self) -> Option<usize> {
        let elems = self.num_elements();
        let blk_elems = self.ggml_type.block_elements()?;
        let blk_bytes = self.ggml_type.block_bytes()?;
        if elems % blk_elems != 0 {
            return None;
        }
        (elems / blk_elems).checked_mul(blk_bytes)
    }
}

/// Errors returned while parsing or indexing a GGUF file.
#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    UnexpectedEof,
    InvalidMagic(u32),
    UnsupportedVersion(u32),
    InvalidUtf8,
    InvalidValueType(u32),
    InvalidDimensions(u32),
    TensorNotFound(String),
    TypeMismatch {
        name: String,
        expected: GgmlType,
        actual: GgmlType,
    },
    OutOfBounds(String),
    UnalignedTensor(String),
}

impl From<io::Error> for GgufError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::UnexpectedEof)?;
        if end > self.bytes.len() {
            return Err(GgufError::UnexpectedEof);
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn read_str(&mut self) -> Result<&'a str, GgufError> {
        let len = usize::try_from(self.read_u64()?).map_err(|_| GgufError::UnexpectedEof)?;
        let raw = self.take(len)?;
        std::str::from_utf8(raw).map_err(|_| GgufError::InvalidUtf8)
    }

    fn skip_scalar_or_str(&mut self, type_code: u32) -> Result<(), GgufError> {
        match type_code {
            0 | 1 | 7 => {
                self.take(1)?;
            }
            2 | 3 => {
                self.take(2)?;
            }
            4 | 5 | 6 => {
                self.take(4)?;
            }
            8 => {
                let len = usize::try_from(self.read_u64()?).map_err(|_| GgufError::UnexpectedEof)?;
                self.take(len)?;
            }
            9 => {
                let elem_type = self.read_u32()?;
                let count =
                    usize::try_from(self.read_u64()?).map_err(|_| GgufError::UnexpectedEof)?;
                for _ in 0..count {
                    self.skip_scalar_or_str(elem_type)?;
                }
            }
            10 | 11 | 12 => {
                self.take(8)?;
            }
            other => return Err(GgufError::InvalidValueType(other)),
        }
        Ok(())
    }

    fn read_value(&mut self) -> Result<GgufValue, GgufError> {
        let type_code = self.read_u32()?;
        match type_code {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufValue::Str(self.read_str()?.to_owned())),
            9 => {
                let elem_type = self.read_u32()?;
                let len =
                    usize::try_from(self.read_u64()?).map_err(|_| GgufError::UnexpectedEof)?;
                let byte_offset = self.pos;
                for _ in 0..len {
                    self.skip_scalar_or_str(elem_type)?;
                }
                let byte_len = self.pos - byte_offset;
                Ok(GgufValue::Array {
                    elem_type,
                    len,
                    byte_offset,
                    byte_len,
                })
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            other => Err(GgufError::InvalidValueType(other)),
        }
    }
}

/// Memory-mapped GGUF model file providing zero-copy views over `&[BlockQ4_0]` and `&[f32]` tensors.
pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub alignment: usize,
    pub data_offset: usize,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorInfo>,
}

impl GgufFile {
    /// Opens and memory-maps a `.gguf` v2/v3 file, parsing its metadata and tensor directory.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        // SAFETY: Backing file is opened read-only and treated as immutable while mapped.
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = ByteCursor::new(&mmap);
        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = cursor.read_u32()?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let n_tensors =
            usize::try_from(cursor.read_u64()?).map_err(|_| GgufError::UnexpectedEof)?;
        let n_kv = usize::try_from(cursor.read_u64()?).map_err(|_| GgufError::UnexpectedEof)?;

        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = cursor.read_str()?.to_owned();
            let val = cursor.read_value()?;
            metadata.insert(key, val);
        }

        let mut tensors = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = cursor.read_str()?.to_owned();
            let n_dims = cursor.read_u32()?;
            if n_dims == 0 || n_dims > 4 {
                return Err(GgufError::InvalidDimensions(n_dims));
            }
            let mut dims = [1u64; 4];
            for d in dims.iter_mut().take(n_dims as usize) {
                *d = cursor.read_u64()?;
            }
            let ggml_type = GgmlType::from_u32(cursor.read_u32()?);
            let offset = cursor.read_u64()?;

            tensors.insert(
                name,
                TensorInfo {
                    ggml_type,
                    n_dims,
                    dims,
                    offset,
                },
            );
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_usize)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT)
            .max(1);
        let data_offset = cursor.pos.next_multiple_of(alignment);

        Ok(Self {
            mmap,
            version,
            alignment,
            data_offset,
            metadata,
            tensors,
        })
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    #[inline]
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(GgufValue::as_u32)
    }

    #[inline]
    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.metadata.get(key).and_then(GgufValue::as_usize)
    }

    #[inline]
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(GgufValue::as_f32)
    }

    #[inline]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(GgufValue::as_str)
    }

    fn raw_tensor_bytes(&self, name: &str, expected: GgmlType) -> Result<&[u8], GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_owned()))?;
        if info.ggml_type != expected {
            return Err(GgufError::TypeMismatch {
                name: name.to_owned(),
                expected,
                actual: info.ggml_type,
            });
        }
        let byte_len = info
            .byte_len()
            .ok_or_else(|| GgufError::OutOfBounds(name.to_owned()))?;
        let start = self
            .data_offset
            .checked_add(usize::try_from(info.offset).map_err(|_| GgufError::OutOfBounds(name.to_owned()))?)
            .ok_or_else(|| GgufError::OutOfBounds(name.to_owned()))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| GgufError::OutOfBounds(name.to_owned()))?;
        if end > self.mmap.len() {
            return Err(GgufError::OutOfBounds(name.to_owned()));
        }
        Ok(&self.mmap[start..end])
    }

    /// Returns a zero-copy view of a `Q4_0` tensor by name.
    pub fn tensor_q4_0(&self, name: &str) -> Result<&[BlockQ4_0], GgufError> {
        let bytes = self.raw_tensor_bytes(name, GgmlType::Q4_0)?;
        bytemuck::try_cast_slice(bytes).map_err(|_| GgufError::UnalignedTensor(name.to_owned()))
    }

    /// Returns a zero-copy view of an `F32` tensor by name.
    pub fn tensor_f32(&self, name: &str) -> Result<&[f32], GgufError> {
        let bytes = self.raw_tensor_bytes(name, GgmlType::F32)?;
        bytemuck::try_cast_slice(bytes).map_err(|_| GgufError::UnalignedTensor(name.to_owned()))
    }

    /// Resolves standard GGUF layer tensor names (`blk.{layer_idx}.*`) into [`TransformerWeights`].
    pub fn layer_weights(&self, layer_idx: usize) -> Result<TransformerWeights<'_>, GgufError> {
        let prefix = format!("blk.{layer_idx}");
        Ok(TransformerWeights {
            rms_att_weight: self.tensor_f32(&format!("{prefix}.attn_norm.weight"))?,
            wq: self.tensor_q4_0(&format!("{prefix}.attn_q.weight"))?,
            wk: self.tensor_q4_0(&format!("{prefix}.attn_k.weight"))?,
            wv: self.tensor_q4_0(&format!("{prefix}.attn_v.weight"))?,
            wo: self.tensor_q4_0(&format!("{prefix}.attn_output.weight"))?,
            rms_ffn_weight: self.tensor_f32(&format!("{prefix}.ffn_norm.weight"))?,
            w_gate: self.tensor_q4_0(&format!("{prefix}.ffn_gate.weight"))?,
            w_up: self.tensor_q4_0(&format!("{prefix}.ffn_up.weight"))?,
            w_down: self.tensor_q4_0(&format!("{prefix}.ffn_down.weight"))?,
        })
    }

    /// Resolves full GGUF model tensors (`token_embd.weight`, `blk.{0..num_layers}.*`,
    /// `output_norm.weight`, and `output.weight`) into a zero-copy [`ModelWeights`] view.
    pub fn model_weights(&self, num_layers: usize) -> Result<ModelWeights<'_>, GgufError> {
        let embd_info = self
            .tensors
            .get("token_embd.weight")
            .ok_or_else(|| GgufError::TensorNotFound("token_embd.weight".to_owned()))?;
        let vocab_size = usize::try_from(embd_info.dims[1])
            .map_err(|_| GgufError::OutOfBounds("token_embd.weight".to_owned()))?;
        let token_embd = self.tensor_q4_0("token_embd.weight")?;
        let output_norm = self.tensor_f32("output_norm.weight")?;
        let lm_head = if self.tensors.contains_key("output.weight") {
            self.tensor_q4_0("output.weight")?
        } else {
            token_embd
        };

        let mut layers = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            layers.push(self.layer_weights(l)?);
        }

        Ok(ModelWeights {
            token_embd,
            layers,
            output_norm,
            lm_head,
            vocab_size,
        })
    }
}

/// Minimal GGUF v3 file writer for self-contained Rust tests and benchmark generation.
pub struct GgufWriter {
    kv_entries: Vec<(String, GgufValue)>,
    tensors: Vec<(String, GgmlType, Vec<u64>, Vec<u8>)>,
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    pub fn new() -> Self {
        Self {
            kv_entries: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn add_u32(&mut self, key: &str, val: u32) {
        self.kv_entries.push((key.to_owned(), GgufValue::U32(val)));
    }

    pub fn add_f32(&mut self, key: &str, val: f32) {
        self.kv_entries.push((key.to_owned(), GgufValue::F32(val)));
    }

    pub fn add_str(&mut self, key: &str, val: &str) {
        self.kv_entries
            .push((key.to_owned(), GgufValue::Str(val.to_owned())));
    }

    pub fn add_tensor_f32(&mut self, name: &str, dims: &[u64], data: &[f32]) {
        self.tensors.push((
            name.to_owned(),
            GgmlType::F32,
            dims.to_vec(),
            bytemuck::cast_slice(data).to_vec(),
        ));
    }

    pub fn add_tensor_q4_0(&mut self, name: &str, dims: &[u64], data: &[BlockQ4_0]) {
        self.tensors.push((
            name.to_owned(),
            GgmlType::Q4_0,
            dims.to_vec(),
            bytemuck::cast_slice(data).to_vec(),
        ));
    }

    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // GGUF v3
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.kv_entries.len() as u64).to_le_bytes());

        let write_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };

        for (k, v) in &self.kv_entries {
            write_str(&mut buf, k);
            match v {
                GgufValue::U32(x) => {
                    buf.extend_from_slice(&4u32.to_le_bytes());
                    buf.extend_from_slice(&x.to_le_bytes());
                }
                GgufValue::F32(x) => {
                    buf.extend_from_slice(&6u32.to_le_bytes());
                    buf.extend_from_slice(&x.to_le_bytes());
                }
                GgufValue::Str(s) => {
                    buf.extend_from_slice(&8u32.to_le_bytes());
                    write_str(&mut buf, s);
                }
                _ => unimplemented!("GgufWriter only supports U32, F32, Str metadata"),
            }
        }

        let mut rel_offset = 0usize;
        let mut tensor_offsets = Vec::with_capacity(self.tensors.len());
        for (name, ggml_type, dims, raw_bytes) in &self.tensors {
            rel_offset = rel_offset.next_multiple_of(GGUF_DEFAULT_ALIGNMENT);
            tensor_offsets.push(rel_offset);

            write_str(&mut buf, name);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&ggml_type.to_u32().to_le_bytes());
            buf.extend_from_slice(&(rel_offset as u64).to_le_bytes());

            rel_offset += raw_bytes.len();
        }

        let data_offset = buf.len().next_multiple_of(GGUF_DEFAULT_ALIGNMENT);
        buf.resize(data_offset, 0);

        for ((_, _, _, raw_bytes), &off) in self.tensors.iter().zip(tensor_offsets.iter()) {
            let target_len = data_offset + off;
            if buf.len() < target_len {
                buf.resize(target_len, 0);
            }
            buf.extend_from_slice(raw_bytes);
        }

        let mut file = File::create(path)?;
        file.write_all(&buf)?;
        file.sync_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::blocks::{BlockQ8_0, QK4_0, QK8_0, quantize_row_q4_0, quantize_row_q8_0};
    use crate::quant::kernels::gemv_q4_0;
    use bytemuck::Zeroable;

    #[test]
    fn test_gguf_parse_metadata_and_zero_copy_tensors() {
        let rows = 8;
        let cols = 64;
        let num_blocks = rows * (cols / QK4_0);

        let w_f32: Vec<f32> = (0..(rows * cols))
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let norm_f32: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32) * 0.001).collect();
        let x_f32: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.05).cos()).collect();

        let mut w_q4 = vec![BlockQ4_0::zeroed(); num_blocks];
        let mut x_q8 = vec![BlockQ8_0::zeroed(); cols / QK8_0];
        quantize_row_q4_0(&w_f32, &mut w_q4);
        quantize_row_q8_0(&x_f32, &mut x_q8);

        let mut expected = vec![0.0f32; rows];
        gemv_q4_0(&w_q4, &x_q8, &mut expected, rows, cols);

        let tmp_path = std::env::temp_dir().join(format!(
            "inference_kernel_test_{}.gguf",
            std::process::id()
        ));

        let mut writer = GgufWriter::new();
        writer.add_str("general.architecture", "llama");
        writer.add_u32("llama.embedding_length", cols as u32);
        writer.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
        writer.add_tensor_f32("blk.0.attn_norm.weight", &[cols as u64], &norm_f32);
        writer.add_tensor_q4_0("blk.0.attn_q.weight", &[cols as u64, rows as u64], &w_q4);
        writer
            .write_to_file(&tmp_path)
            .expect("failed to write test GGUF file");

        let gguf = GgufFile::open(&tmp_path).expect("failed to open GGUF file");
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.get_str("general.architecture"), Some("llama"));
        assert_eq!(gguf.get_u32("llama.embedding_length"), Some(cols as u32));
        assert_eq!(
            gguf.get_f32("llama.attention.layer_norm_rms_epsilon"),
            Some(1e-5)
        );

        let mmap_start = gguf.as_bytes().as_ptr() as usize;
        let mmap_end = mmap_start + gguf.as_bytes().len();

        let mapped_norm = gguf
            .tensor_f32("blk.0.attn_norm.weight")
            .expect("missing norm tensor");
        assert_eq!(mapped_norm, norm_f32.as_slice());
        let norm_ptr = mapped_norm.as_ptr() as usize;
        assert!(
            norm_ptr >= mmap_start && norm_ptr + std::mem::size_of_val(mapped_norm) <= mmap_end,
            "tensor_f32 slice is not zero-copy inside mmap region"
        );

        let mapped_wq = gguf
            .tensor_q4_0("blk.0.attn_q.weight")
            .expect("missing wq tensor");
        assert_eq!(mapped_wq, w_q4.as_slice());
        let wq_ptr = mapped_wq.as_ptr() as usize;
        assert!(
            wq_ptr >= mmap_start && wq_ptr + std::mem::size_of_val(mapped_wq) <= mmap_end,
            "tensor_q4_0 slice is not zero-copy inside mmap region"
        );

        let mut actual = vec![0.0f32; rows];
        gemv_q4_0(mapped_wq, &x_q8, &mut actual, rows, cols);
        assert_eq!(actual, expected);

        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_gguf_all_13_metadata_types_and_error_paths() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // Test GGUF v2 as well
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&13u64.to_le_bytes()); // All 13 KV types

        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };

        // 0: U8, 1: I8, 2: U16, 3: I16, 4: U32, 5: I32, 6: F32, 7: Bool, 8: Str, 9: Array, 10: U64, 11: I64, 12: F64
        push_str(&mut buf, "k_u8");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.push(42u8);

        push_str(&mut buf, "k_i8");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.push((-7i8) as u8);

        push_str(&mut buf, "k_u16");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&1024u16.to_le_bytes());

        push_str(&mut buf, "k_i16");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(-512i16).to_le_bytes());

        push_str(&mut buf, "k_u32");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&999u32.to_le_bytes());

        push_str(&mut buf, "k_i32");
        buf.extend_from_slice(&5u32.to_le_bytes());
        buf.extend_from_slice(&(-999i32).to_le_bytes());

        push_str(&mut buf, "k_f32");
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&3.5f32.to_le_bytes());

        push_str(&mut buf, "k_bool");
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.push(1u8);

        push_str(&mut buf, "k_str");
        buf.extend_from_slice(&8u32.to_le_bytes());
        push_str(&mut buf, "hello");

        push_str(&mut buf, "k_arr");
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // Array of strings
        buf.extend_from_slice(&2u64.to_le_bytes());
        push_str(&mut buf, "tok0");
        push_str(&mut buf, "tok1");

        push_str(&mut buf, "k_u64");
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&123_456u64.to_le_bytes());

        push_str(&mut buf, "k_i64");
        buf.extend_from_slice(&11u32.to_le_bytes());
        buf.extend_from_slice(&(-123_456i64).to_le_bytes());

        push_str(&mut buf, "k_f64");
        buf.extend_from_slice(&12u32.to_le_bytes());
        buf.extend_from_slice(&2.25f64.to_le_bytes());

        // 1 Tensor with deliberately misaligned offset (offset = 1) to test UnalignedTensor
        push_str(&mut buf, "misaligned.weight");
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1D
        buf.extend_from_slice(&4u64.to_le_bytes()); // 4 f32s
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&1u64.to_le_bytes()); // offset = 1 (unaligned for f32)

        let data_off = buf.len().next_multiple_of(32);
        buf.resize(data_off + 32, 0);

        let path = std::env::temp_dir().join(format!(
            "inference_kernel_13types_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &buf).unwrap();

        let gguf = GgufFile::open(&path).expect("failed to parse 13-type GGUF v2");
        assert_eq!(gguf.version, 2);
        assert_eq!(gguf.metadata.get("k_u8"), Some(&GgufValue::U8(42)));
        assert_eq!(gguf.metadata.get("k_i8"), Some(&GgufValue::I8(-7)));
        assert_eq!(gguf.metadata.get("k_u16"), Some(&GgufValue::U16(1024)));
        assert_eq!(gguf.metadata.get("k_i16"), Some(&GgufValue::I16(-512)));
        assert_eq!(gguf.get_u32("k_u32"), Some(999));
        assert_eq!(gguf.metadata.get("k_i32"), Some(&GgufValue::I32(-999)));
        assert_eq!(gguf.get_f32("k_f32"), Some(3.5));
        assert_eq!(gguf.metadata.get("k_bool"), Some(&GgufValue::Bool(true)));
        assert_eq!(gguf.get_str("k_str"), Some("hello"));
        assert_eq!(gguf.get_usize("k_u64"), Some(123_456));
        assert_eq!(gguf.metadata.get("k_i64"), Some(&GgufValue::I64(-123_456)));
        assert_eq!(gguf.get_f32("k_f64"), Some(2.25));
        assert!(matches!(
            gguf.metadata.get("k_arr"),
            Some(GgufValue::Array { elem_type: 8, len: 2, .. })
        ));

        // Verify error cases
        assert!(matches!(
            gguf.tensor_f32("misaligned.weight"),
            Err(GgufError::UnalignedTensor(_))
        ));
        assert!(matches!(
            gguf.tensor_q4_0("misaligned.weight"),
            Err(GgufError::TypeMismatch { .. })
        ));
        assert!(matches!(
            gguf.tensor_f32("nonexistent.weight"),
            Err(GgufError::TensorNotFound(_))
        ));

        // Invalid magic
        std::fs::write(&path, b"BAD!0000").unwrap();
        assert!(matches!(
            GgufFile::open(&path),
            Err(GgufError::InvalidMagic(_))
        ));

        // Unsupported version
        let mut bad_ver = Vec::new();
        bad_ver.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bad_ver.extend_from_slice(&99u32.to_le_bytes());
        std::fs::write(&path, &bad_ver).unwrap();
        assert!(matches!(
            GgufFile::open(&path),
            Err(GgufError::UnsupportedVersion(99))
        ));

        // Unexpected EOF
        std::fs::write(&path, &GGUF_MAGIC.to_le_bytes()).unwrap();
        assert!(matches!(
            GgufFile::open(&path),
            Err(GgufError::UnexpectedEof)
        ));

        let _ = std::fs::remove_file(&path);
    }
}
