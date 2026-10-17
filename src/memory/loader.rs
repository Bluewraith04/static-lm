use std::fs::File;
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;

use crate::quant::blocks::BlockQ4_0;

pub const WEIGHT_FILE_MAGIC: u32 = 0x3034_5149; // "IQ40" in little-endian
pub const WEIGHT_FILE_VERSION: u32 = 1;

/// Fixed 16-byte header for flat binary weight files.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct WeightFileHeader {
    pub magic: u32,
    pub version: u32,
    pub num_blocks: u64,
}

const _: () = assert!(std::mem::size_of::<WeightFileHeader>() == 16);
const _: () = assert!(std::mem::align_of::<WeightFileHeader>() == 8);

/// Memory-mapped weight file backing zero-copy `&[BlockQ4_0]` slices.
pub struct MappedWeights {
    mmap: Mmap,
}

impl MappedWeights {
    /// Maps a binary weight file into the process virtual address space.
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: The backing file is opened read-only and treated as immutable while mapped.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    /// Returns the raw mapped bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Parses and validates the 16-byte [`WeightFileHeader`] at byte offset 0.
    pub fn header(&self) -> Result<&WeightFileHeader, &'static str> {
        let header_size = std::mem::size_of::<WeightFileHeader>();
        if self.mmap.len() < header_size {
            return Err("mapped file smaller than WeightFileHeader (16 bytes)");
        }
        let header: &WeightFileHeader = bytemuck::try_from_bytes(&self.mmap[..header_size])
            .map_err(|_| "unaligned WeightFileHeader")?;
        if header.magic != WEIGHT_FILE_MAGIC {
            return Err("invalid weight file magic");
        }
        if header.version != WEIGHT_FILE_VERSION {
            return Err("unsupported weight file version");
        }
        Ok(header)
    }

    /// Returns a zero-copy view of `num_blocks` [`BlockQ4_0`] elements starting at `byte_offset`.
    #[inline]
    pub fn view_q4_0(&self, byte_offset: usize, num_blocks: usize) -> &[BlockQ4_0] {
        let byte_len = num_blocks
            .checked_mul(std::mem::size_of::<BlockQ4_0>())
            .expect("block byte length overflow");
        let end = byte_offset
            .checked_add(byte_len)
            .expect("byte offset overflow");
        assert!(
            end <= self.mmap.len(),
            "view_q4_0 out of bounds: requested {}..{}, file len {}",
            byte_offset,
            end,
            self.mmap.len()
        );
        bytemuck::try_cast_slice(&self.mmap[byte_offset..end])
            .expect("view_q4_0 byte_offset must be 2-byte aligned for BlockQ4_0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::blocks::{quantize_row_q4_0, quantize_row_q8_0, BlockQ8_0, QK4_0, QK8_0};
    use crate::quant::kernels::gemv_q4_0;
    use std::io::Write;

    #[test]
    fn test_mmap_zero_copy_gemv() {
        let rows = 8;
        let cols = 64;
        let num_blocks = rows * (cols / QK4_0);

        let w_f32: Vec<f32> = (0..(rows * cols))
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let x_f32: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.05).cos()).collect();

        let mut w_q4 = vec![BlockQ4_0::zeroed(); num_blocks];
        let mut x_q8 = vec![BlockQ8_0::zeroed(); cols / QK8_0];
        quantize_row_q4_0(&w_f32, &mut w_q4);
        quantize_row_q8_0(&x_f32, &mut x_q8);

        let mut expected = vec![0.0f32; rows];
        gemv_q4_0(&w_q4, &x_q8, &mut expected, rows, cols);

        let tmp_path = std::env::temp_dir().join(format!(
            "inference_kernel_test_{}.bin",
            std::process::id()
        ));
        {
            let mut file = File::create(&tmp_path).expect("failed to create temp weight file");
            let header = WeightFileHeader {
                magic: WEIGHT_FILE_MAGIC,
                version: WEIGHT_FILE_VERSION,
                num_blocks: num_blocks as u64,
            };
            file.write_all(bytemuck::bytes_of(&header))
                .expect("failed to write header");
            file.write_all(bytemuck::cast_slice(&w_q4))
                .expect("failed to write blocks");
            file.sync_all().expect("failed to sync file");
        }

        let mapped = MappedWeights::open(&tmp_path).expect("failed to mmap weight file");
        let hdr = mapped.header().expect("valid header");
        assert_eq!(hdr.num_blocks, num_blocks as u64);

        let mapped_slice = mapped.view_q4_0(std::mem::size_of::<WeightFileHeader>(), num_blocks);
        assert_eq!(mapped_slice, w_q4.as_slice());

        let mut actual = vec![0.0f32; rows];
        gemv_q4_0(mapped_slice, &x_q8, &mut actual, rows, cols);
        assert_eq!(actual, expected);

        let _ = std::fs::remove_file(&tmp_path);
    }
}
