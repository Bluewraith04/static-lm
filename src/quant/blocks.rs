use bytemuck::{Pod, Zeroable};
use half::f16;

pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;

/// 32-weight 4-bit block (`Q4_0`, 18 bytes).
///
/// * `d`: IEEE 754 `f16` scale factor stored as `u16` bits (`half::f16::from_bits(d)`).
/// * `qs`: 32 signed 4-bit weights packed two per byte (low nibble = weight `i`, high nibble = weight `i + 16`, offset by -8).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct BlockQ4_0 {
    pub d: u16,
    pub qs: [u8; QK4_0 / 2],
}

const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);
const _: () = assert!(std::mem::align_of::<BlockQ4_0>() == 2);

impl BlockQ4_0 {
    #[inline(always)]
    pub fn scale(&self) -> f32 {
        f16::from_bits(self.d).to_f32()
    }
}

/// 32-element 8-bit quantized activation block (`Q8_0`, 34 bytes).
///
/// * `d`: IEEE 754 `f16` scale factor stored as `u16` bits (`half::f16::from_bits(d)`).
/// * `qs`: 32 signed `i8` quantized values.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct BlockQ8_0 {
    pub d: u16,
    pub qs: [i8; QK8_0],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);
const _: () = assert!(std::mem::align_of::<BlockQ8_0>() == 2);

impl BlockQ8_0 {
    #[inline(always)]
    pub fn scale(&self) -> f32 {
        f16::from_bits(self.d).to_f32()
    }
}

/// Quantizes a contiguous `f32` slice (length must be a multiple of 32) into `BlockQ4_0` blocks.
pub fn quantize_row_q4_0(src: &[f32], dst: &mut [BlockQ4_0]) {
    assert_eq!(src.len() % QK4_0, 0, "src length must be a multiple of 32");
    assert_eq!(dst.len(), src.len() / QK4_0, "dst block count mismatch");

    for (chunk, block) in src.chunks_exact(QK4_0).zip(dst.iter_mut()) {
        let mut min_val = 0.0f32;
        let mut max_val = 0.0f32;
        for &v in chunk {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        let d_pos = (-min_val / 8.0).max(max_val / 7.0);
        let d_neg = (max_val / 8.0).max(-min_val / 7.0);
        let d = if d_pos <= d_neg { d_pos } else { -d_neg };
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = f16::from_f32(d).to_bits();

        for j in 0..(QK4_0 / 2) {
            let x0 = chunk[j] * id;
            let x1 = chunk[j + QK4_0 / 2] * id;

            let q0 = (x0.round() as i32 + 8).clamp(0, 15) as u8;
            let q1 = (x1.round() as i32 + 8).clamp(0, 15) as u8;

            block.qs[j] = q0 | (q1 << 4);
        }
    }
}

/// Dequantizes `BlockQ4_0` blocks back into `f32` values for reference verification.
pub fn dequantize_row_q4_0(src: &[BlockQ4_0], dst: &mut [f32]) {
    assert_eq!(dst.len(), src.len() * QK4_0, "dst length mismatch");

    for (block, chunk) in src.iter().zip(dst.chunks_exact_mut(QK4_0)) {
        let d = block.scale();
        for j in 0..(QK4_0 / 2) {
            let packed = block.qs[j];
            let q0 = (packed & 0x0F) as i32 - 8;
            let q1 = (packed >> 4) as i32 - 8;

            chunk[j] = (q0 as f32) * d;
            chunk[j + QK4_0 / 2] = (q1 as f32) * d;
        }
    }
}

/// Quantizes a contiguous `f32` slice (length must be a multiple of 32) into `BlockQ8_0` blocks.
pub fn quantize_row_q8_0(src: &[f32], dst: &mut [BlockQ8_0]) {
    assert_eq!(src.len() % QK8_0, 0, "src length must be a multiple of 32");
    assert_eq!(dst.len(), src.len() / QK8_0, "dst block count mismatch");

    for (chunk, block) in src.chunks_exact(QK8_0).zip(dst.iter_mut()) {
        let mut amax = 0.0f32;
        for &v in chunk {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }

        let d = amax / 127.0;
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
        block.d = f16::from_f32(d).to_bits();

        for j in 0..QK8_0 {
            block.qs[j] = (chunk[j] * id).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// Dequantizes `BlockQ8_0` blocks back into `f32` values for reference verification.
pub fn dequantize_row_q8_0(src: &[BlockQ8_0], dst: &mut [f32]) {
    assert_eq!(dst.len(), src.len() * QK8_0, "dst length mismatch");

    for (block, chunk) in src.iter().zip(dst.chunks_exact_mut(QK8_0)) {
        let d = block.scale();
        for j in 0..QK8_0 {
            chunk[j] = (block.qs[j] as f32) * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random_f32(len: usize, mut seed: u32) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let unit = (seed as f32) / (u32::MAX as f32);
            out.push(unit * 2.0 - 1.0);
        }
        out
    }

    #[test]
    fn test_q4_0_roundtrip_error() {
        let input = pseudo_random_f32(256, 0x1234_5678);
        let mut blocks = vec![BlockQ4_0::zeroed(); 256 / QK4_0];
        let mut reconstructed = vec![0.0f32; 256];

        quantize_row_q4_0(&input, &mut blocks);
        dequantize_row_q4_0(&blocks, &mut reconstructed);

        for (&orig, &rec) in input.iter().zip(reconstructed.iter()) {
            let diff = (orig - rec).abs();
            // Max step size for [-1, 1] is 1/7 = 0.142857, half-step = 0.07143 + f16 rounding
            assert!(
                diff <= 0.0725,
                "Q4_0 roundtrip diff too high: {diff} ({orig} vs {rec})"
            );
        }
    }

    #[test]
    fn test_q8_0_roundtrip_error() {
        let input = pseudo_random_f32(256, 0x8765_4321);
        let mut blocks = vec![BlockQ8_0::zeroed(); 256 / QK8_0];
        let mut reconstructed = vec![0.0f32; 256];

        quantize_row_q8_0(&input, &mut blocks);
        dequantize_row_q8_0(&blocks, &mut reconstructed);

        for (&orig, &rec) in input.iter().zip(reconstructed.iter()) {
            let diff = (orig - rec).abs();
            assert!(diff <= 0.005, "Q8_0 roundtrip diff too high: {diff} ({orig} vs {rec})");
        }
    }

    #[test]
    fn test_quantize_edge_cases_zero_and_extremes() {
        // 1. All-zero input must produce 0 scale and reconstruct to exact 0.0
        let zeros = [0.0f32; 64];
        let mut q4 = [BlockQ4_0::zeroed(); 2];
        let mut q8 = [BlockQ8_0::zeroed(); 2];
        let mut rec4 = [1.0f32; 64];
        let mut rec8 = [1.0f32; 64];

        quantize_row_q4_0(&zeros, &mut q4);
        quantize_row_q8_0(&zeros, &mut q8);
        dequantize_row_q4_0(&q4, &mut rec4);
        dequantize_row_q8_0(&q8, &mut rec8);

        assert_eq!(q4[0].scale(), 0.0);
        assert_eq!(q8[0].scale(), 0.0);
        assert!(rec4.iter().all(|&v| v == 0.0));
        assert!(rec8.iter().all(|&v| v == 0.0));

        // 2. Strictly negative and large-magnitude blocks
        let mut neg_block = [0.0f32; 32];
        for (i, v) in neg_block.iter_mut().enumerate() {
            *v = -((i as f32) + 1.0) * 10.0; // [-320.0 .. -10.0]
        }
        let mut q4_neg = [BlockQ4_0::zeroed(); 1];
        let mut rec4_neg = [0.0f32; 32];
        quantize_row_q4_0(&neg_block, &mut q4_neg);
        dequantize_row_q4_0(&q4_neg, &mut rec4_neg);
        // Max step is 320 / 8 = 40, half-step = 20
        for (&o, &r) in neg_block.iter().zip(rec4_neg.iter()) {
            assert!((o - r).abs() <= 20.5, "neg block mismatch: {o} vs {r}");
        }
    }
}
