use crate::quant::blocks::{BlockQ4_0, BlockQ8_0, QK4_0};

/// Reference scalar dot product between `Q4_0` weights and `Q8_0` activations.
pub fn vec_dot_q4_0_q8_0_scalar(weights: &[BlockQ4_0], activations: &[BlockQ8_0]) -> f32 {
    assert_eq!(weights.len(), activations.len(), "block count mismatch");

    let mut sumf = 0.0f32;
    for (w_block, a_block) in weights.iter().zip(activations.iter()) {
        let mut sum_i = 0i32;
        for j in 0..(QK4_0 / 2) {
            let packed = w_block.qs[j];
            let v0 = (packed & 0x0F) as i32 - 8;
            let v1 = (packed >> 4) as i32 - 8;

            sum_i += v0 * (a_block.qs[j] as i32) + v1 * (a_block.qs[j + QK4_0 / 2] as i32);
        }
        sumf += (sum_i as f32) * (w_block.scale() * a_block.scale());
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_0_q8_0_avx2(weights: &[BlockQ4_0], activations: &[BlockQ8_0]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(weights.len(), activations.len());

    let mask_low = _mm_set1_epi8(0x0F);
    let offset_8 = _mm256_set1_epi8(8);
    let ones_16 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();

    for (w_block, a_block) in weights.iter().zip(activations.iter()) {
        let d = w_block.scale() * a_block.scale();
        let d_vec = _mm256_set1_ps(d);

        // Load 16 packed bytes (32 4-bit weights)
        let raw_q4 = unsafe { _mm_loadu_si128(w_block.qs.as_ptr() as *const __m128i) };
        let q4_lo = _mm_and_si128(raw_q4, mask_low);
        let q4_hi = _mm_and_si128(_mm_srli_epi16(raw_q4, 4), mask_low);

        // Combine low 16 weights and high 16 weights into 32 signed i8 lanes in [-8, 7]
        let q4_u8 = _mm256_set_m128i(q4_hi, q4_lo);
        let w_i8 = _mm256_sub_epi8(q4_u8, offset_8);

        // Load 32 signed i8 activations
        let a_i8 = unsafe { _mm256_loadu_si256(a_block.qs.as_ptr() as *const __m256i) };

        // Unsigned * signed multiply-add via sign-transfer trick
        let ax = _mm256_sign_epi8(w_i8, w_i8);
        let sy = _mm256_sign_epi8(a_i8, w_i8);
        let dot_i16 = _mm256_maddubs_epi16(ax, sy);
        let dot_i32 = _mm256_madd_epi16(ones_16, dot_i16);

        let dot_f32 = _mm256_cvtepi32_ps(dot_i32);
        acc = _mm256_fmadd_ps(d_vec, dot_f32, acc);
    }

    // Horizontal reduction of 8 f32 lanes
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let total = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(total)
}

/// Hardware-accelerated dot product between `Q4_0` weights and `Q8_0` activations.
#[inline]
pub fn vec_dot_q4_0_q8_0(weights: &[BlockQ4_0], activations: &[BlockQ8_0]) -> f32 {
    assert_eq!(weights.len(), activations.len(), "block count mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 and FMA support verified via runtime feature detection.
            return unsafe { vec_dot_q4_0_q8_0_avx2(weights, activations) };
        }
    }

    vec_dot_q4_0_q8_0_scalar(weights, activations)
}

/// Quantized Matrix-Vector multiplication (`output = weights * input_q8`).
pub fn gemv_q4_0(
    weights: &[BlockQ4_0],
    input_q8: &[BlockQ8_0],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    assert_eq!(cols % QK4_0, 0, "cols must be a multiple of 32");
    let blocks_per_row = cols / QK4_0;
    assert_eq!(input_q8.len(), blocks_per_row, "input_q8 length mismatch");
    assert_eq!(output.len(), rows, "output length mismatch");
    assert_eq!(
        weights.len(),
        rows * blocks_per_row,
        "weights block count mismatch"
    );

    for (row_weights, out_val) in weights
        .chunks_exact(blocks_per_row)
        .zip(output.iter_mut())
    {
        *out_val = vec_dot_q4_0_q8_0(row_weights, input_q8);
    }
}

/// Zero-allocation arena-backed quantized Matrix-Vector multiplication (`output = weights * input`).
/// Dynamically quantizes `input` into `q8_scratch` inside `arena` before executing `gemv_q4_0`.
pub fn gemv_q4_0_arena(
    arena: &mut crate::memory::arena::AlignedArena,
    weights: &[BlockQ4_0],
    input: crate::memory::arena::ArenaSlice,
    q8_scratch: crate::memory::arena::ArenaSlice,
    output: crate::memory::arena::ArenaSlice,
    rows: usize,
    cols: usize,
) {
    assert_eq!(cols % QK4_0, 0, "cols must be a multiple of 32");
    let blocks_per_row = cols / QK4_0;
    let (x_f32, x_q8, y_f32) = arena.gemv_slices(input, q8_scratch, output, blocks_per_row);
    crate::quant::blocks::quantize_row_q8_0(x_f32, x_q8);
    gemv_q4_0(weights, x_q8, y_f32, rows, cols);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::blocks::{quantize_row_q4_0, quantize_row_q8_0, QK8_0};
    use bytemuck::Zeroable;

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

    fn naive_f32_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    #[test]
    fn test_vec_dot_scalar_vs_f32_reference() {
        let n = 4096;
        // Offset positive distribution so expected dot product has strong magnitude
        let w_f32: Vec<f32> = pseudo_random_f32(n, 0xCAFE_BABE)
            .into_iter()
            .map(|x| x + 0.5)
            .collect();
        let a_f32: Vec<f32> = pseudo_random_f32(n, 0xDEAD_BEEF)
            .into_iter()
            .map(|x| x + 0.5)
            .collect();

        let mut w_q4 = vec![BlockQ4_0::zeroed(); n / QK4_0];
        let mut a_q8 = vec![BlockQ8_0::zeroed(); n / QK8_0];
        quantize_row_q4_0(&w_f32, &mut w_q4);
        quantize_row_q8_0(&a_f32, &mut a_q8);

        let ref_dot = naive_f32_dot(&w_f32, &a_f32);
        let quant_dot = vec_dot_q4_0_q8_0_scalar(&w_q4, &a_q8);

        let rel_err = (ref_dot - quant_dot).abs() / ref_dot.abs();
        assert!(
            rel_err < 1e-2,
            "relative error {rel_err} exceeded 1e-2 (ref={ref_dot}, quant={quant_dot})"
        );
    }

    #[test]
    fn test_vec_dot_simd_matches_scalar() {
        let n = 1024;
        let w_f32 = pseudo_random_f32(n, 0x1122_3344);
        let a_f32 = pseudo_random_f32(n, 0x5566_7788);

        let mut w_q4 = vec![BlockQ4_0::zeroed(); n / QK4_0];
        let mut a_q8 = vec![BlockQ8_0::zeroed(); n / QK8_0];
        quantize_row_q4_0(&w_f32, &mut w_q4);
        quantize_row_q8_0(&a_f32, &mut a_q8);

        let scalar = vec_dot_q4_0_q8_0_scalar(&w_q4, &a_q8);
        let simd = vec_dot_q4_0_q8_0(&w_q4, &a_q8);

        let diff = (scalar - simd).abs();
        assert!(
            diff < 1e-5,
            "SIMD result diverged from scalar: scalar={scalar}, simd={simd}, diff={diff}"
        );
    }

    #[test]
    fn test_gemv_q4_0_against_naive_f32() {
        let rows = 32;
        let cols = 1024;
        let blocks_per_row = cols / QK4_0;
        let w_f32: Vec<f32> = pseudo_random_f32(rows * cols, 0x99AA_BBCC)
            .into_iter()
            .map(|x| x + 0.5)
            .collect();
        let x_f32: Vec<f32> = pseudo_random_f32(cols, 0xDDEE_FF11)
            .into_iter()
            .map(|x| x + 0.5)
            .collect();

        let mut w_q4 = vec![BlockQ4_0::zeroed(); rows * blocks_per_row];
        let mut x_q8 = vec![BlockQ8_0::zeroed(); cols / QK8_0];
        quantize_row_q4_0(&w_f32, &mut w_q4);
        quantize_row_q8_0(&x_f32, &mut x_q8);

        let mut out_quant = vec![0.0f32; rows];
        gemv_q4_0(&w_q4, &x_q8, &mut out_quant, rows, cols);

        let mut total_rel_err = 0.0f32;
        for r in 0..rows {
            let row_q4 = &w_q4[r * blocks_per_row..(r + 1) * blocks_per_row];
            let scalar_val = vec_dot_q4_0_q8_0_scalar(row_q4, &x_q8);
            assert!(
                (scalar_val - out_quant[r]).abs() < 1e-4,
                "row {r} SIMD GEMV diverged from scalar: {scalar_val} vs {}",
                out_quant[r]
            );

            let row_slice = &w_f32[r * cols..(r + 1) * cols];
            let ref_val = naive_f32_dot(row_slice, &x_f32);
            let rel_err = (ref_val - out_quant[r]).abs() / ref_val.abs();
            total_rel_err += rel_err;
            assert!(
                rel_err < 1.5e-2,
                "row {r} rel_err {rel_err} too high (ref={ref_val}, quant={})",
                out_quant[r]
            );
        }
        let mean_rel_err = total_rel_err / (rows as f32);
        assert!(
            mean_rel_err < 1e-2,
            "mean GEMV rel_err {mean_rel_err} exceeded 1e-2"
        );
    }
}
