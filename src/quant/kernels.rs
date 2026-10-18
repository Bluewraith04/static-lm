use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::{self, Thread};

use crate::quant::blocks::{BlockQ4_0, BlockQ8_0, QK4_0};

const PARALLEL_THRESHOLD_BLOCKS: usize = 8_192;

struct GemvJobState {
    weights_ptr: AtomicUsize,
    input_q8_ptr: AtomicUsize,
    output_ptr: AtomicUsize,
    batch_size: AtomicUsize,
    rows: AtomicUsize,
    blocks_per_row: AtomicUsize,
    next_row: AtomicUsize,
    done_workers: AtomicUsize,
    generation: AtomicUsize,
}

struct GemvPool {
    num_workers: usize,
    worker_threads: Vec<Thread>,
    dispatch_lock: Mutex<()>,
    state: &'static GemvJobState,
}

static GEMV_POOL: OnceLock<GemvPool> = OnceLock::new();

#[inline]
unsafe fn execute_row_chunks(
    w_ptr: *const BlockQ4_0,
    in_ptr: *const BlockQ8_0,
    out_ptr: *mut f32,
    batch_size: usize,
    rows: usize,
    blocks_per_row: usize,
    next_row: &AtomicUsize,
    chunk_size: usize,
) {
    loop {
        let r_start = next_row.fetch_add(chunk_size, Ordering::Relaxed);
        if r_start >= rows {
            break;
        }
        let r_end = (r_start + chunk_size).min(rows);
        for r in r_start..r_end {
            let row_w =
                unsafe { std::slice::from_raw_parts(w_ptr.add(r * blocks_per_row), blocks_per_row) };
            for b in 0..batch_size {
                let in_q8 = unsafe {
                    std::slice::from_raw_parts(in_ptr.add(b * blocks_per_row), blocks_per_row)
                };
                let val = vec_dot_q4_0_q8_0(row_w, in_q8);
                unsafe {
                    *out_ptr.add(b * rows + r) = val;
                }
            }
        }
    }
}

/// Initializes the persistent background worker threads for parallel `GEMV`/`GEMM`.
/// Safe to call multiple times; performs allocation only on the first call.
pub fn init_gemv_pool() {
    let _ = GEMV_POOL.get_or_init(|| {
        let logical_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Target physical cores (logical_cpus / 2) to avoid SMT FMA contention and VM preemptions
        let total_threads = (logical_cpus / 2).max(1);
        let num_workers = total_threads.saturating_sub(1);

        let state: &'static GemvJobState = Box::leak(Box::new(GemvJobState {
            weights_ptr: AtomicUsize::new(0),
            input_q8_ptr: AtomicUsize::new(0),
            output_ptr: AtomicUsize::new(0),
            batch_size: AtomicUsize::new(1),
            rows: AtomicUsize::new(0),
            blocks_per_row: AtomicUsize::new(0),
            next_row: AtomicUsize::new(0),
            done_workers: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }));

        let mut worker_threads = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let handle = thread::spawn(move || {
                let mut seen_gen = 0usize;
                loop {
                    let mut cur_gen = state.generation.load(Ordering::Acquire);
                    while cur_gen == seen_gen {
                        thread::park();
                        cur_gen = state.generation.load(Ordering::Acquire);
                    }
                    seen_gen = cur_gen;

                    let w_ptr = state.weights_ptr.load(Ordering::Acquire) as *const BlockQ4_0;
                    let in_ptr = state.input_q8_ptr.load(Ordering::Acquire) as *const BlockQ8_0;
                    let out_ptr = state.output_ptr.load(Ordering::Acquire) as *mut f32;
                    let batch_size = state.batch_size.load(Ordering::Acquire);
                    let rows = state.rows.load(Ordering::Acquire);
                    let blocks_per_row = state.blocks_per_row.load(Ordering::Acquire);
                    let chunk_size = (rows / (num_workers + 1)).max(16);

                    unsafe {
                        execute_row_chunks(
                            w_ptr,
                            in_ptr,
                            out_ptr,
                            batch_size,
                            rows,
                            blocks_per_row,
                            &state.next_row,
                            chunk_size,
                        );
                    }

                    state.done_workers.fetch_add(1, Ordering::Release);
                }
            });
            worker_threads.push(handle.thread().clone());
        }

        GemvPool {
            num_workers,
            worker_threads,
            dispatch_lock: Mutex::new(()),
            state,
        }
    });
}

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
    let clamp_min_i8 = _mm256_set1_epi8(-127);
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

        // Load 32 signed i8 activations and clamp -128 to -127 so _mm256_sign_epi8 never wraps
        let raw_a_i8 = unsafe { _mm256_loadu_si256(a_block.qs.as_ptr() as *const __m256i) };
        let a_i8 = _mm256_max_epi8(raw_a_i8, clamp_min_i8);

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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_dot_q4_0_q8_0_neon(weights: &[BlockQ4_0], activations: &[BlockQ8_0]) -> f32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(weights.len(), activations.len());

    let mask_low = vdupq_n_u8(0x0F);
    let offset_8 = vdupq_n_s8(8);
    let mut sumf = 0.0f32;

    for (w_block, a_block) in weights.iter().zip(activations.iter()) {
        let d = w_block.scale() * a_block.scale();

        let raw_q4 = unsafe { vld1q_u8(w_block.qs.as_ptr()) };
        let q4_lo = vreinterpretq_s8_u8(vandq_u8(raw_q4, mask_low));
        let q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(raw_q4, 4));
        let w_lo = vsubq_s8(q4_lo, offset_8);
        let w_hi = vsubq_s8(q4_hi, offset_8);

        let a_lo = unsafe { vld1q_s8(a_block.qs.as_ptr()) };
        let a_hi = unsafe { vld1q_s8(a_block.qs.as_ptr().add(16)) };

        let p0 = vmull_s8(vget_low_s8(w_lo), vget_low_s8(a_lo));
        let p0 = vmlal_s8(p0, vget_high_s8(w_lo), vget_high_s8(a_lo));
        let p1 = vmull_s8(vget_low_s8(w_hi), vget_low_s8(a_hi));
        let p1 = vmlal_s8(p1, vget_high_s8(w_hi), vget_high_s8(a_hi));

        let sum_i16 = vaddq_s16(p0, p1);
        let sum_i32 = vpaddlq_s16(sum_i16);
        let block_sum = vaddvq_s32(sum_i32);
        sumf += (block_sum as f32) * d;
    }

    sumf
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

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON support verified via runtime feature detection.
            return unsafe { vec_dot_q4_0_q8_0_neon(weights, activations) };
        }
    }

    vec_dot_q4_0_q8_0_scalar(weights, activations)
}

/// Parallel row-wise Quantized Matrix-Vector multiplication (`output = weights * input_q8`).
pub fn gemv_q4_0(
    weights: &[BlockQ4_0],
    input_q8: &[BlockQ8_0],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    gemm_q4_0(weights, input_q8, output, 1, rows, cols);
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

/// Quantized Matrix-Matrix multiplication (`output[b, r] = weights[r, :] * inputs_q8[b, :]`).
/// Loops over weight rows in the outer loop (`r in 0..rows`) and batch tokens in the inner loop (`b in 0..batch_size`)
/// so each `Q4_0` weight row is loaded from DRAM once and reused across all batch tokens while hot in L1 cache.
pub fn gemm_q4_0(
    weights: &[BlockQ4_0],
    inputs_q8: &[BlockQ8_0],
    output: &mut [f32],
    batch_size: usize,
    rows: usize,
    cols: usize,
) {
    assert_eq!(cols % QK4_0, 0, "cols must be a multiple of 32");
    let blocks_per_row = cols / QK4_0;
    assert_eq!(
        inputs_q8.len(),
        batch_size * blocks_per_row,
        "inputs_q8 batch block count mismatch"
    );
    assert_eq!(
        output.len(),
        batch_size * rows,
        "output batch length mismatch"
    );
    assert_eq!(
        weights.len(),
        rows * blocks_per_row,
        "weights block count mismatch"
    );
    if batch_size == 0 || rows == 0 {
        return;
    }

    if weights.len() * batch_size >= PARALLEL_THRESHOLD_BLOCKS && rows >= 64 {
        if let Some(pool) = GEMV_POOL.get() {
            if pool.num_workers > 0 {
                if let Ok(_guard) = pool.dispatch_lock.try_lock() {
                    let st = pool.state;
                    st.weights_ptr.store(weights.as_ptr() as usize, Ordering::Release);
                    st.input_q8_ptr
                        .store(inputs_q8.as_ptr() as usize, Ordering::Release);
                    st.output_ptr
                        .store(output.as_mut_ptr() as usize, Ordering::Release);
                    st.batch_size.store(batch_size, Ordering::Release);
                    st.rows.store(rows, Ordering::Release);
                    st.blocks_per_row.store(blocks_per_row, Ordering::Release);
                    st.next_row.store(0, Ordering::Release);
                    st.done_workers.store(0, Ordering::Release);

                    st.generation.fetch_add(1, Ordering::AcqRel);
                    for w in &pool.worker_threads {
                        w.unpark();
                    }

                    let chunk_size = (rows / (pool.num_workers + 1)).max(16);
                    unsafe {
                        execute_row_chunks(
                            weights.as_ptr(),
                            inputs_q8.as_ptr(),
                            output.as_mut_ptr(),
                            batch_size,
                            rows,
                            blocks_per_row,
                            &st.next_row,
                            chunk_size,
                        );
                    }

                    while st.done_workers.load(Ordering::Acquire) < pool.num_workers {
                        thread::yield_now();
                    }
                    return;
                }
            }
        }
    }

    if batch_size == 1 {
        for (row_weights, out_val) in weights
            .chunks_exact(blocks_per_row)
            .zip(output.iter_mut())
        {
            *out_val = vec_dot_q4_0_q8_0(row_weights, inputs_q8);
        }
    } else {
        for (r, row_weights) in weights.chunks_exact(blocks_per_row).enumerate() {
            for b in 0..batch_size {
                let in_row_q8 = &inputs_q8[b * blocks_per_row..(b + 1) * blocks_per_row];
                output[b * rows + r] = vec_dot_q4_0_q8_0(row_weights, in_row_q8);
            }
        }
    }
}

/// Zero-allocation arena-backed quantized Matrix-Matrix multiplication (`output = inputs * weights^T`).
/// Dynamically quantizes `batch_size * cols` floats from `inputs` into `q8_scratch` before executing `gemm_q4_0`.
pub fn gemm_q4_0_arena(
    arena: &mut crate::memory::arena::AlignedArena,
    weights: &[BlockQ4_0],
    inputs: crate::memory::arena::ArenaSlice,
    q8_scratch: crate::memory::arena::ArenaSlice,
    output: crate::memory::arena::ArenaSlice,
    batch_size: usize,
    rows: usize,
    cols: usize,
) {
    assert_eq!(cols % QK4_0, 0, "cols must be a multiple of 32");
    assert_eq!(inputs.len, batch_size * cols, "inputs batch size mismatch");
    assert_eq!(output.len, batch_size * rows, "output batch size mismatch");
    let total_q8_blocks = batch_size * (cols / QK4_0);
    let (x_f32, x_q8, y_f32) = arena.gemv_slices(inputs, q8_scratch, output, total_q8_blocks);
    crate::quant::blocks::quantize_row_q8_0(x_f32, x_q8);
    gemm_q4_0(weights, x_q8, y_f32, batch_size, rows, cols);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::arena::AlignedArena;
    use crate::memory::arena::tests::current_thread_heap_ops;
    use crate::quant::blocks::{QK8_0, quantize_row_q4_0, quantize_row_q8_0};
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
    fn test_gemv_q4_0_against_naive_f32_and_parallel_zero_alloc() {
        let rows = 512;
        let cols = 4096;
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

        init_gemv_pool();
        let mut out_quant = vec![0.0f32; rows];
        let ops_before = current_thread_heap_ops();
        gemv_q4_0(&w_q4, &x_q8, &mut out_quant, rows, cols);
        let ops_after = current_thread_heap_ops();
        assert_eq!(ops_after, ops_before, "parallel gemv_q4_0 must not allocate");

        let mut total_rel_err = 0.0f32;
        for r in 0..rows {
            let row_q4 = &w_q4[r * blocks_per_row..(r + 1) * blocks_per_row];
            let scalar_val = vec_dot_q4_0_q8_0_scalar(row_q4, &x_q8);
            let simd_rel_diff = (scalar_val - out_quant[r]).abs() / scalar_val.abs().max(1.0);
            assert!(
                simd_rel_diff < 1e-5,
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

    #[test]
    fn test_gemm_q4_0_against_naive_f32_and_zero_alloc() {
        let batch_size = 4;
        let rows = 32;
        let cols = 256;
        let blocks_per_row = cols / QK4_0;

        let w_f32: Vec<f32> = pseudo_random_f32(rows * cols, 0xABCD_1234)
            .into_iter()
            .map(|x| x + 0.5)
            .collect();
        let x_f32: Vec<f32> = pseudo_random_f32(batch_size * cols, 0x5678_EF90)
            .into_iter()
            .map(|x| x + 0.5)
            .collect();

        let mut w_q4 = vec![BlockQ4_0::zeroed(); rows * blocks_per_row];
        quantize_row_q4_0(&w_f32, &mut w_q4);

        let mut arena = AlignedArena::with_capacity(8192);
        let s_in = arena.alloc_slice(batch_size * cols);
        let s_q8 = arena.alloc_q8_scratch(batch_size * blocks_per_row);
        let s_out = arena.alloc_slice(batch_size * rows);
        arena.get_mut(s_in).copy_from_slice(&x_f32);

        let ops_before = current_thread_heap_ops();
        gemm_q4_0_arena(&mut arena, &w_q4, s_in, s_q8, s_out, batch_size, rows, cols);
        let ops_after = current_thread_heap_ops();
        assert_eq!(ops_after, ops_before, "gemm_q4_0_arena must perform 0 heap ops");

        let out = arena.get(s_out);
        let mut x_q8_ref = vec![BlockQ8_0::zeroed(); batch_size * blocks_per_row];
        quantize_row_q8_0(&x_f32, &mut x_q8_ref);

        for b in 0..batch_size {
            let b_q8 = &x_q8_ref[b * blocks_per_row..(b + 1) * blocks_per_row];
            for r in 0..rows {
                let row_w = &w_q4[r * blocks_per_row..(r + 1) * blocks_per_row];
                let expected = vec_dot_q4_0_q8_0_scalar(row_w, b_q8);
                let actual = out[b * rows + r];
                assert!(
                    (expected - actual).abs() < 1e-4,
                    "GEMM batch {b} row {r} mismatch: expected={expected}, actual={actual}"
                );
            }
        }
    }

    #[test]
    fn test_vec_dot_odd_block_counts_and_saturation() {
        use half::f16;

        for num_blocks in [1usize, 3, 7, 17] {
            let n = num_blocks * QK4_0;
            let w_f32 = pseudo_random_f32(n, 0x1000 + num_blocks as u32);
            let a_f32 = pseudo_random_f32(n, 0x2000 + num_blocks as u32);
            let mut w_q4 = vec![BlockQ4_0::zeroed(); num_blocks];
            let mut a_q8 = vec![BlockQ8_0::zeroed(); num_blocks];
            quantize_row_q4_0(&w_f32, &mut w_q4);
            quantize_row_q8_0(&a_f32, &mut a_q8);

            let s = vec_dot_q4_0_q8_0_scalar(&w_q4, &a_q8);
            let v = vec_dot_q4_0_q8_0(&w_q4, &a_q8);
            assert!(
                (s - v).abs() < 1e-5,
                "num_blocks={num_blocks} mismatch: scalar={s}, simd={v}"
            );
        }

        let d_one = f16::from_f32(1.0).to_bits();
        let w_min = [BlockQ4_0 { d: d_one, qs: [0x00; 16] }];
        let w_max = [BlockQ4_0 { d: d_one, qs: [0xFF; 16] }];
        let a_min = [BlockQ8_0 { d: d_one, qs: [-127; 32] }];
        let a_max = [BlockQ8_0 { d: d_one, qs: [127; 32] }];

        assert_eq!(vec_dot_q4_0_q8_0(&w_min, &a_min), 32_512.0);
        assert_eq!(vec_dot_q4_0_q8_0_scalar(&w_min, &a_min), 32_512.0);
        assert_eq!(vec_dot_q4_0_q8_0(&w_max, &a_max), 28_448.0);
        assert_eq!(vec_dot_q4_0_q8_0_scalar(&w_max, &a_max), 28_448.0);

        let a_i8_min = [BlockQ8_0 { d: d_one, qs: [-128; 32] }];
        assert_eq!(vec_dot_q4_0_q8_0(&w_min, &a_i8_min), 32_512.0);
    }
}
