use crate::jit::engine::{JitEngine, JitKernelFn};
use crate::jit::graph::FuseGraph;
use crate::memory::arena::{AlignedArena, ArenaSlice};
use crate::quant::blocks::{BlockQ4_0, QK4_0};
use crate::quant::kernels::gemv_q4_0_arena;

/// Flat concrete hyperparameters for a single Transformer block.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TransformerConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub max_seq_len: usize,
    pub rms_eps: f32,
}

impl TransformerConfig {
    #[inline(always)]
    pub const fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }
}

/// Zero-copy views of a single layer's quantized weights and `f32` RMSNorm scales.
#[derive(Copy, Clone, Debug)]
pub struct TransformerWeights<'a> {
    pub rms_att_weight: &'a [f32],
    pub wq: &'a [BlockQ4_0],
    pub wk: &'a [BlockQ4_0],
    pub wv: &'a [BlockQ4_0],
    pub wo: &'a [BlockQ4_0],
    pub rms_ffn_weight: &'a [f32],
    pub w_gate: &'a [BlockQ4_0],
    pub w_up: &'a [BlockQ4_0],
    pub w_down: &'a [BlockQ4_0],
}

/// Pre-allocated [`ArenaSlice`] handles for zero-allocation execution of a Transformer layer.
#[derive(Copy, Clone, Debug)]
pub struct LayerWorkspace {
    pub x: ArenaSlice,
    pub x_mid: ArenaSlice,
    pub xb: ArenaSlice,
    pub xb2: ArenaSlice,
    pub q: ArenaSlice,
    pub k_cache: ArenaSlice,
    pub v_cache: ArenaSlice,
    pub att: ArenaSlice,
    pub hb: ArenaSlice,
    pub hb2: ArenaSlice,
    pub hb_act: ArenaSlice,
    pub q8_scratch: ArenaSlice,
}

impl LayerWorkspace {
    /// Bump-allocates all intermediate activation, scratch, and KV-cache slices from `arena`.
    pub fn allocate(arena: &mut AlignedArena, cfg: &TransformerConfig) -> Self {
        assert_eq!(cfg.dim % QK4_0, 0, "dim must be a multiple of 32");
        assert_eq!(
            cfg.hidden_dim % QK4_0,
            0,
            "hidden_dim must be a multiple of 32"
        );
        assert_eq!(
            cfg.dim % cfg.num_heads,
            0,
            "dim must be divisible by num_heads"
        );
        assert_eq!(
            cfg.head_dim() % 2,
            0,
            "head_dim must be even for RoPE rotation"
        );

        let max_q8_blocks = cfg.dim.max(cfg.hidden_dim) / QK4_0;

        Self {
            x: arena.alloc_slice(cfg.dim),
            x_mid: arena.alloc_slice(cfg.dim),
            xb: arena.alloc_slice(cfg.dim),
            xb2: arena.alloc_slice(cfg.dim),
            q: arena.alloc_slice(cfg.dim),
            k_cache: arena.alloc_slice(cfg.max_seq_len * cfg.dim),
            v_cache: arena.alloc_slice(cfg.max_seq_len * cfg.dim),
            att: arena.alloc_slice(cfg.num_heads * cfg.max_seq_len),
            hb: arena.alloc_slice(cfg.hidden_dim),
            hb2: arena.alloc_slice(cfg.hidden_dim),
            hb_act: arena.alloc_slice(cfg.hidden_dim),
            q8_scratch: arena.alloc_q8_scratch(max_q8_blocks),
        }
    }
}

/// Applies Rotary Position Embedding (RoPE) in place to a multi-head `Q` or `K` vector at `pos`.
#[inline]
pub fn apply_rope_in_place(vec: &mut [f32], pos: usize, head_dim: usize) {
    let pos_f = pos as f32;
    let head_dim_f = head_dim as f32;
    for head_slice in vec.chunks_exact_mut(head_dim) {
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0f32 / 10000.0f32.powf((i as f32) / head_dim_f);
            let angle = pos_f * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let v0 = head_slice[i];
            let v1 = head_slice[i + 1];
            head_slice[i] = v0 * cos_a - v1 * sin_a;
            head_slice[i + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

/// Executes causal multi-head scaled dot-product attention over `0..=pos`.
pub fn causal_multi_head_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    att: &mut [f32],
    out: &mut [f32],
    cfg: &TransformerConfig,
    pos: usize,
) {
    let head_dim = cfg.head_dim();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let seq_len = pos + 1;

    out.fill(0.0);

    for h in 0..cfg.num_heads {
        let head_off = h * head_dim;
        let q_h = &q[head_off..head_off + head_dim];
        let att_h = &mut att[h * cfg.max_seq_len..h * cfg.max_seq_len + seq_len];

        let mut max_score = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let k_th = &k_cache[t * cfg.dim + head_off..t * cfg.dim + head_off + head_dim];
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += q_h[i] * k_th[i];
            }
            let score = dot * scale;
            att_h[t] = score;
            if score > max_score {
                max_score = score;
            }
        }

        let mut sum_exp = 0.0f32;
        for score in att_h.iter_mut() {
            let e = (*score - max_score).exp();
            *score = e;
            sum_exp += e;
        }
        let inv_sum = 1.0f32 / sum_exp;
        for score in att_h.iter_mut() {
            *score *= inv_sum;
        }

        let out_h = &mut out[head_off..head_off + head_dim];
        for (t, &prob) in att_h.iter().enumerate() {
            let v_th = &v_cache[t * cfg.dim + head_off..t * cfg.dim + head_off + head_dim];
            for i in 0..head_dim {
                out_h[i] += prob * v_th[i];
            }
        }
    }
}

/// Self-contained Transformer layer wiring quantized GEMV projections and JIT-fused CLIR kernels
/// on [`AlignedArena`].
pub struct TransformerLayer {
    pub config: TransformerConfig,
    pub workspace: LayerWorkspace,
    pub rmsnorm_fn: JitKernelFn,
    pub swiglu_fn: JitKernelFn,
    pub residual_add_fn: JitKernelFn,
}

impl TransformerLayer {
    /// Compiles (or fetches cached) JIT kernels and binds the layer to `workspace`.
    pub fn new(
        engine: &mut JitEngine,
        config: TransformerConfig,
        workspace: LayerWorkspace,
    ) -> Self {
        let rmsnorm_fn = engine.compile(&FuseGraph::rmsnorm(config.dim, config.rms_eps));
        let swiglu_fn = engine.compile(&FuseGraph::swiglu());
        let residual_add_fn = engine.compile(&FuseGraph::residual_add());

        Self {
            config,
            workspace,
            rmsnorm_fn,
            swiglu_fn,
            residual_add_fn,
        }
    }

    /// Executes one full Transformer block (`Attention + SwiGLU FFN`) at token position `pos`
    /// with zero heap allocations. Reads and updates `self.workspace.x` in place inside `arena`.
    pub fn forward_token(
        &self,
        engine: &JitEngine,
        arena: &mut AlignedArena,
        weights: &TransformerWeights<'_>,
        pos: usize,
    ) {
        let dim = self.config.dim;
        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim();
        assert!(pos < self.config.max_seq_len, "sequence position out of bounds");

        let ws = &self.workspace;

        // 1. Pre-Attention JIT RMSNorm: xb = rmsnorm(x, rms_att_weight)
        engine.exec_with_weights(self.rmsnorm_fn, arena, ws.x, weights.rms_att_weight, ws.xb);

        // 2. Quantized QKV Projections directly into q, k_cache[pos], v_cache[pos]
        let k_pos = ws.k_cache.subslice(pos * dim, dim);
        let v_pos = ws.v_cache.subslice(pos * dim, dim);
        gemv_q4_0_arena(arena, weights.wq, ws.xb, ws.q8_scratch, ws.q, dim, dim);
        gemv_q4_0_arena(arena, weights.wk, ws.xb, ws.q8_scratch, k_pos, dim, dim);
        gemv_q4_0_arena(arena, weights.wv, ws.xb, ws.q8_scratch, v_pos, dim, dim);

        // 3. Apply RoPE in place to Q and K[pos]
        apply_rope_in_place(arena.get_mut(ws.q), pos, head_dim);
        apply_rope_in_place(arena.get_mut(k_pos), pos, head_dim);

        // 4. Causal Multi-Head Attention over 0..=pos -> xb
        {
            let (q_slice, kc_slice, vc_slice, att_slice, xb_slice) =
                arena.attention_slices(ws.q, ws.k_cache, ws.v_cache, ws.att, ws.xb);
            causal_multi_head_attention(
                q_slice,
                kc_slice,
                vc_slice,
                att_slice,
                xb_slice,
                &self.config,
                pos,
            );
        }

        // 5. Output Projection: xb2 = Wo * xb, then JIT Residual Add: x_mid = x + xb2
        gemv_q4_0_arena(arena, weights.wo, ws.xb, ws.q8_scratch, ws.xb2, dim, dim);
        engine.exec_binary(self.residual_add_fn, arena, ws.x, ws.xb2, ws.x_mid);

        // 6. Pre-FFN JIT RMSNorm: xb = rmsnorm(x_mid, rms_ffn_weight)
        engine.exec_with_weights(
            self.rmsnorm_fn,
            arena,
            ws.x_mid,
            weights.rms_ffn_weight,
            ws.xb,
        );

        // 7. FFN Gate & Up Projections: hb = W_gate * xb, hb2 = W_up * xb
        gemv_q4_0_arena(
            arena,
            weights.w_gate,
            ws.xb,
            ws.q8_scratch,
            ws.hb,
            hidden_dim,
            dim,
        );
        gemv_q4_0_arena(
            arena,
            weights.w_up,
            ws.xb,
            ws.q8_scratch,
            ws.hb2,
            hidden_dim,
            dim,
        );

        // 8. Fused JIT SwiGLU: hb_act = silu(hb) * hb2
        engine.exec_binary(self.swiglu_fn, arena, ws.hb, ws.hb2, ws.hb_act);

        // 9. FFN Down Projection: xb = W_down * hb_act, then JIT Residual Add: x = x_mid + xb
        gemv_q4_0_arena(
            arena,
            weights.w_down,
            ws.hb_act,
            ws.q8_scratch,
            ws.xb,
            dim,
            hidden_dim,
        );
        engine.exec_binary(self.residual_add_fn, arena, ws.x_mid, ws.xb, ws.x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::arena::tests::current_thread_allocs;
    use crate::quant::blocks::{BlockQ8_0, QK8_0, quantize_row_q4_0, quantize_row_q8_0};
    use crate::quant::kernels::vec_dot_q4_0_q8_0_scalar;
    use bytemuck::Zeroable;

    fn pseudo_random_f32(len: usize, mut seed: u32) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let unit = (seed as f32) / (u32::MAX as f32);
            out.push((unit * 2.0 - 1.0) * 0.25);
        }
        out
    }

    fn quantize_matrix(rows: usize, cols: usize, seed: u32) -> Vec<BlockQ4_0> {
        let f32_data = pseudo_random_f32(rows * cols, seed);
        let mut blocks = vec![BlockQ4_0::zeroed(); rows * (cols / QK4_0)];
        quantize_row_q4_0(&f32_data, &mut blocks);
        blocks
    }

    fn scalar_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = 1.0 / (sum_sq / (x.len() as f32) + eps).sqrt();
        x.iter().zip(weight.iter()).map(|(&xi, &wi)| xi * rms * wi).collect()
    }

    fn scalar_gemv(w: &[BlockQ4_0], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let blocks_per_row = cols / QK4_0;
        let mut x_q8 = vec![BlockQ8_0::zeroed(); cols / QK8_0];
        quantize_row_q8_0(x, &mut x_q8);
        let mut out = vec![0.0f32; rows];
        for r in 0..rows {
            let row_w = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
            out[r] = vec_dot_q4_0_q8_0_scalar(row_w, &x_q8);
        }
        out
    }

    #[test]
    fn test_transformer_layer_matches_scalar_reference() {
        let cfg = TransformerConfig {
            dim: 64,
            hidden_dim: 128,
            num_heads: 4,
            max_seq_len: 8,
            rms_eps: 1e-5,
        };

        let rms_att_w = pseudo_random_f32(cfg.dim, 0x1001);
        let rms_ffn_w = pseudo_random_f32(cfg.dim, 0x1002);
        let wq = quantize_matrix(cfg.dim, cfg.dim, 0x2001);
        let wk = quantize_matrix(cfg.dim, cfg.dim, 0x2002);
        let wv = quantize_matrix(cfg.dim, cfg.dim, 0x2003);
        let wo = quantize_matrix(cfg.dim, cfg.dim, 0x2004);
        let w_gate = quantize_matrix(cfg.hidden_dim, cfg.dim, 0x3001);
        let w_up = quantize_matrix(cfg.hidden_dim, cfg.dim, 0x3002);
        let w_down = quantize_matrix(cfg.dim, cfg.hidden_dim, 0x3003);

        let weights = TransformerWeights {
            rms_att_weight: &rms_att_w,
            wq: &wq,
            wk: &wk,
            wv: &wv,
            wo: &wo,
            rms_ffn_weight: &rms_ffn_w,
            w_gate: &w_gate,
            w_up: &w_up,
            w_down: &w_down,
        };

        let mut engine = JitEngine::new();
        let mut arena = AlignedArena::with_capacity(8192);
        let ws = LayerWorkspace::allocate(&mut arena, &cfg);
        let layer = TransformerLayer::new(&mut engine, cfg, ws);

        let init_x = pseudo_random_f32(cfg.dim, 0x9999);
        arena.get_mut(layer.workspace.x).copy_from_slice(&init_x);

        let mut ref_x = init_x;
        let mut ref_k_cache = vec![0.0f32; cfg.max_seq_len * cfg.dim];
        let mut ref_v_cache = vec![0.0f32; cfg.max_seq_len * cfg.dim];
        let mut ref_att = vec![0.0f32; cfg.num_heads * cfg.max_seq_len];

        for pos in 0..4 {
            layer.forward_token(&engine, &mut arena, &weights, pos);

            // Scalar reference step
            let xb = scalar_rmsnorm(&ref_x, &rms_att_w, cfg.rms_eps);
            let mut q = scalar_gemv(&wq, &xb, cfg.dim, cfg.dim);
            let mut k = scalar_gemv(&wk, &xb, cfg.dim, cfg.dim);
            let v = scalar_gemv(&wv, &xb, cfg.dim, cfg.dim);

            apply_rope_in_place(&mut q, pos, cfg.head_dim());
            apply_rope_in_place(&mut k, pos, cfg.head_dim());

            ref_k_cache[pos * cfg.dim..(pos + 1) * cfg.dim].copy_from_slice(&k);
            ref_v_cache[pos * cfg.dim..(pos + 1) * cfg.dim].copy_from_slice(&v);

            let mut att_out = vec![0.0f32; cfg.dim];
            causal_multi_head_attention(
                &q,
                &ref_k_cache,
                &ref_v_cache,
                &mut ref_att,
                &mut att_out,
                &cfg,
                pos,
            );

            let wo_out = scalar_gemv(&wo, &att_out, cfg.dim, cfg.dim);
            let x_mid: Vec<f32> = ref_x.iter().zip(wo_out.iter()).map(|(a, b)| a + b).collect();

            let ffn_in = scalar_rmsnorm(&x_mid, &rms_ffn_w, cfg.rms_eps);
            let gate = scalar_gemv(&w_gate, &ffn_in, cfg.hidden_dim, cfg.dim);
            let up = scalar_gemv(&w_up, &ffn_in, cfg.hidden_dim, cfg.dim);
            let swiglu: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let down = scalar_gemv(&w_down, &swiglu, cfg.dim, cfg.hidden_dim);
            ref_x = x_mid.iter().zip(down.iter()).map(|(a, b)| a + b).collect();

            let actual_x = arena.get(layer.workspace.x);
            for i in 0..cfg.dim {
                let diff = (actual_x[i] - ref_x[i]).abs();
                assert!(
                    diff < 1e-4,
                    "pos {pos} dim {i} mismatch: actual={}, ref={}, diff={diff}",
                    actual_x[i],
                    ref_x[i]
                );
            }
        }
    }

    #[test]
    fn test_transformer_layer_zero_heap_allocations() {
        let cfg = TransformerConfig {
            dim: 128,
            hidden_dim: 256,
            num_heads: 4,
            max_seq_len: 16,
            rms_eps: 1e-5,
        };

        let rms_att_w = pseudo_random_f32(cfg.dim, 0x4001);
        let rms_ffn_w = pseudo_random_f32(cfg.dim, 0x4002);
        let wq = quantize_matrix(cfg.dim, cfg.dim, 0x5001);
        let wk = quantize_matrix(cfg.dim, cfg.dim, 0x5002);
        let wv = quantize_matrix(cfg.dim, cfg.dim, 0x5003);
        let wo = quantize_matrix(cfg.dim, cfg.dim, 0x5004);
        let w_gate = quantize_matrix(cfg.hidden_dim, cfg.dim, 0x6001);
        let w_up = quantize_matrix(cfg.hidden_dim, cfg.dim, 0x6002);
        let w_down = quantize_matrix(cfg.dim, cfg.hidden_dim, 0x6003);

        let weights = TransformerWeights {
            rms_att_weight: &rms_att_w,
            wq: &wq,
            wk: &wk,
            wv: &wv,
            wo: &wo,
            rms_ffn_weight: &rms_ffn_w,
            w_gate: &w_gate,
            w_up: &w_up,
            w_down: &w_down,
        };

        let mut engine = JitEngine::new();
        let mut arena = AlignedArena::with_capacity(32768);
        let ws = LayerWorkspace::allocate(&mut arena, &cfg);
        let layer = TransformerLayer::new(&mut engine, cfg, ws);

        let init_x = pseudo_random_f32(cfg.dim, 0x7777);
        arena.get_mut(layer.workspace.x).copy_from_slice(&init_x);

        let allocs_before = current_thread_allocs();
        for pos in 0..cfg.max_seq_len {
            layer.forward_token(&engine, &mut arena, &weights, pos);
        }
        let allocs_after = current_thread_allocs();

        assert_eq!(
            allocs_after, allocs_before,
            "Expected 0 heap allocations during Transformer token generation loop, found {}",
            allocs_after - allocs_before
        );
    }
}
