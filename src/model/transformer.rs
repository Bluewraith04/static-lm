use crate::jit::engine::{JitEngine, JitKernelFn};
use crate::jit::graph::FuseGraph;
use crate::memory::arena::{AlignedArena, ArenaSlice};
use crate::quant::blocks::{BlockQ4_0, QK4_0, dequantize_row_q4_0};
use crate::quant::kernels::{gemm_q4_0_arena, gemv_q4_0_arena};

/// Flat concrete hyperparameters for a Transformer block (supports both MHA and GQA).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TransformerConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub max_seq_len: usize,
    pub rms_eps: f32,
}

impl TransformerConfig {
    #[inline(always)]
    pub const fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }

    #[inline(always)]
    pub const fn kv_dim(&self) -> usize {
        (self.dim / self.num_heads) * self.num_kv_heads
    }

    #[inline(always)]
    pub const fn kv_mul(&self) -> usize {
        self.num_heads / self.num_kv_heads
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

/// Zero-copy views of a complete Transformer model (`token_embd`, layers, `output_norm`, `output`).
#[derive(Clone, Debug)]
pub struct ModelWeights<'a> {
    pub token_embd: &'a [BlockQ4_0],
    pub layers: Vec<TransformerWeights<'a>>,
    pub output_norm: &'a [f32],
    pub lm_head: &'a [BlockQ4_0],
    pub vocab_size: usize,
}

/// Pre-allocated [`ArenaSlice`] handles for zero-allocation execution of a single-token Transformer step.
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
    pub rope_cos: ArenaSlice,
    pub rope_sin: ArenaSlice,
}

impl LayerWorkspace {
    /// Bump-allocates all intermediate activation, scratch, KV-cache, and precomputed RoPE tables from `arena`.
    pub fn allocate(arena: &mut AlignedArena, cfg: &TransformerConfig) -> Self {
        assert_eq!(cfg.dim % QK4_0, 0, "dim must be a multiple of 32");
        assert_eq!(
            cfg.hidden_dim % QK4_0,
            0,
            "hidden_dim must be a multiple of 32"
        );
        assert!(cfg.num_heads > 0 && cfg.num_kv_heads > 0);
        assert_eq!(
            cfg.dim % cfg.num_heads,
            0,
            "dim must be divisible by num_heads"
        );
        assert_eq!(
            cfg.num_heads % cfg.num_kv_heads,
            0,
            "num_heads must be divisible by num_kv_heads"
        );
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.kv_dim();
        assert_eq!(head_dim % 2, 0, "head_dim must be even for RoPE rotation");
        assert_eq!(kv_dim % QK4_0, 0, "kv_dim must be a multiple of 32");

        let max_q8_blocks = cfg.dim.max(cfg.hidden_dim) / QK4_0;
        let half_head = head_dim / 2;
        let rope_table_len = cfg.max_seq_len * half_head;

        let ws = Self {
            x: arena.alloc_slice(cfg.dim),
            x_mid: arena.alloc_slice(cfg.dim),
            xb: arena.alloc_slice(cfg.dim),
            xb2: arena.alloc_slice(cfg.dim),
            q: arena.alloc_slice(cfg.dim),
            k_cache: arena.alloc_slice(cfg.max_seq_len * kv_dim),
            v_cache: arena.alloc_slice(cfg.max_seq_len * kv_dim),
            att: arena.alloc_slice(cfg.num_heads * cfg.max_seq_len),
            hb: arena.alloc_slice(cfg.hidden_dim),
            hb2: arena.alloc_slice(cfg.hidden_dim),
            hb_act: arena.alloc_slice(cfg.hidden_dim),
            q8_scratch: arena.alloc_q8_scratch(max_q8_blocks),
            rope_cos: arena.alloc_slice(rope_table_len),
            rope_sin: arena.alloc_slice(rope_table_len),
        };

        let head_dim_f = head_dim as f32;
        for pos in 0..cfg.max_seq_len {
            let pos_f = pos as f32;
            for j in 0..half_head {
                let freq = 1.0f32 / 10000.0f32.powf(((2 * j) as f32) / head_dim_f);
                let (sin_a, cos_a) = (pos_f * freq).sin_cos();
                arena.get_mut(ws.rope_cos)[pos * half_head + j] = cos_a;
                arena.get_mut(ws.rope_sin)[pos * half_head + j] = sin_a;
            }
        }

        ws
    }
}

/// Pre-allocated [`ArenaSlice`] handles for batched prompt prefill via [`gemm_q4_0_arena`].
#[derive(Copy, Clone, Debug)]
pub struct BatchWorkspace {
    pub max_batch_size: usize,
    pub x_batch: ArenaSlice,
    pub x_mid_batch: ArenaSlice,
    pub xb_batch: ArenaSlice,
    pub xb2_batch: ArenaSlice,
    pub q_batch: ArenaSlice,
    pub hb_batch: ArenaSlice,
    pub hb2_batch: ArenaSlice,
    pub hb_act_batch: ArenaSlice,
    pub q8_scratch: ArenaSlice,
}

impl BatchWorkspace {
    /// Bump-allocates batched activation and `Q8_0` scratch buffers for up to `max_batch_size` tokens.
    pub fn allocate(
        arena: &mut AlignedArena,
        cfg: &TransformerConfig,
        max_batch_size: usize,
    ) -> Self {
        let max_b = max_batch_size.max(1);
        let max_q8_blocks = max_b * (cfg.dim.max(cfg.hidden_dim) / QK4_0);
        Self {
            max_batch_size: max_b,
            x_batch: arena.alloc_slice(max_b * cfg.dim),
            x_mid_batch: arena.alloc_slice(max_b * cfg.dim),
            xb_batch: arena.alloc_slice(max_b * cfg.dim),
            xb2_batch: arena.alloc_slice(max_b * cfg.dim),
            q_batch: arena.alloc_slice(max_b * cfg.dim),
            hb_batch: arena.alloc_slice(max_b * cfg.hidden_dim),
            hb2_batch: arena.alloc_slice(max_b * cfg.hidden_dim),
            hb_act_batch: arena.alloc_slice(max_b * cfg.hidden_dim),
            q8_scratch: arena.alloc_q8_scratch(max_q8_blocks),
        }
    }
}

/// Applies Rotary Position Embedding (RoPE) in place using precomputed `cos`/`sin` tables for `pos`.
#[inline]
pub fn apply_rope_cached(vec: &mut [f32], cos_row: &[f32], sin_row: &[f32], head_dim: usize) {
    let half_head = head_dim / 2;
    debug_assert_eq!(cos_row.len(), half_head);
    debug_assert_eq!(sin_row.len(), half_head);
    for head_slice in vec.chunks_exact_mut(head_dim) {
        for j in 0..half_head {
            let cos_a = cos_row[j];
            let sin_a = sin_row[j];
            let v0 = head_slice[2 * j];
            let v1 = head_slice[2 * j + 1];
            head_slice[2 * j] = v0 * cos_a - v1 * sin_a;
            head_slice[2 * j + 1] = v0 * sin_a + v1 * cos_a;
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

/// Executes causal Multi-Head / Grouped-Query scaled dot-product attention over `0..=pos` (scalar reference).
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
    let kv_dim = cfg.kv_dim();
    let kv_mul = cfg.kv_mul();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let seq_len = pos + 1;

    out.fill(0.0);

    for h in 0..cfg.num_heads {
        let q_head_off = h * head_dim;
        let kv_head_off = (h / kv_mul) * head_dim;
        let q_h = &q[q_head_off..q_head_off + head_dim];
        let att_h = &mut att[h * cfg.max_seq_len..h * cfg.max_seq_len + seq_len];

        let mut max_score = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let k_th = &k_cache[t * kv_dim + kv_head_off..t * kv_dim + kv_head_off + head_dim];
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

        let out_h = &mut out[q_head_off..q_head_off + head_dim];
        for (t, &prob) in att_h.iter().enumerate() {
            let v_th = &v_cache[t * kv_dim + kv_head_off..t * kv_dim + kv_head_off + head_dim];
            for i in 0..head_dim {
                out_h[i] += prob * v_th[i];
            }
        }
    }
}

/// Executes causal Multi-Head / Grouped-Query scaled dot-product attention over `0..=pos`
/// using the JIT-compiled `Softmax` kernel.
pub fn causal_multi_head_attention_jit(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    att: &mut [f32],
    out: &mut [f32],
    cfg: &TransformerConfig,
    pos: usize,
    softmax_fn: JitKernelFn,
) {
    let head_dim = cfg.head_dim();
    let kv_dim = cfg.kv_dim();
    let kv_mul = cfg.kv_mul();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let seq_len = pos + 1;

    out.fill(0.0);

    for h in 0..cfg.num_heads {
        let q_head_off = h * head_dim;
        let kv_head_off = (h / kv_mul) * head_dim;
        let q_h = &q[q_head_off..q_head_off + head_dim];
        let att_h = &mut att[h * cfg.max_seq_len..h * cfg.max_seq_len + seq_len];

        for t in 0..seq_len {
            let k_th = &k_cache[t * kv_dim + kv_head_off..t * kv_dim + kv_head_off + head_dim];
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += q_h[i] * k_th[i];
            }
            att_h[t] = dot * scale;
        }

        let inputs = [att_h.as_ptr()];
        unsafe {
            softmax_fn(inputs.as_ptr(), att_h.as_mut_ptr(), seq_len);
        }

        let out_h = &mut out[q_head_off..q_head_off + head_dim];
        for (t, &prob) in att_h.iter().enumerate() {
            let v_th = &v_cache[t * kv_dim + kv_head_off..t * kv_dim + kv_head_off + head_dim];
            for i in 0..head_dim {
                out_h[i] += prob * v_th[i];
            }
        }
    }
}

/// Dedicated per-layer KV-cache slices inside [`AlignedArena`].
#[derive(Copy, Clone, Debug)]
pub struct LayerKvCache {
    pub k_cache: ArenaSlice,
    pub v_cache: ArenaSlice,
}

impl LayerKvCache {
    pub fn allocate(arena: &mut AlignedArena, cfg: &TransformerConfig) -> Self {
        let kv_dim = cfg.kv_dim();
        Self {
            k_cache: arena.alloc_slice(cfg.max_seq_len * kv_dim),
            v_cache: arena.alloc_slice(cfg.max_seq_len * kv_dim),
        }
    }
}

/// Self-contained Transformer layer wiring quantized GEMV/GEMM projections and JIT-fused CLIR kernels
/// on [`AlignedArena`].
pub struct TransformerLayer {
    pub config: TransformerConfig,
    pub workspace: LayerWorkspace,
    pub rmsnorm_fn: JitKernelFn,
    pub swiglu_fn: JitKernelFn,
    pub residual_add_fn: JitKernelFn,
    pub softmax_fn: JitKernelFn,
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
        let softmax_fn = engine.compile(&FuseGraph::softmax());

        Self {
            config,
            workspace,
            rmsnorm_fn,
            swiglu_fn,
            residual_add_fn,
            softmax_fn,
        }
    }

    /// Executes one full Transformer block (`Attention + SwiGLU FFN`) at token position `pos`
    /// using the layer's default workspace KV-cache.
    pub fn forward_token(
        &self,
        engine: &JitEngine,
        arena: &mut AlignedArena,
        weights: &TransformerWeights<'_>,
        pos: usize,
    ) {
        let kv = LayerKvCache {
            k_cache: self.workspace.k_cache,
            v_cache: self.workspace.v_cache,
        };
        self.forward_token_with_kv(engine, arena, weights, &kv, pos);
    }

    /// Executes one full Transformer block (`Attention + SwiGLU FFN`) at token position `pos`
    /// using an explicit per-layer [`LayerKvCache`] with zero heap allocations.
    pub fn forward_token_with_kv(
        &self,
        engine: &JitEngine,
        arena: &mut AlignedArena,
        weights: &TransformerWeights<'_>,
        kv: &LayerKvCache,
        pos: usize,
    ) {
        let dim = self.config.dim;
        let kv_dim = self.config.kv_dim();
        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim();
        let half_head = head_dim / 2;
        assert!(
            pos < self.config.max_seq_len,
            "sequence position out of bounds"
        );

        let ws = &self.workspace;

        engine.exec_with_weights(self.rmsnorm_fn, arena, ws.x, weights.rms_att_weight, ws.xb);

        let k_pos = kv.k_cache.subslice(pos * kv_dim, kv_dim);
        let v_pos = kv.v_cache.subslice(pos * kv_dim, kv_dim);
        gemv_q4_0_arena(arena, weights.wq, ws.xb, ws.q8_scratch, ws.q, dim, dim);
        gemv_q4_0_arena(arena, weights.wk, ws.xb, ws.q8_scratch, k_pos, kv_dim, dim);
        gemv_q4_0_arena(arena, weights.wv, ws.xb, ws.q8_scratch, v_pos, kv_dim, dim);

        let cos_pos = ws.rope_cos.subslice(pos * half_head, half_head);
        let sin_pos = ws.rope_sin.subslice(pos * half_head, half_head);
        {
            let (cos_s, sin_s, q_s) = arena.io2_slices(cos_pos, sin_pos, ws.q);
            apply_rope_cached(q_s, cos_s, sin_s, head_dim);
        }
        {
            let (cos_s, sin_s, k_s) = arena.io2_slices(cos_pos, sin_pos, k_pos);
            apply_rope_cached(k_s, cos_s, sin_s, head_dim);
        }

        {
            let (q_slice, kc_slice, vc_slice, att_slice, xb_slice) =
                arena.attention_slices(ws.q, kv.k_cache, kv.v_cache, ws.att, ws.xb);
            causal_multi_head_attention_jit(
                q_slice,
                kc_slice,
                vc_slice,
                att_slice,
                xb_slice,
                &self.config,
                pos,
                self.softmax_fn,
            );
        }

        gemv_q4_0_arena(arena, weights.wo, ws.xb, ws.q8_scratch, ws.xb2, dim, dim);
        engine.exec_binary(self.residual_add_fn, arena, ws.x, ws.xb2, ws.x_mid);

        engine.exec_with_weights(
            self.rmsnorm_fn,
            arena,
            ws.x_mid,
            weights.rms_ffn_weight,
            ws.xb,
        );

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

        engine.exec_binary(self.swiglu_fn, arena, ws.hb, ws.hb2, ws.hb_act);

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

    /// Executes a batched Transformer block over `batch_size` tokens starting at `start_pos`
    /// using [`gemm_q4_0_arena`] and batched JIT kernels with zero heap allocations.
    pub fn forward_batch_with_kv(
        &self,
        engine: &JitEngine,
        arena: &mut AlignedArena,
        bws: &BatchWorkspace,
        weights: &TransformerWeights<'_>,
        kv: &LayerKvCache,
        start_pos: usize,
        batch_size: usize,
    ) {
        assert!(batch_size > 0 && batch_size <= bws.max_batch_size);
        assert!(start_pos + batch_size <= self.config.max_seq_len);

        let dim = self.config.dim;
        let kv_dim = self.config.kv_dim();
        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim();
        let half_head = head_dim / 2;
        let ws = &self.workspace;

        let x_b = bws.x_batch.subslice(0, batch_size * dim);
        let x_mid_b = bws.x_mid_batch.subslice(0, batch_size * dim);
        let xb_b = bws.xb_batch.subslice(0, batch_size * dim);
        let xb2_b = bws.xb2_batch.subslice(0, batch_size * dim);
        let q_b = bws.q_batch.subslice(0, batch_size * dim);
        let hb_b = bws.hb_batch.subslice(0, batch_size * hidden_dim);
        let hb2_b = bws.hb2_batch.subslice(0, batch_size * hidden_dim);
        let hb_act_b = bws.hb_act_batch.subslice(0, batch_size * hidden_dim);

        for b in 0..batch_size {
            engine.exec_with_weights(
                self.rmsnorm_fn,
                arena,
                x_b.subslice(b * dim, dim),
                weights.rms_att_weight,
                xb_b.subslice(b * dim, dim),
            );
        }

        let k_batch = kv.k_cache.subslice(start_pos * kv_dim, batch_size * kv_dim);
        let v_batch = kv.v_cache.subslice(start_pos * kv_dim, batch_size * kv_dim);
        gemm_q4_0_arena(
            arena,
            weights.wq,
            xb_b,
            bws.q8_scratch,
            q_b,
            batch_size,
            dim,
            dim,
        );
        gemm_q4_0_arena(
            arena,
            weights.wk,
            xb_b,
            bws.q8_scratch,
            k_batch,
            batch_size,
            kv_dim,
            dim,
        );
        gemm_q4_0_arena(
            arena,
            weights.wv,
            xb_b,
            bws.q8_scratch,
            v_batch,
            batch_size,
            kv_dim,
            dim,
        );

        for b in 0..batch_size {
            let pos = start_pos + b;
            let q_pos = q_b.subslice(b * dim, dim);
            let k_pos = kv.k_cache.subslice(pos * kv_dim, kv_dim);
            let xb_pos = xb_b.subslice(b * dim, dim);
            let cos_pos = ws.rope_cos.subslice(pos * half_head, half_head);
            let sin_pos = ws.rope_sin.subslice(pos * half_head, half_head);

            {
                let (cos_s, sin_s, q_s) = arena.io2_slices(cos_pos, sin_pos, q_pos);
                apply_rope_cached(q_s, cos_s, sin_s, head_dim);
            }
            {
                let (cos_s, sin_s, k_s) = arena.io2_slices(cos_pos, sin_pos, k_pos);
                apply_rope_cached(k_s, cos_s, sin_s, head_dim);
            }
            {
                let (q_slice, kc_slice, vc_slice, att_slice, xb_slice) =
                    arena.attention_slices(q_pos, kv.k_cache, kv.v_cache, ws.att, xb_pos);
                causal_multi_head_attention_jit(
                    q_slice,
                    kc_slice,
                    vc_slice,
                    att_slice,
                    xb_slice,
                    &self.config,
                    pos,
                    self.softmax_fn,
                );
            }
        }

        gemm_q4_0_arena(
            arena,
            weights.wo,
            xb_b,
            bws.q8_scratch,
            xb2_b,
            batch_size,
            dim,
            dim,
        );
        engine.exec_binary(self.residual_add_fn, arena, x_b, xb2_b, x_mid_b);

        for b in 0..batch_size {
            engine.exec_with_weights(
                self.rmsnorm_fn,
                arena,
                x_mid_b.subslice(b * dim, dim),
                weights.rms_ffn_weight,
                xb_b.subslice(b * dim, dim),
            );
        }

        gemm_q4_0_arena(
            arena,
            weights.w_gate,
            xb_b,
            bws.q8_scratch,
            hb_b,
            batch_size,
            hidden_dim,
            dim,
        );
        gemm_q4_0_arena(
            arena,
            weights.w_up,
            xb_b,
            bws.q8_scratch,
            hb2_b,
            batch_size,
            hidden_dim,
            dim,
        );

        engine.exec_binary(self.swiglu_fn, arena, hb_b, hb2_b, hb_act_b);

        gemm_q4_0_arena(
            arena,
            weights.w_down,
            hb_act_b,
            bws.q8_scratch,
            xb_b,
            batch_size,
            dim,
            hidden_dim,
        );
        engine.exec_binary(self.residual_add_fn, arena, x_mid_b, xb_b, x_b);
    }
}

/// Returns the index of the maximum logit value (greedy argmax token selection).
#[inline]
pub fn sample_argmax(logits: &[f32]) -> usize {
    assert!(!logits.is_empty(), "logits cannot be empty");
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (i, &val) in logits.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

/// Computes the cosine similarity ($\mathbf{a} \cdot \mathbf{b}$) between two L2-normalized embedding slices.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "embedding dimension mismatch");
    let mut dot = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
    }
    dot
}

/// Ranks `num_docs` L2-normalized document embeddings (`doc_embs` of length `num_docs * dim`)
/// against `query_emb` in descending cosine similarity order with zero heap allocations.
pub fn rank_documents_by_cosine(
    query_emb: &[f32],
    doc_embs: &[f32],
    dim: usize,
    scores_out: &mut [(usize, f32)],
) {
    assert_eq!(query_emb.len(), dim);
    let num_docs = scores_out.len();
    assert_eq!(doc_embs.len(), num_docs * dim);

    for (doc_idx, doc_vec) in doc_embs.chunks_exact(dim).enumerate() {
        scores_out[doc_idx] = (doc_idx, cosine_similarity(query_emb, doc_vec));
    }
    scores_out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
}

/// End-to-end zero-allocation Transformer model executing `token_embd` lookup, `L` Transformer layers,
/// batched `GEMM` prompt prefill, greedy token generation, and JIT L2-normalized embedding extraction.
pub struct TransformerModel {
    pub layer: TransformerLayer,
    pub batch_workspace: BatchWorkspace,
    pub kv_caches: Vec<LayerKvCache>,
    pub logits: ArenaSlice,
    pub embed_accum: ArenaSlice,
    pub embed_norm: ArenaSlice,
    pub l2_norm_fn: JitKernelFn,
    pub vocab_size: usize,
}

impl TransformerModel {
    /// Allocates single-token and batch workspaces, per-layer KV-caches, logits, and embedding slices
    /// in `arena`, and compiles all JIT kernels.
    pub fn new(
        engine: &mut JitEngine,
        arena: &mut AlignedArena,
        config: TransformerConfig,
        num_layers: usize,
        vocab_size: usize,
    ) -> Self {
        assert!(vocab_size > 0, "vocab_size must be positive");
        let ws = LayerWorkspace::allocate(arena, &config);
        let batch_workspace = BatchWorkspace::allocate(arena, &config, config.max_seq_len);
        let kv_caches = (0..num_layers)
            .map(|_| LayerKvCache::allocate(arena, &config))
            .collect();
        let logits = arena.alloc_slice(vocab_size);
        let embed_accum = arena.alloc_slice(config.dim);
        let embed_norm = arena.alloc_slice(config.dim);
        let l2_norm_fn = engine.compile(&FuseGraph::l2_norm(config.rms_eps));
        let layer = TransformerLayer::new(engine, config, ws);

        Self {
            layer,
            batch_workspace,
            kv_caches,
            logits,
            embed_accum,
            embed_norm,
            l2_norm_fn,
            vocab_size,
        }
    }

    /// Executes a complete token-to-token forward pass for `token_id` at sequence position `pos`,
    /// returning the greedy argmax next-token ID with zero heap allocations.
    pub fn forward_token(
        &self,
        engine: &JitEngine,
        arena: &mut AlignedArena,
        weights: &ModelWeights<'_>,
        token_id: usize,
        pos: usize,
    ) -> usize {
        let dim = self.layer.config.dim;
        let blocks_per_row = dim / QK4_0;
        assert!(
            token_id < self.vocab_size,
            "token_id out of vocabulary bounds"
        );
        assert_eq!(weights.vocab_size, self.vocab_size, "vocab_size mismatch");
        assert_eq!(
            weights.layers.len(),
            self.kv_caches.len(),
            "layer count mismatch"
        );

        let embd_row =
            &weights.token_embd[token_id * blocks_per_row..(token_id + 1) * blocks_per_row];
        dequantize_row_q4_0(embd_row, arena.get_mut(self.layer.workspace.x));

        for (lw, kv) in weights.layers.iter().zip(self.kv_caches.iter()) {
            self.layer.forward_token_with_kv(engine, arena, lw, kv, pos);
        }

        engine.exec_with_weights(
            self.layer.rmsnorm_fn,
            arena,
            self.layer.workspace.x,
            weights.output_norm,
            self.layer.workspace.xb,
        );

        gemv_q4_0_arena(
            arena,
            weights.lm_head,
            self.layer.workspace.xb,
            self.layer.workspace.q8_scratch,
            self.logits,
            self.vocab_size,
            dim,
        );

        sample_argmax(arena.get(self.logits))
    }

    /// Executes batched prompt prefill over `prompt_tokens` starting at `start_pos` using [`gemm_q4_0_arena`],
    /// returning the greedy argmax next-token ID following the prompt with zero heap allocations.
    pub fn prefill_tokens(
        &self,
        engine: &JitEngine,
        arena: &mut AlignedArena,
        weights: &ModelWeights<'_>,
        prompt_tokens: &[usize],
        start_pos: usize,
    ) -> usize {
        let batch_size = prompt_tokens.len();
        assert!(batch_size > 0, "prompt_tokens cannot be empty");
        let dim = self.layer.config.dim;
        let blocks_per_row = dim / QK4_0;

        for (b, &tok) in prompt_tokens.iter().enumerate() {
            assert!(tok < self.vocab_size, "token_id out of vocabulary bounds");
            let embd_row = &weights.token_embd[tok * blocks_per_row..(tok + 1) * blocks_per_row];
            let dst = self.batch_workspace.x_batch.subslice(b * dim, dim);
            dequantize_row_q4_0(embd_row, arena.get_mut(dst));
        }

        for (lw, kv) in weights.layers.iter().zip(self.kv_caches.iter()) {
            self.layer.forward_batch_with_kv(
                engine,
                arena,
                &self.batch_workspace,
                lw,
                kv,
                start_pos,
                batch_size,
            );
        }

        let last_x = self
            .batch_workspace
            .x_batch
            .subslice((batch_size - 1) * dim, dim);
        engine.exec_with_weights(
            self.layer.rmsnorm_fn,
            arena,
            last_x,
            weights.output_norm,
            self.layer.workspace.xb,
        );

        gemv_q4_0_arena(
            arena,
            weights.lm_head,
            self.layer.workspace.xb,
            self.layer.workspace.q8_scratch,
            self.logits,
            self.vocab_size,
            dim,
        );

        sample_argmax(arena.get(self.logits))
    }

    /// Runs batched prompt prefill on `prompt_tokens` followed by autoregressive generation into `out_tokens`
    /// with zero heap allocations. Returns the number of generated tokens written to `out_tokens`.
    pub fn generate(
        &self,
        engine: &JitEngine,
        arena: &mut AlignedArena,
        weights: &ModelWeights<'_>,
        prompt_tokens: &[usize],
        out_tokens: &mut [usize],
        eos_token_id: Option<usize>,
    ) -> usize {
        if out_tokens.is_empty() || prompt_tokens.is_empty() {
            return 0;
        }
        let max_seq = self.layer.config.max_seq_len;
        assert!(prompt_tokens.len() <= max_seq);

        let mut next_tok = self.prefill_tokens(engine, arena, weights, prompt_tokens, 0);
        out_tokens[0] = next_tok;
        let mut generated = 1usize;

        if eos_token_id == Some(next_tok) {
            return generated;
        }

        let mut pos = prompt_tokens.len();
        while generated < out_tokens.len() && pos < max_seq {
            next_tok = self.forward_token(engine, arena, weights, next_tok, pos);
            out_tokens[generated] = next_tok;
            generated += 1;
            pos += 1;
            if eos_token_id == Some(next_tok) {
                break;
            }
        }
        generated
    }

    /// Computes a mean-pooled, JIT L2-normalized semantic embedding vector (`out_embedding` of length `dim`)
    /// for `tokens` using batched `GEMM` prefill with zero heap allocations.
    pub fn embed_tokens(
        &self,
        engine: &JitEngine,
        arena: &mut AlignedArena,
        weights: &ModelWeights<'_>,
        tokens: &[usize],
        out_embedding: &mut [f32],
    ) {
        let batch_size = tokens.len();
        assert!(batch_size > 0, "tokens cannot be empty");
        let dim = self.layer.config.dim;
        assert_eq!(out_embedding.len(), dim, "out_embedding length mismatch");
        let blocks_per_row = dim / QK4_0;

        for (b, &tok) in tokens.iter().enumerate() {
            assert!(tok < self.vocab_size, "token_id out of vocabulary bounds");
            let embd_row = &weights.token_embd[tok * blocks_per_row..(tok + 1) * blocks_per_row];
            let dst = self.batch_workspace.x_batch.subslice(b * dim, dim);
            dequantize_row_q4_0(embd_row, arena.get_mut(dst));
        }

        for (lw, kv) in weights.layers.iter().zip(self.kv_caches.iter()) {
            self.layer.forward_batch_with_kv(
                engine,
                arena,
                &self.batch_workspace,
                lw,
                kv,
                0,
                batch_size,
            );
        }

        arena.get_mut(self.embed_accum).fill(0.0);
        let inv_batch = 1.0f32 / (batch_size as f32);
        for b in 0..batch_size {
            let x_b = self.batch_workspace.x_batch.subslice(b * dim, dim);
            engine.exec_with_weights(
                self.layer.rmsnorm_fn,
                arena,
                x_b,
                weights.output_norm,
                self.layer.workspace.xb,
            );
            let (normed_b, accum) = arena.io_slices(self.layer.workspace.xb, self.embed_accum);
            for i in 0..dim {
                accum[i] += normed_b[i] * inv_batch;
            }
        }

        engine.exec_unary(self.l2_norm_fn, arena, self.embed_accum, self.embed_norm);
        out_embedding.copy_from_slice(arena.get(self.embed_norm));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::arena::tests::current_thread_heap_ops;
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
        x.iter()
            .zip(weight.iter())
            .map(|(&xi, &wi)| xi * rms * wi)
            .collect()
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
            num_kv_heads: 4,
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
            let x_mid: Vec<f32> = ref_x
                .iter()
                .zip(wo_out.iter())
                .map(|(a, b)| a + b)
                .collect();

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
    fn test_gqa_transformer_layer_matches_scalar_reference() {
        let cfg = TransformerConfig {
            dim: 128,
            hidden_dim: 256,
            num_heads: 8,
            num_kv_heads: 2, // 4:1 GQA ratio (kv_dim = 32)
            max_seq_len: 8,
            rms_eps: 1e-5,
        };
        let kv_dim = cfg.kv_dim();
        assert_eq!(kv_dim, 32);

        let rms_att_w = pseudo_random_f32(cfg.dim, 0x1101);
        let rms_ffn_w = pseudo_random_f32(cfg.dim, 0x1102);
        let wq = quantize_matrix(cfg.dim, cfg.dim, 0x2101);
        let wk = quantize_matrix(kv_dim, cfg.dim, 0x2102);
        let wv = quantize_matrix(kv_dim, cfg.dim, 0x2103);
        let wo = quantize_matrix(cfg.dim, cfg.dim, 0x2104);
        let w_gate = quantize_matrix(cfg.hidden_dim, cfg.dim, 0x3101);
        let w_up = quantize_matrix(cfg.hidden_dim, cfg.dim, 0x3102);
        let w_down = quantize_matrix(cfg.dim, cfg.hidden_dim, 0x3103);

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
        let mut arena = AlignedArena::with_capacity(16384);
        let ws = LayerWorkspace::allocate(&mut arena, &cfg);
        let layer = TransformerLayer::new(&mut engine, cfg, ws);

        let init_x = pseudo_random_f32(cfg.dim, 0x8888);
        arena.get_mut(layer.workspace.x).copy_from_slice(&init_x);

        let mut ref_x = init_x;
        let mut ref_k_cache = vec![0.0f32; cfg.max_seq_len * kv_dim];
        let mut ref_v_cache = vec![0.0f32; cfg.max_seq_len * kv_dim];
        let mut ref_att = vec![0.0f32; cfg.num_heads * cfg.max_seq_len];

        let ops_before = current_thread_heap_ops();
        for pos in 0..4 {
            layer.forward_token(&engine, &mut arena, &weights, pos);
        }
        let ops_after = current_thread_heap_ops();
        assert_eq!(ops_after, ops_before, "GQA forward_token must not allocate");

        for pos in 0..4 {
            let xb = scalar_rmsnorm(&ref_x, &rms_att_w, cfg.rms_eps);
            let mut q = scalar_gemv(&wq, &xb, cfg.dim, cfg.dim);
            let mut k = scalar_gemv(&wk, &xb, kv_dim, cfg.dim);
            let v = scalar_gemv(&wv, &xb, kv_dim, cfg.dim);

            apply_rope_in_place(&mut q, pos, cfg.head_dim());
            apply_rope_in_place(&mut k, pos, cfg.head_dim());

            ref_k_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
            ref_v_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);

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
            let x_mid: Vec<f32> = ref_x
                .iter()
                .zip(wo_out.iter())
                .map(|(a, b)| a + b)
                .collect();

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
        }

        let actual_x = arena.get(layer.workspace.x);
        for i in 0..cfg.dim {
            let diff = (actual_x[i] - ref_x[i]).abs();
            assert!(
                diff < 1e-4,
                "GQA dim {i} mismatch: actual={}, ref={}, diff={diff}",
                actual_x[i],
                ref_x[i]
            );
        }
    }

    #[test]
    fn test_transformer_layer_zero_heap_allocations() {
        let cfg = TransformerConfig {
            dim: 128,
            hidden_dim: 256,
            num_heads: 4,
            num_kv_heads: 4,
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
        let mut arena = AlignedArena::with_capacity(65536);
        let ws = LayerWorkspace::allocate(&mut arena, &cfg);
        let kv0 = LayerKvCache::allocate(&mut arena, &cfg);
        let kv1 = LayerKvCache::allocate(&mut arena, &cfg);
        let layer = TransformerLayer::new(&mut engine, cfg, ws);

        let init_x = pseudo_random_f32(cfg.dim, 0x7777);
        arena.get_mut(layer.workspace.x).copy_from_slice(&init_x);

        let ops_before = current_thread_heap_ops();
        for pos in 0..cfg.max_seq_len {
            layer.forward_token_with_kv(&engine, &mut arena, &weights, &kv0, pos);
            layer.forward_token_with_kv(&engine, &mut arena, &weights, &kv1, pos);
        }
        let ops_after = current_thread_heap_ops();

        assert_eq!(
            ops_after,
            ops_before,
            "Expected 0 heap allocs/deallocs during Transformer token generation loop, found {:?}",
            (ops_after.0 - ops_before.0, ops_after.1 - ops_before.1)
        );
    }

    #[test]
    fn test_prefill_batch_matches_sequential_forward_token_and_zero_alloc() {
        let cfg = TransformerConfig {
            dim: 64,
            hidden_dim: 128,
            num_heads: 4,
            num_kv_heads: 2,
            max_seq_len: 16,
            rms_eps: 1e-5,
        };
        let vocab_size = 128;
        let num_layers = 2;

        let token_embd = quantize_matrix(vocab_size, cfg.dim, 0x0100);
        let output_norm = vec![1.0f32; cfg.dim];
        let lm_head = quantize_matrix(vocab_size, cfg.dim, 0x0200);

        let rms_att_w = pseudo_random_f32(cfg.dim, 0x1001);
        let rms_ffn_w = pseudo_random_f32(cfg.dim, 0x1002);
        let wq = quantize_matrix(cfg.dim, cfg.dim, 0x2001);
        let wk = quantize_matrix(cfg.kv_dim(), cfg.dim, 0x2002);
        let wv = quantize_matrix(cfg.kv_dim(), cfg.dim, 0x2003);
        let wo = quantize_matrix(cfg.dim, cfg.dim, 0x2004);
        let w_gate = quantize_matrix(cfg.hidden_dim, cfg.dim, 0x3001);
        let w_up = quantize_matrix(cfg.hidden_dim, cfg.dim, 0x3002);
        let w_down = quantize_matrix(cfg.dim, cfg.hidden_dim, 0x3003);

        let lw = TransformerWeights {
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

        let model_weights = ModelWeights {
            token_embd: &token_embd,
            layers: vec![lw; num_layers],
            output_norm: &output_norm,
            lm_head: &lm_head,
            vocab_size,
        };

        let prompt = [1usize, 7, 19, 42];

        let mut engine_seq = JitEngine::new();
        let mut arena_seq = AlignedArena::with_capacity(131072);
        let model_seq =
            TransformerModel::new(&mut engine_seq, &mut arena_seq, cfg, num_layers, vocab_size);
        let mut seq_next = 0;
        for (pos, &tok) in prompt.iter().enumerate() {
            seq_next =
                model_seq.forward_token(&engine_seq, &mut arena_seq, &model_weights, tok, pos);
        }

        let mut engine_batch = JitEngine::new();
        let mut arena_batch = AlignedArena::with_capacity(131072);
        let model_batch = TransformerModel::new(
            &mut engine_batch,
            &mut arena_batch,
            cfg,
            num_layers,
            vocab_size,
        );

        let mut out_tokens = [0usize; 4];
        let ops_before = current_thread_heap_ops();
        let count = model_batch.generate(
            &engine_batch,
            &mut arena_batch,
            &model_weights,
            &prompt,
            &mut out_tokens,
            None,
        );
        let ops_after = current_thread_heap_ops();

        assert_eq!(count, 4);
        assert_eq!(
            ops_after, ops_before,
            "TransformerModel::generate (prefill + decode) must perform 0 heap ops"
        );
        assert_eq!(
            out_tokens[0], seq_next,
            "Batched GEMM prefill first generated token must match sequential forward_token"
        );
    }

    #[test]
    fn test_jit_l2_norm_and_semantic_embedding_search_zero_alloc() {
        let cfg = TransformerConfig {
            dim: 64,
            hidden_dim: 128,
            num_heads: 4,
            num_kv_heads: 2,
            max_seq_len: 16,
            rms_eps: 1e-5,
        };
        let vocab_size = 64;
        let num_layers = 2;

        let mut engine = JitEngine::new();
        let (_, l2_stats) = engine.compile_with_stats(&FuseGraph::l2_norm(cfg.rms_eps));
        assert_eq!(l2_stats.num_stack_slots, 0);
        assert!(!l2_stats.has_call_insts);
        assert_eq!(l2_stats.num_loops, 2);
        assert_eq!(
            l2_stats.final_loop_sqrt_or_div, 0,
            "l2_norm sqrt/recip must be hoisted out of Pass 2 loop"
        );

        let token_embd = quantize_matrix(vocab_size, cfg.dim, 0x0500);
        let output_norm = vec![1.0f32; cfg.dim];
        let lm_head = quantize_matrix(vocab_size, cfg.dim, 0x0600);
        let rms_att_w = pseudo_random_f32(cfg.dim, 0x1501);
        let rms_ffn_w = pseudo_random_f32(cfg.dim, 0x1502);
        let wq = quantize_matrix(cfg.dim, cfg.dim, 0x2501);
        let wk = quantize_matrix(cfg.kv_dim(), cfg.dim, 0x2502);
        let wv = quantize_matrix(cfg.kv_dim(), cfg.dim, 0x2503);
        let wo = quantize_matrix(cfg.dim, cfg.dim, 0x2504);
        let w_gate = quantize_matrix(cfg.hidden_dim, cfg.dim, 0x3501);
        let w_up = quantize_matrix(cfg.hidden_dim, cfg.dim, 0x3502);
        let w_down = quantize_matrix(cfg.dim, cfg.hidden_dim, 0x3503);

        let lw = TransformerWeights {
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
        let model_weights = ModelWeights {
            token_embd: &token_embd,
            layers: vec![lw; num_layers],
            output_norm: &output_norm,
            lm_head: &lm_head,
            vocab_size,
        };

        let mut arena = AlignedArena::with_capacity(131072);
        let model = TransformerModel::new(&mut engine, &mut arena, cfg, num_layers, vocab_size);

        let doc0_tokens = [1usize, 4, 8, 12];
        let doc1_tokens = [1usize, 20, 35, 50];
        let query_tokens = [1usize, 4, 8, 12]; // Identical to doc0

        let mut doc_embs = vec![0.0f32; 2 * cfg.dim];
        let mut query_emb = vec![0.0f32; cfg.dim];
        let mut ranked = [(0usize, 0.0f32); 2];

        let ops_before = current_thread_heap_ops();
        model.embed_tokens(
            &engine,
            &mut arena,
            &model_weights,
            &doc0_tokens,
            &mut doc_embs[0..cfg.dim],
        );
        model.embed_tokens(
            &engine,
            &mut arena,
            &model_weights,
            &doc1_tokens,
            &mut doc_embs[cfg.dim..2 * cfg.dim],
        );
        model.embed_tokens(
            &engine,
            &mut arena,
            &model_weights,
            &query_tokens,
            &mut query_emb,
        );
        rank_documents_by_cosine(&query_emb, &doc_embs, cfg.dim, &mut ranked);
        let ops_after = current_thread_heap_ops();

        assert_eq!(
            ops_after, ops_before,
            "embed_tokens and rank_documents_by_cosine must perform 0 heap ops"
        );

        let norm_sq: f32 = query_emb.iter().map(|&v| v * v).sum();
        assert!(
            (norm_sq.sqrt() - 1.0).abs() < 1e-4,
            "Embedding must have unit L2 norm, got {}",
            norm_sq.sqrt()
        );
        assert_eq!(ranked[0].0, 0, "Exact match doc0 must rank first");
        assert!(
            (ranked[0].1 - 1.0).abs() < 1e-4,
            "Exact match cosine similarity must be ~1.0, got {}",
            ranked[0].1
        );
        assert!(ranked[0].1 > ranked[1].1);
    }
}
