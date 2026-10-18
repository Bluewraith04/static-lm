use std::hint::black_box;
use std::time::Instant;

use crate::jit::engine::JitEngine;
use crate::memory::arena::{AlignedArena, current_thread_heap_stats};
use crate::model::transformer::{
    LayerKvCache, LayerWorkspace, ModelWeights, TransformerConfig, TransformerModel,
    apply_rope_cached, apply_rope_in_place, causal_multi_head_attention, cosine_similarity,
    rank_documents_by_cosine,
};
use crate::quant::blocks::{BlockQ4_0, QK4_0, dequantize_row_q4_0, quantize_row_q8_0};
use crate::quant::kernels::vec_dot_q4_0_q8_0_scalar;

#[inline]
fn scalar_rmsnorm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let rms = 1.0f32 / (sum_sq / (x.len() as f32) + eps).sqrt();
    for ((dst, &xi), &wi) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *dst = xi * rms * wi;
    }
}

#[inline]
fn scalar_swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((dst, &g), &u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        let silu = g / (1.0f32 + (-g).exp());
        *dst = silu * u;
    }
}

#[inline]
fn scalar_residual_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((dst, &ai), &bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *dst = ai + bi;
    }
}

/// Comparative benchmark metrics for Application 1: Local `.gguf` Text Generation (`generate`).
#[derive(Clone, Debug)]
pub struct GenerateBenchMetrics {
    pub engine_name: &'static str,
    pub arch_summary: &'static str,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub prefill_tok_per_sec: f64,
    pub decode_tok_per_sec: f64,
    pub speedup_vs_static_lm: f64,
    pub heap_allocs: usize,
    pub heap_bytes: usize,
    pub static_memory_bytes: usize,
    pub output_tokens: Vec<usize>,
}

/// Comparative benchmark metrics for Application 2: Semantic Embedding & Cosine Search (`embed`).
#[derive(Clone, Debug)]
pub struct EmbedBenchMetrics {
    pub engine_name: &'static str,
    pub arch_summary: &'static str,
    pub num_docs: usize,
    pub total_tokens: usize,
    pub total_ms: f64,
    pub ms_per_doc: f64,
    pub docs_per_sec: f64,
    pub tok_per_sec: f64,
    pub speedup_vs_static_lm: f64,
    pub heap_allocs: usize,
    pub heap_bytes: usize,
    pub static_memory_bytes: usize,
    pub top_doc_idx: usize,
    pub top_cosine: f32,
    pub ranking: Vec<(usize, f32)>,
}

#[inline]
fn argmax_f32(slice: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in slice.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx
}

#[inline]
fn q4_model_weight_bytes(weights: &ModelWeights<'_>) -> usize {
    let mut blocks = weights.token_embd.len() + weights.lm_head.len();
    for l in &weights.layers {
        blocks +=
            l.wq.len() + l.wk.len() + l.wv.len() + l.wo.len() + l.w_gate.len() + l.w_up.len() + l.w_down.len();
    }
    let norm_floats = weights.output_norm.len()
        + weights
            .layers
            .iter()
            .map(|l| l.rms_att_weight.len() + l.rms_ffn_weight.len())
            .sum::<usize>();
    blocks * std::mem::size_of::<BlockQ4_0>() + norm_floats * std::mem::size_of::<f32>()
}

// ============================================================================
// Baseline 1A & 2A: `candle`-style / `fastembed`-style Heap-Allocating Tensor Engine
// ============================================================================

struct HeapKvCache {
    k: Vec<f32>,
    v: Vec<f32>,
}

fn heap_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    scalar_rmsnorm(x, weight, &mut out, eps);
    out
}

fn heap_residual_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; a.len()];
    scalar_residual_add(a, b, &mut out);
    out
}

fn heap_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; gate.len()];
    scalar_swiglu(gate, up, &mut out);
    out
}

fn heap_l2_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let inv_norm = 1.0f32 / (sum_sq + eps).sqrt();
    x.iter().map(|&v| v * inv_norm).collect()
}

/// Idiomatic heap-allocating quantized matmul: dequantizes a `Q4_0` row buffer on the fly and returns a new `Vec<f32>`.
fn heap_matmul_q4_dequant(weights: &[BlockQ4_0], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let blocks_per_row = cols / QK4_0;
    let mut row_f32 = vec![0.0f32; cols];
    let mut out = vec![0.0f32; rows];
    for (r, out_val) in out.iter_mut().enumerate() {
        let w_row = &weights[r * blocks_per_row..(r + 1) * blocks_per_row];
        dequantize_row_q4_0(w_row, &mut row_f32);
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += row_f32[c] * x[c];
        }
        *out_val = sum;
    }
    out
}

fn heap_forward_hidden(
    cfg: &TransformerConfig,
    weights: &ModelWeights<'_>,
    kv_caches: &mut [HeapKvCache],
    token_id: usize,
    pos: usize,
) -> Vec<f32> {
    let blocks_per_dim = cfg.dim / QK4_0;
    let mut x = vec![0.0f32; cfg.dim];
    dequantize_row_q4_0(
        &weights.token_embd[token_id * blocks_per_dim..(token_id + 1) * blocks_per_dim],
        &mut x,
    );

    let head_dim = cfg.head_dim();
    let kv_dim = cfg.kv_dim();

    for (l_idx, lw) in weights.layers.iter().enumerate() {
        let xb = heap_rmsnorm(&x, lw.rms_att_weight, cfg.rms_eps);
        let mut q = heap_matmul_q4_dequant(lw.wq, &xb, cfg.dim, cfg.dim);
        let mut k = heap_matmul_q4_dequant(lw.wk, &xb, kv_dim, cfg.dim);
        let v = heap_matmul_q4_dequant(lw.wv, &xb, kv_dim, cfg.dim);

        apply_rope_in_place(&mut q, pos, head_dim);
        apply_rope_in_place(&mut k, pos, head_dim);

        let kv = &mut kv_caches[l_idx];
        kv.k[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
        kv.v[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);

        let mut att = vec![0.0f32; cfg.num_heads * cfg.max_seq_len];
        let mut attn_out = vec![0.0f32; cfg.dim];
        causal_multi_head_attention(&q, &kv.k, &kv.v, &mut att, &mut attn_out, cfg, pos);

        let attn_proj = heap_matmul_q4_dequant(lw.wo, &attn_out, cfg.dim, cfg.dim);
        let x_mid = heap_residual_add(&x, &attn_proj);

        let ffn_in = heap_rmsnorm(&x_mid, lw.rms_ffn_weight, cfg.rms_eps);
        let gate = heap_matmul_q4_dequant(lw.w_gate, &ffn_in, cfg.hidden_dim, cfg.dim);
        let up = heap_matmul_q4_dequant(lw.w_up, &ffn_in, cfg.hidden_dim, cfg.dim);
        let act = heap_swiglu(&gate, &up);
        let down = heap_matmul_q4_dequant(lw.w_down, &act, cfg.dim, cfg.hidden_dim);
        x = heap_residual_add(&x_mid, &down);
    }

    heap_rmsnorm(&x, weights.output_norm, cfg.rms_eps)
}

// ============================================================================
// Baseline 1B & 2B: `llama2.c`-style Pre-Allocated `f32` RunState Engine
// ============================================================================

struct F32LayerWeights {
    rms_att: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    rms_ffn: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

struct F32ModelWeights {
    token_embd: Vec<f32>,
    layers: Vec<F32LayerWeights>,
    output_norm: Vec<f32>,
    lm_head: Vec<f32>,
}

impl F32ModelWeights {
    fn from_q4(weights: &ModelWeights<'_>, cfg: &TransformerConfig) -> Self {
        let dequant = |blocks: &[BlockQ4_0]| -> Vec<f32> {
            let mut out = vec![0.0f32; blocks.len() * QK4_0];
            dequantize_row_q4_0(blocks, &mut out);
            out
        };
        let _ = cfg;
        let layers = weights
            .layers
            .iter()
            .map(|l| F32LayerWeights {
                rms_att: l.rms_att_weight.to_vec(),
                wq: dequant(l.wq),
                wk: dequant(l.wk),
                wv: dequant(l.wv),
                wo: dequant(l.wo),
                rms_ffn: l.rms_ffn_weight.to_vec(),
                w_gate: dequant(l.w_gate),
                w_up: dequant(l.w_up),
                w_down: dequant(l.w_down),
            })
            .collect();
        Self {
            token_embd: dequant(weights.token_embd),
            layers,
            output_norm: weights.output_norm.to_vec(),
            lm_head: dequant(weights.lm_head),
        }
    }

    fn total_bytes(&self) -> usize {
        let mut floats =
            self.token_embd.len() + self.output_norm.len() + self.lm_head.len();
        for l in &self.layers {
            floats += l.rms_att.len()
                + l.wq.len()
                + l.wk.len()
                + l.wv.len()
                + l.wo.len()
                + l.rms_ffn.len()
                + l.w_gate.len()
                + l.w_up.len()
                + l.w_down.len();
        }
        floats * std::mem::size_of::<f32>()
    }
}

struct F32RunState {
    x: Vec<f32>,
    x_mid: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    q: Vec<f32>,
    k_caches: Vec<Vec<f32>>,
    v_caches: Vec<Vec<f32>>,
    att: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    hb_act: Vec<f32>,
    logits: Vec<f32>,
    embed_pool: Vec<f32>,
}

impl F32RunState {
    fn new(cfg: &TransformerConfig, num_layers: usize, vocab_size: usize) -> Self {
        let kv_dim = cfg.kv_dim();
        Self {
            x: vec![0.0; cfg.dim],
            x_mid: vec![0.0; cfg.dim],
            xb: vec![0.0; cfg.dim],
            xb2: vec![0.0; cfg.dim],
            q: vec![0.0; cfg.dim],
            k_caches: (0..num_layers)
                .map(|_| vec![0.0; cfg.max_seq_len * kv_dim])
                .collect(),
            v_caches: (0..num_layers)
                .map(|_| vec![0.0; cfg.max_seq_len * kv_dim])
                .collect(),
            att: vec![0.0; cfg.num_heads * cfg.max_seq_len],
            hb: vec![0.0; cfg.hidden_dim],
            hb2: vec![0.0; cfg.hidden_dim],
            hb_act: vec![0.0; cfg.hidden_dim],
            logits: vec![0.0; vocab_size],
            embed_pool: vec![0.0; cfg.dim],
        }
    }

    fn total_bytes(&self) -> usize {
        let kv_floats: usize = self.k_caches.iter().map(Vec::len).sum::<usize>()
            + self.v_caches.iter().map(Vec::len).sum::<usize>();
        let vec_floats = self.x.len()
            + self.x_mid.len()
            + self.xb.len()
            + self.xb2.len()
            + self.q.len()
            + self.att.len()
            + self.hb.len()
            + self.hb2.len()
            + self.hb_act.len()
            + self.logits.len()
            + self.embed_pool.len()
            + kv_floats;
        vec_floats * std::mem::size_of::<f32>()
    }
}

#[inline(never)]
fn f32_matmul_vec(weights: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(weights.len(), rows * cols);
    for (r, out_val) in out.iter_mut().enumerate() {
        let row = &weights[r * cols..(r + 1) * cols];
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += row[c] * x[c];
        }
        *out_val = sum;
    }
    black_box(&mut *out);
}

fn f32_forward_hidden(
    cfg: &TransformerConfig,
    weights: &F32ModelWeights,
    state: &mut F32RunState,
    token_id: usize,
    pos: usize,
) {
    let dim = cfg.dim;
    let kv_dim = cfg.kv_dim();
    let head_dim = cfg.head_dim();
    state
        .x
        .copy_from_slice(&weights.token_embd[token_id * dim..(token_id + 1) * dim]);

    for (l_idx, lw) in weights.layers.iter().enumerate() {
        scalar_rmsnorm(&state.x, &lw.rms_att, &mut state.xb, cfg.rms_eps);
        f32_matmul_vec(&lw.wq, &state.xb, &mut state.q, dim, dim);
        f32_matmul_vec(
            &lw.wk,
            &state.xb,
            &mut state.k_caches[l_idx][pos * kv_dim..(pos + 1) * kv_dim],
            kv_dim,
            dim,
        );
        f32_matmul_vec(
            &lw.wv,
            &state.xb,
            &mut state.v_caches[l_idx][pos * kv_dim..(pos + 1) * kv_dim],
            kv_dim,
            dim,
        );

        apply_rope_in_place(&mut state.q, pos, head_dim);
        apply_rope_in_place(
            &mut state.k_caches[l_idx][pos * kv_dim..(pos + 1) * kv_dim],
            pos,
            head_dim,
        );

        causal_multi_head_attention(
            &state.q,
            &state.k_caches[l_idx],
            &state.v_caches[l_idx],
            &mut state.att,
            &mut state.xb,
            cfg,
            pos,
        );

        f32_matmul_vec(&lw.wo, &state.xb, &mut state.xb2, dim, dim);
        scalar_residual_add(&state.x, &state.xb2, &mut state.x_mid);

        scalar_rmsnorm(&state.x_mid, &lw.rms_ffn, &mut state.xb, cfg.rms_eps);
        f32_matmul_vec(&lw.w_gate, &state.xb, &mut state.hb, cfg.hidden_dim, dim);
        f32_matmul_vec(&lw.w_up, &state.xb, &mut state.hb2, cfg.hidden_dim, dim);
        scalar_swiglu(&state.hb, &state.hb2, &mut state.hb_act);
        f32_matmul_vec(&lw.w_down, &state.hb_act, &mut state.xb, dim, cfg.hidden_dim);
        scalar_residual_add(&state.x_mid, &state.xb, &mut state.x);
    }

    scalar_rmsnorm(&state.x, &weights.output_norm, &mut state.xb, cfg.rms_eps);
}

// ============================================================================
// Baseline 1C & 2C: Unbatched Scalar Quantized Engine (No SIMD, No JIT, No GEMM)
// ============================================================================

fn scalar_gemv_q4_0_arena(
    arena: &mut AlignedArena,
    weights: &[BlockQ4_0],
    input: crate::memory::arena::ArenaSlice,
    q8_scratch: crate::memory::arena::ArenaSlice,
    output: crate::memory::arena::ArenaSlice,
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / QK4_0;
    let (x_f32, x_q8, y_f32) = arena.gemv_slices(input, q8_scratch, output, blocks_per_row);
    quantize_row_q8_0(x_f32, x_q8);
    for (row_w, out_val) in weights.chunks_exact(blocks_per_row).zip(y_f32.iter_mut()) {
        *out_val = vec_dot_q4_0_q8_0_scalar(row_w, x_q8);
    }
}

fn scalar_quant_forward_hidden(
    cfg: &TransformerConfig,
    ws: &LayerWorkspace,
    kv_caches: &[LayerKvCache],
    arena: &mut AlignedArena,
    weights: &ModelWeights<'_>,
    token_id: usize,
    pos: usize,
) {
    let blocks_per_dim = cfg.dim / QK4_0;
    dequantize_row_q4_0(
        &weights.token_embd[token_id * blocks_per_dim..(token_id + 1) * blocks_per_dim],
        arena.get_mut(ws.x),
    );

    let head_dim = cfg.head_dim();
    let half_head = head_dim / 2;
    let kv_dim = cfg.kv_dim();
    let rope_off = pos * half_head;

    for (l_idx, lw) in weights.layers.iter().enumerate() {
        let kv = &kv_caches[l_idx];
        {
            let (x_in, xb_out) = arena.io_slices(ws.x, ws.xb);
            scalar_rmsnorm(x_in, lw.rms_att_weight, xb_out, cfg.rms_eps);
        }
        let k_pos = kv.k_cache.subslice(pos * kv_dim, kv_dim);
        let v_pos = kv.v_cache.subslice(pos * kv_dim, kv_dim);
        scalar_gemv_q4_0_arena(arena, lw.wq, ws.xb, ws.q8_scratch, ws.q, cfg.dim, cfg.dim);
        scalar_gemv_q4_0_arena(arena, lw.wk, ws.xb, ws.q8_scratch, k_pos, kv_dim, cfg.dim);
        scalar_gemv_q4_0_arena(arena, lw.wv, ws.xb, ws.q8_scratch, v_pos, kv_dim, cfg.dim);

        let cos_pos = ws.rope_cos.subslice(rope_off, half_head);
        let sin_pos = ws.rope_sin.subslice(rope_off, half_head);
        {
            let (cos_s, sin_s, q_mut) = arena.io2_slices(cos_pos, sin_pos, ws.q);
            apply_rope_cached(q_mut, cos_s, sin_s, head_dim);
        }
        {
            let (cos_s, sin_s, k_mut) = arena.io2_slices(cos_pos, sin_pos, k_pos);
            apply_rope_cached(k_mut, cos_s, sin_s, head_dim);
        }

        {
            let (q_s, k_s, v_s, att_s, xb_out) =
                arena.attention_slices(ws.q, kv.k_cache, kv.v_cache, ws.att, ws.xb);
            causal_multi_head_attention(q_s, k_s, v_s, att_s, xb_out, cfg, pos);
        }

        scalar_gemv_q4_0_arena(arena, lw.wo, ws.xb, ws.q8_scratch, ws.xb2, cfg.dim, cfg.dim);
        {
            let (a, b, dst) = arena.io2_slices(ws.x, ws.xb2, ws.x_mid);
            scalar_residual_add(a, b, dst);
        }
        {
            let (x_mid, xb_out) = arena.io_slices(ws.x_mid, ws.xb);
            scalar_rmsnorm(x_mid, lw.rms_ffn_weight, xb_out, cfg.rms_eps);
        }
        scalar_gemv_q4_0_arena(
            arena,
            lw.w_gate,
            ws.xb,
            ws.q8_scratch,
            ws.hb,
            cfg.hidden_dim,
            cfg.dim,
        );
        scalar_gemv_q4_0_arena(
            arena,
            lw.w_up,
            ws.xb,
            ws.q8_scratch,
            ws.hb2,
            cfg.hidden_dim,
            cfg.dim,
        );
        {
            let (gate, up, act) = arena.io2_slices(ws.hb, ws.hb2, ws.hb_act);
            scalar_swiglu(gate, up, act);
        }
        scalar_gemv_q4_0_arena(
            arena,
            lw.w_down,
            ws.hb_act,
            ws.q8_scratch,
            ws.xb,
            cfg.dim,
            cfg.hidden_dim,
        );
        {
            let (a, b, dst) = arena.io2_slices(ws.x_mid, ws.xb, ws.x);
            scalar_residual_add(a, b, dst);
        }
    }

    let (x_final, xb_norm) = arena.io_slices(ws.x, ws.xb);
    scalar_rmsnorm(x_final, weights.output_norm, xb_norm, cfg.rms_eps);
}

// ============================================================================
// Public Comparative Benchmark Runners
// ============================================================================

/// Runs Application 1 (`generate`) across `static-lm` and 3 competing baselines on the same model and prompt.
pub fn run_generate_comparative_bench(
    engine: &JitEngine,
    model: &TransformerModel,
    arena: &mut AlignedArena,
    weights: &ModelWeights<'_>,
    prompt_tokens: &[usize],
    num_decode_tokens: usize,
    iterations: usize,
) -> [GenerateBenchMetrics; 4] {
    let iters = iterations.max(1);
    let cfg = &model.layer.config;
    let num_layers = model.kv_caches.len();
    let q4_weight_bytes = q4_model_weight_bytes(weights);
    let arena_used_bytes = arena.used_floats_or_bytes(cfg, num_layers, weights.vocab_size);

    // 1. static-lm (Ours: Pre-allocated Arena + Batched GEMM Prefill + SIMD Q4xQ8 + Cranelift JIT)
    let mut static_out = vec![0usize; num_decode_tokens];
    let (a0, _, b0) = current_thread_heap_stats();
    let mut prefill_dur = 0.0f64;
    let mut decode_dur = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        let mut next_tok = model.prefill_tokens(engine, arena, weights, prompt_tokens, 0);
        let t1 = Instant::now();
        static_out[0] = next_tok;
        for step in 1..num_decode_tokens {
            let pos = prompt_tokens.len() + step - 1;
            next_tok = model.forward_token(engine, arena, weights, next_tok, pos);
            static_out[step] = next_tok;
        }
        let t2 = Instant::now();
        prefill_dur += (t1 - t0).as_secs_f64() * 1000.0;
        decode_dur += (t2 - t1).as_secs_f64() * 1000.0;
    }
    let (a1, _, b1) = current_thread_heap_stats();
    let static_prefill_ms = prefill_dur / (iters as f64);
    let static_decode_ms = decode_dur / (iters as f64);
    let static_total_ms = static_prefill_ms + static_decode_ms;

    // 2. Baseline 1A: `candle`-style Heap Tensor Engine (per-op Vec<f32> allocs + Q4->f32 dequant)
    let mut heap_kv: Vec<HeapKvCache> = (0..num_layers)
        .map(|_| HeapKvCache {
            k: vec![0.0; cfg.max_seq_len * cfg.kv_dim()],
            v: vec![0.0; cfg.max_seq_len * cfg.kv_dim()],
        })
        .collect();
    let mut b1a_out = vec![0usize; num_decode_tokens];
    let (h_a0, _, h_b0) = current_thread_heap_stats();
    let mut b1a_prefill = 0.0f64;
    let mut b1a_decode = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        let mut last_hidden = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            last_hidden = heap_forward_hidden(cfg, weights, &mut heap_kv, tok, pos);
        }
        let logits = heap_matmul_q4_dequant(weights.lm_head, &last_hidden, weights.vocab_size, cfg.dim);
        let mut next_tok = argmax_f32(&logits);
        let t1 = Instant::now();
        b1a_out[0] = next_tok;
        for step in 1..num_decode_tokens {
            let pos = prompt_tokens.len() + step - 1;
            let hidden = heap_forward_hidden(cfg, weights, &mut heap_kv, next_tok, pos);
            let step_logits =
                heap_matmul_q4_dequant(weights.lm_head, &hidden, weights.vocab_size, cfg.dim);
            next_tok = argmax_f32(&step_logits);
            b1a_out[step] = next_tok;
        }
        let t2 = Instant::now();
        b1a_prefill += (t1 - t0).as_secs_f64() * 1000.0;
        b1a_decode += (t2 - t1).as_secs_f64() * 1000.0;
    }
    let (h_a1, _, h_b1) = current_thread_heap_stats();
    let b1a_prefill_ms = b1a_prefill / (iters as f64);
    let b1a_decode_ms = b1a_decode / (iters as f64);
    let b1a_total_ms = b1a_prefill_ms + b1a_decode_ms;

    // 3. Baseline 1B: `llama2.c`-style Pre-Allocated `f32` RunState Engine
    let f32_weights = F32ModelWeights::from_q4(weights, cfg);
    let mut f32_state = F32RunState::new(cfg, num_layers, weights.vocab_size);
    let mut b1b_out = vec![0usize; num_decode_tokens];
    let (f_a0, _, f_b0) = current_thread_heap_stats();
    let mut b1b_prefill = 0.0f64;
    let mut b1b_decode = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            f32_forward_hidden(cfg, &f32_weights, &mut f32_state, tok, pos);
        }
        f32_matmul_vec(
            &f32_weights.lm_head,
            &f32_state.xb,
            &mut f32_state.logits,
            weights.vocab_size,
            cfg.dim,
        );
        let mut next_tok = argmax_f32(&f32_state.logits);
        let t1 = Instant::now();
        b1b_out[0] = next_tok;
        for step in 1..num_decode_tokens {
            let pos = prompt_tokens.len() + step - 1;
            f32_forward_hidden(cfg, &f32_weights, &mut f32_state, next_tok, pos);
            f32_matmul_vec(
                &f32_weights.lm_head,
                &f32_state.xb,
                &mut f32_state.logits,
                weights.vocab_size,
                cfg.dim,
            );
            next_tok = argmax_f32(&f32_state.logits);
            b1b_out[step] = next_tok;
        }
        let t2 = Instant::now();
        b1b_prefill += (t1 - t0).as_secs_f64() * 1000.0;
        b1b_decode += (t2 - t1).as_secs_f64() * 1000.0;
    }
    let (f_a1, _, f_b1) = current_thread_heap_stats();
    let b1b_prefill_ms = b1b_prefill / (iters as f64);
    let b1b_decode_ms = b1b_decode / (iters as f64);
    let b1b_total_ms = b1b_prefill_ms + b1b_decode_ms;

    // 4. Baseline 1C: Unbatched Scalar Quantized Engine (No GEMM prefill, No SIMD, No JIT)
    let mut b1c_out = vec![0usize; num_decode_tokens];
    let (s_a0, _, s_b0) = current_thread_heap_stats();
    let mut b1c_prefill = 0.0f64;
    let mut b1c_decode = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            scalar_quant_forward_hidden(
                cfg,
                &model.layer.workspace,
                &model.kv_caches,
                arena,
                weights,
                tok,
                pos,
            );
        }
        scalar_gemv_q4_0_arena(
            arena,
            weights.lm_head,
            model.layer.workspace.xb,
            model.layer.workspace.q8_scratch,
            model.logits,
            weights.vocab_size,
            cfg.dim,
        );
        let mut next_tok = argmax_f32(arena.get(model.logits));
        let t1 = Instant::now();
        b1c_out[0] = next_tok;
        for step in 1..num_decode_tokens {
            let pos = prompt_tokens.len() + step - 1;
            scalar_quant_forward_hidden(
                cfg,
                &model.layer.workspace,
                &model.kv_caches,
                arena,
                weights,
                next_tok,
                pos,
            );
            scalar_gemv_q4_0_arena(
                arena,
                weights.lm_head,
                model.layer.workspace.xb,
                model.layer.workspace.q8_scratch,
                model.logits,
                weights.vocab_size,
                cfg.dim,
            );
            next_tok = argmax_f32(arena.get(model.logits));
            b1c_out[step] = next_tok;
        }
        let t2 = Instant::now();
        b1c_prefill += (t1 - t0).as_secs_f64() * 1000.0;
        b1c_decode += (t2 - t1).as_secs_f64() * 1000.0;
    }
    let (s_a1, _, s_b1) = current_thread_heap_stats();
    let b1c_prefill_ms = b1c_prefill / (iters as f64);
    let b1c_decode_ms = b1c_decode / (iters as f64);
    let b1c_total_ms = b1c_prefill_ms + b1c_decode_ms;

    let decode_steps = (num_decode_tokens.saturating_sub(1)).max(1) as f64;
    let prompt_len_f = prompt_tokens.len() as f64;

    [
        GenerateBenchMetrics {
            engine_name: "static-lm (Ours)",
            arch_summary: "Arena + Batch GEMM + SIMD Q4xQ8 + Cranelift JIT",
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: num_decode_tokens,
            prefill_ms: static_prefill_ms,
            decode_ms: static_decode_ms,
            total_ms: static_total_ms,
            prefill_tok_per_sec: (prompt_len_f * 1000.0) / static_prefill_ms,
            decode_tok_per_sec: (decode_steps * 1000.0) / static_decode_ms,
            speedup_vs_static_lm: 1.0,
            heap_allocs: (a1 - a0) / iters,
            heap_bytes: (b1 - b0) / iters,
            static_memory_bytes: q4_weight_bytes + arena_used_bytes,
            output_tokens: static_out,
        },
        GenerateBenchMetrics {
            engine_name: "Baseline 1A (candle-style)",
            arch_summary: "Heap Vec<f32> Tensors + On-the-Fly Q4 Dequant",
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: num_decode_tokens,
            prefill_ms: b1a_prefill_ms,
            decode_ms: b1a_decode_ms,
            total_ms: b1a_total_ms,
            prefill_tok_per_sec: (prompt_len_f * 1000.0) / b1a_prefill_ms,
            decode_tok_per_sec: (decode_steps * 1000.0) / b1a_decode_ms,
            speedup_vs_static_lm: b1a_total_ms / static_total_ms,
            heap_allocs: (h_a1 - h_a0) / iters,
            heap_bytes: (h_b1 - h_b0) / iters,
            static_memory_bytes: q4_weight_bytes
                + heap_kv
                    .iter()
                    .map(|kv| (kv.k.len() + kv.v.len()) * 4)
                    .sum::<usize>(),
            output_tokens: b1a_out,
        },
        GenerateBenchMetrics {
            engine_name: "Baseline 1B (llama2.c-style)",
            arch_summary: "Static f32 RunState + Pre-Dequantized f32 GEMV",
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: num_decode_tokens,
            prefill_ms: b1b_prefill_ms,
            decode_ms: b1b_decode_ms,
            total_ms: b1b_total_ms,
            prefill_tok_per_sec: (prompt_len_f * 1000.0) / b1b_prefill_ms,
            decode_tok_per_sec: (decode_steps * 1000.0) / b1b_decode_ms,
            speedup_vs_static_lm: b1b_total_ms / static_total_ms,
            heap_allocs: (f_a1 - f_a0) / iters,
            heap_bytes: (f_b1 - f_b0) / iters,
            static_memory_bytes: f32_weights.total_bytes() + f32_state.total_bytes(),
            output_tokens: b1b_out,
        },
        GenerateBenchMetrics {
            engine_name: "Baseline 1C (Unfused Q4)",
            arch_summary: "Sequential Scalar Q4xQ8 + Unfused Scalar Ops",
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: num_decode_tokens,
            prefill_ms: b1c_prefill_ms,
            decode_ms: b1c_decode_ms,
            total_ms: b1c_total_ms,
            prefill_tok_per_sec: (prompt_len_f * 1000.0) / b1c_prefill_ms,
            decode_tok_per_sec: (decode_steps * 1000.0) / b1c_decode_ms,
            speedup_vs_static_lm: b1c_total_ms / static_total_ms,
            heap_allocs: (s_a1 - s_a0) / iters,
            heap_bytes: (s_b1 - s_b0) / iters,
            static_memory_bytes: q4_weight_bytes + arena_used_bytes,
            output_tokens: b1c_out,
        },
    ]
}

/// Runs Application 2 (`embed`) across `static-lm` and 3 competing baselines on the same corpus and query.
pub fn run_embed_comparative_bench(
    engine: &JitEngine,
    model: &TransformerModel,
    arena: &mut AlignedArena,
    weights: &ModelWeights<'_>,
    doc_tokens: &[Vec<usize>],
    query_tokens: &[usize],
    iterations: usize,
) -> [EmbedBenchMetrics; 4] {
    let iters = iterations.max(1);
    let cfg = &model.layer.config;
    let dim = cfg.dim;
    let num_layers = model.kv_caches.len();
    let num_docs = doc_tokens.len();
    let total_tokens: usize =
        doc_tokens.iter().map(Vec::len).sum::<usize>() + query_tokens.len();
    let q4_weight_bytes = q4_model_weight_bytes(weights);
    let arena_used_bytes = arena.used_floats_or_bytes(cfg, num_layers, weights.vocab_size);

    // 1. static-lm (Ours: In-Arena Streaming Mean Pool + Cranelift JIT 2-Pass L2 Norm + In-Place Rank)
    let mut static_doc_embs = vec![0.0f32; num_docs * dim];
    let mut static_q_emb = vec![0.0f32; dim];
    let mut static_scores = vec![(0usize, 0.0f32); num_docs];

    let (a0, _, b0) = current_thread_heap_stats();
    let t0 = Instant::now();
    for _ in 0..iters {
        for (i, d_toks) in doc_tokens.iter().enumerate() {
            model.embed_tokens(
                engine,
                arena,
                weights,
                d_toks,
                &mut static_doc_embs[i * dim..(i + 1) * dim],
            );
        }
        model.embed_tokens(engine, arena, weights, query_tokens, &mut static_q_emb);
        rank_documents_by_cosine(&static_q_emb, &static_doc_embs, dim, &mut static_scores);
    }
    let static_ms = t0.elapsed().as_secs_f64() * 1000.0 / (iters as f64);
    let (a1, _, b1) = current_thread_heap_stats();

    // 2. Baseline 2A: `fastembed`/`candle`-style Heap Tensor Pipeline
    let mut heap_kv: Vec<HeapKvCache> = (0..num_layers)
        .map(|_| HeapKvCache {
            k: vec![0.0; cfg.max_seq_len * cfg.kv_dim()],
            v: vec![0.0; cfg.max_seq_len * cfg.kv_dim()],
        })
        .collect();
    let mut b2a_scores = Vec::new();
    let (h_a0, _, h_b0) = current_thread_heap_stats();
    let t_2a = Instant::now();
    for _ in 0..iters {
        let mut doc_embs_heap: Vec<Vec<f32>> = Vec::with_capacity(num_docs);
        for d_toks in doc_tokens {
            let mut token_hiddens: Vec<Vec<f32>> = Vec::with_capacity(d_toks.len());
            for (pos, &tok) in d_toks.iter().enumerate() {
                token_hiddens.push(heap_forward_hidden(cfg, weights, &mut heap_kv, tok, pos));
            }
            let inv_len = 1.0f32 / (d_toks.len() as f32);
            let mut pooled = vec![0.0f32; dim];
            for h in &token_hiddens {
                for (dst, &val) in pooled.iter_mut().zip(h.iter()) {
                    *dst += val * inv_len;
                }
            }
            doc_embs_heap.push(heap_l2_norm(&pooled, cfg.rms_eps));
        }

        let mut q_hiddens: Vec<Vec<f32>> = Vec::with_capacity(query_tokens.len());
        for (pos, &tok) in query_tokens.iter().enumerate() {
            q_hiddens.push(heap_forward_hidden(cfg, weights, &mut heap_kv, tok, pos));
        }
        let inv_q = 1.0f32 / (query_tokens.len() as f32);
        let mut q_pooled = vec![0.0f32; dim];
        for h in &q_hiddens {
            for (dst, &val) in q_pooled.iter_mut().zip(h.iter()) {
                *dst += val * inv_q;
            }
        }
        let q_norm = heap_l2_norm(&q_pooled, cfg.rms_eps);

        let mut ranked: Vec<(usize, f32)> = doc_embs_heap
            .iter()
            .enumerate()
            .map(|(idx, d_emb)| (idx, cosine_similarity(&q_norm, d_emb)))
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        b2a_scores = ranked;
    }
    let b2a_ms = t_2a.elapsed().as_secs_f64() * 1000.0 / (iters as f64);
    let (h_a1, _, h_b1) = current_thread_heap_stats();

    // 3. Baseline 2B: `llama2.c`-style Static `f32` Embedder
    let f32_weights = F32ModelWeights::from_q4(weights, cfg);
    let mut f32_state = F32RunState::new(cfg, num_layers, weights.vocab_size);
    let mut b2b_doc_embs = vec![0.0f32; num_docs * dim];
    let mut b2b_q_emb = vec![0.0f32; dim];
    let mut b2b_scores = vec![(0usize, 0.0f32); num_docs];

    let f32_embed_seq = |toks: &[usize],
                         state: &mut F32RunState,
                         out_emb: &mut [f32]| {
        state.embed_pool.fill(0.0);
        for (pos, &tok) in toks.iter().enumerate() {
            f32_forward_hidden(cfg, &f32_weights, state, tok, pos);
            for (dst, &val) in state.embed_pool.iter_mut().zip(state.xb.iter()) {
                *dst += val;
            }
        }
        let inv_len = 1.0f32 / (toks.len() as f32);
        let mut sum_sq = 0.0f32;
        for v in state.embed_pool.iter_mut() {
            *v *= inv_len;
            sum_sq += *v * *v;
        }
        let inv_norm = 1.0f32 / (sum_sq + cfg.rms_eps).sqrt();
        for (dst, &v) in out_emb.iter_mut().zip(state.embed_pool.iter()) {
            *dst = v * inv_norm;
        }
    };

    let (f_a0, _, f_b0) = current_thread_heap_stats();
    let t_2b = Instant::now();
    for _ in 0..iters {
        for (i, d_toks) in doc_tokens.iter().enumerate() {
            f32_embed_seq(
                d_toks,
                &mut f32_state,
                &mut b2b_doc_embs[i * dim..(i + 1) * dim],
            );
        }
        f32_embed_seq(query_tokens, &mut f32_state, &mut b2b_q_emb);
        rank_documents_by_cosine(&b2b_q_emb, &b2b_doc_embs, dim, &mut b2b_scores);
    }
    let b2b_ms = t_2b.elapsed().as_secs_f64() * 1000.0 / (iters as f64);
    let (f_a1, _, f_b1) = current_thread_heap_stats();

    // 4. Baseline 2C: Unfused Scalar Quantized Embedder (No JIT l2_norm / No SIMD)
    let mut b2c_doc_embs = vec![0.0f32; num_docs * dim];
    let mut b2c_q_emb = vec![0.0f32; dim];
    let mut b2c_scores = vec![(0usize, 0.0f32); num_docs];

    let scalar_quant_embed_seq =
        |toks: &[usize], arena: &mut AlignedArena, out_emb: &mut [f32]| {
            arena.get_mut(model.embed_accum).fill(0.0);
            for (pos, &tok) in toks.iter().enumerate() {
                scalar_quant_forward_hidden(
                    cfg,
                    &model.layer.workspace,
                    &model.kv_caches,
                    arena,
                    weights,
                    tok,
                    pos,
                );
                let (xb_s, pool_mut) = arena.io_slices(model.layer.workspace.xb, model.embed_accum);
                for (dst, &val) in pool_mut.iter_mut().zip(xb_s.iter()) {
                    *dst += val;
                }
            }
            let inv_len = 1.0f32 / (toks.len() as f32);
            let mut sum_sq = 0.0f32;
            for v in arena.get_mut(model.embed_accum).iter_mut() {
                *v *= inv_len;
                sum_sq += *v * *v;
            }
            let inv_norm = 1.0f32 / (sum_sq + cfg.rms_eps).sqrt();
            for (dst, &v) in out_emb.iter_mut().zip(arena.get(model.embed_accum).iter()) {
                *dst = v * inv_norm;
            }
        };

    let (s_a0, _, s_b0) = current_thread_heap_stats();
    let t_2c = Instant::now();
    for _ in 0..iters {
        for (i, d_toks) in doc_tokens.iter().enumerate() {
            scalar_quant_embed_seq(d_toks, arena, &mut b2c_doc_embs[i * dim..(i + 1) * dim]);
        }
        scalar_quant_embed_seq(query_tokens, arena, &mut b2c_q_emb);
        rank_documents_by_cosine(&b2c_q_emb, &b2c_doc_embs, dim, &mut b2c_scores);
    }
    let b2c_ms = t_2c.elapsed().as_secs_f64() * 1000.0 / (iters as f64);
    let (s_a1, _, s_b1) = current_thread_heap_stats();

    let total_sequences = (num_docs + 1) as f64;
    let total_tokens_f = total_tokens as f64;

    [
        EmbedBenchMetrics {
            engine_name: "static-lm (Ours)",
            arch_summary: "In-Arena Mean Pool + Cranelift JIT 2-Pass L2 Norm",
            num_docs,
            total_tokens,
            total_ms: static_ms,
            ms_per_doc: static_ms / total_sequences,
            docs_per_sec: (total_sequences * 1000.0) / static_ms,
            tok_per_sec: (total_tokens_f * 1000.0) / static_ms,
            speedup_vs_static_lm: 1.0,
            heap_allocs: (a1 - a0) / iters,
            heap_bytes: (b1 - b0) / iters,
            static_memory_bytes: q4_weight_bytes + arena_used_bytes,
            top_doc_idx: static_scores[0].0,
            top_cosine: static_scores[0].1,
            ranking: static_scores,
        },
        EmbedBenchMetrics {
            engine_name: "Baseline 2A (fastembed/candle)",
            arch_summary: "Heap Vec<Vec<f32>> Pooling + Heap L2 Norm & Sort",
            num_docs,
            total_tokens,
            total_ms: b2a_ms,
            ms_per_doc: b2a_ms / total_sequences,
            docs_per_sec: (total_sequences * 1000.0) / b2a_ms,
            tok_per_sec: (total_tokens_f * 1000.0) / b2a_ms,
            speedup_vs_static_lm: b2a_ms / static_ms,
            heap_allocs: (h_a1 - h_a0) / iters,
            heap_bytes: (h_b1 - h_b0) / iters,
            static_memory_bytes: q4_weight_bytes
                + heap_kv
                    .iter()
                    .map(|kv| (kv.k.len() + kv.v.len()) * 4)
                    .sum::<usize>(),
            top_doc_idx: b2a_scores[0].0,
            top_cosine: b2a_scores[0].1,
            ranking: b2a_scores,
        },
        EmbedBenchMetrics {
            engine_name: "Baseline 2B (llama2.c-style)",
            arch_summary: "Static f32 RunState + Scalar 2-Pass L2 Norm",
            num_docs,
            total_tokens,
            total_ms: b2b_ms,
            ms_per_doc: b2b_ms / total_sequences,
            docs_per_sec: (total_sequences * 1000.0) / b2b_ms,
            tok_per_sec: (total_tokens_f * 1000.0) / b2b_ms,
            speedup_vs_static_lm: b2b_ms / static_ms,
            heap_allocs: (f_a1 - f_a0) / iters,
            heap_bytes: (f_b1 - f_b0) / iters,
            static_memory_bytes: f32_weights.total_bytes() + f32_state.total_bytes(),
            top_doc_idx: b2b_scores[0].0,
            top_cosine: b2b_scores[0].1,
            ranking: b2b_scores,
        },
        EmbedBenchMetrics {
            engine_name: "Baseline 2C (Unfused Q4)",
            arch_summary: "Scalar Q4xQ8 Forward + Unfused Scalar L2 Norm",
            num_docs,
            total_tokens,
            total_ms: b2c_ms,
            ms_per_doc: b2c_ms / total_sequences,
            docs_per_sec: (total_sequences * 1000.0) / b2c_ms,
            tok_per_sec: (total_tokens_f * 1000.0) / b2c_ms,
            speedup_vs_static_lm: b2c_ms / static_ms,
            heap_allocs: (s_a1 - s_a0) / iters,
            heap_bytes: (s_b1 - s_b0) / iters,
            static_memory_bytes: q4_weight_bytes + arena_used_bytes,
            top_doc_idx: b2c_scores[0].0,
            top_cosine: b2c_scores[0].1,
            ranking: b2c_scores,
        },
    ]
}

trait ArenaFootprintExt {
    fn used_floats_or_bytes(
        &self,
        cfg: &TransformerConfig,
        num_layers: usize,
        vocab_size: usize,
    ) -> usize;
}

impl ArenaFootprintExt for AlignedArena {
    fn used_floats_or_bytes(
        &self,
        cfg: &TransformerConfig,
        num_layers: usize,
        vocab_size: usize,
    ) -> usize {
        let kv_floats = num_layers * 2 * cfg.max_seq_len * cfg.kv_dim();
        let ws_floats = 5 * cfg.dim
            + cfg.num_heads * cfg.max_seq_len
            + 3 * cfg.hidden_dim
            + vocab_size
            + cfg.dim;
        (kv_floats + ws_floats) * std::mem::size_of::<f32>()
    }
}

/// Prints a formatted ASCII comparative table for Application 1 (`generate`).
pub fn print_generate_bench_table(results: &[GenerateBenchMetrics; 4]) {
    println!(
        "\n=== Application 1 Comparative Benchmark: Text Generation (prompt={} toks, gen={} toks) ===",
        results[0].prompt_tokens, results[0].generated_tokens
    );
    println!(
        "{:<29} | {:>11} | {:>11} | {:>9} | {:>8} | {:>10} | {:>10} | {:>10}",
        "Engine / Baseline",
        "Prefill t/s",
        "Decode t/s",
        "Total ms",
        "Rel Time",
        "Heap Alloc",
        "Heap KB",
        "Static KB"
    );
    println!("{}", "-".repeat(118));
    for r in results {
        println!(
            "{:<29} | {:>11.1} | {:>11.1} | {:>9.2} | {:>7.2}x | {:>10} | {:>10.1} | {:>10.1}",
            r.engine_name,
            r.prefill_tok_per_sec,
            r.decode_tok_per_sec,
            r.total_ms,
            r.speedup_vs_static_lm,
            r.heap_allocs,
            (r.heap_bytes as f64) / 1024.0,
            (r.static_memory_bytes as f64) / 1024.0,
        );
    }
}

/// Prints a formatted ASCII comparative table for Application 2 (`embed`).
pub fn print_embed_bench_table(results: &[EmbedBenchMetrics; 4]) {
    println!(
        "\n=== Application 2 Comparative Benchmark: Semantic Embedding & Search (docs={}, total_toks={}) ===",
        results[0].num_docs, results[0].total_tokens
    );
    println!(
        "{:<29} | {:>11} | {:>11} | {:>9} | {:>8} | {:>10} | {:>10} | {:>10}",
        "Engine / Baseline",
        "Embed doc/s",
        "Embed tok/s",
        "Total ms",
        "Rel Time",
        "Heap Alloc",
        "Heap KB",
        "Top Cosine"
    );
    println!("{}", "-".repeat(118));
    for r in results {
        println!(
            "{:<29} | {:>11.1} | {:>11.1} | {:>9.2} | {:>7.2}x | {:>10} | {:>10.1} | {:>10.4}",
            r.engine_name,
            r.docs_per_sec,
            r.tok_per_sec,
            r.total_ms,
            r.speedup_vs_static_lm,
            r.heap_allocs,
            (r.heap_bytes as f64) / 1024.0,
            r.top_cosine,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use crate::model::transformer::TransformerWeights;
    use crate::quant::blocks::quantize_row_q4_0;

    fn pseudo_random_f32(len: usize, mut seed: u32) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let unit = (seed as f32) / (u32::MAX as f32);
            out.push((unit * 2.0 - 1.0) * 0.05);
        }
        out
    }

    fn quantize_vec(f32_data: &[f32]) -> Vec<BlockQ4_0> {
        let mut blocks = vec![BlockQ4_0::zeroed(); f32_data.len() / QK4_0];
        quantize_row_q4_0(f32_data, &mut blocks);
        blocks
    }

    #[test]
    fn test_comparative_benchmarks_parity_and_allocations() {
        let cfg = TransformerConfig {
            dim: 64,
            hidden_dim: 128,
            num_heads: 4,
            num_kv_heads: 2,
            max_seq_len: 16,
            rms_eps: 1e-5,
        };
        let vocab_size = 64usize;
        let kv_dim = cfg.kv_dim();

        let token_embd = quantize_vec(&pseudo_random_f32(vocab_size * cfg.dim, 101));
        let rms_att = vec![1.0f32; cfg.dim];
        let wq = quantize_vec(&pseudo_random_f32(cfg.dim * cfg.dim, 102));
        let wk = quantize_vec(&pseudo_random_f32(kv_dim * cfg.dim, 103));
        let wv = quantize_vec(&pseudo_random_f32(kv_dim * cfg.dim, 104));
        let wo = quantize_vec(&pseudo_random_f32(cfg.dim * cfg.dim, 105));
        let rms_ffn = vec![1.0f32; cfg.dim];
        let w_gate = quantize_vec(&pseudo_random_f32(cfg.hidden_dim * cfg.dim, 106));
        let w_up = quantize_vec(&pseudo_random_f32(cfg.hidden_dim * cfg.dim, 107));
        let w_down = quantize_vec(&pseudo_random_f32(cfg.dim * cfg.hidden_dim, 108));
        let output_norm = vec![1.0f32; cfg.dim];
        let lm_head = quantize_vec(&pseudo_random_f32(vocab_size * cfg.dim, 109));

        let model_weights = ModelWeights {
            token_embd: &token_embd,
            layers: vec![TransformerWeights {
                rms_att_weight: &rms_att,
                wq: &wq,
                wk: &wk,
                wv: &wv,
                wo: &wo,
                rms_ffn_weight: &rms_ffn,
                w_gate: &w_gate,
                w_up: &w_up,
                w_down: &w_down,
            }],
            output_norm: &output_norm,
            lm_head: &lm_head,
            vocab_size,
        };

        let mut engine = JitEngine::new();
        let mut arena = AlignedArena::with_capacity(512 * 1024);
        let model = TransformerModel::new(&mut engine, &mut arena, cfg, 1, vocab_size);

        // 1. Test Application 1 (`generate`) Comparative Harness
        let prompt = [1usize, 5, 9, 13];
        let gen_results =
            run_generate_comparative_bench(&engine, &model, &mut arena, &model_weights, &prompt, 4, 1);

        // Verify zero heap allocations for static-lm, Baseline 1B (llama2.c-style), and Baseline 1C
        assert_eq!(gen_results[0].heap_allocs, 0, "static-lm must have 0 heap allocs");
        assert_eq!(gen_results[0].heap_bytes, 0, "static-lm must allocate 0 heap bytes");
        assert!(
            gen_results[1].heap_allocs > 50,
            "Baseline 1A (candle-style) must allocate heap tensors per op"
        );
        assert!(gen_results[1].heap_bytes > 4096);
        assert_eq!(gen_results[2].heap_allocs, 0, "Baseline 1B (llama2.c-style) has 0 runtime allocs");
        assert_eq!(gen_results[3].heap_allocs, 0, "Baseline 1C has 0 runtime allocs");
        assert!(
            gen_results[2].static_memory_bytes > gen_results[0].static_memory_bytes,
            "Baseline 1B f32 weights must consume more static memory than Q4_0 static-lm"
        );

        // Verify exact generated token parity between static-lm and quantized Baseline 1C
        assert_eq!(
            gen_results[0].output_tokens, gen_results[3].output_tokens,
            "static-lm and Baseline 1C must generate identical tokens"
        );
        assert_eq!(
            gen_results[1].output_tokens, gen_results[2].output_tokens,
            "Baseline 1A and 1B (dequantized f32) must generate identical tokens"
        );

        // 2. Test Application 2 (`embed`) Comparative Harness
        let docs = vec![vec![1usize, 2, 3, 4], vec![10usize, 20, 30, 40], vec![1usize, 2, 3, 4]];
        let query = [1usize, 2, 3, 4];
        let embed_results = run_embed_comparative_bench(
            &engine,
            &model,
            &mut arena,
            &model_weights,
            &docs,
            &query,
            1,
        );

        assert_eq!(embed_results[0].heap_allocs, 0, "static-lm embed must have 0 heap allocs");
        assert_eq!(embed_results[0].heap_bytes, 0, "static-lm embed must allocate 0 heap bytes");
        assert!(
            embed_results[1].heap_allocs > 50,
            "Baseline 2A (fastembed/candle-style) must allocate heap tensors"
        );
        assert_eq!(embed_results[2].heap_allocs, 0);
        assert_eq!(embed_results[3].heap_allocs, 0);

        for r in &embed_results {
            assert!(
                r.top_doc_idx == 0 || r.top_doc_idx == 2,
                "Engine {} failed to rank matching doc first",
                r.engine_name
            );
            assert!(
                (r.top_cosine - 1.0).abs() < 1e-4,
                "Engine {} top cosine {} should be ~1.0",
                r.engine_name,
                r.top_cosine
            );
            // Non-matching doc[1] must be ranked last across all 4 engines
            assert_eq!(r.ranking[2].0, 1);
            assert!(
                (r.ranking[2].1 - embed_results[0].ranking[2].1).abs() < 5e-3,
                "Cosine similarity parity mismatch for {}: {} vs {}",
                r.engine_name,
                r.ranking[2].1,
                embed_results[0].ranking[2].1
            );
        }
    }
}
