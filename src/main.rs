use std::path::PathBuf;
use std::time::Instant;

use bytemuck::Zeroable;
use inference_kernel::jit::engine::JitEngine;
use inference_kernel::memory::arena::AlignedArena;
use inference_kernel::memory::loader::{GgufFile, GgufWriter};
use inference_kernel::model::transformer::{TransformerConfig, TransformerModel};
use inference_kernel::quant::blocks::{BlockQ4_0, QK4_0, quantize_row_q4_0};

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

fn build_synthetic_gguf(
    path: &std::path::Path,
    cfg: &TransformerConfig,
    num_layers: usize,
    vocab_size: usize,
) {
    let mut writer = GgufWriter::new();
    writer.add_str("general.architecture", "llama");
    writer.add_u32("llama.block_count", num_layers as u32);
    writer.add_u32("llama.embedding_length", cfg.dim as u32);
    writer.add_u32("llama.feed_forward_length", cfg.hidden_dim as u32);
    writer.add_u32("llama.attention.head_count", cfg.num_heads as u32);
    writer.add_u32("llama.context_length", cfg.max_seq_len as u32);
    writer.add_f32("llama.attention.layer_norm_rms_epsilon", cfg.rms_eps);

    let norm_ones = vec![1.0f32; cfg.dim];
    let vocab_dims = [cfg.dim as u64, vocab_size as u64];
    let qkv_dims = [cfg.dim as u64, cfg.dim as u64];
    let ffn_up_dims = [cfg.dim as u64, cfg.hidden_dim as u64];
    let ffn_down_dims = [cfg.hidden_dim as u64, cfg.dim as u64];

    writer.add_tensor_q4_0(
        "token_embd.weight",
        &vocab_dims,
        &quantize_vec(&pseudo_random_f32(vocab_size * cfg.dim, 0x0101)),
    );

    for l in 0..num_layers {
        let prefix = format!("blk.{l}");
        let base_seed = (l as u32 + 1) * 0x1000;

        writer.add_tensor_f32(&format!("{prefix}.attn_norm.weight"), &[cfg.dim as u64], &norm_ones);
        writer.add_tensor_q4_0(
            &format!("{prefix}.attn_q.weight"),
            &qkv_dims,
            &quantize_vec(&pseudo_random_f32(cfg.dim * cfg.dim, base_seed + 1)),
        );
        writer.add_tensor_q4_0(
            &format!("{prefix}.attn_k.weight"),
            &qkv_dims,
            &quantize_vec(&pseudo_random_f32(cfg.dim * cfg.dim, base_seed + 2)),
        );
        writer.add_tensor_q4_0(
            &format!("{prefix}.attn_v.weight"),
            &qkv_dims,
            &quantize_vec(&pseudo_random_f32(cfg.dim * cfg.dim, base_seed + 3)),
        );
        writer.add_tensor_q4_0(
            &format!("{prefix}.attn_output.weight"),
            &qkv_dims,
            &quantize_vec(&pseudo_random_f32(cfg.dim * cfg.dim, base_seed + 4)),
        );

        writer.add_tensor_f32(&format!("{prefix}.ffn_norm.weight"), &[cfg.dim as u64], &norm_ones);
        writer.add_tensor_q4_0(
            &format!("{prefix}.ffn_gate.weight"),
            &ffn_up_dims,
            &quantize_vec(&pseudo_random_f32(cfg.hidden_dim * cfg.dim, base_seed + 5)),
        );
        writer.add_tensor_q4_0(
            &format!("{prefix}.ffn_up.weight"),
            &ffn_up_dims,
            &quantize_vec(&pseudo_random_f32(cfg.hidden_dim * cfg.dim, base_seed + 6)),
        );
        writer.add_tensor_q4_0(
            &format!("{prefix}.ffn_down.weight"),
            &ffn_down_dims,
            &quantize_vec(&pseudo_random_f32(cfg.dim * cfg.hidden_dim, base_seed + 7)),
        );
    }

    writer.add_tensor_f32("output_norm.weight", &[cfg.dim as u64], &norm_ones);
    writer.add_tensor_q4_0(
        "output.weight",
        &vocab_dims,
        &quantize_vec(&pseudo_random_f32(vocab_size * cfg.dim, 0x0202)),
    );

    writer
        .write_to_file(path)
        .expect("failed to write synthetic GGUF model");
}

fn main() {
    let cli_path = std::env::args().nth(1).map(PathBuf::from);
    let is_temp = cli_path.is_none();
    let gguf_path = cli_path
        .unwrap_or_else(|| std::env::temp_dir().join("inference_kernel_bench.gguf"));

    if is_temp {
        let default_cfg = TransformerConfig {
            dim: 512,
            hidden_dim: 1536,
            num_heads: 8,
            max_seq_len: 128,
            rms_eps: 1e-5,
        };
        build_synthetic_gguf(&gguf_path, &default_cfg, 4, 2048);
    }

    let gguf = GgufFile::open(&gguf_path).expect("failed to open GGUF file");
    let arch = gguf.get_str("general.architecture").unwrap_or("llama");
    let num_layers = gguf
        .get_usize(&format!("{arch}.block_count"))
        .unwrap_or(4);
    let cfg = TransformerConfig {
        dim: gguf
            .get_usize(&format!("{arch}.embedding_length"))
            .unwrap_or(512),
        hidden_dim: gguf
            .get_usize(&format!("{arch}.feed_forward_length"))
            .unwrap_or(1536),
        num_heads: gguf
            .get_usize(&format!("{arch}.attention.head_count"))
            .unwrap_or(8),
        max_seq_len: gguf
            .get_usize(&format!("{arch}.context_length"))
            .unwrap_or(128)
            .min(512),
        rms_eps: gguf
            .get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5),
    };
    let steps = 64.min(cfg.max_seq_len);

    let model_weights = gguf
        .model_weights(num_layers)
        .expect("failed to resolve GGUF model tensors");

    let mut engine = JitEngine::new();
    let mut arena = AlignedArena::with_capacity(8 * 1024 * 1024);
    let model = TransformerModel::new(
        &mut engine,
        &mut arena,
        cfg,
        num_layers,
        model_weights.vocab_size,
    );

    let mut generated_tokens = Vec::with_capacity(steps);
    let mut cur_token = 1usize;

    let start = Instant::now();
    for pos in 0..steps {
        cur_token = model.forward_token(&engine, &mut arena, &model_weights, cur_token, pos);
        generated_tokens.push(cur_token);
    }
    let elapsed = start.elapsed();

    if is_temp {
        let _ = std::fs::remove_file(&gguf_path);
    }

    let qkv_blocks = cfg.dim * (cfg.dim / QK4_0);
    let ffn_blocks = cfg.hidden_dim * (cfg.dim / QK4_0);
    let lm_head_blocks = model_weights.vocab_size * (cfg.dim / QK4_0);
    let total_blocks = num_layers * (4 * qkv_blocks + 3 * ffn_blocks) + lm_head_blocks;
    let secs = elapsed.as_secs_f64();
    let ms_per_tok = (secs * 1000.0) / (steps as f64);
    let tok_per_sec = (steps as f64) / secs;
    let bytes_per_tok = (total_blocks * std::mem::size_of::<BlockQ4_0>()) as f64;
    let gb_per_sec = (bytes_per_tok * (steps as f64)) / (secs * 1e9);

    println!(
        "inference-kernel GGUF v{} | arch={} layers={} dim={} hidden={} vocab={} steps={}",
        gguf.version, arch, num_layers, cfg.dim, cfg.hidden_dim, model_weights.vocab_size, steps
    );
    println!(
        "latency: {:.3} ms/tok | throughput: {:.1} tok/s | weight bandwidth: {:.2} GB/s | tokens[0..8]: {:?}",
        ms_per_tok,
        tok_per_sec,
        gb_per_sec,
        &generated_tokens[..steps.min(8)]
    );
}
