use std::fs::File;
use std::io::Write;
use std::time::Instant;

use bytemuck::Zeroable;
use inference_kernel::jit::engine::JitEngine;
use inference_kernel::memory::arena::AlignedArena;
use inference_kernel::memory::loader::{
    MappedWeights, WEIGHT_FILE_MAGIC, WEIGHT_FILE_VERSION, WeightFileHeader,
};
use inference_kernel::model::transformer::{
    LayerWorkspace, TransformerConfig, TransformerLayer, TransformerWeights,
};
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

fn main() {
    let cfg = TransformerConfig {
        dim: 512,
        hidden_dim: 1536,
        num_heads: 8,
        max_seq_len: 128,
        rms_eps: 1e-5,
    };
    let num_layers = 4;
    let steps = 64;

    let qkv_blocks = cfg.dim * (cfg.dim / QK4_0);
    let ffn_blocks = cfg.hidden_dim * (cfg.dim / QK4_0);
    let blocks_per_layer = 4 * qkv_blocks + 3 * ffn_blocks;
    let total_blocks = num_layers * blocks_per_layer;

    // Generate and serialize synthetic Q4_0 model weights to a binary file
    let weight_path = std::env::temp_dir().join("inference_kernel_bench.bin");
    {
        let raw_f32 = pseudo_random_f32(total_blocks * QK4_0, 0xCAFE_F00D);
        let mut all_blocks = vec![BlockQ4_0::zeroed(); total_blocks];
        quantize_row_q4_0(&raw_f32, &mut all_blocks);

        let header = WeightFileHeader {
            magic: WEIGHT_FILE_MAGIC,
            version: WEIGHT_FILE_VERSION,
            num_blocks: total_blocks as u64,
        };
        let mut file = File::create(&weight_path).expect("failed to create weight file");
        file.write_all(bytemuck::bytes_of(&header))
            .expect("failed to write header");
        file.write_all(bytemuck::cast_slice(&all_blocks))
            .expect("failed to write blocks");
    }

    // Zero-copy mmap weight file
    let mapped = MappedWeights::open(&weight_path).expect("failed to mmap weight file");
    let hdr = mapped.header().expect("invalid weight file header");
    let all_q4 = mapped.view_q4_0(
        std::mem::size_of::<WeightFileHeader>(),
        hdr.num_blocks as usize,
    );

    let rms_att_w = vec![1.0f32; cfg.dim];
    let rms_ffn_w = vec![1.0f32; cfg.dim];

    let mut layer_weights = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let base = l * blocks_per_layer;
        let wq = &all_q4[base..base + qkv_blocks];
        let wk = &all_q4[base + qkv_blocks..base + 2 * qkv_blocks];
        let wv = &all_q4[base + 2 * qkv_blocks..base + 3 * qkv_blocks];
        let wo = &all_q4[base + 3 * qkv_blocks..base + 4 * qkv_blocks];
        let ffn_base = base + 4 * qkv_blocks;
        let w_gate = &all_q4[ffn_base..ffn_base + ffn_blocks];
        let w_up = &all_q4[ffn_base + ffn_blocks..ffn_base + 2 * ffn_blocks];
        let w_down = &all_q4[ffn_base + 2 * ffn_blocks..ffn_base + 3 * ffn_blocks];

        layer_weights.push(TransformerWeights {
            rms_att_weight: &rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight: &rms_ffn_w,
            w_gate,
            w_up,
            w_down,
        });
    }

    // Initialize JIT engine and static activation arena
    let mut engine = JitEngine::new();
    let mut arena = AlignedArena::with_capacity(2 * 1024 * 1024);
    let ws = LayerWorkspace::allocate(&mut arena, &cfg);
    let layer = TransformerLayer::new(&mut engine, cfg, ws);

    let init_x = pseudo_random_f32(cfg.dim, 0x1234_5678);
    arena.get_mut(layer.workspace.x).copy_from_slice(&init_x);

    // Execute autoregressive token generation benchmark
    let start = Instant::now();
    for pos in 0..steps {
        for lw in &layer_weights {
            layer.forward_token(&engine, &mut arena, lw, pos);
        }
    }
    let elapsed = start.elapsed();

    let _ = std::fs::remove_file(&weight_path);

    let secs = elapsed.as_secs_f64();
    let ms_per_tok = (secs * 1000.0) / (steps as f64);
    let tok_per_sec = (steps as f64) / secs;
    let bytes_per_tok = (total_blocks * std::mem::size_of::<BlockQ4_0>()) as f64;
    let gb_per_sec = (bytes_per_tok * (steps as f64)) / (secs * 1e9);

    println!(
        "inference-kernel benchmark | layers={} dim={} hidden={} steps={}",
        num_layers, cfg.dim, cfg.hidden_dim, steps
    );
    println!(
        "latency: {:.3} ms/tok | throughput: {:.1} tok/s | weight bandwidth: {:.2} GB/s | out[0]: {:.6}",
        ms_per_tok,
        tok_per_sec,
        gb_per_sec,
        arena.get(layer.workspace.x)[0]
    );
}
