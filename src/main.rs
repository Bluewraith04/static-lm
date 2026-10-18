use std::path::PathBuf;
use std::time::Instant;

use bytemuck::Zeroable;
use static_lm::jit::engine::JitEngine;
use static_lm::memory::arena::AlignedArena;
use static_lm::memory::loader::{GgufFile, GgufWriter};
use static_lm::model::bench::{
    print_embed_bench_table, print_generate_bench_table, run_embed_comparative_bench,
    run_generate_comparative_bench,
};
use static_lm::model::tokenizer::GgufTokenizer;
use static_lm::model::transformer::{
    TransformerConfig, TransformerModel, rank_documents_by_cosine,
};
use static_lm::quant::blocks::{BlockQ4_0, QK4_0, quantize_row_q4_0};

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

fn build_synthetic_gguf(path: &std::path::Path, cfg: &TransformerConfig, num_layers: usize) {
    let mut writer = GgufWriter::new();
    writer.add_str("general.architecture", "llama");
    writer.add_u32("llama.block_count", num_layers as u32);
    writer.add_u32("llama.embedding_length", cfg.dim as u32);
    writer.add_u32("llama.feed_forward_length", cfg.hidden_dim as u32);
    writer.add_u32("llama.attention.head_count", cfg.num_heads as u32);
    writer.add_u32("llama.attention.head_count_kv", cfg.num_kv_heads as u32);
    writer.add_u32("llama.rope.dimension_count", cfg.head_dim() as u32);
    writer.add_u32("llama.context_length", cfg.max_seq_len as u32);
    writer.add_f32("llama.attention.layer_norm_rms_epsilon", cfg.rms_eps);
    writer.add_str("tokenizer.ggml.model", "llama");

    // Build embedded GGUF vocabulary + BPE merge rules
    let base_tokens = [
        "<unk>", "<s>", "</s>", "▁", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3",
        "4", "5", "6", "7", "8", "9", "-", "_", ".", ",", ":", "/", "▁s", "▁st", "▁sta", "▁stat",
        "▁stati", "▁static", "▁l", "▁lm", "▁g", "▁gg", "▁ggu", "▁gguf", "▁j", "▁ji", "▁jit",
        "▁m", "▁mm", "▁mma", "▁mmap", "▁a", "▁ar", "▁are", "▁aren", "▁arena", "▁q", "▁q4",
        "▁q8", "▁q40", "▁si", "▁sim", "▁simd",
    ];
    let vocab_size = 512usize;
    let mut owned_vocab: Vec<String> = base_tokens.iter().map(|&s| s.to_owned()).collect();
    while owned_vocab.len() < vocab_size {
        owned_vocab.push(format!("▁tok{}", owned_vocab.len()));
    }
    let vocab_refs: Vec<&str> = owned_vocab.iter().map(String::as_str).collect();
    let scores: Vec<f32> = (0..vocab_size).map(|i| i as f32).collect();
    let merges = [
        "▁ s",
        "▁s t",
        "▁st a",
        "▁sta t",
        "▁stat i",
        "▁stati c",
        "▁ l",
        "▁l m",
        "▁ g",
        "▁g g",
        "▁gg u",
        "▁ggu f",
        "▁ j",
        "▁j i",
        "▁ji t",
        "▁ m",
        "▁m m",
        "▁mm a",
        "▁mma p",
        "▁ a",
        "▁a r",
        "▁ar e",
        "▁are n",
        "▁aren a",
        "▁s i",
        "▁si m",
        "▁sim d",
    ];

    writer.add_str_array("tokenizer.ggml.tokens", &vocab_refs);
    writer.add_f32_array("tokenizer.ggml.scores", &scores);
    writer.add_str_array("tokenizer.ggml.merges", &merges);
    writer.add_u32("tokenizer.ggml.bos_token_id", 1);
    writer.add_u32("tokenizer.ggml.eos_token_id", 2);
    writer.add_u32("tokenizer.ggml.unknown_token_id", 0);

    let kv_dim = cfg.kv_dim();
    let norm_ones = vec![1.0f32; cfg.dim];
    let vocab_dims = [cfg.dim as u64, vocab_size as u64];
    let q_dims = [cfg.dim as u64, cfg.dim as u64];
    let kv_dims = [cfg.dim as u64, kv_dim as u64];
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

        writer.add_tensor_f32(
            &format!("{prefix}.attn_norm.weight"),
            &[cfg.dim as u64],
            &norm_ones,
        );
        writer.add_tensor_q4_0(
            &format!("{prefix}.attn_q.weight"),
            &q_dims,
            &quantize_vec(&pseudo_random_f32(cfg.dim * cfg.dim, base_seed + 1)),
        );
        writer.add_tensor_q4_0(
            &format!("{prefix}.attn_k.weight"),
            &kv_dims,
            &quantize_vec(&pseudo_random_f32(kv_dim * cfg.dim, base_seed + 2)),
        );
        writer.add_tensor_q4_0(
            &format!("{prefix}.attn_v.weight"),
            &kv_dims,
            &quantize_vec(&pseudo_random_f32(kv_dim * cfg.dim, base_seed + 3)),
        );
        writer.add_tensor_q4_0(
            &format!("{prefix}.attn_output.weight"),
            &q_dims,
            &quantize_vec(&pseudo_random_f32(cfg.dim * cfg.dim, base_seed + 4)),
        );

        writer.add_tensor_f32(
            &format!("{prefix}.ffn_norm.weight"),
            &[cfg.dim as u64],
            &norm_ones,
        );
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
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("all");
    let cli_path = if matches!(mode, "bench" | "generate" | "embed" | "all" | "write-synthetic") {
        args.get(2).map(PathBuf::from)
    } else {
        args.get(1).map(PathBuf::from)
    };

    let is_temp = cli_path.is_none();
    let gguf_path =
        cli_path.unwrap_or_else(|| std::env::temp_dir().join("static_lm_bench.gguf"));

    if is_temp || mode == "write-synthetic" {
        let default_cfg = TransformerConfig {
            dim: 256,
            hidden_dim: 768,
            num_heads: 8,
            num_kv_heads: 2,
            max_seq_len: 64,
            rms_eps: 1e-5,
        };
        build_synthetic_gguf(&gguf_path, &default_cfg, 2);
        if mode == "write-synthetic" {
            println!("wrote synthetic GGUF to {}", gguf_path.display());
            return;
        }
    }

    let gguf = GgufFile::open(&gguf_path).expect("failed to open GGUF file");
    let arch = gguf.get_str("general.architecture").unwrap_or("llama");
    let num_layers = gguf
        .get_usize(&format!("{arch}.block_count"))
        .unwrap_or(2);
    let num_heads = gguf
        .get_usize(&format!("{arch}.attention.head_count"))
        .unwrap_or(8);
    let cfg = TransformerConfig {
        dim: gguf
            .get_usize(&format!("{arch}.embedding_length"))
            .unwrap_or(256),
        hidden_dim: gguf
            .get_usize(&format!("{arch}.feed_forward_length"))
            .unwrap_or(768),
        num_heads,
        num_kv_heads: gguf
            .get_usize(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(num_heads),
        max_seq_len: gguf
            .get_usize(&format!("{arch}.context_length"))
            .unwrap_or(64)
            .min(512),
        rms_eps: gguf
            .get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5),
    };

    let model_weights = gguf
        .model_weights(num_layers)
        .expect("failed to resolve GGUF model tensors");

    let mut engine = JitEngine::new();
    let kv_floats = num_layers * 2 * cfg.max_seq_len * cfg.kv_dim();
    let batch_floats = cfg.max_seq_len * (6 * cfg.dim + 4 * cfg.hidden_dim);
    let arena_cap =
        (kv_floats + batch_floats + model_weights.vocab_size * 2 + 1024 * 1024).max(8 * 1024 * 1024);
    let mut arena = AlignedArena::with_capacity(arena_cap);
    let model = TransformerModel::new(
        &mut engine,
        &mut arena,
        cfg,
        num_layers,
        model_weights.vocab_size,
    );

    println!(
        "static-lm GGUF v{} | arch={} layers={} dim={} hidden={} heads={}/{} vocab={}",
        gguf.version,
        arch,
        num_layers,
        cfg.dim,
        cfg.hidden_dim,
        cfg.num_heads,
        cfg.num_kv_heads,
        model_weights.vocab_size
    );

    if matches!(mode, "generate" | "all") {
        if let Ok(tokenizer) = GgufTokenizer::from_gguf(&gguf) {
            let prompt = "static lm jit mmap arena";
            let prompt_ids = tokenizer.encode(prompt, true);
            let mut gen_ids = [0usize; 8];
            let start = Instant::now();
            let count = model.generate(
                &engine,
                &mut arena,
                &model_weights,
                &prompt_ids,
                &mut gen_ids,
                Some(tokenizer.eos_token_id),
            );
            let elapsed = start.elapsed();
            let decoded = tokenizer.decode(&gen_ids[..count]);
            println!(
                "[generate] prompt={:?} (ids={:?}) -> generated={} tokens in {:.2} ms: {:?}",
                prompt,
                prompt_ids,
                count,
                elapsed.as_secs_f64() * 1000.0,
                decoded
            );
        }
    }

    if matches!(mode, "embed" | "all") {
        if let Ok(tokenizer) = GgufTokenizer::from_gguf(&gguf) {
            let docs = [
                "static lm mmap arena",
                "jit simd kernel",
                "static lm mmap arena",
            ];
            let query = "static lm mmap arena";
            let mut doc_embs = vec![0.0f32; docs.len() * cfg.dim];
            let mut query_emb = vec![0.0f32; cfg.dim];
            let mut scores = vec![(0usize, 0.0f32); docs.len()];

            for (i, doc) in docs.iter().enumerate() {
                let ids = tokenizer.encode(doc, true);
                model.embed_tokens(
                    &engine,
                    &mut arena,
                    &model_weights,
                    &ids,
                    &mut doc_embs[i * cfg.dim..(i + 1) * cfg.dim],
                );
            }
            let q_ids = tokenizer.encode(query, true);
            model.embed_tokens(&engine, &mut arena, &model_weights, &q_ids, &mut query_emb);
            rank_documents_by_cosine(&query_emb, &doc_embs, cfg.dim, &mut scores);

            println!(
                "[embed] query={:?} | top_match=doc[{}] ({:?}, cosine={:.4}) | runner_up=doc[{}] (cosine={:.4})",
                query,
                scores[0].0,
                docs[scores[0].0],
                scores[0].1,
                scores[1].0,
                scores[1].1
            );
        }
    }

    if matches!(mode, "bench" | "all") || !matches!(mode, "generate" | "embed") {
        let steps = 64.min(cfg.max_seq_len);
        let mut cur_token = 1usize;
        let start = Instant::now();
        for pos in 0..steps {
            cur_token = model.forward_token(&engine, &mut arena, &model_weights, cur_token, pos);
        }
        let elapsed = start.elapsed();

        let q_blocks = cfg.dim * (cfg.dim / QK4_0);
        let kv_blocks = cfg.kv_dim() * (cfg.dim / QK4_0);
        let ffn_blocks = cfg.hidden_dim * (cfg.dim / QK4_0);
        let lm_head_blocks = model_weights.vocab_size * (cfg.dim / QK4_0);
        let total_blocks =
            num_layers * (2 * q_blocks + 2 * kv_blocks + 3 * ffn_blocks) + lm_head_blocks;
        let secs = elapsed.as_secs_f64();
        let ms_per_tok = (secs * 1000.0) / (steps as f64);
        let tok_per_sec = (steps as f64) / secs;
        let bytes_per_tok = (total_blocks * std::mem::size_of::<BlockQ4_0>()) as f64;
        let gb_per_sec = (bytes_per_tok * (steps as f64)) / (secs * 1e9);

        println!(
            "[bench] steps={} | latency: {:.3} ms/tok | throughput: {:.1} tok/s | weight bandwidth: {:.2} GB/s | last_token={}",
            steps, ms_per_tok, tok_per_sec, gb_per_sec, cur_token
        );

        let bench_iters = args
            .get(3)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(if num_layers >= 6 { 1 } else { 3 });

        let prompt_ids: Vec<usize> = (1..=16.min(cfg.max_seq_len / 2)).collect();
        let decode_len = 8.min(cfg.max_seq_len - prompt_ids.len());
        let gen_comp = run_generate_comparative_bench(
            &engine,
            &model,
            &mut arena,
            &model_weights,
            &prompt_ids,
            decode_len,
            bench_iters,
        );
        print_generate_bench_table(&gen_comp);

        let doc_seqs = vec![
            vec![1usize, 2, 3, 4, 5, 6, 7, 8],
            vec![9usize, 10, 11, 12, 13, 14, 15, 16],
            vec![17usize, 18, 19, 20, 21, 22, 23, 24],
            vec![1usize, 2, 3, 4, 5, 6, 7, 8],
        ];
        let query_seq = [1usize, 2, 3, 4, 5, 6, 7, 8];
        let embed_comp = run_embed_comparative_bench(
            &engine,
            &model,
            &mut arena,
            &model_weights,
            &doc_seqs,
            &query_seq,
            bench_iters,
        );
        print_embed_bench_table(&embed_comp);
    }

    if is_temp {
        let _ = std::fs::remove_file(&gguf_path);
    }
}

