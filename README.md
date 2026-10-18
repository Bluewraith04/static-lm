# `static-lm`

CPU-only GGUF language and embedding model runtime in Rust. It maps quantized weights directly from disk via `mmap`, pre-allocates all working memory upfront in a single 64-byte-aligned arena, executes `AVX2`/`NEON` `Q4_0 × Q8_0` matrix kernels, and JIT-compiles vector operators (`RMSNorm`, `SwiGLU`, `Softmax`, `L2Norm`, `ResidualAdd`) to native machine code using Cranelift.

## Features & Scope

- **Zero runtime heap allocations:** Once initialized, token generation, prompt prefill, and embedding extraction perform `0` heap allocations (`malloc` and `free` are never called in the hot loop).
- **Fixed memory footprint:** All activation buffers, `Q8_0` quantization scratch space, and KV-caches live in a single pre-allocated buffer (`AlignedArena`). Memory usage never grows during execution.
- **Zero-copy weights:** `.gguf` (v2/v3) weight tensors are referenced directly from the OS page cache as `&[BlockQ4_0]` and `&[f32]` slices.
- **Supported models:** Llama-family architectures (`RMSNorm`, `RoPE`, Multi-Head / Grouped-Query Attention, `SwiGLU`) quantized with `Q4_0` weights and `F32` normalization scales.

## Installation

Clone the repository:

```bash
git clone https://github.com/Bluewraith04/static-lm.git
```

Clone with GitHub CLI:

```bash
gh repo clone Bluewraith04/static-lm
```

Add as a dependency in `Cargo.toml`:

```toml
[dependencies]
static-lm = { git = "https://github.com/Bluewraith04/static-lm.git" }
```

## CLI Usage

Run the full suite (text generation, embedding search, and comparative benchmarks) using a self-generated synthetic GGUF model:

```bash
cargo run --release
```

Run text generation with an embedded GGUF BPE tokenizer, batched `GEMM` prompt prefill, and autoregressive decoding:

```bash
cargo run --release -- generate path/to/model.gguf
```

Compute JIT L2-normalized mean-pooled embeddings and rank documents by cosine similarity:

```bash
cargo run --release -- embed path/to/model.gguf
```

Run throughput and baseline comparison benchmarks:

```bash
cargo run --release -- bench path/to/model.gguf
```

Load a model, prefill a prompt of N tokens, decode M tokens, and exit (used for end-to-end timing):

```bash
cargo run --release -- run path/to/model.gguf N M
```

Run the test suite (verifies quantization error bounds, SIMD parity, Cranelift IR register/loop invariants, and zero runtime heap allocations):

```bash
cargo test
```

## Rust Library Usage

```rust
use std::path::Path;
use static_lm::jit::engine::JitEngine;
use static_lm::memory::arena::AlignedArena;
use static_lm::memory::loader::GgufFile;
use static_lm::model::tokenizer::GgufTokenizer;
use static_lm::model::transformer::{TransformerConfig, TransformerModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = GgufFile::open(Path::new("model.gguf"))?;
    let arch = gguf.get_str("general.architecture").unwrap_or("llama");
    let num_layers = gguf
        .get_u32(&format!("{arch}.block_count"))
        .unwrap_or(1) as usize;
    let model_weights = gguf.model_weights(num_layers)?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf)?;

    let cfg = TransformerConfig {
        dim: gguf.get_u32(&format!("{arch}.embedding_length")).unwrap_or(256) as usize,
        hidden_dim: gguf.get_u32(&format!("{arch}.feed_forward_length")).unwrap_or(512) as usize,
        num_heads: gguf.get_u32(&format!("{arch}.attention.head_count")).unwrap_or(4) as usize,
        num_kv_heads: gguf.get_u32(&format!("{arch}.attention.head_count_kv")).unwrap_or(4) as usize,
        max_seq_len: 512,
        rms_eps: gguf.get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-5),
    };

    let kv_floats = num_layers * 2 * cfg.max_seq_len * cfg.kv_dim();
    let batch_floats = cfg.max_seq_len * (6 * cfg.dim + 4 * cfg.hidden_dim);
    let arena_cap = (kv_floats + batch_floats + model_weights.vocab_size * 2 + 1024 * 1024)
        .max(8 * 1024 * 1024);
    let mut arena = AlignedArena::with_capacity(arena_cap);

    let mut engine = JitEngine::new();
    let model = TransformerModel::new(
        &mut engine,
        &mut arena,
        cfg,
        num_layers,
        model_weights.vocab_size,
    );

    let prompt = tokenizer.encode("Hello world", true);
    let mut output = [0usize; 32];
    let generated = model.generate(
        &engine,
        &mut arena,
        &model_weights,
        &prompt,
        &mut output,
        Some(tokenizer.eos_token_id),
    );

    println!("Decoded: {}", tokenizer.decode(&output[..generated]));
    Ok(())
}
```

## Benchmarks

`AMD Ryzen 5 PRO 4650U` (6 cores / 12 threads), pure `Q4_0` models, measured with `hyperfine -N --warmup 3` (mean ± σ, 10–30 runs). Each run is a full process: load the model, prefill a 16-token prompt, decode 8 tokens. `llama.cpp` is `llama-bench` built from source (`-DGGML_NATIVE=ON`, 3 threads); `candle` is `0.8.4` with `quantized_llama`.

| Model | `static-lm` | `llama.cpp` | `candle` |
| :--- | ---: | ---: | ---: |
| Synthetic (`1.8M` params, `0.95 MiB`) | 76.0 ± 9.8 ms | **49.1 ± 8.5 ms** | 279.1 ± 29.6 ms |
| Stories-15M (`24.4M`, `13.1 MiB`) | **114.6 ± 9.9 ms** | 200.8 ± 13.3 ms | 1,231 ± 65 ms |
| SmolLM2-135M (`134.5M`, `72.3 MiB`) | **703 ± 82 ms** | 790 ± 131 ms | 6,520 ± 172 ms |

Splitting the same measurement into startup and steady-state work (`static-lm` / `llama.cpp`):

| Model | Startup (1-token prompt, 1 token decoded) | 16-token prompt + 64 tokens decoded |
| :--- | ---: | ---: |
| Stories-15M | **69 ms** / 197 ms | 340 ms / **289 ms** |
| SmolLM2-135M | **165 ms** / 580 ms | 2.51 s / **1.30 s** |

- **Startup:** `static-lm` reaches its first token 3× faster than `llama.cpp` on the real models, since it only maps the file, parses the header, and JIT-compiles a handful of small kernels.
- **Sustained decoding:** `llama.cpp` is about 1.2× faster on Stories-15M and 1.9× faster on SmolLM2-135M. Total time favors `static-lm` only when a run is short enough for startup to dominate.
- **CPU use:** `static-lm` workers spin between matrix multiplications, so it consumes roughly 2.5× the CPU time of `llama.cpp` for the same work (6.6 s user time vs. 2.6 s on the SmolLM2-135M run above).
- **vs. `candle`:** `static-lm` is 9×–11× faster end to end on the real models.
- **Memory:** weights stay `Q4_0` and all working memory is one arena; SmolLM2-135M needs about `110 MB` total versus about `644 MB` for the same model dequantized to `f32`.
