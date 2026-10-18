# `static-lm`

CPU-only GGUF language and embedding model runtime in Rust. It maps quantized weights directly from disk via `mmap`, pre-allocates all working memory upfront in a single 64-byte-aligned arena, executes `AVX2`/`NEON` `Q4_0 × Q8_0` matrix kernels, and JIT-compiles vector operators (`RMSNorm`, `SwiGLU`, `Softmax`, `L2Norm`, `ResidualAdd`) to native machine code using Cranelift.

## Guarantees & Scope

- **Zero runtime heap allocations:** Once initialized, token generation, prompt prefill, and embedding extraction perform `0` heap allocations (`malloc` and `free` are never called in the hot loop).
- **Fixed memory footprint:** All activation buffers, `Q8_0` quantization scratch space, and KV-caches live in a single pre-allocated buffer (`AlignedArena`). Memory usage never grows during execution.
- **Zero-copy weights:** `.gguf` (v2/v3) weight tensors are referenced directly from the OS page cache as `&[BlockQ4_0]` and `&[f32]` slices.
- **Supported models:** Llama-family architectures (`RMSNorm`, `RoPE`, Multi-Head / Grouped-Query Attention, `SwiGLU`) quantized with `Q4_0` weights and `F32` normalization scales.

## Usage

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

Run the test suite (verifies quantization error bounds, SIMD parity, Cranelift IR register/loop invariants, and zero runtime heap allocations):

```bash
cargo test
```

## Performance Characteristics

On a 2-layer `Q4_0` model (`dim = 256, hidden_dim = 768, heads = 8/2 GQA, vocab = 512`) running on an AMD Ryzen 5 PRO 4650U (`cargo run --release`), `static-lm` compared against in-tree reference baselines:

- **4.8×–5.1× faster** than unfused scalar `Q4_0 × Q8_0` execution on the same static arena, driven by `AVX2`/`NEON` integer dot products, multi-threaded `GEMV`/`GEMM`, and Cranelift JIT-fused vector loops.
- **7.1×–7.5× faster** than a heap-allocating `Vec<f32>` tensor pipeline that dequantizes `Q4_0` weights on the fly (`0` heap allocations vs. `1,000+` allocations per run).
- **6.6× smaller static memory footprint** (`1.06 MB` vs. `7.00 MB`) than a pre-dequantized `f32` runtime (`4.5 bits/weight` vs. `32 bits/weight`).

*Note: Default benchmark numbers reflect an L2/L3-cache-resident synthetic model against in-process reference baselines, not external C++ runtimes (`llama.cpp`) or multi-gigabyte DRAM-bound models.*
