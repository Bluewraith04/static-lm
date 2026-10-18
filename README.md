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

## Benchmarks (`AMD Ryzen 5 PRO 4650U`, Pure `Q4_0` CPU, `prompt = 16`, `decode = 8`)

| Model | Runtime | Prefill (`tok/s`) | Decode (`tok/s`) | Runtime Heap Allocs | Static Memory |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Synthetic (`1.8M`, `0.95 MiB`)** | `llama.cpp` (`llama-bench`) | 42,385.9 | 11,209.4 | — | — |
| | **`static-lm`** | **5,150.1** | **4,144.4** | **0 (`0 KB`)** | **1.0 MB** |
| | HuggingFace `candle` (`0.8.4`) | 206.6 | 130.7 | Heap per op | — |
| | In-tree static `f32` (`llama2.c`-style) | 633.6 | 583.8 | 0 (`0 KB`) | 6.8 MB |
| | In-tree unfused scalar `Q4_0` | 514.7 | 398.3 | 0 (`0 KB`) | 1.0 MB |
| **Stories-15M (`24.4M`, `13.1 MiB`)** | `llama.cpp` (`llama-bench`) | 4,251.8 | 636.6 | — | — |
| | **`static-lm`** | **1,271.7** | **469.4** | **0 (`0 KB`)** | **16.6 MB** |
| | HuggingFace `candle` (`0.8.4`) | 36.4 | 26.9 | Heap per op | — |
| | In-tree static `f32` (`llama2.c`-style) | 147.2 | 65.7 | 0 (`0 KB`) | 96.6 MB |
| | In-tree unfused scalar `Q4_0` | 129.7 | 65.9 | 0 (`0 KB`) | 16.6 MB |
| **SmolLM2-135M (`134.5M`, `72.3 MiB`)** | `llama.cpp` (`llama-bench`) | 442.8 | 125.2 | — | — |
| | **`static-lm`** | **78.4** | **54.6** | **0 (`0 KB`)** | **110.2 MB** |
| | HuggingFace `candle` (`0.8.4`) | 7.1 | 5.1 | Heap per op | — |
| | In-tree static `f32` (`llama2.c`-style) | 8.4 | 6.7 | 0 (`0 KB`) | 643.9 MB |
| | In-tree unfused scalar `Q4_0` | 5.6 | 5.2 | 0 (`0 KB`) | 110.2 MB |

- **vs. HuggingFace `candle` (`0.8.4` CPU):** `static-lm` is **10.7×–17.4× faster on decode** and **11×–35× faster on prefill** on the two real-world models (`Stories-15M` and `SmolLM2-135M`).
- **vs. `llama.cpp` (`ggml-cpu`):** `static-lm` reaches **44%–74% of `llama.cpp` decode throughput** (`469.4` vs. `636.6 tok/s` on `Stories-15M`; `54.6` vs. `125.2 tok/s` on `SmolLM2-135M`) and **18%–30% of its prefill throughput**. The remaining gap comes from `ggml`'s multi-row register-blocked `GEMV`/`GEMM` microkernels and SIMD attention scoring (`static-lm` uses single-row `Q4_0 × Q8_0` dot products and scalar `QK^T`/`AV` attention loops). `llama.cpp` figures were measured on the same machine with run-to-run variance of up to ±50% at these small sizes.
- **vs. In-tree baselines:** `static-lm` is **7×–14× faster** than unfused scalar `Q4_0` and uses **5.8× less RAM** (`110.2 MB` vs. `643.9 MB` on `SmolLM2-135M`) than pre-dequantized `f32`, while avoiding the `14,552` heap allocations (`53.4 MB` of transient heap churn) of per-operator `Vec<f32>` execution.
