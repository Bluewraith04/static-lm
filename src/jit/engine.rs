use std::collections::HashMap;

use cranelift_codegen::Context;
use cranelift_codegen::ir::{AbiParam, Opcode};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;

use crate::jit::graph::FuseGraph;
use crate::jit::lower::lower_fuse_graph;
use crate::memory::arena::{AlignedArena, ArenaSlice};

/// Raw JIT-compiled kernel entrypoint: `(inputs_ptr_array, output_ptr, len)`.
pub type JitKernelFn = unsafe extern "C" fn(*const *const f32, *mut f32, usize);

/// Structural and machine-code metrics inspected from a compiled Cranelift IR function.
#[derive(Clone, Debug)]
pub struct JitCompileStats {
    pub code_size: usize,
    pub num_stack_slots: usize,
    pub num_loops: usize,
    pub has_call_insts: bool,
    pub final_loop_sqrt_or_div: usize,
    pub clir_text: String,
}

/// Cranelift JIT compilation engine and kernel cache.
pub struct JitEngine {
    module: JITModule,
    ctx: Context,
    fb_ctx: FunctionBuilderContext,
    cache: HashMap<FuseGraph, JitKernelFn>,
}

impl Default for JitEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl JitEngine {
    pub fn new() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .expect("failed to set opt_level");

        let isa_builder =
            cranelift_native::builder().expect("host architecture not supported by cranelift");
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .expect("failed to build target ISA");

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);
        let ctx = module.make_context();

        Self {
            module,
            ctx,
            fb_ctx: FunctionBuilderContext::new(),
            cache: HashMap::new(),
        }
    }

    /// Compiles a [`FuseGraph`] into a native [`JitKernelFn`], or returns the cached function pointer.
    pub fn compile(&mut self, graph: &FuseGraph) -> JitKernelFn {
        if let Some(&func_ptr) = self.cache.get(graph) {
            return func_ptr;
        }
        self.compile_with_stats(graph).0
    }

    /// Compiles a [`FuseGraph`] and returns both the [`JitKernelFn`] and [`JitCompileStats`]
    /// inspecting the lowered Cranelift IR and emitted machine code.
    pub fn compile_with_stats(&mut self, graph: &FuseGraph) -> (JitKernelFn, JitCompileStats) {
        let ptr_ty = self.module.target_config().pointer_type();
        self.module.clear_context(&mut self.ctx);
        self.ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // inputs: *const *const f32
        self.ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // output: *mut f32
        self.ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // len: usize

        lower_fuse_graph(graph, &mut self.ctx.func, &mut self.fb_ctx);

        let clir_text = self.ctx.func.display().to_string();
        let blocks: Vec<_> = self.ctx.func.layout.blocks().collect();
        // Entry (1) + 6 blocks per ReduceOp pass + 5 blocks for final Store pass = 6 * num_passes
        let num_loops = blocks.len() / 6;
        let final_vec_body_block = blocks.get(blocks.len().saturating_sub(4)).copied();
        let final_tail_body_block = blocks.get(blocks.len().saturating_sub(2)).copied();

        let mut has_call_insts = false;
        let mut final_loop_sqrt_or_div = 0usize;

        for &blk in &blocks {
            let is_final_body =
                Some(blk) == final_vec_body_block || Some(blk) == final_tail_body_block;
            for inst in self.ctx.func.layout.block_insts(blk) {
                let op = self.ctx.func.dfg.insts[inst].opcode();
                if matches!(op, Opcode::Call | Opcode::CallIndirect) {
                    has_call_insts = true;
                }
                if is_final_body && matches!(op, Opcode::Sqrt | Opcode::Fdiv) {
                    final_loop_sqrt_or_div += 1;
                }
            }
        }

        let func_id = self
            .module
            .declare_anonymous_function(&self.ctx.func.signature)
            .expect("failed to declare JIT function");

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("failed to define JIT function");

        let num_stack_slots =
            self.ctx.func.sized_stack_slots.len() + self.ctx.func.dynamic_stack_slots.len();
        let code_size = self
            .ctx
            .compiled_code()
            .expect("missing compiled machine code")
            .buffer
            .data()
            .len();

        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("failed to finalize JIT definitions");

        let code_ptr = self.module.get_finalized_function(func_id);
        // SAFETY: `code_ptr` was compiled with the exact signature of `JitKernelFn`.
        let kernel_fn: JitKernelFn = unsafe { std::mem::transmute(code_ptr) };

        self.cache.insert(graph.clone(), kernel_fn);

        let stats = JitCompileStats {
            code_size,
            num_stack_slots,
            num_loops,
            has_call_insts,
            final_loop_sqrt_or_div,
            clir_text,
        };
        (kernel_fn, stats)
    }

    /// Executes a 1-input, 1-output JIT kernel on [`AlignedArena`] slices with zero allocations.
    #[inline]
    pub fn exec_unary(
        &self,
        kernel: JitKernelFn,
        arena: &mut AlignedArena,
        src: ArenaSlice,
        dst: ArenaSlice,
    ) {
        assert_eq!(src.len, dst.len, "src and dst length mismatch");
        let (s, d) = arena.io_slices(src, dst);
        let inputs = [s.as_ptr()];
        unsafe {
            kernel(inputs.as_ptr(), d.as_mut_ptr(), d.len());
        }
    }

    /// Executes a 2-input, 1-output JIT kernel on [`AlignedArena`] slices with zero allocations.
    #[inline]
    pub fn exec_binary(
        &self,
        kernel: JitKernelFn,
        arena: &mut AlignedArena,
        src_a: ArenaSlice,
        src_b: ArenaSlice,
        dst: ArenaSlice,
    ) {
        assert_eq!(src_a.len, dst.len, "src_a and dst length mismatch");
        assert_eq!(src_b.len, dst.len, "src_b and dst length mismatch");
        let (sa, sb, d) = arena.io2_slices(src_a, src_b, dst);
        let inputs = [sa.as_ptr(), sb.as_ptr()];
        unsafe {
            kernel(inputs.as_ptr(), d.as_mut_ptr(), d.len());
        }
    }

    /// Executes a 2-input, 1-output JIT kernel where input 0 is an [`ArenaSlice`] and input 1
    /// is an external read-only `f32` weight slice (e.g. RMSNorm scale weights).
    #[inline]
    pub fn exec_with_weights(
        &self,
        kernel: JitKernelFn,
        arena: &mut AlignedArena,
        src: ArenaSlice,
        weights: &[f32],
        dst: ArenaSlice,
    ) {
        assert_eq!(src.len, dst.len, "src and dst length mismatch");
        assert_eq!(weights.len(), dst.len, "weights and dst length mismatch");
        let (s, d) = arena.io_slices(src, dst);
        let inputs = [s.as_ptr(), weights.as_ptr()];
        unsafe {
            kernel(inputs.as_ptr(), d.as_mut_ptr(), d.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::graph::{InputId, OutputId};
    use crate::memory::arena::tests::current_thread_heap_ops;

    fn pseudo_random_f32(len: usize, mut seed: u32) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let unit = (seed as f32) / (u32::MAX as f32);
            out.push(unit * 4.0 - 2.0);
        }
        out
    }

    #[test]
    fn test_jit_swiglu_against_scalar_reference() {
        let n = 1024;
        let gate = pseudo_random_f32(n, 0x1357_9BDF);
        let up = pseudo_random_f32(n, 0x2468_ACE0);
        let mut out_jit = vec![0.0f32; n];

        let mut engine = JitEngine::new();
        let (kernel, stats) = engine.compile_with_stats(&FuseGraph::swiglu());

        assert_eq!(stats.num_stack_slots, 0, "SwiGLU must not spill to stack");
        assert!(!stats.has_call_insts, "SwiGLU must not emit external calls");
        assert_eq!(stats.num_loops, 1, "SwiGLU must lower to a 1-pass loop");
        assert!(stats.code_size > 0);

        let inputs = [gate.as_ptr(), up.as_ptr()];
        unsafe {
            kernel(inputs.as_ptr(), out_jit.as_mut_ptr(), n);
        }

        for i in 0..n {
            let ref_val = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
            let diff = (out_jit[i] - ref_val).abs();
            assert!(
                diff < 1e-5,
                "SwiGLU mismatch at {i}: jit={}, ref={ref_val}, diff={diff}",
                out_jit[i]
            );
        }
    }

    #[test]
    fn test_jit_rmsnorm_against_scalar_reference_and_hoisting() {
        let n = 1024;
        let eps = 1e-5f32;
        let x = pseudo_random_f32(n, 0xAAAA_5555);
        let gamma = pseudo_random_f32(n, 0x5555_AAAA);
        let mut out_jit = vec![0.0f32; n];

        let mut engine = JitEngine::new();
        let (kernel, stats) = engine.compile_with_stats(&FuseGraph::rmsnorm(n, eps));

        assert_eq!(stats.num_stack_slots, 0, "RMSNorm must not spill to stack");
        assert!(
            !stats.has_call_insts,
            "RMSNorm must not emit external calls"
        );
        assert_eq!(stats.num_loops, 2, "RMSNorm must lower to a 2-pass loop");
        assert_eq!(
            stats.final_loop_sqrt_or_div, 0,
            "RMSNorm Pass 2 loop must have 0 sqrt/fdiv instructions (hoisted to Pass 1 exit):\n{}",
            stats.clir_text
        );

        let inputs = [x.as_ptr(), gamma.as_ptr()];
        unsafe {
            kernel(inputs.as_ptr(), out_jit.as_mut_ptr(), n);
        }

        let mut sum_sq = 0.0f32;
        for i in 0..n {
            sum_sq += x[i] * x[i];
        }
        let rms = 1.0 / (sum_sq / (n as f32) + eps).sqrt();
        for i in 0..n {
            let ref_val = x[i] * rms * gamma[i];
            let diff = (out_jit[i] - ref_val).abs();
            assert!(
                diff < 1e-5,
                "RMSNorm mismatch at {i}: jit={}, ref={ref_val}, diff={diff}",
                out_jit[i]
            );
        }
    }

    #[test]
    fn test_jit_softmax_and_all_graph_ops() {
        let n = 256;
        let x = pseudo_random_f32(n, 0x7788_99AA);
        let mut out_jit = vec![0.0f32; n];

        let mut engine = JitEngine::new();
        let (softmax_fn, stats) = engine.compile_with_stats(&FuseGraph::softmax());

        assert_eq!(stats.num_stack_slots, 0, "Softmax must not spill to stack");
        assert!(
            !stats.has_call_insts,
            "Softmax must not emit external calls"
        );
        assert_eq!(
            stats.num_loops, 3,
            "Softmax (ReduceMax + ReduceSum + Normalize) must lower to 3 loops"
        );
        assert_eq!(
            stats.final_loop_sqrt_or_div, 0,
            "Softmax recip(sum_ex) must be hoisted out of Pass 3 loop:\n{}",
            stats.clir_text
        );

        let inputs = [x.as_ptr()];
        unsafe {
            softmax_fn(inputs.as_ptr(), out_jit.as_mut_ptr(), n);
        }

        let max_x = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_ex: f32 = x.iter().map(|&v| (v - max_x).exp()).sum();
        for i in 0..n {
            let ref_val = (x[i] - max_x).exp() / sum_ex;
            let diff = (out_jit[i] - ref_val).abs();
            assert!(
                diff < 1e-5,
                "Softmax mismatch at {i}: jit={}, ref={ref_val}, diff={diff}",
                out_jit[i]
            );
        }

        let mut g = FuseGraph::new();
        let a = g.load(InputId(0));
        let b = g.load(InputId(1));
        let neg_a = g.neg(a);
        let diff_ab = g.sub(b, neg_a); // b - (-a) = b + a
        let c_ten = g.constant(10.0);
        let pos_val = g.add(diff_ab, c_ten); // strictly positive (> 6.0)
        let root = g.sqrt(pos_val);
        let inv = g.recip(root);
        let div_res = g.div(pos_val, root);
        let combined = g.add(inv, div_res);
        g.store(OutputId(0), combined);

        let (op_fn, op_stats) = engine.compile_with_stats(&g);
        assert_eq!(op_stats.num_stack_slots, 0);
        assert!(!op_stats.has_call_insts);

        let y = pseudo_random_f32(n, 0xBBCC_DDEE);
        let inputs2 = [x.as_ptr(), y.as_ptr()];
        unsafe {
            op_fn(inputs2.as_ptr(), out_jit.as_mut_ptr(), n);
        }

        for i in 0..n {
            let pv = (y[i] - (-x[i])) + 10.0;
            let rt = pv.sqrt();
            let ref_val = (1.0 / rt) + (pv / rt);
            assert!((out_jit[i] - ref_val).abs() < 1e-5);
        }
    }

    #[test]
    fn test_jit_cache_and_zero_heap_ops_on_arena() {
        let n = 256;
        let mut engine = JitEngine::new();
        let g_swiglu = FuseGraph::swiglu();
        let g_rmsnorm = FuseGraph::rmsnorm(n, 1e-5);
        let g_softmax = FuseGraph::softmax();

        let k_swiglu = engine.compile(&g_swiglu);
        let k_rmsnorm = engine.compile(&g_rmsnorm);
        let k_softmax = engine.compile(&g_softmax);

        let mut arena = AlignedArena::with_capacity(2048);
        let s_a = arena.alloc_slice(n);
        let s_b = arena.alloc_slice(n);
        let s_out = arena.alloc_slice(n);

        let data_a = pseudo_random_f32(n, 0xCAFE_1234);
        let data_b = pseudo_random_f32(n, 0xBEEF_5678);
        let ext_weights = vec![1.0f32; n];
        arena.get_mut(s_a).copy_from_slice(&data_a);
        arena.get_mut(s_b).copy_from_slice(&data_b);

        let ops_before = current_thread_heap_ops();

        // Cache hit lookups + all 3 arena execution entrypoints must perform 0 heap ops
        let k_swiglu_cached = engine.compile(&g_swiglu);
        assert_eq!(k_swiglu as usize, k_swiglu_cached as usize);

        for _ in 0..10 {
            engine.exec_unary(k_softmax, &mut arena, s_a, s_out);
            engine.exec_binary(k_swiglu, &mut arena, s_a, s_b, s_out);
            engine.exec_with_weights(k_rmsnorm, &mut arena, s_a, &ext_weights, s_out);
        }

        let ops_after = current_thread_heap_ops();
        assert_eq!(
            ops_after,
            ops_before,
            "Expected 0 heap allocs/deallocs on JIT cache hit and exec_* calls, found {:?}",
            (ops_after.0 - ops_before.0, ops_after.1 - ops_before.1)
        );
    }

    #[test]
    fn test_jit_unaligned_lengths_tail_loop() {
        let mut engine = JitEngine::new();
        let k_swiglu = engine.compile(&FuseGraph::swiglu());
        let k_softmax = engine.compile(&FuseGraph::softmax());

        for &len in &[1usize, 2, 3, 5, 7, 15, 33] {
            let a = pseudo_random_f32(len, 0x1111_2222 ^ (len as u32));
            let b = pseudo_random_f32(len, 0x3333_4444 ^ (len as u32));

            let mut out_swiglu = vec![-999.0f32; len + 4];
            let inputs_swiglu = [a.as_ptr(), b.as_ptr()];
            unsafe {
                k_swiglu(inputs_swiglu.as_ptr(), out_swiglu.as_mut_ptr(), len);
            }
            for i in 0..len {
                let ref_val = (a[i] / (1.0 + (-a[i]).exp())) * b[i];
                assert!(
                    (out_swiglu[i] - ref_val).abs() < 1e-5,
                    "Unaligned SwiGLU mismatch at len={len}, i={i}: jit={}, ref={ref_val}",
                    out_swiglu[i]
                );
            }
            assert_eq!(
                out_swiglu[len], -999.0,
                "Tail loop overwrote past len={len}"
            );

            let mut out_softmax = vec![-999.0f32; len + 4];
            let inputs_softmax = [a.as_ptr()];
            unsafe {
                k_softmax(inputs_softmax.as_ptr(), out_softmax.as_mut_ptr(), len);
            }
            let max_a = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_ea: f32 = a.iter().map(|&v| (v - max_a).exp()).sum();
            for i in 0..len {
                let ref_val = (a[i] - max_a).exp() / sum_ea;
                assert!(
                    (out_softmax[i] - ref_val).abs() < 1e-5,
                    "Unaligned Softmax mismatch at len={len}, i={i}: jit={}, ref={ref_val}",
                    out_softmax[i]
                );
            }
            assert_eq!(
                out_softmax[len], -999.0,
                "Softmax tail loop overwrote past len={len}"
            );

            let eps = 1e-5f32;
            let k_rmsnorm = engine.compile(&FuseGraph::rmsnorm(len, eps));
            let mut out_rms = vec![-999.0f32; len + 4];
            let inputs_rms = [a.as_ptr(), b.as_ptr()];
            unsafe {
                k_rmsnorm(inputs_rms.as_ptr(), out_rms.as_mut_ptr(), len);
            }
            let sum_sq: f32 = a.iter().map(|&v| v * v).sum();
            let rms = 1.0 / (sum_sq / (len as f32) + eps).sqrt();
            for i in 0..len {
                let ref_val = a[i] * rms * b[i];
                assert!(
                    (out_rms[i] - ref_val).abs() < 1e-5,
                    "Unaligned RMSNorm mismatch at len={len}, i={i}: jit={}, ref={ref_val}",
                    out_rms[i]
                );
            }
            assert_eq!(
                out_rms[len], -999.0,
                "RMSNorm tail loop overwrote past len={len}"
            );
        }
    }
}
