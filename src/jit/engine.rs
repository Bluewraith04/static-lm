use std::collections::HashMap;

use cranelift_codegen::Context;
use cranelift_codegen::ir::AbiParam;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;

use crate::jit::graph::FuseGraph;
use crate::jit::lower::lower_fuse_graph;
use crate::memory::arena::{AlignedArena, ArenaSlice};

/// Raw JIT-compiled kernel entrypoint: `(inputs_ptr_array, output_ptr, len)`.
pub type JitKernelFn = unsafe extern "C" fn(*const *const f32, *mut f32, usize);

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

        let ptr_ty = self.module.target_config().pointer_type();
        self.module.clear_context(&mut self.ctx);
        self.ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // inputs: *const *const f32
        self.ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // output: *mut f32
        self.ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // len: usize

        lower_fuse_graph(graph, &mut self.ctx.func, &mut self.fb_ctx);

        let func_id = self
            .module
            .declare_anonymous_function(&self.ctx.func.signature)
            .expect("failed to declare JIT function");

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("failed to define JIT function");

        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("failed to finalize JIT definitions");

        let code_ptr = self.module.get_finalized_function(func_id);
        // SAFETY: `code_ptr` was compiled with the exact signature of `JitKernelFn`.
        let kernel_fn: JitKernelFn = unsafe { std::mem::transmute(code_ptr) };

        self.cache.insert(graph.clone(), kernel_fn);
        kernel_fn
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
        assert_eq!(src.len % 4, 0, "SIMD kernel length must be a multiple of 4");
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
        assert_eq!(dst.len % 4, 0, "SIMD kernel length must be a multiple of 4");
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
        assert_eq!(dst.len % 4, 0, "SIMD kernel length must be a multiple of 4");
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
        let graph = FuseGraph::swiglu();
        let kernel = engine.compile(&graph);

        let inputs = [gate.as_ptr(), up.as_ptr()];
        unsafe {
            kernel(inputs.as_ptr(), out_jit.as_mut_ptr(), n);
        }

        // Naive 5-line scalar f32 reference loop
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
    fn test_jit_rmsnorm_against_scalar_reference() {
        let n = 1024;
        let eps = 1e-5f32;
        let x = pseudo_random_f32(n, 0xAAAA_5555);
        let gamma = pseudo_random_f32(n, 0x5555_AAAA);
        let mut out_jit = vec![0.0f32; n];

        let mut engine = JitEngine::new();
        let graph = FuseGraph::rmsnorm(n, eps);
        let kernel = engine.compile(&graph);

        let inputs = [x.as_ptr(), gamma.as_ptr()];
        unsafe {
            kernel(inputs.as_ptr(), out_jit.as_mut_ptr(), n);
        }

        // Naive scalar f32 reference loop
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
    fn test_jit_cache_and_arena_execution() {
        let n = 256;
        let mut engine = JitEngine::new();
        let g1 = FuseGraph::swiglu();
        let g2 = FuseGraph::swiglu();

        let k1 = engine.compile(&g1);
        let k2 = engine.compile(&g2);
        assert_eq!(k1 as usize, k2 as usize, "expected cached function pointer");

        let mut arena = AlignedArena::with_capacity(1024);
        let s_gate = arena.alloc_slice(n);
        let s_up = arena.alloc_slice(n);
        let s_out = arena.alloc_slice(n);

        let gate_data = pseudo_random_f32(n, 0xCAFE_1234);
        let up_data = pseudo_random_f32(n, 0xBEEF_5678);
        arena.get_mut(s_gate).copy_from_slice(&gate_data);
        arena.get_mut(s_up).copy_from_slice(&up_data);

        engine.exec_binary(k1, &mut arena, s_gate, s_up, s_out);

        let out_slice = arena.get(s_out);
        for i in 0..n {
            let ref_val = (gate_data[i] / (1.0 + (-gate_data[i]).exp())) * up_data[i];
            assert!((out_slice[i] - ref_val).abs() < 1e-5);
        }
    }
}
