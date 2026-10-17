use crate::jit::graph::FuseGraph;

/// Raw JIT-compiled kernel entrypoint: `(inputs_ptr_array, output_ptr, len)`.
pub type JitKernelFn = unsafe extern "C" fn(*const *const f32, *mut f32, usize);

/// Cranelift JIT compilation engine and kernel cache.
pub struct JitEngine;

impl JitEngine {
    pub fn new() -> Self {
        todo!("initialize cranelift-jit module and cache")
    }

    pub fn compile(&mut self, _graph: &FuseGraph) -> JitKernelFn {
        todo!("compile or fetch cached JitKernelFn for FuseGraph")
    }
}
