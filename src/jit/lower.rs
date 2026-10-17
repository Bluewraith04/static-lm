use cranelift_codegen::ir::Function;
use cranelift_frontend::FunctionBuilderContext;

use crate::jit::graph::FuseGraph;

/// Lowers a [`FuseGraph`] into a SIMD-vectorized (`F32X4`) Cranelift IR function.
pub fn lower_fuse_graph(
    _graph: &FuseGraph,
    _func: &mut Function,
    _fb_ctx: &mut FunctionBuilderContext,
) {
    todo!("lower FuseGraph DAG to Cranelift IR SIMD loop")
}
