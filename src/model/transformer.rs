use crate::memory::arena::{AlignedArena, ArenaSlice};

/// Flat configuration and arena bindings for a single Transformer block.
pub struct TransformerLayer {
    pub dim: usize,
    pub hidden_dim: usize,
}

impl TransformerLayer {
    pub fn forward(&self, _arena: &mut AlignedArena, _input: ArenaSlice, _output: ArenaSlice) {
        todo!("wire GEMV and JIT kernels on AlignedArena")
    }
}
