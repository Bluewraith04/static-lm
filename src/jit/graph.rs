#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u16);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct InputId(pub u8);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OutputId(pub u8);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnaryKind {
    Neg,
    Exp,
    Recip,
    Sqrt,
    SiLU,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    Sum,
    Max,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpNode {
    Load(InputId),
    Const(u32),
    UnaryOp {
        op: UnaryKind,
        src: NodeId,
    },
    BinaryOp {
        op: BinaryKind,
        lhs: NodeId,
        rhs: NodeId,
    },
    ReduceOp {
        op: ReduceKind,
        src: NodeId,
    },
    Store {
        dst: OutputId,
        src: NodeId,
    },
}

/// Lightweight SSA expression DAG for elementwise and reduction fusion.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FuseGraph {
    pub nodes: Vec<OpNode>,
    pub num_inputs: u8,
}

impl FuseGraph {
    pub fn new() -> Self {
        Self::default()
    }
}
