#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u16);
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct InputId(pub u8);
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OutputId(pub u8);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnaryKind { Neg, Exp, Recip, Sqrt, SiLU }

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryKind { Add, Sub, Mul, Div }

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceKind { Sum, Max }

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpNode {
    Load(InputId),
    Const(u32),
    UnaryOp { op: UnaryKind, src: NodeId },
    BinaryOp { op: BinaryKind, lhs: NodeId, rhs: NodeId },
    ReduceOp { op: ReduceKind, src: NodeId },
    Store { dst: OutputId, src: NodeId },
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

    #[inline]
    fn push(&mut self, node: OpNode) -> NodeId {
        let id = NodeId(self.nodes.len() as u16);
        self.nodes.push(node);
        id
    }

    pub fn load(&mut self, input: InputId) -> NodeId {
        self.num_inputs = self.num_inputs.max(input.0 + 1);
        self.push(OpNode::Load(input))
    }

    pub fn constant(&mut self, val: f32) -> NodeId {
        self.push(OpNode::Const(val.to_bits()))
    }

    pub fn unary(&mut self, op: UnaryKind, src: NodeId) -> NodeId {
        self.push(OpNode::UnaryOp { op, src })
    }

    pub fn neg(&mut self, src: NodeId) -> NodeId { self.unary(UnaryKind::Neg, src) }
    pub fn exp(&mut self, src: NodeId) -> NodeId { self.unary(UnaryKind::Exp, src) }
    pub fn recip(&mut self, src: NodeId) -> NodeId { self.unary(UnaryKind::Recip, src) }
    pub fn sqrt(&mut self, src: NodeId) -> NodeId { self.unary(UnaryKind::Sqrt, src) }
    pub fn silu(&mut self, src: NodeId) -> NodeId { self.unary(UnaryKind::SiLU, src) }

    pub fn binary(&mut self, op: BinaryKind, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.push(OpNode::BinaryOp { op, lhs, rhs })
    }

    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId { self.binary(BinaryKind::Add, lhs, rhs) }
    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId { self.binary(BinaryKind::Sub, lhs, rhs) }
    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId { self.binary(BinaryKind::Mul, lhs, rhs) }
    pub fn div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId { self.binary(BinaryKind::Div, lhs, rhs) }

    pub fn reduce_sum(&mut self, src: NodeId) -> NodeId {
        self.push(OpNode::ReduceOp { op: ReduceKind::Sum, src })
    }

    pub fn reduce_max(&mut self, src: NodeId) -> NodeId {
        self.push(OpNode::ReduceOp { op: ReduceKind::Max, src })
    }

    pub fn store(&mut self, dst: OutputId, src: NodeId) -> NodeId {
        self.push(OpNode::Store { dst, src })
    }

    /// Single-pass fused SwiGLU graph: `out[i] = silu(gate[i]) * up[i]`.
    pub fn swiglu() -> Self {
        let mut g = Self::new();
        let (gate, up) = (g.load(InputId(0)), g.load(InputId(1)));
        let act = g.silu(gate);
        let out = g.mul(act, up);
        g.store(OutputId(0), out);
        g
    }

    /// Two-pass fused RMSNorm graph: `rms = 1.0 / sqrt(mean(x^2) + eps)`, `out[i] = x[i] * rms * weight[i]`.
    pub fn rmsnorm(n: usize, eps: f32) -> Self {
        let mut g = Self::new();
        let (x, weight) = (g.load(InputId(0)), g.load(InputId(1)));
        let x2 = g.mul(x, x);
        let sum_x2 = g.reduce_sum(x2);
        let (inv_n, eps_c) = (g.constant(1.0 / (n as f32)), g.constant(eps));
        let mean_x2 = g.mul(sum_x2, inv_n);
        let var_eps = g.add(mean_x2, eps_c);
        let std_dev = g.sqrt(var_eps);
        let rms = g.recip(std_dev);
        let norm_x = g.mul(x, rms);
        let out = g.mul(norm_x, weight);
        g.store(OutputId(0), out);
        g
    }

    /// Single-pass fused residual addition graph: `out[i] = a[i] + b[i]`.
    pub fn residual_add() -> Self {
        let mut g = Self::new();
        let (a, b) = (g.load(InputId(0)), g.load(InputId(1)));
        let out = g.add(a, b);
        g.store(OutputId(0), out);
        g
    }

    /// Multi-pass numerically stable fused Softmax graph:
    /// `max_x = max(x)`, `sum_ex = sum(exp(x - max_x))`, `out[i] = exp(x[i] - max_x) * (1 / sum_ex)`.
    pub fn softmax() -> Self {
        let mut g = Self::new();
        let x = g.load(InputId(0));
        let max_x = g.reduce_max(x);
        let shifted = g.sub(x, max_x);
        let ex = g.exp(shifted);
        let sum_ex = g.reduce_sum(ex);
        let inv_sum = g.recip(sum_ex);
        let out = g.mul(ex, inv_sum);
        g.store(OutputId(0), out);
        g
    }
}
