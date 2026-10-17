use cranelift_codegen::ir::{
    Function, InstBuilder, MemFlags, Value, condcodes::IntCC, immediates::Ieee32, types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};

use crate::jit::graph::{BinaryKind, FuseGraph, OpNode, ReduceKind, UnaryKind};

#[inline]
fn splat_f32(builder: &mut FunctionBuilder, val: f32) -> Value {
    let scalar = builder.ins().f32const(Ieee32::with_bits(val.to_bits()));
    builder.ins().splat(types::F32X4, scalar)
}

#[inline]
fn splat_i32(builder: &mut FunctionBuilder, val: i32) -> Value {
    let scalar = builder.ins().iconst(types::I32, val as i64);
    builder.ins().splat(types::I32X4, scalar)
}

/// Emits a register-only SIMD `exp(x)` approximation on `F32X4` via Cody-Waite range reduction
/// and a degree-5 Horner polynomial on `[-ln(2)/2, ln(2)/2]`.
fn emit_exp_f32x4(builder: &mut FunctionBuilder, x: Value) -> Value {
    let min_x = splat_f32(builder, -87.3f32);
    let max_x = splat_f32(builder, 88.3f32);
    let x_clamped = builder.ins().fmax(x, min_x);
    let x_clamped = builder.ins().fmin(x_clamped, max_x);

    let log2_e = splat_f32(builder, std::f32::consts::LOG2_E);
    let fx = builder.ins().fmul(x_clamped, log2_e);
    let n_f32 = builder.ins().nearest(fx);

    let ln2_hi = splat_f32(builder, 0.693_145_75_f32);
    let ln2_lo = splat_f32(builder, 1.428_606_8e-6_f32);
    let n_hi = builder.ins().fmul(n_f32, ln2_hi);
    let n_lo = builder.ins().fmul(n_f32, ln2_lo);
    let r_hi = builder.ins().fsub(x_clamped, n_hi);
    let r = builder.ins().fsub(r_hi, n_lo);

    let c5 = splat_f32(builder, 1.0 / 120.0);
    let c4 = splat_f32(builder, 1.0 / 24.0);
    let c3 = splat_f32(builder, 1.0 / 6.0);
    let c2 = splat_f32(builder, 0.5);
    let one = splat_f32(builder, 1.0);

    let p = builder.ins().fmul(r, c5);
    let p = builder.ins().fadd(p, c4);
    let p = builder.ins().fmul(r, p);
    let p = builder.ins().fadd(p, c3);
    let p = builder.ins().fmul(r, p);
    let p = builder.ins().fadd(p, c2);
    let p = builder.ins().fmul(r, p);
    let p = builder.ins().fadd(p, one);
    let p = builder.ins().fmul(r, p);
    let poly = builder.ins().fadd(p, one);

    let n_i32 = builder.ins().fcvt_to_sint_sat(types::I32X4, n_f32);
    let bias = splat_i32(builder, 127);
    let biased = builder.ins().iadd(n_i32, bias);
    let pow2_bits = builder.ins().ishl_imm(biased, 23);
    let pow2 = builder
        .ins()
        .bitcast(types::F32X4, MemFlags::new(), pow2_bits);

    builder.ins().fmul(poly, pow2)
}

fn emit_node_f32x4(
    builder: &mut FunctionBuilder,
    node: OpNode,
    vals: &[Option<Value>],
    input_ptrs: &[Value],
    out_ptr: Value,
    byte_offset: Value,
) -> Option<Value> {
    match node {
        OpNode::Load(in_id) => {
            let base = input_ptrs[in_id.0 as usize];
            let addr = builder.ins().iadd(base, byte_offset);
            Some(
                builder
                    .ins()
                    .load(types::F32X4, MemFlags::trusted(), addr, 0),
            )
        }
        OpNode::Const(bits) => {
            let scalar = builder.ins().f32const(Ieee32::with_bits(bits));
            Some(builder.ins().splat(types::F32X4, scalar))
        }
        OpNode::UnaryOp { op, src } => {
            let v = vals[src.0 as usize].expect("missing unary src value");
            let res = match op {
                UnaryKind::Neg => builder.ins().fneg(v),
                UnaryKind::Sqrt => builder.ins().sqrt(v),
                UnaryKind::Recip => {
                    let one = splat_f32(builder, 1.0);
                    builder.ins().fdiv(one, v)
                }
                UnaryKind::Exp => emit_exp_f32x4(builder, v),
                UnaryKind::SiLU => {
                    let neg_v = builder.ins().fneg(v);
                    let exp_neg = emit_exp_f32x4(builder, neg_v);
                    let one = splat_f32(builder, 1.0);
                    let denom = builder.ins().fadd(one, exp_neg);
                    builder.ins().fdiv(v, denom)
                }
            };
            Some(res)
        }
        OpNode::BinaryOp { op, lhs, rhs } => {
            let l = vals[lhs.0 as usize].expect("missing binary lhs value");
            let r = vals[rhs.0 as usize].expect("missing binary rhs value");
            let res = match op {
                BinaryKind::Add => builder.ins().fadd(l, r),
                BinaryKind::Sub => builder.ins().fsub(l, r),
                BinaryKind::Mul => builder.ins().fmul(l, r),
                BinaryKind::Div => builder.ins().fdiv(l, r),
            };
            Some(res)
        }
        OpNode::ReduceOp { .. } => None,
        OpNode::Store { dst: _, src } => {
            let v = vals[src.0 as usize].expect("missing store src value");
            let addr = builder.ins().iadd(out_ptr, byte_offset);
            builder.ins().store(MemFlags::trusted(), v, addr, 0);
            None
        }
    }
}

/// Lowers a [`FuseGraph`] into a SIMD-vectorized (`F32X4`) Cranelift IR function
/// with signature `unsafe extern "C" fn(inputs: *const *const f32, output: *mut f32, len: usize)`.
pub fn lower_fuse_graph(
    graph: &FuseGraph,
    func: &mut Function,
    fb_ctx: &mut FunctionBuilderContext,
) {
    let ptr_ty = func.signature.params[0].value_type;
    let mut builder = FunctionBuilder::new(func, fb_ctx);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let inputs_ptr = builder.block_params(entry_block)[0];
    let out_ptr = builder.block_params(entry_block)[1];
    let len = builder.block_params(entry_block)[2];

    let ptr_size = ptr_ty.bytes() as i32;
    let mut input_ptrs = Vec::with_capacity(graph.num_inputs as usize);
    for i in 0..graph.num_inputs {
        let p = builder.ins().load(
            ptr_ty,
            MemFlags::trusted(),
            inputs_ptr,
            (i as i32) * ptr_size,
        );
        input_ptrs.push(p);
    }

    let zero_idx = builder.ins().iconst(ptr_ty, 0);
    let step_four = builder.ins().iconst(ptr_ty, 4);

    let reduce_pos = graph
        .nodes
        .iter()
        .position(|n| matches!(n, OpNode::ReduceOp { .. }));

    let mut precomputed_vals: Vec<Option<Value>> = vec![None; graph.nodes.len()];

    if let Some(r_idx) = reduce_pos {
        let (reduce_kind, reduce_src) = match graph.nodes[r_idx] {
            OpNode::ReduceOp { op, src } => (op, src),
            _ => unreachable!(),
        };

        // Mark ancestor nodes needed to evaluate `reduce_src` in Pass 1
        let mut needed_pass1 = vec![false; graph.nodes.len()];
        needed_pass1[reduce_src.0 as usize] = true;
        for idx in (0..=reduce_src.0 as usize).rev() {
            if !needed_pass1[idx] {
                continue;
            }
            match graph.nodes[idx] {
                OpNode::UnaryOp { src, .. } => needed_pass1[src.0 as usize] = true,
                OpNode::BinaryOp { lhs, rhs, .. } => {
                    needed_pass1[lhs.0 as usize] = true;
                    needed_pass1[rhs.0 as usize] = true;
                }
                _ => {}
            }
        }

        let idx_var = Variable::from_u32(0);
        let acc_var = Variable::from_u32(1);
        builder.declare_var(idx_var, ptr_ty);
        builder.declare_var(acc_var, types::F32X4);

        builder.def_var(idx_var, zero_idx);
        let init_acc = match reduce_kind {
            ReduceKind::Sum => splat_f32(&mut builder, 0.0),
            ReduceKind::Max => splat_f32(&mut builder, f32::NEG_INFINITY),
        };
        builder.def_var(acc_var, init_acc);

        let pass1_header = builder.create_block();
        let pass1_body = builder.create_block();
        let pass1_exit = builder.create_block();

        builder.ins().jump(pass1_header, &[]);

        builder.switch_to_block(pass1_header);
        let cur_idx = builder.use_var(idx_var);
        let cond = builder.ins().icmp(IntCC::UnsignedLessThan, cur_idx, len);
        builder.ins().brif(cond, pass1_body, &[], pass1_exit, &[]);

        builder.switch_to_block(pass1_body);
        let byte_off = builder.ins().ishl_imm(cur_idx, 2);
        let mut pass1_vals: Vec<Option<Value>> = vec![None; graph.nodes.len()];
        for idx in 0..=reduce_src.0 as usize {
            if needed_pass1[idx] {
                pass1_vals[idx] = emit_node_f32x4(
                    &mut builder,
                    graph.nodes[idx],
                    &pass1_vals,
                    &input_ptrs,
                    out_ptr,
                    byte_off,
                );
            }
        }

        let src_vec = pass1_vals[reduce_src.0 as usize].expect("missing reduce src");
        let prev_acc = builder.use_var(acc_var);
        let next_acc = match reduce_kind {
            ReduceKind::Sum => builder.ins().fadd(prev_acc, src_vec),
            ReduceKind::Max => builder.ins().fmax(prev_acc, src_vec),
        };
        builder.def_var(acc_var, next_acc);

        let next_idx = builder.ins().iadd(cur_idx, step_four);
        builder.def_var(idx_var, next_idx);
        builder.ins().jump(pass1_header, &[]);

        builder.seal_block(pass1_header);
        builder.seal_block(pass1_body);

        // Horizontal 4-lane reduction in pass1_exit
        builder.switch_to_block(pass1_exit);
        builder.seal_block(pass1_exit);
        let final_acc = builder.use_var(acc_var);
        let l0 = builder.ins().extractlane(final_acc, 0);
        let l1 = builder.ins().extractlane(final_acc, 1);
        let l2 = builder.ins().extractlane(final_acc, 2);
        let l3 = builder.ins().extractlane(final_acc, 3);
        let scalar_red = match reduce_kind {
            ReduceKind::Sum => {
                let s01 = builder.ins().fadd(l0, l1);
                let s23 = builder.ins().fadd(l2, l3);
                builder.ins().fadd(s01, s23)
            }
            ReduceKind::Max => {
                let m01 = builder.ins().fmax(l0, l1);
                let m23 = builder.ins().fmax(l2, l3);
                builder.ins().fmax(m01, m23)
            }
        };
        precomputed_vals[r_idx] = Some(builder.ins().splat(types::F32X4, scalar_red));
    }

    // Determine nodes needed for the final Store pass
    let mut needed_pass2 = vec![false; graph.nodes.len()];
    for (idx, node) in graph.nodes.iter().enumerate() {
        if matches!(node, OpNode::Store { .. }) {
            needed_pass2[idx] = true;
        }
    }
    for idx in (0..graph.nodes.len()).rev() {
        if !needed_pass2[idx] || precomputed_vals[idx].is_some() {
            continue;
        }
        match graph.nodes[idx] {
            OpNode::Store { src, .. } | OpNode::UnaryOp { src, .. } => {
                needed_pass2[src.0 as usize] = true;
            }
            OpNode::BinaryOp { lhs, rhs, .. } => {
                needed_pass2[lhs.0 as usize] = true;
                needed_pass2[rhs.0 as usize] = true;
            }
            _ => {}
        }
    }

    let pass2_idx_var = Variable::from_u32(2);
    builder.declare_var(pass2_idx_var, ptr_ty);
    builder.def_var(pass2_idx_var, zero_idx);

    let pass2_header = builder.create_block();
    let pass2_body = builder.create_block();
    let pass2_exit = builder.create_block();

    builder.ins().jump(pass2_header, &[]);

    builder.switch_to_block(pass2_header);
    let cur_idx = builder.use_var(pass2_idx_var);
    let cond = builder.ins().icmp(IntCC::UnsignedLessThan, cur_idx, len);
    builder.ins().brif(cond, pass2_body, &[], pass2_exit, &[]);

    builder.switch_to_block(pass2_body);
    let byte_off = builder.ins().ishl_imm(cur_idx, 2);
    let mut pass2_vals = precomputed_vals;
    for idx in 0..graph.nodes.len() {
        if needed_pass2[idx] && pass2_vals[idx].is_none() {
            pass2_vals[idx] = emit_node_f32x4(
                &mut builder,
                graph.nodes[idx],
                &pass2_vals,
                &input_ptrs,
                out_ptr,
                byte_off,
            );
        }
    }

    let next_idx = builder.ins().iadd(cur_idx, step_four);
    builder.def_var(pass2_idx_var, next_idx);
    builder.ins().jump(pass2_header, &[]);

    builder.seal_block(pass2_header);
    builder.seal_block(pass2_body);

    builder.switch_to_block(pass2_exit);
    builder.seal_block(pass2_exit);
    builder.ins().return_(&[]);

    builder.finalize();
}
