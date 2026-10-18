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

fn emit_pure_op_f32x4(
    builder: &mut FunctionBuilder,
    node: OpNode,
    vals: &[Option<Value>],
) -> Option<Value> {
    match node {
        OpNode::Const(bits) => {
            let scalar = builder.ins().f32const(Ieee32::with_bits(bits));
            Some(builder.ins().splat(types::F32X4, scalar))
        }
        OpNode::UnaryOp { op, src } => {
            let v = vals[src.0 as usize]?;
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
            let l = vals[lhs.0 as usize]?;
            let r = vals[rhs.0 as usize]?;
            let res = match op {
                BinaryKind::Add => builder.ins().fadd(l, r),
                BinaryKind::Sub => builder.ins().fsub(l, r),
                BinaryKind::Mul => builder.ins().fmul(l, r),
                BinaryKind::Div => builder.ins().fdiv(l, r),
            };
            Some(res)
        }
        _ => None,
    }
}

/// Evaluates and hoists any node whose operands are already loop-invariant (`Const`, completed `ReduceOp`,
/// or pure operations on already-hoisted values) outside the inner SIMD loop.
fn hoist_invariant_nodes(
    builder: &mut FunctionBuilder,
    graph: &FuseGraph,
    precomputed_vals: &mut [Option<Value>],
) {
    for idx in 0..graph.nodes.len() {
        if precomputed_vals[idx].is_some() {
            continue;
        }
        if let Some(val) = emit_pure_op_f32x4(builder, graph.nodes[idx], precomputed_vals) {
            precomputed_vals[idx] = Some(val);
        }
    }
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
        OpNode::Store { dst: _, src } => {
            let v = vals[src.0 as usize].expect("missing store src value");
            let addr = builder.ins().iadd(out_ptr, byte_offset);
            builder.ins().store(MemFlags::trusted(), v, addr, 0);
            None
        }
        other => emit_pure_op_f32x4(builder, other, vals),
    }
}

fn emit_node_scalar(
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
            let scalar = builder.ins().load(types::F32, MemFlags::trusted(), addr, 0);
            Some(builder.ins().splat(types::F32X4, scalar))
        }
        OpNode::Store { dst: _, src } => {
            let v = vals[src.0 as usize].expect("missing store src value");
            let lane0 = builder.ins().extractlane(v, 0);
            let addr = builder.ins().iadd(out_ptr, byte_offset);
            builder.ins().store(MemFlags::trusted(), lane0, addr, 0);
            None
        }
        other => emit_pure_op_f32x4(builder, other, vals),
    }
}

/// Marks all non-precomputed ancestor nodes required to compute `target_idx`.
fn mark_ancestors(
    graph: &FuseGraph,
    target_idx: usize,
    precomputed: &[Option<Value>],
) -> Vec<bool> {
    let mut needed = vec![false; graph.nodes.len()];
    needed[target_idx] = true;
    for idx in (0..=target_idx).rev() {
        if !needed[idx] || precomputed[idx].is_some() {
            continue;
        }
        match graph.nodes[idx] {
            OpNode::Store { src, .. } | OpNode::UnaryOp { src, .. } => {
                needed[src.0 as usize] = true;
            }
            OpNode::BinaryOp { lhs, rhs, .. } => {
                needed[lhs.0 as usize] = true;
                needed[rhs.0 as usize] = true;
            }
            _ => {}
        }
    }
    needed
}

/// Lowers a [`FuseGraph`] into a SIMD-vectorized (`F32X4`) Cranelift IR function with a scalar tail loop
/// for arbitrary lengths (`len % 4 != 0`), with signature:
/// `unsafe extern "C" fn(inputs: *const *const f32, output: *mut f32, len: usize)`.
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
    let step_one = builder.ins().iconst(ptr_ty, 1);
    let vec_len = builder.ins().band_imm(len, -4);

    let mut precomputed_vals: Vec<Option<Value>> = vec![None; graph.nodes.len()];
    hoist_invariant_nodes(&mut builder, graph, &mut precomputed_vals);

    let mut next_var_id = 0u32;

    // Emit a vectorized F32X4 reduction loop + scalar tail loop for each ReduceOp in topological order
    for r_idx in 0..graph.nodes.len() {
        let (reduce_kind, reduce_src) = match graph.nodes[r_idx] {
            OpNode::ReduceOp { op, src } => (op, src),
            _ => continue,
        };

        let needed_pass = mark_ancestors(graph, reduce_src.0 as usize, &precomputed_vals);

        let idx_var = Variable::from_u32(next_var_id);
        let acc_var = Variable::from_u32(next_var_id + 1);
        let scalar_acc_var = Variable::from_u32(next_var_id + 2);
        next_var_id += 3;

        builder.declare_var(idx_var, ptr_ty);
        builder.declare_var(acc_var, types::F32X4);
        builder.declare_var(scalar_acc_var, types::F32);

        builder.def_var(idx_var, zero_idx);
        let init_acc = match reduce_kind {
            ReduceKind::Sum => splat_f32(&mut builder, 0.0),
            ReduceKind::Max => splat_f32(&mut builder, f32::NEG_INFINITY),
        };
        builder.def_var(acc_var, init_acc);

        let pass_header = builder.create_block();
        let pass_body = builder.create_block();
        let vec_exit = builder.create_block();
        let tail_header = builder.create_block();
        let tail_body = builder.create_block();
        let pass_exit = builder.create_block();

        builder.ins().jump(pass_header, &[]);

        builder.switch_to_block(pass_header);
        let cur_idx = builder.use_var(idx_var);
        let cond = builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, cur_idx, vec_len);
        builder.ins().brif(cond, pass_body, &[], vec_exit, &[]);

        builder.switch_to_block(pass_body);
        let byte_off = builder.ins().ishl_imm(cur_idx, 2);
        let mut pass_vals = precomputed_vals.clone();
        for idx in 0..=reduce_src.0 as usize {
            if needed_pass[idx] && pass_vals[idx].is_none() {
                pass_vals[idx] = emit_node_f32x4(
                    &mut builder,
                    graph.nodes[idx],
                    &pass_vals,
                    &input_ptrs,
                    out_ptr,
                    byte_off,
                );
            }
        }

        let src_vec = pass_vals[reduce_src.0 as usize].expect("missing reduce src");
        let prev_acc = builder.use_var(acc_var);
        let next_acc = match reduce_kind {
            ReduceKind::Sum => builder.ins().fadd(prev_acc, src_vec),
            ReduceKind::Max => builder.ins().fmax(prev_acc, src_vec),
        };
        builder.def_var(acc_var, next_acc);

        let next_idx = builder.ins().iadd(cur_idx, step_four);
        builder.def_var(idx_var, next_idx);
        builder.ins().jump(pass_header, &[]);

        builder.seal_block(pass_header);
        builder.seal_block(pass_body);

        // Horizontal 4-lane reduction in vec_exit before entering scalar tail loop
        builder.switch_to_block(vec_exit);
        builder.seal_block(vec_exit);
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
        builder.def_var(scalar_acc_var, scalar_red);
        builder.ins().jump(tail_header, &[]);

        builder.switch_to_block(tail_header);
        let tail_idx = builder.use_var(idx_var);
        let tail_cond = builder.ins().icmp(IntCC::UnsignedLessThan, tail_idx, len);
        builder
            .ins()
            .brif(tail_cond, tail_body, &[], pass_exit, &[]);

        builder.switch_to_block(tail_body);
        let tail_byte_off = builder.ins().ishl_imm(tail_idx, 2);
        let mut tail_vals = precomputed_vals.clone();
        for idx in 0..=reduce_src.0 as usize {
            if needed_pass[idx] && tail_vals[idx].is_none() {
                tail_vals[idx] = emit_node_scalar(
                    &mut builder,
                    graph.nodes[idx],
                    &tail_vals,
                    &input_ptrs,
                    out_ptr,
                    tail_byte_off,
                );
            }
        }
        let tail_src_vec = tail_vals[reduce_src.0 as usize].expect("missing tail reduce src");
        let tail_src_scalar = builder.ins().extractlane(tail_src_vec, 0);
        let prev_scalar_acc = builder.use_var(scalar_acc_var);
        let next_scalar_acc = match reduce_kind {
            ReduceKind::Sum => builder.ins().fadd(prev_scalar_acc, tail_src_scalar),
            ReduceKind::Max => builder.ins().fmax(prev_scalar_acc, tail_src_scalar),
        };
        builder.def_var(scalar_acc_var, next_scalar_acc);
        let next_tail_idx = builder.ins().iadd(tail_idx, step_one);
        builder.def_var(idx_var, next_tail_idx);
        builder.ins().jump(tail_header, &[]);

        builder.seal_block(tail_header);
        builder.seal_block(tail_body);

        // Broadcast completed reduction and hoist any newly invariant scalar/broadcast ops
        builder.switch_to_block(pass_exit);
        builder.seal_block(pass_exit);
        let total_scalar = builder.use_var(scalar_acc_var);
        precomputed_vals[r_idx] = Some(builder.ins().splat(types::F32X4, total_scalar));
        hoist_invariant_nodes(&mut builder, graph, &mut precomputed_vals);
    }

    // Determine nodes needed for the final Store pass
    let store_idx = graph
        .nodes
        .iter()
        .position(|n| matches!(n, OpNode::Store { .. }))
        .expect("FuseGraph must contain a Store node");
    let needed_store = mark_ancestors(graph, store_idx, &precomputed_vals);

    let store_idx_var = Variable::from_u32(next_var_id);
    builder.declare_var(store_idx_var, ptr_ty);
    builder.def_var(store_idx_var, zero_idx);

    let store_header = builder.create_block();
    let store_body = builder.create_block();
    let store_tail_header = builder.create_block();
    let store_tail_body = builder.create_block();
    let store_exit = builder.create_block();

    builder.ins().jump(store_header, &[]);

    builder.switch_to_block(store_header);
    let cur_idx = builder.use_var(store_idx_var);
    let cond = builder
        .ins()
        .icmp(IntCC::UnsignedLessThan, cur_idx, vec_len);
    builder
        .ins()
        .brif(cond, store_body, &[], store_tail_header, &[]);

    builder.switch_to_block(store_body);
    let byte_off = builder.ins().ishl_imm(cur_idx, 2);
    let mut store_vals = precomputed_vals.clone();
    for idx in 0..=store_idx {
        if needed_store[idx] && store_vals[idx].is_none() {
            store_vals[idx] = emit_node_f32x4(
                &mut builder,
                graph.nodes[idx],
                &store_vals,
                &input_ptrs,
                out_ptr,
                byte_off,
            );
        }
    }

    let next_idx = builder.ins().iadd(cur_idx, step_four);
    builder.def_var(store_idx_var, next_idx);
    builder.ins().jump(store_header, &[]);

    builder.seal_block(store_header);
    builder.seal_block(store_body);

    builder.switch_to_block(store_tail_header);
    let tail_cur_idx = builder.use_var(store_idx_var);
    let tail_cond = builder
        .ins()
        .icmp(IntCC::UnsignedLessThan, tail_cur_idx, len);
    builder
        .ins()
        .brif(tail_cond, store_tail_body, &[], store_exit, &[]);

    builder.switch_to_block(store_tail_body);
    let tail_byte_off = builder.ins().ishl_imm(tail_cur_idx, 2);
    let mut tail_store_vals = precomputed_vals;
    for idx in 0..=store_idx {
        if needed_store[idx] && tail_store_vals[idx].is_none() {
            tail_store_vals[idx] = emit_node_scalar(
                &mut builder,
                graph.nodes[idx],
                &tail_store_vals,
                &input_ptrs,
                out_ptr,
                tail_byte_off,
            );
        }
    }

    let next_tail_idx = builder.ins().iadd(tail_cur_idx, step_one);
    builder.def_var(store_idx_var, next_tail_idx);
    builder.ins().jump(store_tail_header, &[]);

    builder.seal_block(store_tail_header);
    builder.seal_block(store_tail_body);

    builder.switch_to_block(store_exit);
    builder.seal_block(store_exit);
    builder.ins().return_(&[]);

    builder.finalize();
}
