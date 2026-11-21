//! Cranelift lowering of the whole-function v2 IR (`napalm_wfjit_ir.rs`).
//!
//! Body ABI (Cranelift `tail` convention):
//! `(vm: ptr, frame: ptr, depth: i32, arg0..argN: i64) -> i64`.
//! `frame` is the interpreter-owned GPR bank of the top frame, or null in a
//! nested native frame; it is written only when a deopt materializes the
//! live GPRs for the interpreter. `depth` is the native chain depth. The
//! result is a tagged word, or `WFJIT_BAILOUT` after a deoptimization.

use super::wf_ir::*;
use super::*;
use cranelift::codegen::ir::{BlockArg, FuncRef};
use cranelift::prelude::Value as ClValue;

#[inline]
fn cl_type(ty: WfTy) -> cranelift::prelude::Type {
    match ty {
        WfTy::Tagged => types::I64,
        WfTy::F64 => types::F64,
        WfTy::Bool => types::I8,
    }
}

struct Lowering<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    ir: &'a WfFunction,
    function: &'a PreparedFunction,
    values: Vec<Option<ClValue>>,
    blocks: Vec<cranelift::prelude::Block>,
    deopt_blocks: Vec<Option<cranelift::prelude::Block>>,
    vm: ClValue,
    /// The top frame's GPR bank (null in nested frames). A host helper may move
    /// the bank, so it is an SSA variable redefined after each RuntimeOp.
    frame: cranelift::frontend::Variable,
    depth: ClValue,
    params: Vec<ClValue>,
    self_ref: FuncRef,
    speculation_failed_ref: FuncRef,
    runtime_op_ref: FuncRef,
    direct_call_ref: FuncRef,
    /// Count native self-calls in the WFJIT statistics (tests, NAPALM_JIT_STATS).
    count_calls: bool,
}

impl Lowering<'_, '_> {
    fn value(&self, value: WfValue) -> ClValue {
        self.values[value.0 as usize].expect("IR value used before its definition")
    }

    fn define(&mut self, value: WfValue, lowered: ClValue) {
        self.values[value.0 as usize] = Some(lowered);
    }

    fn deopt_block(&mut self, deopt: WfDeopt) -> cranelift::prelude::Block {
        if let Some(block) = self.deopt_blocks[deopt.0 as usize] {
            return block;
        }
        let block = self.b.create_block();
        self.b.set_cold_block(block);
        // The bank pointer current at the guard (it can move across helpers).
        self.b.append_block_param(block, types::I64);
        self.deopt_blocks[deopt.0 as usize] = Some(block);
        block
    }

    fn frame_arg(&mut self) -> [BlockArg; 1] {
        [BlockArg::Value(self.b.use_var(self.frame))]
    }

    /// Store `registers` (boxing unboxed values) into the bank `frame`.
    fn materialize(&mut self, frame: ClValue, registers: &[(u8, WfValue)]) {
        for &(reg, value) in registers {
            let word = self.tagged(value);
            self.b.ins().store(MemFlagsData::trusted(), word, frame, reg as i32 * 8);
        }
    }

    /// Branch to the deopt unless `condition` holds; continue in a new block.
    fn guard(&mut self, condition: ClValue, deopt: WfDeopt) {
        let target = self.deopt_block(deopt);
        let next = self.b.create_block();
        let frame = self.frame_arg();
        self.b.ins().brif(condition, next, &[], target, &frame);
        self.b.switch_to_block(next);
    }

    fn tagged(&mut self, value: WfValue) -> ClValue {
        let lowered = self.value(value);
        match self.ir.ty(value) {
            WfTy::Tagged => lowered,
            WfTy::F64 => self.box_f64(lowered),
            WfTy::Bool => self.box_bool(lowered),
        }
    }

    fn box_f64(&mut self, value: ClValue) -> ClValue {
        // Napalm keeps exactly one NaN bit pattern in tagged words.
        let is_nan = self.b.ins().fcmp(FloatCC::Unordered, value, value);
        let bits = self.b.ins().bitcast(types::I64, MemFlagsData::new(), value);
        let canonical = self.b.ins().iconst(types::I64, CANONICAL_NAN as i64);
        self.b.ins().select(is_nan, canonical, bits)
    }

    fn box_bool(&mut self, value: ClValue) -> ClValue {
        let bit = self.b.ins().uextend(types::I64, value);
        self.b.ins().iadd_imm(bit, TAG_FALSE as i64)
    }

    /// Sidecar entry of a heap-tagged word (`TraceHeapEntry`).
    fn heap_entry(&mut self, tagged: ClValue) -> ClValue {
        let handle32 = self.b.ins().ireduce(types::I32, tagged);
        let handle = self.b.ins().uextend(types::I64, handle32);
        let chunk_index = self.b.ins().ushr_imm_u(handle, TRACE_HEAP_CHUNK_BITS as i64);
        let slot = self.b.ins().band_imm(handle, TRACE_HEAP_CHUNK_MASK as i64);
        let chunks = self.b.ins().iconst(types::I64, handles().jit_chunks() as usize as i64);
        let chunk_offset = self.b.ins().ishl_imm(chunk_index, 3);
        let chunk_slot = self.b.ins().iadd(chunks, chunk_offset);
        let chunk = self.b.ins().load(types::I64, MemFlagsData::trusted(), chunk_slot, 0);
        let offset = self.b.ins().imul_imm(slot, std::mem::size_of::<TraceHeapEntry>() as i64);
        self.b.ins().iadd(chunk, offset)
    }

    fn is_heap(&mut self, tagged: ClValue) -> ClValue {
        let tag = self.b.ins().band_imm(tagged, TAG_HEAP_MASK as i64);
        self.b.ins().icmp_imm(IntCC::Equal, tag, TAG_HEAP as i64)
    }

    fn lower_inst(&mut self, result: WfValue, inst: &WfInst) {
        let flags = MemFlagsData::trusted();
        let lowered = match inst {
            WfInst::Param(index) => self.params[*index as usize],
            WfInst::ConstTagged(word) => self.b.ins().iconst(types::I64, *word as i64),
            WfInst::ConstBool(value) => self.b.ins().iconst(types::I8, *value as i64),
            WfInst::ConstF64(bits) => self.b.ins().f64const(f64::from_bits(*bits)),
            WfInst::LoadAbsolute(address) => {
                let slot = self.b.ins().iconst(types::I64, *address as i64);
                self.b.ins().load(types::I64, flags, slot, 0)
            }
            WfInst::UnboxNumber { value, deopt } => {
                let word = self.value(*value);
                let tag = self.b.ins().band_imm(word, TAG_MASK as i64);
                let is_number = self.b.ins().icmp_imm(IntCC::NotEqual, tag, TAG_BASE as i64);
                self.guard(is_number, *deopt);
                self.b.ins().bitcast(types::F64, MemFlagsData::new(), word)
            }
            WfInst::AssumeNumber(value) => {
                let word = self.value(*value);
                self.b.ins().bitcast(types::F64, MemFlagsData::new(), word)
            }
            WfInst::BoxF64(value) => { let value = self.value(*value); self.box_f64(value) }
            WfInst::BoxBool(value) => { let value = self.value(*value); self.box_bool(value) }
            WfInst::Truthy(value) => {
                // Only null and false are falsey; they are adjacent tags.
                let word = self.value(*value);
                let shifted = self.b.ins().iadd_imm(word, -(TAG_NULL as i64));
                self.b.ins().icmp_imm(IntCC::UnsignedGreaterThan, shifted, 1)
            }
            WfInst::FloatBin { op, lhs, rhs } => {
                let (lhs, rhs) = (self.value(*lhs), self.value(*rhs));
                match op {
                    WfFloatBin::Add => self.b.ins().fadd(lhs, rhs),
                    WfFloatBin::Sub => self.b.ins().fsub(lhs, rhs),
                    WfFloatBin::Mul => self.b.ins().fmul(lhs, rhs),
                    WfFloatBin::Div => self.b.ins().fdiv(lhs, rhs),
                }
            }
            WfInst::FloatNeg(value) => { let value = self.value(*value); self.b.ins().fneg(value) }
            WfInst::FloatCmp { cc, lhs, rhs } => {
                let (lhs, rhs) = (self.value(*lhs), self.value(*rhs));
                let cc = match cc {
                    WfFloatCmp::Eq => FloatCC::Equal,
                    WfFloatCmp::Ne => FloatCC::NotEqual,
                    WfFloatCmp::Lt => FloatCC::LessThan,
                    WfFloatCmp::Le => FloatCC::LessThanOrEqual,
                    WfFloatCmp::Gt => FloatCC::GreaterThan,
                    WfFloatCmp::Ge => FloatCC::GreaterThanOrEqual,
                };
                self.b.ins().fcmp(cc, lhs, rhs)
            }
            WfInst::IdentityCompare { lhs, rhs, classes, negate, deopt } => {
                let (lhs, rhs) = (self.value(*lhs), self.value(*rhs));
                let calculate = self.b.create_block();
                let heap = self.b.create_block();
                let target = self.deopt_block(*deopt);
                // Immediate receivers (null, Booleans) are accepted by tag.
                let mut immediate = self.b.ins().iconst(types::I8, 0);
                for &class in classes {
                    let tags: &[u64] = if class == CLASS_KIND_BUILTIN | BuiltinClass::Null as u64 {
                        &[TAG_NULL]
                    } else if class == CLASS_KIND_BUILTIN | BuiltinClass::Bool as u64 {
                        &[TAG_FALSE, TAG_TRUE]
                    } else { &[] };
                    for &tag in tags {
                        let matches = self.b.ins().icmp_imm(IntCC::Equal, lhs, tag as i64);
                        immediate = self.b.ins().bor(immediate, matches);
                    }
                }
                self.b.ins().brif(immediate, calculate, &[], heap, &[]);
                self.b.switch_to_block(heap);
                let resolve = self.b.create_block();
                let is_heap = self.is_heap(lhs);
                let frame = self.frame_arg();
                self.b.ins().brif(is_heap, resolve, &[], target, &frame);
                self.b.switch_to_block(resolve);
                let entry = self.heap_entry(lhs);
                let class = self.b.ins().load(types::I64, flags, entry,
                    offset_of!(TraceHeapEntry, receiver_class) as i32);
                let mut accepted = self.b.ins().iconst(types::I8, 0);
                for &expected in classes {
                    let matches = self.b.ins().icmp_imm(IntCC::Equal, class, expected as i64);
                    accepted = self.b.ins().bor(accepted, matches);
                }
                let frame = self.frame_arg();
                self.b.ins().brif(accepted, calculate, &[], target, &frame);
                self.b.switch_to_block(calculate);
                let cc = if *negate { IntCC::NotEqual } else { IntCC::Equal };
                self.b.ins().icmp(cc, lhs, rhs)
            }
            WfInst::MapPropertyGet { object, entries, deopt } => {
                let object = self.value(*object);
                let is_heap = self.is_heap(object);
                self.guard(is_heap, *deopt);
                let entry = self.heap_entry(object);
                let address = self.b.ins().load(types::I64, flags, entry,
                    offset_of!(TraceHeapEntry, address) as i32);
                let live = self.b.ins().icmp_imm(IntCC::NotEqual, address, 0);
                self.guard(live, *deopt);
                let metadata = self.b.ins().load(types::I32, flags, entry,
                    offset_of!(TraceHeapEntry, kind) as i32);
                let kind = self.b.ins().band_imm(metadata, TRACE_HEAP_KIND_MASK as i64);
                let is_map = self.b.ins().icmp_imm(IntCC::Equal, kind, TraceHeapKind::Map as i64);
                self.guard(is_map, *deopt);
                let shape = self.b.ins().ushr_imm_u(metadata, TRACE_HEAP_METADATA_SHIFT as i64);
                let data = emit_heap_data_base(self.b, entry, address);
                let done = self.b.create_block();
                self.b.append_block_param(done, types::I64);
                let target = self.deopt_block(*deopt);
                for (position, &(expected_shape, slot)) in entries.iter().enumerate() {
                    let hit = self.b.create_block();
                    let last = position + 1 == entries.len();
                    let miss = if last { target } else { self.b.create_block() };
                    let matches = self.b.ins().icmp_imm(IntCC::Equal, shape, expected_shape as i64);
                    let miss_args = if last { self.frame_arg().to_vec() } else { Vec::new() };
                    self.b.ins().brif(matches, hit, &[], miss, &miss_args);
                    self.b.switch_to_block(hit);
                    let value = self.b.ins().load(types::I64, flags, data,
                        slot as i32 * std::mem::size_of::<RegisterWord>() as i32);
                    self.b.ins().jump(done, &[BlockArg::Value(value)]);
                    if miss != target { self.b.switch_to_block(miss); }
                }
                self.b.switch_to_block(done);
                self.b.block_params(done)[0]
            }
            WfInst::SelfCall { args, deopt } => {
                let allowed = self.b.ins().icmp_imm(IntCC::UnsignedLessThan, self.depth, MAX_NATIVE_CALL_DEPTH as i64);
                self.guard(allowed, *deopt);
                if self.count_calls {
                    let jit = self.b.ins().load(types::I64, flags, self.vm,
                        offset_of!(Interpreter<'static>, whole_function_jit) as i32);
                    for counter in [offset_of!(WholeFunctionJitStats, native_entries),
                        offset_of!(WholeFunctionJitStats, native_recursive_calls)] {
                        let offset = (offset_of!(WholeFunctionJit, stats) + counter) as i32;
                        let old = self.b.ins().load(types::I64, flags, jit, offset);
                        let next = self.b.ins().iadd_imm(old, 1);
                        self.b.ins().store(flags, next, jit, offset);
                    }
                }
                let child_depth = self.b.ins().iadd_imm(self.depth, 1);
                let nested_frame = self.b.ins().iconst(types::I64, 0);
                let mut arguments = vec![self.vm, nested_frame, child_depth];
                arguments.extend(args.iter().map(|arg| self.values[arg.0 as usize].expect("self-call argument")));
                let call = self.b.ins().call(self.self_ref, &arguments);
                let value = self.b.inst_results(call)[0];
                let completed = self.b.ins().icmp_imm(IntCC::NotEqual, value, WFJIT_BAILOUT as i64);
                self.guard(completed, *deopt);
                value
            }
            WfInst::InstanceFieldGet { object, slot, nullable, deopt } => {
                let tracking = self.b.ins().load(types::I8, flags, self.vm,
                    offset_of!(Interpreter<'static>, reactive_tracking_active) as i32);
                let untracked = self.b.ins().icmp_imm(IntCC::Equal, tracking, 0);
                self.guard(untracked, *deopt);
                let object = self.value(*object);
                let (entry, address) = self.live_instance(object, *slot, *deopt);
                let data = emit_heap_data_base(self.b, entry, address);
                let value = self.b.ins().load(types::I64, flags, data,
                    *slot as i32 * std::mem::size_of::<RegisterWord>() as i32);
                if !*nullable {
                    // An uninitialized field traps in baseline: let it do so.
                    let initialized = self.b.ins().icmp_imm(IntCC::NotEqual, value, TAG_NULL as i64);
                    self.guard(initialized, *deopt);
                }
                value
            }
            WfInst::InstanceFieldSet { object, value, slot, class_key, value_immediate, deopt } => {
                let reactive = self.b.ins().load(types::I8, flags, self.vm,
                    offset_of!(Interpreter<'static>, reactive_enabled) as i32);
                let plain = self.b.ins().icmp_imm(IntCC::Equal, reactive, 0);
                self.guard(plain, *deopt);
                let stored = self.value(*value);
                if !*value_immediate {
                    // A heap reference would need the generational barrier.
                    let is_heap = self.is_heap(stored);
                    let immediate = self.b.ins().bxor_imm(is_heap, 1);
                    self.guard(immediate, *deopt);
                }
                let object = self.value(*object);
                let (entry, address) = self.live_instance(object, *slot, *deopt);
                let class = self.b.ins().load(types::I64, flags, entry,
                    offset_of!(TraceHeapEntry, receiver_class) as i32);
                let exact = self.b.ins().icmp_imm(IntCC::Equal, class, *class_key as i64);
                self.guard(exact, *deopt);
                let data = emit_heap_data_base(self.b, entry, address);
                self.b.ins().store(flags, stored, data,
                    *slot as i32 * std::mem::size_of::<RegisterWord>() as i32);
                self.b.ins().iconst(types::I8, 0)
            }
            WfInst::GuardClass { value, class, deopt } => {
                let value = self.value(*value);
                let is_heap = self.is_heap(value);
                self.guard(is_heap, *deopt);
                let entry = self.heap_entry(value);
                let address = self.b.ins().load(types::I64, flags, entry,
                    offset_of!(TraceHeapEntry, address) as i32);
                let live = self.b.ins().icmp_imm(IntCC::NotEqual, address, 0);
                self.guard(live, *deopt);
                let actual = self.b.ins().load(types::I64, flags, entry,
                    offset_of!(TraceHeapEntry, receiver_class) as i32);
                let matches = self.b.ins().icmp_imm(IntCC::Equal, actual, *class as i64);
                self.guard(matches, *deopt);
                self.b.ins().iconst(types::I8, 0)
            }
            WfInst::GuardEpoch { expected, deopt } => {
                let epoch = self.b.ins().load(types::I64, flags, self.vm,
                    offset_of!(Interpreter<'static>, method_epoch) as i32);
                let valid = self.b.ins().icmp_imm(IntCC::Equal, epoch, *expected as i64);
                self.guard(valid, *deopt);
                self.b.ins().iconst(types::I8, 0)
            }
            WfInst::RuntimeOp { helper, op_index, state, dest } => {
                // Nested frames have no bank: deoptimize (unwind) instead.
                let frame = self.b.use_var(self.frame);
                let target = self.deopt_block(*state);
                let top = self.b.create_block();
                self.b.ins().brif(frame, top, &[], target, &[BlockArg::Value(frame)]);
                self.b.switch_to_block(top);
                let registers = self.ir.deopts[state.0 as usize].registers.clone();
                self.materialize(frame, &registers);
                let helper = match helper {
                    WfRuntimeHelper::Op => self.runtime_op_ref,
                    WfRuntimeHelper::DirectCall => self.direct_call_ref,
                };
                let op = self.b.ins().iconst(types::I32, *op_index as i64);
                let call = self.b.ins().call(helper, &[self.vm, op]);
                let bank = self.b.inst_results(call)[0];
                // Null: the helper left the interpreter at this op (or an error
                // is pending) with the bank already materialized.
                let failed = self.b.create_block();
                self.b.set_cold_block(failed);
                let next = self.b.create_block();
                self.b.ins().brif(bank, next, &[], failed, &[]);
                self.b.switch_to_block(failed);
                let sentinel = self.b.ins().iconst(types::I64, WFJIT_BAILOUT as i64);
                self.b.ins().return_(&[sentinel]);
                self.b.switch_to_block(next);
                self.b.def_var(self.frame, bank);
                match dest {
                    Some(dest) => self.b.ins().load(types::I64, flags, bank, *dest as i32 * 8),
                    None => self.b.ins().iconst(types::I8, 0),
                }
            }
        };
        self.define(result, lowered);
    }

    /// Guard that `object` is a live Instance whose field array holds `slot`;
    /// returns its sidecar entry and object address.
    fn live_instance(&mut self, object: ClValue, slot: u32, deopt: WfDeopt) -> (ClValue, ClValue) {
        let flags = MemFlagsData::trusted();
        let is_heap = self.is_heap(object);
        self.guard(is_heap, deopt);
        let entry = self.heap_entry(object);
        let address = self.b.ins().load(types::I64, flags, entry, offset_of!(TraceHeapEntry, address) as i32);
        let live = self.b.ins().icmp_imm(IntCC::NotEqual, address, 0);
        self.guard(live, deopt);
        let kind = self.b.ins().load(types::I32, flags, entry, offset_of!(TraceHeapEntry, kind) as i32);
        let kind = self.b.ins().band_imm(kind, TRACE_HEAP_KIND_MASK as i64);
        let is_instance = self.b.ins().icmp_imm(IntCC::Equal, kind, TraceHeapKind::Instance as i64);
        self.guard(is_instance, deopt);
        let len = self.b.ins().load(types::I64, flags, entry, offset_of!(TraceHeapEntry, len) as i32);
        let in_bounds = self.b.ins().icmp_imm(IntCC::UnsignedGreaterThan, len, slot as i64);
        self.guard(in_bounds, deopt);
        (entry, address)
    }

    fn edge_args(&self, edge: &WfEdge) -> Vec<BlockArg> {
        edge.args.iter().map(|arg| BlockArg::Value(self.value(*arg))).collect()
    }

    fn lower_terminator(&mut self, terminator: &WfTerminator) {
        match terminator {
            WfTerminator::Jump(edge) => {
                let args = self.edge_args(edge);
                self.b.ins().jump(self.blocks[edge.target.0 as usize], &args);
            }
            WfTerminator::Branch { condition, then, otherwise } => {
                let condition = self.value(*condition);
                let then_args = self.edge_args(then);
                let else_args = self.edge_args(otherwise);
                self.b.ins().brif(condition,
                    self.blocks[then.target.0 as usize], &then_args,
                    self.blocks[otherwise.target.0 as usize], &else_args);
            }
            WfTerminator::Return(value) => {
                // An F64 return (`returns_number`) leaves as raw bits; the host
                // entry canonicalizes NaN before handing it to the interpreter.
                let lowered = self.value(*value);
                let word = if self.ir.ty(*value) == WfTy::F64 {
                    self.b.ins().bitcast(types::I64, MemFlagsData::new(), lowered)
                } else { lowered };
                self.b.ins().return_(&[word]);
            }
        }
    }

    /// Materialize a deopt state for the interpreter (top frame only), then
    /// return the bailout sentinel.
    fn lower_deopt(&mut self, deopt: WfDeopt, block: cranelift::prelude::Block) {
        let state = &self.ir.deopts[deopt.0 as usize];
        let flags = MemFlagsData::trusted();
        self.b.switch_to_block(block);
        let frame = self.b.block_params(block)[0];
        if state.speculative {
            let function = self.b.ins().iconst(types::I32, self.ir.function_index as i64);
            self.b.ins().call(self.speculation_failed_ref, &[self.vm, function]);
        }
        let bail = self.b.create_block();
        self.b.set_cold_block(bail);
        let materialize = self.b.create_block();
        self.b.set_cold_block(materialize);
        self.b.ins().brif(frame, materialize, &[], bail, &[]);
        self.b.switch_to_block(materialize);
        let registers = state.registers.clone();
        self.materialize(frame, &registers);
        let hot_frame = self.b.ins().load(types::I64, flags, self.vm,
            offset_of!(Interpreter<'static>, hot_frame) as i32);
        let root_map = self.b.ins().iconst(types::I32,
            self.function.gc_map_by_op[state.op_index as usize] as i64);
        self.b.ins().store(flags, root_map, hot_frame, offset_of!(Frame, root_map) as i32);
        let pc = self.b.ins().iconst(types::I32, state.op_index as i64);
        self.b.ins().store(flags, pc, self.vm, offset_of!(Interpreter<'static>, hot_pc) as i32);
        self.b.ins().jump(bail, &[]);
        self.b.switch_to_block(bail);
        let sentinel = self.b.ins().iconst(types::I64, WFJIT_BAILOUT as i64);
        self.b.ins().return_(&[sentinel]);
    }
}

/// Lower `ir` into the (empty) function under construction in `b`, whose
/// signature is the v2 body ABI.
pub(super) fn lower_body(
    b: &mut FunctionBuilder<'_>,
    ir: &WfFunction,
    function: &PreparedFunction,
    self_ref: FuncRef,
    speculation_failed_ref: FuncRef,
    runtime_op_ref: FuncRef,
    direct_call_ref: FuncRef,
) {
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let params = b.block_params(entry).to_vec();
    let frame = b.declare_var(types::I64);
    b.def_var(frame, params[1]);
    let mut blocks = Vec::with_capacity(ir.blocks.len());
    blocks.push(entry);
    for data in &ir.blocks[1..] {
        let block = b.create_block();
        for param in &data.params {
            b.append_block_param(block, cl_type(ir.ty(*param)));
        }
        blocks.push(block);
    }
    let mut lowering = Lowering {
        b,
        ir,
        function,
        values: vec![None; ir.value_types.len()],
        blocks,
        deopt_blocks: vec![None; ir.deopts.len()],
        vm: params[0],
        frame,
        depth: params[2],
        params: params[3..].to_vec(),
        self_ref,
        speculation_failed_ref,
        runtime_op_ref,
        direct_call_ref,
        count_calls: wfjit_call_counters_enabled(),
    };
    // The method epoch is guarded by the host entry (see `compile_v2`): the
    // body is otherwise entered only by its own self-calls, and a body with
    // self-calls runs no code that could rebind methods.
    for (index, data) in ir.blocks.iter().enumerate() {
        if index != 0 {
            let block = lowering.blocks[index];
            lowering.b.switch_to_block(block);
            let lowered: Vec<ClValue> = lowering.b.block_params(block).to_vec();
            for (param, value) in data.params.iter().zip(lowered) {
                lowering.define(*param, value);
            }
        }
        for (result, inst) in &data.insts {
            lowering.lower_inst(*result, inst);
        }
        lowering.lower_terminator(data.terminator.as_ref().expect("IR block without terminator"));
    }
    for deopt in 0..ir.deopts.len() {
        if let Some(block) = lowering.deopt_blocks[deopt] {
            lowering.lower_deopt(WfDeopt(deopt as u32), block);
        }
    }
    lowering.b.seal_all_blocks();
}
