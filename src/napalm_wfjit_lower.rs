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
use smallvec::{SmallVec, smallvec};

#[inline]
fn cl_type(ty: WfTy) -> cranelift::prelude::Type {
    match ty {
        WfTy::Tagged => types::I64,
        WfTy::F64 => types::F64,
        WfTy::Bool => types::I8,
    }
}

/// Functions a v2 body calls, declared in the function being built.
pub(super) struct WfHelperRefs {
    pub(super) self_ref: FuncRef,
    pub(super) speculation_failed: FuncRef,
    pub(super) runtime_op: FuncRef,
    pub(super) direct_call: FuncRef,
    pub(super) direct_method: FuncRef,
    pub(super) profiled_method: FuncRef,
    pub(super) call_to_return: FuncRef,
    // Allocation (shared with v1's emitters).
    pub(super) profiled_small_map: FuncRef,
    pub(super) profiled_small_map_commit: FuncRef,
    pub(super) bind_factory_contract: FuncRef,
    pub(super) small_map_literal_regs: FuncRef,
    pub(super) specialized_factory: FuncRef,
    pub(super) reserved_typed_array: FuncRef,
    /// Rebuilds inlined frames on a deopt after an effect.
    pub(super) reconstruct: FuncRef,
    /// Calls from inlined bodies (`CallExplicit`).
    pub(super) call_explicit: FuncRef,
    /// Pure leaf helpers of native builtin methods.
    pub(super) string_eq: FuncRef,
    pub(super) string_code_point_at: FuncRef,
    pub(super) bytes_index: FuncRef,
}

/// Only heap-independent words can retain their *bank equality* fact after
/// a safepoint. Heap references in this VM are stable handles rather than raw
/// moving pointers, but the interpreter is nevertheless authoritative after
/// a helper or an allocation: never skip their later publication to the bank.
fn bank_gc_independent(ir: &WfFunction) -> Vec<bool> {
    let mut independent = ir.value_types.iter().map(|ty| *ty != WfTy::Tagged)
        .collect::<Vec<_>>();
    for block in &ir.blocks {
        for &(value, ref inst) in &block.insts {
            independent[value.0 as usize] = match inst {
                WfInst::ConstTagged(word) => *word & TAG_HEAP_MASK != TAG_HEAP,
                WfInst::BoxF64(_) | WfInst::BoxBool(_) => true,
                _ => ir.ty(value) != WfTy::Tagged,
            };
        }
    }
    independent
}

fn retain_gc_independent(independent: &[bool], known: &mut [Option<WfValue>]) {
    for slot in known {
        if slot.is_some_and(|value| !independent[value.0 as usize]) {
            *slot = None;
        }
    }
}

/// Ahead-of-lowering dataflow for the interpreter/shadow GPR bank. An entry
/// fact is retained only if every executable predecessor has materialized
/// the same SSA word in that slot. Edge arguments are renamed to the target
/// block's SSA parameters before the predecessor facts are intersected.
/// This allows a join to skip writes even when its incoming SSA ids differ.
fn bank_entry_facts(ir: &WfFunction) -> Vec<Vec<Option<WfValue>>> {
    const REGISTERS: usize = 256;
    let count = ir.blocks.len();
    let independent = bank_gc_independent(ir);
    let empty = || vec![None; REGISTERS];
    let mut incoming: Vec<SmallVec<[(usize, usize); 2]>> = vec![SmallVec::new(); count];
    for (pred, block) in ir.blocks.iter().enumerate() {
        for (slot, edge) in wf_edges(block.terminator.as_ref().expect("WF terminator")).enumerate() {
            incoming[edge.target.0 as usize].push((pred, slot));
        }
    }
    let mut entry = vec![None::<Vec<Option<WfValue>>>; count];
    let mut exit = vec![None::<Vec<Option<WfValue>>>; count];
    entry[0] = Some(empty());
    // Fixed point with a conservative bailout for pathological cyclic CFGs.
    for _ in 0..(count.saturating_mul(4).saturating_add(16)) {
        let mut changed = false;
        for b in 0..count {
            let input = if b == 0 {
                Some(empty())
            } else {
                let mut merged: Option<Vec<Option<WfValue>>> = None;
                for &(pred, edge_index) in &incoming[b] {
                    let Some(out) = &exit[pred] else { continue; };
                    let edge = wf_edges(ir.blocks[pred].terminator.as_ref().unwrap())
                        .nth(edge_index).expect("WF successor edge");
                    let mut mapped = out.clone();
                    for known in &mut mapped {
                        if let Some(v) = *known {
                            for (&arg, &param) in edge.args.iter().zip(&ir.blocks[b].params) {
                                if v == arg { *known = Some(param); break; }
                            }
                        }
                    }
                    match &mut merged {
                        None => merged = Some(mapped),
                        Some(existing) => {
                            for (fact, next) in existing.iter_mut().zip(mapped) {
                                if *fact != next { *fact = None; }
                            }
                        }
                    }
                }
                merged
            };
            if entry[b] != input { entry[b] = input.clone(); changed = true; }
            let Some(mut known) = input else { continue; };
            for &(result, ref inst) in &ir.blocks[b].insts {
                match inst {
                    WfInst::FrameLoad(reg) => known[*reg as usize] = Some(result),
                    WfInst::SelfCall { .. } | WfInst::AllocObject { .. } => {
                        // These may allocate, collect and publish other roots.
                        // A later guard must be able to reconstruct every root.
                        retain_gc_independent(&independent, &mut known);
                    }
                    // Runs code and writes the root's call destination slot.
                    WfInst::CallExplicit { .. } => known.fill(None),
                    WfInst::RuntimeOp { state, dest, clobbers, .. } => {
                        for &(reg, v) in &ir.deopts[state.0 as usize].registers {
                            known[reg as usize] = Some(v);
                        }
                        if *clobbers {
                            known.fill(None);
                        } else {
                            // A precise helper only writes dest, but moving-GC
                            // publication can change heap roots in the bank.
                            retain_gc_independent(&independent, &mut known);
                        }
                        if let Some(reg) = dest { known[*reg as usize] = Some(result); }
                    }
                    _ => {}
                }
            }
            if exit[b].as_ref() != Some(&known) { exit[b] = Some(known); changed = true; }
        }
        if !changed {
            return entry.into_iter().map(|facts| facts.unwrap_or_else(empty)).collect();
        }
    }
    vec![empty(); count]
}

fn wf_edges(terminator: &WfTerminator) -> impl Iterator<Item = &WfEdge> {
    let edges = match terminator {
        WfTerminator::Jump(edge) => [Some(edge), None],
        WfTerminator::Branch { then, otherwise, .. } => [Some(then), Some(otherwise)],
        WfTerminator::Return(_) => [None, None],
    };
    edges.into_iter().flatten()
}

struct Lowering<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    ir: &'a WfFunction,
    fast_alloc: Option<GcFastAllocationInfo>,
    values: Vec<Option<ClValue>>,
    /// Mutable Cranelift SSA variables for Tagged words that can carry heap
    /// handles. After a safepoint, reads must come from the relocated root
    /// bank, not the pre-GC Cranelift value. Pure immediates use plain SSA.
    relocated: Vec<Option<cranelift::frontend::Variable>>,
    blocks: Vec<cranelift::prelude::Block>,
    deopt_blocks: Vec<Option<cranelift::prelude::Block>>,
    vm: ClValue,
    /// The top frame's GPR bank (null in nested frames). A host helper may move
    /// the bank, so it is an SSA variable redefined after each RuntimeOp.
    frame: cranelift::frontend::Variable,
    depth: ClValue,
    params: SmallVec<[ClValue; 4]>,
    helpers: WfHelperRefs,
    /// Count native self-calls in the WFJIT statistics (tests, NAPALM_JIT_STATS).
    count_calls: bool,
    /// What the current frame's bank is known to hold, per GPR, at the
    /// current point of the IR block being lowered (reset at block starts).
    /// Materializing a register whose slot already holds its value is
    /// skipped: consecutive helpers then store only what changed.
    bank_known: Vec<Option<WfValue>>,
    /// Precomputed safepoint-stable representations (constant-time lookup).
    gc_independent: Vec<bool>,
    /// Intra-block successful stores: (receiver, field slot, tagged SSA value).
    /// The first following field read still executes its tracking guard, but
    /// can forward the stored word without another heap lookup.
    field_written: SmallVec<[(WfValue, u32, WfValue); 4]>,
    /// Fresh non-escaping allocations in the current block. Only these can
    /// be forwarded safely without an explicit shared-object proof in the IR.
    local_objects: SmallVec<[WfValue; 4]>,
    /// Per deopt state: bank contents known at every guard reaching it.
    deopt_known: Vec<Option<Vec<Option<WfValue>>>>,
    /// The body can run as a nested native frame (it has native self-calls):
    /// helpers and deopts must check for the top frame.
    may_nest: bool,
    bail: Option<cranelift::prelude::Block>,
    tails: [Option<cranelift::prelude::Block>; 2],
    nested_spec_bail: Option<cranelift::prelude::Block>,
}

impl Lowering<'_, '_> {
    fn value(&mut self, value: WfValue) -> ClValue {
        let index = value.0 as usize;
        if let Some(variable) = self.relocated[index] {
            self.b.use_var(variable)
        } else {
            self.values[index].expect("IR value used before its definition")
        }
    }

    fn define(&mut self, value: WfValue, lowered: ClValue) {
        let index = value.0 as usize;
        self.values[index] = Some(lowered);
        if let Some(variable) = self.relocated[index] {
            self.b.def_var(variable, lowered);
        }
    }

    /// The GC root map identifies which Tagged values it is allowed to move.
    /// The SSA variable is redefined with the relocated word in the bank;
    /// Cranelift's variable SSA construction joins these definitions correctly
    /// across branches, including the shared cold deoptimization block.
    fn reload_roots_after_safepoint(
        &mut self,
        bank: ClValue,
        registers: &[(u8, WfValue)],
        written: Option<u8>,
    ) {
        let mut refreshed = SmallVec::<[WfValue; 8]>::new();
        for &(reg, value) in registers {
            if Some(reg) == written || refreshed.contains(&value) { continue; }
            let Some(variable) = self.relocated[value.0 as usize] else { continue; };
            let word = self.b.ins().load(types::I64, MemFlagsData::trusted(), bank, reg as i32 * 8);
            self.b.def_var(variable, word);
            refreshed.push(value);
        }
    }

    fn deopt_block(&mut self, deopt: WfDeopt) -> cranelift::prelude::Block {
        // Every reference is a guard site: keep only bank contents known at
        // all of them.
        let slot = &mut self.deopt_known[deopt.0 as usize];
        match slot {
            None => *slot = Some(self.bank_known.clone()),
            Some(known) => for (entry, current) in known.iter_mut().zip(&self.bank_known) {
                if *entry != *current { *entry = None; }
            },
        }
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

    /// A safepoint may replace the interpreter-owned top-level GPR buffer.
    /// Nested shadow frames, by contrast, live in native stack storage.
    fn refresh_frame_after_safepoint(&mut self) {
        let flags = MemFlagsData::trusted();
        let previous = self.b.use_var(self.frame);
        let hot = self.b.ins().load(types::I64, flags, self.vm,
            offset_of!(Interpreter<'static>, hot_registers) as i32);
        let top = self.b.ins().icmp_imm(IntCC::Equal, self.depth, 0);
        let current = self.b.ins().select(top, hot, previous);
        self.b.def_var(self.frame, current);
    }

    fn frame_arg(&mut self) -> [BlockArg; 1] {
        [BlockArg::Value(self.b.use_var(self.frame))]
    }

    /// Bool: `frame` is the interpreter-owned bank of the top frame (not null
    /// and not a nested frame's shadow bank).
    fn is_top(&mut self, frame: ClValue) -> ClValue {
        let hot = self.b.ins().load(types::I64, MemFlagsData::trusted(), self.vm,
            offset_of!(Interpreter<'static>, hot_registers) as i32);
        self.b.ins().icmp(IntCC::Equal, frame, hot)
    }

    /// Store `registers` (boxing unboxed values) into the bank `frame`,
    /// skipping slots known to hold that value already.
    fn materialize(&mut self, frame: ClValue, registers: &[(u8, WfValue)]) {
        for &(reg, value) in registers {
            if self.bank_known[reg as usize] == Some(value) { continue; }
            let word = self.tagged(value);
            self.b.ins().store(MemFlagsData::trusted(), word, frame, reg as i32 * 8);
            self.bank_known[reg as usize] = Some(value);
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
                    let miss_args = self.frame_arg();
                    let miss_args = if last { &miss_args[..] } else { &[] };
                    self.b.ins().brif(matches, hit, &[], miss, miss_args);
                    self.b.switch_to_block(hit);
                    let value = self.b.ins().load(types::I64, flags, data,
                        slot as i32 * std::mem::size_of::<RegisterWord>() as i32);
                    self.b.ins().jump(done, &[BlockArg::Value(value)]);
                    if miss != target { self.b.switch_to_block(miss); }
                }
                self.b.switch_to_block(done);
                self.b.block_params(done)[0]
            }
            WfInst::SelfCall { args, op_index, deopt } => {
                self.field_written.clear();
                self.local_objects.clear();
                let allowed = self.b.ins().icmp_imm(IntCC::UnsignedLessThan, self.depth, MAX_NATIVE_CALL_DEPTH as i64);
                self.guard(allowed, *deopt);
                // With nested banks the chain can allocate (and collect): this
                // frame publishes its live registers and root map, and the
                // child gets a bank on the native stack linked as a shadow frame.
                let (child_frame, previous_shadow) = if self.ir.nested_banks {
                    let frame = self.b.use_var(self.frame);
                    let registers = &self.ir.deopts[deopt.0 as usize].registers;
                    self.materialize(frame, registers);
                    emit_native_root_map_update(self.b, self.vm, frame,
                        self.ir.frame.gc_map_by_op[*op_index as usize], *op_index);
                    let count = self.ir.frame.reg_count;
                    let storage = self.b.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot, u32::from(count.max(1)) * 8, 3));
                    let child = self.b.ins().stack_addr(types::I64, storage, 0);
                    // Arguments sit in their entry registers too, so the entry
                    // root map describes the bank until the child publishes.
                    for (argument, arg) in args.iter().enumerate().take(self.ir.arity) {
                        let destination = self.ir.frame.entry_arg_dsts[argument];
                        if destination == u16::MAX { continue; }
                        let word = self.tagged(*arg);
                        self.b.ins().store(flags, word, child, destination as i32 * 8);
                    }
                    if self.ir.has_this && self.ir.frame.entry_this_dst != u16::MAX {
                        let this = self.tagged(*args.last().expect("method self-call receiver"));
                        self.b.ins().store(flags, this, child, self.ir.frame.entry_this_dst as i32 * 8);
                    }
                    let shadow_slot = self.b.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot, std::mem::size_of::<NativeShadowFrame>() as u32, 3));
                    let shadow = self.b.ins().stack_addr(types::I64, shadow_slot, 0);
                    let previous = self.b.ins().load(types::I64, flags, self.vm,
                        offset_of!(Interpreter<'static>, native_shadow_top) as i32);
                    self.b.ins().store(flags, previous, shadow, offset_of!(NativeShadowFrame, previous) as i32);
                    self.b.ins().store(flags, child, shadow, offset_of!(NativeShadowFrame, registers) as i32);
                    let function_id = self.b.ins().iconst(types::I32, self.ir.function_index as i64);
                    self.b.ins().store(flags, function_id, shadow, offset_of!(NativeShadowFrame, function) as i32);
                    let entry_map = self.b.ins().iconst(types::I32, self.ir.frame.entry_gc_map as i64);
                    self.b.ins().store(flags, entry_map, shadow, offset_of!(NativeShadowFrame, root_map) as i32);
                    let register_count = self.b.ins().iconst(types::I16, count as i64);
                    self.b.ins().store(flags, register_count, shadow, offset_of!(NativeShadowFrame, reg_count) as i32);
                    self.b.ins().store(flags, shadow, self.vm, offset_of!(Interpreter<'static>, native_shadow_top) as i32);
                    (child, Some(previous))
                } else {
                    (self.b.ins().iconst(types::I64, 0), None)
                };
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
                let mut arguments: SmallVec<[ClValue; 8]> = smallvec![self.vm, child_frame, child_depth];
                arguments.extend(args.iter().map(|arg| self.value(*arg)));
                let call = self.b.ins().call(self.helpers.self_ref, &arguments);
                let value = self.b.inst_results(call)[0];
                // The top frame's bank may have moved while the child ran.
                self.refresh_frame_after_safepoint();
                retain_gc_independent(&self.gc_independent, &mut self.bank_known);
                if self.ir.nested_banks {
                    let current = self.b.use_var(self.frame);
                    let registers = &self.ir.deopts[deopt.0 as usize].registers;
                    self.reload_roots_after_safepoint(current, registers, None);
                }
                if let Some(previous) = previous_shadow {
                    self.b.ins().store(flags, previous, self.vm,
                        offset_of!(Interpreter<'static>, native_shadow_top) as i32);
                }
                let completed = self.b.ins().icmp_imm(IntCC::NotEqual, value, WFJIT_BAILOUT as i64);
                self.guard(completed, *deopt);
                value
            }
            WfInst::InstanceFieldGet { object, slot, nullable, deopt } => {
                let tracking = self.b.ins().load(types::I8, flags, self.vm,
                    offset_of!(Interpreter<'static>, reactive_tracking_active) as i32);
                let untracked = self.b.ins().icmp_imm(IntCC::Equal, tracking, 0);
                self.guard(untracked, *deopt);
                let forwarded = if self.local_objects.contains(object) {
                    // Slice iteration: a glob-imported trait shadows `.iter()`.
                    <[(WfValue, u32, WfValue)]>::iter(&self.field_written).rev().find_map(|&(receiver, field, value)| {
                        (receiver == *object && field == *slot).then_some(value)
                    })
                } else {
                    None
                };
                if let Some(value) = forwarded {
                    // A preceding successful store proves that the receiver
                    // and slot were valid. It does NOT prove tracking is off
                    // or that the value is initialized; check both explicitly.
                    let word = self.value(value);
                    if !*nullable {
                        let initialized = self.b.ins().icmp_imm(IntCC::NotEqual, word, TAG_NULL as i64);
                        self.guard(initialized, *deopt);
                    }
                    word
                } else {
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
                let receiver_id = *object;
                let object = self.value(receiver_id);
                let (entry, address) = self.live_instance(object, *slot, *deopt);
                let class = self.b.ins().load(types::I64, flags, entry,
                    offset_of!(TraceHeapEntry, receiver_class) as i32);
                let exact = self.b.ins().icmp_imm(IntCC::Equal, class, *class_key as i64);
                self.guard(exact, *deopt);
                let data = emit_heap_data_base(self.b, entry, address);
                self.b.ins().store(flags, stored, data,
                    *slot as i32 * std::mem::size_of::<RegisterWord>() as i32);
                // Different SSA receivers may alias the same instance.
                self.field_written.retain(|(_, field, _)| *field != *slot);
                if self.local_objects.contains(&receiver_id) {
                    self.field_written.push((receiver_id, *slot, *value));
                }
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
            WfInst::RuntimeOp { helper, op_index, next_op, state, dest, clobbers } => {
                self.field_written.clear();
                self.local_objects.clear();
                // Helpers work on the interpreter's top frame: a nested frame
                // deoptimizes (unwinds) instead.
                let frame = self.b.use_var(self.frame);
                if self.may_nest {
                    let target = self.deopt_block(*state);
                    let top = self.b.create_block();
                    let is_top = self.is_top(frame);
                    self.b.ins().brif(is_top, top, &[], target, &[BlockArg::Value(frame)]);
                    self.b.switch_to_block(top);
                }
                let registers = &self.ir.deopts[state.0 as usize].registers;
                self.materialize(frame, registers);
                let op = self.b.ins().iconst(types::I32, *op_index as i64);
                let call = match helper {
                    WfRuntimeHelper::Op => self.b.ins().call(self.helpers.runtime_op, &[self.vm, op]),
                    WfRuntimeHelper::DirectCall => self.b.ins().call(self.helpers.direct_call, &[self.vm, op]),
                    WfRuntimeHelper::DirectMethod => self.b.ins().call(self.helpers.direct_method, &[self.vm, op]),
                    WfRuntimeHelper::CallToReturn => self.b.ins().call(self.helpers.call_to_return, &[self.vm, op]),
                    WfRuntimeHelper::ProfiledMethod(target) => {
                        let target = self.b.ins().iconst(types::I32, *target as i64);
                        self.b.ins().call(self.helpers.profiled_method, &[self.vm, op, target])
                    }
                };
                let bank = self.b.inst_results(call)[0];
                // Null: the helper left the interpreter at this op (or an error
                // is pending) with the bank already materialized.
                let failed = self.bail_block();
                let next = self.b.create_block();
                self.b.ins().brif(bank, next, &[], failed, &[]);
                self.b.switch_to_block(next);
                if self.ir.check_resume_pc {
                    // A handler of this frame caught an exception: the
                    // interpreter already stands at the handler, so the
                    // body leaves without touching the bank.
                    let pc = self.b.ins().load(types::I32, flags, self.vm,
                        offset_of!(Interpreter<'static>, hot_pc) as i32);
                    let resumed = self.b.ins().icmp_imm(IntCC::Equal, pc, *next_op as i64);
                    let continue_block = self.b.create_block();
                    self.b.ins().brif(resumed, continue_block, &[], failed, &[]);
                    self.b.switch_to_block(continue_block);
                }
                self.b.def_var(self.frame, bank);
                // Precise helpers preserve all GPRs except their output.
                // Generic helpers can overwrite the entire bank: the builder
                // discards those SSA GPR values and creates new FrameLoads.
                // Rebinding pre-op SSA names to their overwritten slots would
                // incorrectly change their meaning on the success path.
                if !*clobbers {
                    let registers = &self.ir.deopts[state.0 as usize].registers;
                    self.reload_roots_after_safepoint(bank, registers, *dest);
                }
                if *clobbers {
                    self.bank_known.fill(None);
                } else {
                    // Preserve only words that cannot be rewritten by GC.
                    retain_gc_independent(&self.gc_independent, &mut self.bank_known);
                }
                match dest {
                    Some(dest) => {
                        self.bank_known[*dest as usize] = Some(result);
                        self.b.ins().load(types::I64, flags, bank, *dest as i32 * 8)
                    }
                    None => self.b.ins().iconst(types::I8, 0),
                }
            }
            WfInst::AllocObject { op_index, kind, state } => {
                self.field_written.clear();
                self.local_objects.clear();
                // Operands and every live reference go to this frame's bank
                // (the interpreter's, or a nested frame's shadow bank), which is
                // what the allocation paths read and what the GC scans.
                let frame = self.b.use_var(self.frame);
                let registers = &self.ir.deopts[state.0 as usize].registers;
                self.materialize(frame, registers);
                let abort = self.b.create_block();
                self.b.set_cold_block(abort);
                let index = *op_index as usize;
                let function_index = self.ir.function_index;
                let h = &self.helpers;
                let (profiled_small_map, commit, bind, small_map_regs, factory, reserved) = (
                    h.profiled_small_map, h.profiled_small_map_commit, h.bind_factory_contract,
                    h.small_map_literal_regs, h.specialized_factory, h.reserved_typed_array);
                let site = self.ir.alloc_sites.iter().find(|(op, _)| op == op_index).map(|(_, site)| site.clone());
                let root_map = self.ir.frame.gc_map_by_op[index];
                let value = match (kind, site) {
                    (WfAllocKind::TypedArray, Some(WfAllocSite::TypedArray(site))) => emit_native_typed_array_alloc(
                        self.b, factory, reserved, self.vm, frame, site, function_index, index, self.fast_alloc, abort),
                    (_, Some(WfAllocSite::Object(Some(site)))) => emit_profiled_small_map_alloc(self.b,
                        profiled_small_map, commit, bind, self.vm, frame, root_map, function_index, index, &site,
                        self.fast_alloc, abort),
                    _ => {
                        let helper = if *kind == WfAllocKind::SmallMapLiteral { small_map_regs } else { factory };
                        call_native_value_helper(self.b, helper, self.vm, function_index, *op_index, frame, abort)
                    }
                };
                let resume = self.b.current_block().expect("allocation continues in a block");
                self.b.switch_to_block(abort);
                // Allocation may have relocated the top frame bank on an
                // unsuccessful path as well as the successful continuation.
                let hot = self.b.ins().load(types::I64, flags, self.vm,
                    offset_of!(Interpreter<'static>, hot_registers) as i32);
                let top = self.b.ins().icmp_imm(IntCC::Equal, self.depth, 0);
                let current = self.b.ins().select(top, hot, frame);
                // The abort edge is after an allocation safepoint too.
                retain_gc_independent(&self.gc_independent, &mut self.bank_known);
                self.reload_roots_after_safepoint(current, registers, None);
                let target = self.deopt_block(*state);
                self.b.ins().jump(target, &[BlockArg::Value(current)]);
                self.b.switch_to_block(resume);
                self.refresh_frame_after_safepoint();
                retain_gc_independent(&self.gc_independent, &mut self.bank_known);
                let current = self.b.use_var(self.frame);
                self.reload_roots_after_safepoint(current, registers, None);
                self.local_objects.push(result);
                value
            }
            WfInst::CallExplicit { target, receiver, args, state } => {
                self.field_written.clear();
                self.local_objects.clear();
                let snapshot = &self.ir.deopts[state.0 as usize];
                // The root frame: live GPRs into its bank (the runtime sets its
                // root map and PC to the outermost call op).
                let frame = self.b.use_var(self.frame);
                self.materialize(frame, &snapshot.registers);
                // Every inlined frame becomes a GC shadow frame on the native
                // stack for the duration of the call.
                let mut first_previous = None;
                for level in &snapshot.frames {
                    let count = u32::from(level.reg_count.max(1));
                    let bank_slot = self.b.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot, count * 8, 3));
                    let bank = self.b.ins().stack_addr(types::I64, bank_slot, 0);
                    // Slots not tracked by the IR must not hold stale words.
                    let null = self.b.ins().iconst(types::I64, TAG_NULL as i64);
                    for reg in 0..level.reg_count {
                        self.b.ins().store(flags, null, bank, i32::from(reg) * 8);
                    }
                    for &(reg, value) in &level.registers {
                        let word = self.tagged(value);
                        self.b.ins().store(flags, word, bank, reg as i32 * 8);
                    }
                    let shadow_slot = self.b.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot, std::mem::size_of::<NativeShadowFrame>() as u32, 3));
                    let shadow = self.b.ins().stack_addr(types::I64, shadow_slot, 0);
                    let previous = self.b.ins().load(types::I64, flags, self.vm,
                        offset_of!(Interpreter<'static>, native_shadow_top) as i32);
                    first_previous.get_or_insert(previous);
                    self.b.ins().store(flags, previous, shadow, offset_of!(NativeShadowFrame, previous) as i32);
                    self.b.ins().store(flags, bank, shadow, offset_of!(NativeShadowFrame, registers) as i32);
                    let function_id = self.b.ins().iconst(types::I32, level.callee as i64);
                    self.b.ins().store(flags, function_id, shadow, offset_of!(NativeShadowFrame, function) as i32);
                    let root_map = self.b.ins().iconst(types::I32, level.root_map as i64);
                    self.b.ins().store(flags, root_map, shadow, offset_of!(NativeShadowFrame, root_map) as i32);
                    let register_count = self.b.ins().iconst(types::I16, level.reg_count as i64);
                    self.b.ins().store(flags, register_count, shadow, offset_of!(NativeShadowFrame, reg_count) as i32);
                    self.b.ins().store(flags, shadow, self.vm, offset_of!(Interpreter<'static>, native_shadow_top) as i32);
                }
                let args_slot = self.b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, (args.len().max(1) * 8) as u32, 3));
                let args_ptr = self.b.ins().stack_addr(types::I64, args_slot, 0);
                for (index, arg) in args.iter().enumerate() {
                    let word = self.tagged(*arg);
                    self.b.ins().store(flags, word, args_ptr, index as i32 * 8);
                }
                let out_slot = self.b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
                let out = self.b.ins().stack_addr(types::I64, out_slot, 0);
                let root_op = self.b.ins().iconst(types::I32, snapshot.op_index as i64);
                let callee = self.b.ins().iconst(types::I32, *target as i64);
                let receiver_word = match receiver {
                    Some(value) => self.tagged(*value),
                    None => self.b.ins().iconst(types::I64, 0),
                };
                let has_receiver = self.b.ins().iconst(types::I32, receiver.is_some() as i64);
                let argc = self.b.ins().iconst(types::I32, args.len() as i64);
                let call = self.b.ins().call(self.helpers.call_explicit,
                    &[self.vm, root_op, callee, receiver_word, has_receiver, args_ptr, argc, out]);
                let status = self.b.inst_results(call)[0];
                if let Some(previous) = first_previous {
                    self.b.ins().store(flags, previous, self.vm,
                        offset_of!(Interpreter<'static>, native_shadow_top) as i32);
                }
                // 2: declined before doing anything: rebuild the inlined frames
                // and let the interpreter make this call. 1: the interpreter
                // took over (exception...): leave without touching the bank.
                let declined = self.b.ins().icmp_imm(IntCC::Equal, status, 2);
                let deopt_target = self.deopt_block(*state);
                let decided = self.b.create_block();
                self.b.ins().brif(declined, deopt_target, &[BlockArg::Value(frame)], decided, &[]);
                self.b.switch_to_block(decided);
                let ok = self.b.ins().icmp_imm(IntCC::Equal, status, 0);
                let bail = self.bail_block();
                let done = self.b.create_block();
                self.b.ins().brif(ok, done, &[], bail, &[]);
                self.b.switch_to_block(done);
                // Entering the callee may have grown (moved) the register stack.
                let hot = self.b.ins().load(types::I64, flags, self.vm,
                    offset_of!(Interpreter<'static>, hot_registers) as i32);
                self.b.def_var(self.frame, hot);
                self.bank_known.fill(None);
                self.b.ins().load(types::I64, flags, out, 0)
            }
            WfInst::FrameLoad(reg) => {
                self.bank_known[*reg as usize] = Some(result);
                let frame = self.b.use_var(self.frame);
                self.b.ins().load(types::I64, flags, frame, *reg as i32 * 8)
            }
            WfInst::AssumeBool(value) => {
                let word = self.value(*value);
                self.b.ins().ireduce(types::I8, word)
            }
            WfInst::Primitive { kind, receiver, arg, deopt } => {
                // Heap receivers were class-guarded and Numbers unboxed;
                // `BoolNot` checks its immediate tags here.
                let receiver = self.value(*receiver);
                match kind {
                    WfPrimitive::NumAbs => self.b.ins().fabs(receiver),
                    WfPrimitive::NumFloor => self.b.ins().floor(receiver),
                    WfPrimitive::NumCeil => self.b.ins().ceil(receiver),
                    WfPrimitive::NumSqrt => self.b.ins().sqrt(receiver),
                    WfPrimitive::BoolNot => {
                        let is_true = self.b.ins().icmp_imm(IntCC::Equal, receiver, TAG_TRUE as i64);
                        let is_false = self.b.ins().icmp_imm(IntCC::Equal, receiver, TAG_FALSE as i64);
                        let is_bool = self.b.ins().bor(is_true, is_false);
                        self.guard(is_bool, *deopt);
                        is_false
                    }
                    WfPrimitive::ListCount => {
                        let entry = self.heap_entry(receiver);
                        let len = self.b.ins().load(types::I64, flags, entry, offset_of!(TraceHeapEntry, len) as i32);
                        self.b.ins().fcvt_from_sint(types::F64, len)
                    }
                    WfPrimitive::ListSubscript | WfPrimitive::ListIteratorValue => {
                        // In-bounds non-negative integer index; anything else
                        // (negative, fractional, out of range) re-executes.
                        let index = self.value(arg.expect("list index"));
                        let entry = self.heap_entry(receiver);
                        let address = self.b.ins().load(types::I64, flags, entry, offset_of!(TraceHeapEntry, address) as i32);
                        let len = self.b.ins().load(types::I64, flags, entry, offset_of!(TraceHeapEntry, len) as i32);
                        let integer = self.b.ins().fcvt_to_sint_sat(types::I64, index);
                        let back = self.b.ins().fcvt_from_sint(types::F64, integer);
                        let exact = self.b.ins().fcmp(FloatCC::Equal, back, index);
                        let in_bounds = self.b.ins().icmp(IntCC::UnsignedLessThan, integer, len);
                        let ok = self.b.ins().band(exact, in_bounds);
                        self.guard(ok, *deopt);
                        let data = emit_heap_data_base(self.b, entry, address);
                        let offset = self.b.ins().imul_imm(integer, std::mem::size_of::<RegisterWord>() as i64);
                        let slot = self.b.ins().iadd(data, offset);
                        self.b.ins().load(types::I64, flags, slot, 0)
                    }
                    WfPrimitive::ListIterate => {
                        // next = cursor == null ? 0 : cursor + 1 (integer cursor);
                        // result: next while it indexes the list, else false.
                        let cursor = self.value(arg.expect("iteration cursor"));
                        let entry = self.heap_entry(receiver);
                        let len = self.b.ins().load(types::I64, flags, entry, offset_of!(TraceHeapEntry, len) as i32);
                        let is_null = self.b.ins().icmp_imm(IntCC::Equal, cursor, TAG_NULL as i64);
                        let tag = self.b.ins().band_imm(cursor, TAG_MASK as i64);
                        let is_number = self.b.ins().icmp_imm(IntCC::NotEqual, tag, TAG_BASE as i64);
                        let number = self.b.ins().bitcast(types::F64, MemFlagsData::new(), cursor);
                        let integer = self.b.ins().fcvt_to_sint_sat(types::I64, number);
                        let back = self.b.ins().fcvt_from_sint(types::F64, integer);
                        let exact = self.b.ins().fcmp(FloatCC::Equal, back, number);
                        let numeric = self.b.ins().band(is_number, exact);
                        let ok = self.b.ins().bor(is_null, numeric);
                        self.guard(ok, *deopt);
                        let following = self.b.ins().iadd_imm(integer, 1);
                        let zero = self.b.ins().iconst(types::I64, 0);
                        let next = self.b.ins().select(is_null, zero, following);
                        let non_negative = self.b.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, next, 0);
                        let below = self.b.ins().icmp(IntCC::SignedLessThan, next, len);
                        let more = self.b.ins().band(non_negative, below);
                        let as_number = self.b.ins().fcvt_from_sint(types::F64, next);
                        let bits = self.b.ins().bitcast(types::I64, MemFlagsData::new(), as_number);
                        let done = self.b.ins().iconst(types::I64, TAG_FALSE as i64);
                        self.b.ins().select(more, bits, done)
                    }
                    WfPrimitive::StringEq | WfPrimitive::StringCodePointAt | WfPrimitive::BytesIndex => {
                        // Pure leaf call: no bank, no root map, no reloads.
                        let argument = self.value(arg.expect("leaf argument"));
                        let argument = if kind.numeric_arg() {
                            self.b.ins().bitcast(types::I64, MemFlagsData::new(), argument)
                        } else { argument };
                        let helper = match kind {
                            WfPrimitive::StringEq => self.helpers.string_eq,
                            WfPrimitive::StringCodePointAt => self.helpers.string_code_point_at,
                            _ => self.helpers.bytes_index,
                        };
                        let call = self.b.ins().call(helper, &[self.vm, receiver, argument]);
                        let word = self.b.inst_results(call)[0];
                        let decided = self.b.ins().icmp_imm(IntCC::NotEqual, word, WFJIT_BAILOUT as i64);
                        self.guard(decided, *deopt);
                        if *kind == WfPrimitive::StringEq { self.b.ins().ireduce(types::I8, word) } else { word }
                    }
                }
            }
            WfInst::LoopBackedge { counter, trace_slot, shared_flag, deopt } => {
                // Count the iteration; hand the loop to the tracer once it is
                // hot (exactly once per site), whenever a loop trace exists
                // for the header, or when the heap went shared (other threads
                // may need this one to reach a safepoint).
                let cell = self.b.ins().iconst(types::I64, *counter as i64);
                let old = self.b.ins().load(types::I32, flags, cell, 0);
                let new = self.b.ins().iadd_imm(old, 1);
                self.b.ins().store(flags, new, cell, 0);
                let handoff = self.b.ins().icmp_imm(IntCC::Equal, new, WF_LOOP_HANDOFF_ITERATIONS as i64);
                let slot = self.b.ins().iconst(types::I64, *trace_slot as i64);
                let trace = self.b.ins().load(types::I32, flags, slot, 0);
                let traced = self.b.ins().icmp_imm(IntCC::NotEqual, trace, u32::MAX as i64);
                let shared = self.b.ins().load(types::I8, flags, self.vm, *shared_flag as i32);
                let shared = self.b.ins().icmp_imm(IntCC::NotEqual, shared, 0);
                let leave = self.b.ins().bor(handoff, traced);
                let leave = self.b.ins().bor(leave, shared);
                let stay = self.b.ins().bxor_imm(leave, 1);
                self.guard(stay, *deopt);
                self.b.ins().iconst(types::I8, 0)
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

    fn edge_args(&mut self, edge: &WfEdge) -> SmallVec<[BlockArg; 4]> {
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
                // Raw returns (`return_ty`): F64 bits, whose NaN the host entry
                // canonicalizes, or 0/1, which it boxes.
                let lowered = self.value(*value);
                let word = match self.ir.ty(*value) {
                    WfTy::F64 => self.b.ins().bitcast(types::I64, MemFlagsData::new(), lowered),
                    WfTy::Bool => self.b.ins().uextend(types::I64, lowered),
                    WfTy::Tagged => lowered,
                };
                self.b.ins().return_(&[word]);
            }
        }
    }

    /// Materialize a deopt state for the interpreter (top frame only), then
    /// return the bailout sentinel.
    /// Shared cold block returning the bailout sentinel.
    fn bail_block(&mut self) -> cranelift::prelude::Block {
        if let Some(block) = self.bail { return block; }
        let block = self.b.create_block();
        self.b.set_cold_block(block);
        self.bail = Some(block);
        block
    }

    /// Shared deopt tail (params: root map, pc): count a failed speculation
    /// if `speculative`, publish the resume point of the top frame and bail.
    fn deopt_tail(&mut self, speculative: bool) -> cranelift::prelude::Block {
        if let Some(block) = self.tails[speculative as usize] { return block; }
        let block = self.b.create_block();
        self.b.set_cold_block(block);
        self.b.append_block_param(block, types::I32);
        self.b.append_block_param(block, types::I32);
        self.tails[speculative as usize] = Some(block);
        block
    }

    /// Shared cold block for a speculative failure in a nested frame: count it
    /// and unwind.
    fn nested_speculative_bail(&mut self) -> cranelift::prelude::Block {
        if let Some(block) = self.nested_spec_bail { return block; }
        let block = self.b.create_block();
        self.b.set_cold_block(block);
        self.nested_spec_bail = Some(block);
        block
    }

    fn lower_shared_blocks(&mut self) {
        let flags = MemFlagsData::trusted();
        for speculative in [false, true] {
            let Some(block) = self.tails[speculative as usize] else { continue; };
            self.b.switch_to_block(block);
            let (root_map, pc) = (self.b.block_params(block)[0], self.b.block_params(block)[1]);
            if speculative {
                let function = self.b.ins().iconst(types::I32, self.ir.function_index as i64);
                self.b.ins().call(self.helpers.speculation_failed, &[self.vm, function]);
            }
            let hot_frame = self.b.ins().load(types::I64, flags, self.vm,
                offset_of!(Interpreter<'static>, hot_frame) as i32);
            self.b.ins().store(flags, root_map, hot_frame, offset_of!(Frame, root_map) as i32);
            self.b.ins().store(flags, pc, self.vm, offset_of!(Interpreter<'static>, hot_pc) as i32);
            let bail = self.bail_block();
            self.b.ins().jump(bail, &[]);
        }
        if let Some(block) = self.nested_spec_bail {
            self.b.switch_to_block(block);
            let function = self.b.ins().iconst(types::I32, self.ir.function_index as i64);
            self.b.ins().call(self.helpers.speculation_failed, &[self.vm, function]);
            let bail = self.bail_block();
            self.b.ins().jump(bail, &[]);
        }
        if let Some(block) = self.bail {
            self.b.switch_to_block(block);
            let sentinel = self.b.ins().iconst(types::I64, WFJIT_BAILOUT as i64);
            self.b.ins().return_(&[sentinel]);
        }
    }

    fn lower_deopt(&mut self, deopt: WfDeopt, block: cranelift::prelude::Block) {
        let state = &self.ir.deopts[deopt.0 as usize];
        let flags = MemFlagsData::trusted();
        self.b.switch_to_block(block);
        let frame = self.b.block_params(block)[0];
        let speculative = state.speculative;
        if self.may_nest {
            // A nested native frame unwinds; only the top frame materializes.
            let materialize = self.b.create_block();
            self.b.set_cold_block(materialize);
            let unwind = if speculative { self.nested_speculative_bail() } else { self.bail_block() };
            let is_top = self.is_top(frame);
            self.b.ins().brif(is_top, materialize, &[], unwind, &[]);
            self.b.switch_to_block(materialize);
        }
        let registers = &state.registers;
        self.bank_known = self.deopt_known[deopt.0 as usize].take().unwrap_or_else(|| vec![None; 256]);
        self.materialize(frame, registers);
        let root_map = self.b.ins().iconst(types::I32,
            self.ir.frame.gc_map_by_op[state.op_index as usize] as i64);
        let pc = self.b.ins().iconst(types::I32, state.op_index as i64);
        if state.frames.is_empty() {
            let tail = self.deopt_tail(speculative);
            self.b.ins().jump(tail, &[BlockArg::Value(root_map), BlockArg::Value(pc)]);
            return;
        }
        if speculative {
            let function = self.b.ins().iconst(types::I32, self.ir.function_index as i64);
            self.b.ins().call(self.helpers.speculation_failed, &[self.vm, function]);
        }
        let hot_frame = self.b.ins().load(types::I64, flags, self.vm,
            offset_of!(Interpreter<'static>, hot_frame) as i32);
        self.b.ins().store(flags, root_map, hot_frame, offset_of!(Frame, root_map) as i32);
        self.b.ins().store(flags, pc, self.vm, offset_of!(Interpreter<'static>, hot_pc) as i32);
        {
            // Push the inlined callee frames (the runtime enters each call
            // op as the interpreter would), then fill their registers from a
            // native buffer of tagged words.
            let frames = &state.frames;
            let total: usize = frames.iter().map(|frame| frame.registers.len()).sum();
            let buffer_slot = self.b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, (total.max(1) * 8) as u32, 3));
            let buffer = self.b.ins().stack_addr(types::I64, buffer_slot, 0);
            let mut offset = 0;
            for frame in frames {
                for &(_, value) in &frame.registers {
                    let word = self.tagged(value);
                    self.b.ins().store(flags, word, buffer, offset * 8);
                    offset += 1;
                }
            }
            let descriptor: &'static WfReconstruct = Box::leak(Box::new(WfReconstruct {
                frames: frames.iter().map(|frame| WfReconstructFrame {
                    callee: frame.callee,
                    call_op: frame.call_op,
                    resume_op: frame.resume_op,
                    registers: frame.registers.iter().map(|&(reg, _)| reg).collect(),
                }).collect(),
            }));
            let descriptor = self.b.ins().iconst(types::I64, descriptor as *const WfReconstruct as i64);
            self.b.ins().call(self.helpers.reconstruct, &[self.vm, descriptor, buffer]);
        }
        let bail = self.bail_block();
        self.b.ins().jump(bail, &[]);
    }
}

/// Lower `ir` into the (empty) function under construction in `b`, whose
/// signature is the v2 body ABI.
pub(super) fn lower_body(
    b: &mut FunctionBuilder<'_>,
    ir: &WfFunction,
    fast_alloc: Option<GcFastAllocationInfo>,
    helpers: WfHelperRefs,
) {
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let params: SmallVec<[ClValue; 8]> = b.block_params(entry).iter().copied().collect();
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
    let gc_independent = bank_gc_independent(ir);
    let relocated = ir.value_types.iter().enumerate().map(|(index, ty)| {
        (*ty == WfTy::Tagged && !gc_independent[index])
            .then(|| b.declare_var(types::I64))
    }).collect();
    let mut lowering = Lowering {
        b,
        ir,
        fast_alloc,
        values: vec![None; ir.value_types.len()],
        relocated,
        blocks,
        deopt_blocks: vec![None; ir.deopts.len()],
        vm: params[0],
        frame,
        depth: params[2],
        params: params[3..].iter().copied().collect(),
        helpers,
        count_calls: wfjit_call_counters_enabled(),
        bank_known: vec![None; 256],
        gc_independent,
        field_written: SmallVec::new(),
        local_objects: SmallVec::new(),
        deopt_known: vec![None; ir.deopts.len()],
        may_nest: ir.blocks.iter().any(|block| block.insts.iter()
            .any(|(_, inst)| matches!(inst, WfInst::SelfCall { .. }))),
        bail: None,
        tails: [None, None],
        nested_spec_bail: None,
    };
    let bank_entries = bank_entry_facts(ir);
    // The method epoch is guarded by the host entry (see `compile_v2`): the
    // body is otherwise entered only by its own self-calls, and a body with
    // self-calls runs no code that could rebind methods.
    for (index, data) in ir.blocks.iter().enumerate() {
        // Facts are valid on every incoming edge, including loop backedges.
        lowering.bank_known.clone_from(&bank_entries[index]);
        lowering.field_written.clear();
        lowering.local_objects.clear();
        if index != 0 {
            let block = lowering.blocks[index];
            lowering.b.switch_to_block(block);
            for (position, param) in data.params.iter().enumerate() {
                let value = lowering.b.block_params(block)[position];
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
    lowering.lower_shared_blocks();
    lowering.b.seal_all_blocks();
}

#[cfg(test)]
mod bank_dataflow_tests {
    use super::*;

    #[test]
    fn moving_gc_safepoints_kill_heap_bank_equalities() {
        let ir = WfFunction {
            function_index: 0,
            arity: 0,
            has_this: false,
            blocks: vec![
                WfBlockData {
                    params: vec![],
                    insts: vec![
                        (WfValue(0), WfInst::ConstTagged(TAG_HEAP | 7)),
                        (WfValue(1), WfInst::ConstTagged(TAG_TRUE)),
                        (WfValue(2), WfInst::RuntimeOp {
                            helper: WfRuntimeHelper::Op,
                            op_index: 0,
                            next_op: 1,
                            state: WfDeopt(0),
                            dest: None,
                            clobbers: false,
                        }),
                    ],
                    terminator: Some(WfTerminator::Jump(WfEdge {
                        target: WfBlock(1), args: vec![],
                    })),
                },
                WfBlockData {
                    params: vec![], insts: vec![],
                    terminator: Some(WfTerminator::Return(WfValue(0))),
                },
            ],
            value_types: vec![WfTy::Tagged, WfTy::Tagged, WfTy::Bool],
            deopts: vec![WfDeoptState {
                op_index: 0, speculative: false,
                registers: vec![(2, WfValue(0)), (3, WfValue(1))],
                frames: vec![],
            }],
            entry_epoch_guard: None,
            return_ty: WfTy::Tagged,
            check_resume_pc: false,
            nested_banks: false,
            param_types: vec![],
            inlined_calls: 0,
            frame: WfFrameInfo::default(),
            alloc_sites: Vec::new(),
        };
        let states = bank_entry_facts(&ir);
        assert_eq!(states[1][2], None);
        assert_eq!(states[1][3], Some(WfValue(1)));
    }

    #[test]
    fn known_materializations_flow_through_phi_joins() {
        let ir = WfFunction {
            function_index: 0,
            arity: 0,
            has_this: false,
            blocks: vec![
                WfBlockData {
                    params: vec![],
                    insts: vec![
                        (WfValue(0), WfInst::ConstTagged(TAG_TRUE)),
                        (WfValue(1), WfInst::ConstBool(true)),
                        (WfValue(2), WfInst::RuntimeOp {
                            helper: WfRuntimeHelper::Op, op_index: 0,
                            next_op: 1, state: WfDeopt(0), dest: None,
                            clobbers: false,
                        }),
                    ],
                    terminator: Some(WfTerminator::Branch {
                        condition: WfValue(1),
                        then: WfEdge { target: WfBlock(1), args: vec![] },
                        otherwise: WfEdge { target: WfBlock(2), args: vec![] },
                    }),
                },
                WfBlockData {
                    params: vec![], insts: vec![],
                    terminator: Some(WfTerminator::Jump(WfEdge {
                        target: WfBlock(3), args: vec![WfValue(0)],
                    })),
                },
                WfBlockData {
                    params: vec![], insts: vec![],
                    terminator: Some(WfTerminator::Jump(WfEdge {
                        target: WfBlock(3), args: vec![WfValue(0)],
                    })),
                },
                WfBlockData {
                    params: vec![WfValue(3)], insts: vec![],
                    terminator: Some(WfTerminator::Return(WfValue(3))),
                },
            ],
            value_types: vec![WfTy::Tagged, WfTy::Bool, WfTy::Bool, WfTy::Tagged],
            deopts: vec![WfDeoptState {
                op_index: 0, speculative: false,
                registers: vec![(2, WfValue(0))], frames: vec![],
            }],
            entry_epoch_guard: None,
            return_ty: WfTy::Tagged,
            check_resume_pc: false,
            nested_banks: false,
            param_types: vec![],
            inlined_calls: 0,
            frame: WfFrameInfo::default(),
            alloc_sites: Vec::new(),
        };
        let states = bank_entry_facts(&ir);
        assert_eq!(states[3][2], Some(WfValue(3)));
    }
}
