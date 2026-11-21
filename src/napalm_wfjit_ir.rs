//! Whole-function JIT v2: a typed SSA IR for one bytecode function.
//!
//! The IR is backend-neutral (no Cranelift or LLVM types): blocks with typed
//! block parameters, one value per instruction, explicit deoptimization
//! states. `build` translates prepared bytecode into it; a backend lowers it
//! (see `napalm_wfjit_lower.rs` for Cranelift).
//!
//! Model:
//! - Napalm GPRs do not exist at run time. Each GPR maps to an SSA value whose
//!   type is `Tagged` (NaN-boxed word), `F64` (unboxed Number) or `Bool`.
//!   Boxing happens only where a tagged word is required (call arguments,
//!   returns, deoptimization), so NaN canonicalization happens only there.
//! - The supported operations never allocate, collect or run arbitrary code,
//!   so the only observer of a frame's GPRs is the interpreter after a
//!   deoptimization. A deopt state lists the GPRs live at its resume op and
//!   their SSA values; only the interpreter-owned top frame materializes them.
//!   Nested native frames have no register bank at all: a deopt there unwinds
//!   the whole native chain and the top frame replays its outermost self-call
//!   (every supported operation is replay-safe).
//! - The method epoch can only change by running code, which no supported
//!   operation does, so one epoch guard at entry covers every speculative
//!   dispatch decision in the body.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WfTy {
    /// A NaN-boxed Napalm value word.
    Tagged,
    /// An unboxed Number (NaN payload not yet canonicalized).
    F64,
    /// A Boolean condition.
    Bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct WfValue(pub(super) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct WfBlock(pub(super) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct WfDeopt(pub(super) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WfFloatBin { Add, Sub, Mul, Div }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WfFloatCmp { Eq, Ne, Lt, Le, Gt, Ge }

/// Host helper that executes one bytecode op against the top frame's bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WfRuntimeHelper {
    /// `napalm_wfjit_runtime_op`: the op's own handler.
    Op,
    /// `napalm_wfjit_direct_call`: a direct call to another function.
    DirectCall,
}

#[derive(Clone, Debug)]
pub(super) enum WfInst {
    /// Declared argument `n` (Tagged).
    Param(u16),
    ConstTagged(u64),
    ConstBool(bool),
    /// Number constant, as IEEE bits.
    ConstF64(u64),
    /// Tagged word loaded from a stable absolute address (module variable).
    LoadAbsolute(usize),
    /// F64 from a Tagged value; deoptimizes unless it is a Number.
    UnboxNumber { value: WfValue, deopt: WfDeopt },
    /// F64 from a Tagged value proven to be a Number by construction.
    AssumeNumber(WfValue),
    /// Tagged from F64, canonicalizing NaN.
    BoxF64(WfValue),
    /// Tagged from Bool.
    BoxBool(WfValue),
    /// Bool: the Tagged value is neither null nor false.
    Truthy(WfValue),
    FloatBin { op: WfFloatBin, lhs: WfValue, rhs: WfValue },
    FloatNeg(WfValue),
    /// Bool from an IEEE comparison.
    FloatCmp { cc: WfFloatCmp, lhs: WfValue, rhs: WfValue },
    /// Bool: stock identity `==`/`!=` once `lhs` is proven to be one of
    /// `classes` (receiver classes); deoptimizes otherwise.
    IdentityCompare { lhs: WfValue, rhs: WfValue, classes: Vec<u64>, negate: bool, deopt: WfDeopt },
    /// Tagged: shaped-Map property load through a monomorphic/polymorphic
    /// inline cache of `(shape, slot)`; deoptimizes on any miss.
    MapPropertyGet { object: WfValue, entries: Vec<(u32, u32)>, deopt: WfDeopt },
    /// Tagged result of a native self-call with Tagged arguments. `deopt` is
    /// the caller's state at the call op (taken on depth limit or when the
    /// callee chain bailed out).
    SelfCall { args: Vec<WfValue>, deopt: WfDeopt },
    /// Tagged: class instance field `slot` (absolute). Deoptimizes unless the
    /// object is a live Instance with that slot, reactive tracking is off, and
    /// (unless `nullable`) the field is initialized (non-null).
    InstanceFieldGet { object: WfValue, slot: u32, nullable: bool, deopt: WfDeopt },
    /// Stores an immediate (non-heap) Tagged value into field `slot` of an
    /// instance of exactly `class_key`; deoptimizes on any other case
    /// (heap value needing a barrier, reactivity, other class). The result
    /// value is a meaningless Bool.
    InstanceFieldSet { object: WfValue, value: WfValue, slot: u32, class_key: u64, value_immediate: bool, deopt: WfDeopt },
    /// Run op `op_index` through a host helper (top frame only; a nested
    /// frame deoptimizes instead). `state` lists the live GPRs, which are
    /// materialized first because the helper reads operands from the bank and
    /// may collect or run code. The helper may move the bank. The result is
    /// the reloaded `dest` GPR (Tagged), or a meaningless Bool.
    RuntimeOp { helper: WfRuntimeHelper, op_index: u32, state: WfDeopt, dest: Option<u8> },
    /// Re-check the method epoch after code may have run (Bool, meaningless).
    GuardEpoch { expected: u64, deopt: WfDeopt },
    /// Deoptimize unless the Tagged `value` is a heap object of exactly
    /// receiver class `class` (inlined method dispatch). Bool, meaningless.
    GuardClass { value: WfValue, class: u64, deopt: WfDeopt },
}

#[derive(Clone, Debug)]
pub(super) struct WfEdge {
    pub(super) target: WfBlock,
    pub(super) args: Vec<WfValue>,
}

#[derive(Clone, Debug)]
pub(super) enum WfTerminator {
    Jump(WfEdge),
    Branch { condition: WfValue, then: WfEdge, otherwise: WfEdge },
    /// Return a Tagged value.
    Return(WfValue),
}

#[derive(Clone, Debug)]
pub(super) struct WfDeoptState {
    /// Prepared op index the interpreter resumes at.
    pub(super) op_index: u32,
    /// Counts against speculation (a failed type/shape/epoch guard), as
    /// opposed to depth limits and child bailouts.
    pub(super) speculative: bool,
    /// GPRs live at `op_index` that hold SSA values, with those values.
    pub(super) registers: Vec<(u8, WfValue)>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct WfBlockData {
    pub(super) params: Vec<WfValue>,
    pub(super) insts: Vec<(WfValue, WfInst)>,
    pub(super) terminator: Option<WfTerminator>,
}

#[derive(Clone, Debug)]
pub(super) struct WfFunction {
    pub(super) function_index: u32,
    pub(super) arity: usize,
    /// Methods receive `this` as an extra trailing parameter (`Param(arity)`).
    pub(super) has_this: bool,
    pub(super) blocks: Vec<WfBlockData>,
    pub(super) value_types: Vec<WfTy>,
    pub(super) deopts: Vec<WfDeoptState>,
    /// Guard the method epoch at entry (deopt state at op 0).
    pub(super) entry_epoch_guard: Option<(u64, WfDeopt)>,
    /// Every Return returns a Number (used for self-call results). The body
    /// then returns raw `f64` bits; the host entry canonicalizes them.
    pub(super) returns_number: bool,
    /// Type of each declared parameter: `F64` when every native self-call
    /// passes an unboxed Number (the host entry unboxes it with a guard),
    /// otherwise `Tagged`.
    pub(super) param_types: Vec<WfTy>,
    /// Calls inlined into this body (statistics).
    pub(super) inlined_calls: u32,
}

impl WfFunction {
    #[inline]
    pub(super) fn ty(&self, value: WfValue) -> WfTy { self.value_types[value.0 as usize] }
}

/// One bytecode function as the builder sees it (the compiled root or an
/// inlining candidate), gathered from the VM by the caller.
pub(super) struct WfFunctionData<'a> {
    pub(super) function_index: u32,
    pub(super) function: &'a PreparedFunction,
    pub(super) bytecode: &'a BytecodeFunction,
    pub(super) constants: &'a [RegisterWord],
    pub(super) reachable: Vec<bool>,
    pub(super) identity_sites: Vec<Option<NativeIdentitySite>>,
    /// SetFieldClass sites with a native-safe exact receiver class.
    pub(super) plain_store_classes: crate::napalm_collections::HashMap<usize, u64>,
}

/// Inputs of one v2 compilation.
pub(super) struct WfSource<'a> {
    pub(super) root: &'a WfFunctionData<'a>,
    /// Functions the root may inline (transitively), by prepared index.
    pub(super) callees: &'a crate::napalm_collections::HashMap<u32, WfFunctionData<'a>>,
    pub(super) expected_method_epoch: u64,
    /// InvokeMethodDirect resolution (owner class slot, method table).
    pub(super) class_slots: &'a crate::napalm_collections::HashMap<ClassId, u32>,
    pub(super) method_targets: &'a crate::napalm_collections::HashMap<(ReceiverClass, MethodSymbol), CallTarget>,
}

/// Inlining limits: nesting depth, reachable ops per callee, total per root.
const MAX_INLINE_DEPTH: u32 = 3;
const MAX_INLINE_CALLEE_OPS: usize = 96;
const MAX_INLINE_TOTAL_OPS: usize = 480;

/// A call the builder may inline: its prepared target, receiver, arguments
/// and the receiver class to guard (method dispatch).
#[derive(Clone, Debug)]
struct InlineTarget {
    callee: u32,
    receiver: Option<u8>,
    args: Vec<u8>,
    class_guard: Option<u64>,
}

/// What an op does, as far as the v2 tier is concerned.
#[derive(Clone, Debug)]
enum OpKind {
    Nop,
    Move { dst: u8, src: u8 },
    LoadConst { dst: u8, word: u64 },
    LoadAbsolute { dst: u8, address: usize },
    Jump { target: usize },
    JumpIf { cond: u8, target: usize, when_true: bool },
    Return { src: u8 },
    Numeric { dst: u8, lhs: u8, rhs: Option<u8>, quick: NumericQuickOp },
    Identity { dst: u8, lhs: u8, rhs: u8, classes: Vec<u64>, negate: bool },
    MapGet { dst: u8, object: u8, entries: Vec<(u32, u32)> },
    SelfCall { dst: u8, args: Vec<Option<u8>> },
    FieldGet { dst: u8, object: u8, slot: u32, nullable: bool },
    FieldSet { object: u8, value: u8, slot: u32, class_key: u64 },
    /// Executed by a host helper unless `inline` is set and inlining succeeds.
    Runtime { helper: WfRuntimeHelper, reads: Vec<u8>, write: Option<u8>, inline: Option<InlineTarget> },
}

impl OpKind {
    fn reads(&self) -> Vec<u8> {
        match self {
            OpKind::Move { src, .. } | OpKind::Return { src } => vec![*src],
            OpKind::JumpIf { cond, .. } => vec![*cond],
            OpKind::Numeric { lhs, rhs, .. } => std::iter::once(*lhs).chain(*rhs).collect(),
            OpKind::Identity { lhs, rhs, .. } => vec![*lhs, *rhs],
            OpKind::MapGet { object, .. } => vec![*object],
            OpKind::SelfCall { args, .. } => args.iter().flatten().copied().collect(),
            OpKind::FieldGet { object, .. } => vec![*object],
            OpKind::FieldSet { object, value, .. } => vec![*object, *value],
            OpKind::Runtime { reads, .. } => reads.clone(),
            _ => Vec::new(),
        }
    }

    fn writes(&self) -> Option<u8> {
        match self {
            OpKind::Move { dst, .. } | OpKind::LoadConst { dst, .. } | OpKind::LoadAbsolute { dst, .. }
            | OpKind::Numeric { dst, .. } | OpKind::Identity { dst, .. } | OpKind::MapGet { dst, .. }
            | OpKind::SelfCall { dst, .. } | OpKind::FieldGet { dst, .. } => Some(*dst),
            OpKind::Runtime { write, .. } => *write,
            _ => None,
        }
    }
}

/// GPRs of a call-like op's arguments.
fn call_args(function: &PreparedFunction, prepared: &PreparedOp) -> Result<Vec<u8>, String> {
    (0..prepared.c as usize)
        .map(|argument| unsafe { wfjit_call_source_reg(function, prepared, argument) }
            .ok_or_else(|| "call argument register out of range".to_string()))
        .collect()
}

/// Classify every reachable op of `data`, or explain why the function is not
/// (yet) supported by the v2 tier.
fn classify_ops(data: &WfFunctionData<'_>, source: &WfSource<'_>) -> Result<Vec<Option<(OpKind, usize)>>, String> {
    let function = data.function;
    let count = function.ops.len();
    let mut kinds = vec![None; count];
    for index in 0..count {
        if !data.reachable[index] { continue; }
        let op = function.original_ops[index];
        let prepared = function.ops[index];
        let opcode = data.bytecode.chunk.code.get(function.raw_pcs[index] as usize)
            .and_then(|word| word.opcode())
            .ok_or_else(|| format!("invalid opcode at op {index}"))?;
        let next = op.next as usize;
        let kind = match opcode {
            Opcode::Nop => OpKind::Nop,
            Opcode::Move => OpKind::Move { dst: op.a, src: op.b },
            Opcode::LoadConst => {
                let word = data.constants.get(op.x as usize)
                    .ok_or_else(|| format!("constant {} out of range", op.x))?.0;
                OpKind::LoadConst { dst: op.a, word }
            }
            Opcode::LoadNull => OpKind::LoadConst { dst: op.a, word: TAG_NULL },
            Opcode::LoadTrue => OpKind::LoadConst { dst: op.a, word: TAG_TRUE },
            Opcode::LoadFalse => OpKind::LoadConst { dst: op.a, word: TAG_FALSE },
            Opcode::LoadModuleVar => OpKind::LoadAbsolute {
                dst: op.a,
                address: unsafe { module_slot_ptr(&prepared) } as usize,
            },
            Opcode::Jump => OpKind::Jump { target: prepared.x as usize },
            Opcode::JumpIfFalse | Opcode::JumpIfTrue => OpKind::JumpIf {
                cond: op.a,
                target: prepared.x as usize,
                when_true: opcode == Opcode::JumpIfTrue,
            },
            Opcode::Return => OpKind::Return { src: op.a },
            Opcode::Invoke => {
                if let Some(site) = &data.identity_sites[index] {
                    OpKind::Identity {
                        dst: op.a, lhs: op.b, rhs: op.args_offset as u8,
                        classes: site.classes.iter().map(|class| class.0).collect(),
                        negate: site.negate,
                    }
                } else if let Some(quick) = native_numeric_quick(function, index) {
                    let rhs = (!matches!(quick, NumericQuickOp::Neg)).then_some(prepared.y as u8);
                    OpKind::Numeric { dst: op.a, lhs: op.b, rhs, quick }
                } else if let Some(pic) = native_map_property_pic(function, index) {
                    OpKind::MapGet {
                        dst: op.a, object: op.b,
                        entries: pic.entries.iter().map(|entry| (entry.shape, entry.slot as u32)).collect(),
                    }
                } else if let Some((class, target)) = profiled_direct_invoke_site(function, index) {
                    // Monomorphic profiled call: inline it behind a receiver
                    // class guard, or let the helper run its handler.
                    let args = call_args(function, &prepared)?;
                    let reads = std::iter::once(op.b).chain(args.iter().copied()).collect();
                    OpKind::Runtime {
                        helper: WfRuntimeHelper::Op, reads, write: Some(op.a),
                        inline: Some(InlineTarget { callee: target, receiver: Some(op.b), args, class_guard: Some(class.0) }),
                    }
                } else {
                    return Err(format!("op {index}: Invoke without a native fast path"));
                }
            }
            Opcode::CallDirect if prepared.x == data.function_index && !function.uses_this => {
                if prepared.c as usize > function.arity as usize {
                    return Err(format!("op {index}: self-call with extra arguments"));
                }
                let mut args = Vec::with_capacity(function.arity as usize);
                for argument in 0..function.arity as usize {
                    args.push(if argument < prepared.c as usize {
                        Some(unsafe { wfjit_call_source_reg(function, &prepared, argument) }
                            .ok_or_else(|| format!("op {index}: unknown argument register"))?)
                    } else {
                        None
                    });
                }
                OpKind::SelfCall { dst: op.a, args }
            }
            Opcode::GetFieldClass => OpKind::FieldGet {
                dst: prepared.a, object: prepared.b, slot: prepared.x,
                nullable: prepared.y & crate::napalm_bytecode::FIELD_CLASS_NULLABLE != 0,
            },
            Opcode::SetFieldClass => match data.plain_store_classes.get(&index) {
                Some(&class_key) => OpKind::FieldSet {
                    object: prepared.a, value: prepared.b, slot: prepared.x, class_key,
                },
                // Contracted/generic store: the runtime validates and barriers it.
                None => OpKind::Runtime { helper: WfRuntimeHelper::Op, reads: vec![prepared.a, prepared.b], write: None, inline: None },
            },
            Opcode::SetFieldShape => OpKind::Runtime { helper: WfRuntimeHelper::Op, reads: vec![prepared.a, prepared.b, prepared.c], write: None, inline: None },
            Opcode::GetFieldShape => OpKind::Runtime { helper: WfRuntimeHelper::Op, reads: vec![prepared.b], write: Some(prepared.a), inline: None },
            Opcode::StoreModuleVar => OpKind::Runtime { helper: WfRuntimeHelper::Op, reads: vec![prepared.a], write: None, inline: None },
            Opcode::InvokeMethodDirect => {
                let args = call_args(function, &prepared)?;
                let reads = std::iter::once(op.b).chain(args.iter().copied()).collect();
                // The prepared op names the bytecode target; it is only valid for
                // the descriptor's owner class while that class still binds it.
                let inline = match data.bytecode.chunk.specializations.get(prepared.x as usize) {
                    Some(crate::napalm_bytecode::SpecializedDescriptor::DirectMethod { owner, symbol, .. }) => {
                        source.class_slots.get(owner).and_then(|&slot| {
                            let key = ReceiverClass(CLASS_KIND_INSTANCE | slot as u64);
                            (source.method_targets.get(&(key, *symbol)) == Some(&CallTarget::Bytecode(prepared.y)))
                                .then(|| InlineTarget { callee: prepared.y, receiver: Some(op.b), args: args.clone(), class_guard: Some(key.0) })
                        })
                    }
                    _ => None,
                };
                OpKind::Runtime { helper: WfRuntimeHelper::Op, reads, write: Some(op.a), inline }
            }
            Opcode::CallDirect => {
                let args = call_args(function, &prepared)?;
                OpKind::Runtime {
                    helper: WfRuntimeHelper::DirectCall, reads: args.clone(), write: Some(op.a),
                    inline: Some(InlineTarget { callee: prepared.x, receiver: None, args, class_guard: None }),
                }
            }
            other => return Err(format!("op {index}: {other:?} is not supported by WFJIT v2")),
        };
        if matches!(kind, OpKind::Jump { .. } | OpKind::JumpIf { .. }) {
            let target = match kind { OpKind::Jump { target } | OpKind::JumpIf { target, .. } => target, _ => 0 };
            if target <= index || target >= count {
                return Err(format!("op {index}: non-forward jump"));
            }
        }
        if !matches!(kind, OpKind::Return { .. } | OpKind::Jump { .. }) && (next <= index || next >= count) {
            return Err(format!("op {index}: invalid fall-through"));
        }
        kinds[index] = Some((kind, next));
    }
    Ok(kinds)
}

/// Successor op indices of a classified op.
fn successors(kind: &OpKind, next: usize) -> smallvec::SmallVec<[usize; 2]> {
    match kind {
        OpKind::Return { .. } => smallvec::SmallVec::new(),
        OpKind::Jump { target } => smallvec::smallvec![*target],
        OpKind::JumpIf { target, .. } => smallvec::smallvec![*target, next],
        _ => smallvec::smallvec![next],
    }
}

type RegSet = [u64; 4];

#[inline]
fn set_has(set: &RegSet, reg: u8) -> bool { set[(reg >> 6) as usize] & (1 << (reg & 63)) != 0 }
#[inline]
fn set_add(set: &mut RegSet, reg: u8) { set[(reg >> 6) as usize] |= 1 << (reg & 63); }
#[inline]
fn set_remove(set: &mut RegSet, reg: u8) { set[(reg >> 6) as usize] &= !(1 << (reg & 63)); }

/// Backward liveness over the acyclic op graph: GPRs live on entry to each op.
fn liveness(kinds: &[Option<(OpKind, usize)>]) -> Vec<RegSet> {
    let mut live_in = vec![[0u64; 4]; kinds.len()];
    for index in (0..kinds.len()).rev() {
        let Some((kind, next)) = &kinds[index] else { continue; };
        let mut live = [0u64; 4];
        for successor in successors(kind, *next) {
            for (word, other) in live.iter_mut().zip(live_in[successor]) { *word |= other; }
        }
        if let Some(dst) = kind.writes() { set_remove(&mut live, dst); }
        for reg in kind.reads() { set_add(&mut live, reg); }
        live_in[index] = live;
    }
    live_in
}

/// An edge whose target block is not built yet: it carries the full GPR
/// state at the branch; its arguments are filled when the target is built.
struct PendingEdge {
    from: WfBlock,
    /// 0 = Jump / Branch.then, 1 = Branch.otherwise.
    slot: u8,
    state: Vec<Option<WfValue>>,
}

/// Translation mode of the function currently being translated.
struct InlineFrame {
    /// Every guard of an inlined body replays the whole call: the caller's
    /// state at its call op.
    deopt: WfDeopt,
    /// An effect (field store) already happened in this inlined body; any
    /// later guard would replay it, so it aborts the inlining attempt.
    effect_done: bool,
    /// Returns of the inlined body: (block ending in a placeholder jump, value).
    returns: Vec<(WfBlock, WfValue)>,
}

struct Builder<'a> {
    source: &'a WfSource<'a>,
    out: WfFunction,
    state: Vec<Option<WfValue>>,
    current: Option<WfBlock>,
    assume_self_call_numbers: bool,
    returns_number: bool,
    /// The root contains native self-calls: inlined bodies must then be free
    /// of effects (an outer replay would repeat them).
    root_self_calls: bool,
    /// Innermost inlined body being translated, if any.
    inline: Option<InlineFrame>,
    inline_depth: u32,
    inlined_ops: usize,
}

impl Builder<'_> {
    fn new_value(&mut self, ty: WfTy) -> WfValue {
        self.out.value_types.push(ty);
        WfValue(self.out.value_types.len() as u32 - 1)
    }

    fn new_block(&mut self) -> WfBlock {
        self.out.blocks.push(WfBlockData::default());
        WfBlock(self.out.blocks.len() as u32 - 1)
    }

    fn emit_in(&mut self, block: WfBlock, ty: WfTy, inst: WfInst) -> WfValue {
        let value = self.new_value(ty);
        self.out.blocks[block.0 as usize].insts.push((value, inst));
        value
    }

    fn emit(&mut self, ty: WfTy, inst: WfInst) -> WfValue {
        let block = self.current.expect("instruction outside a block");
        self.emit_in(block, ty, inst)
    }

    /// The deopt for a guard at `op_index`. Inside an inlined body every guard
    /// replays the call (and must not follow an effect of that body).
    fn deopt(&mut self, op_index: usize, speculative: bool, live: &RegSet) -> Result<WfDeopt, String> {
        if let Some(frame) = &self.inline {
            if frame.effect_done {
                return Err(format!("op {op_index}: guard after an effect in an inlined body"));
            }
            return Ok(frame.deopt);
        }
        let registers = (0..=255u8)
            .filter(|&reg| set_has(live, reg))
            .filter_map(|reg| self.state[reg as usize].map(|value| (reg, value)))
            .collect();
        self.out.deopts.push(WfDeoptState { op_index: op_index as u32, speculative, registers });
        Ok(WfDeopt(self.out.deopts.len() as u32 - 1))
    }

    fn read(&self, reg: u8, index: usize) -> Result<WfValue, String> {
        self.state[reg as usize].ok_or_else(|| format!("op {index}: reads undefined r{reg}"))
    }

    /// Convert `value` to `ty` in `block` (only widening to Tagged is legal).
    fn convert_in(&mut self, block: WfBlock, value: WfValue, ty: WfTy) -> Result<WfValue, String> {
        match (self.out.ty(value), ty) {
            (from, to) if from == to => Ok(value),
            (WfTy::F64, WfTy::Tagged) => Ok(self.emit_in(block, WfTy::Tagged, WfInst::BoxF64(value))),
            (WfTy::Bool, WfTy::Tagged) => Ok(self.emit_in(block, WfTy::Tagged, WfInst::BoxBool(value))),
            (from, to) => Err(format!("cannot convert {from:?} to {to:?}")),
        }
    }

    fn tagged(&mut self, value: WfValue) -> Result<WfValue, String> {
        let block = self.current.expect("conversion outside a block");
        self.convert_in(block, value, WfTy::Tagged)
    }

    /// A Number operand as F64: unboxed values pass through; a Tagged value is
    /// guarded once and the register is rebound to the unboxed result.
    fn number(&mut self, reg: u8, index: usize, deopt: &mut Option<WfDeopt>, live: &RegSet) -> Result<WfValue, String> {
        let value = self.read(reg, index)?;
        match self.out.ty(value) {
            WfTy::F64 => Ok(value),
            WfTy::Tagged => {
                let guard = match *deopt {
                    Some(guard) => guard,
                    None => { let guard = self.deopt(index, true, live)?; *deopt = Some(guard); guard }
                };
                let unboxed = self.emit(WfTy::F64, WfInst::UnboxNumber { value, deopt: guard });
                self.state[reg as usize] = Some(unboxed);
                Ok(unboxed)
            }
            WfTy::Bool => Err(format!("op {index}: Boolean used as a Number")),
        }
    }

    fn set_edge(&mut self, from: WfBlock, slot: u8, target: WfBlock, args: Vec<WfValue>) {
        let block = &mut self.out.blocks[from.0 as usize];
        let edge = match block.terminator.as_mut().expect("pending edge without terminator") {
            WfTerminator::Jump(edge) => edge,
            WfTerminator::Branch { then, otherwise, .. } => if slot == 0 { then } else { otherwise },
            WfTerminator::Return(_) => unreachable!("return has no edges"),
        };
        edge.target = target;
        edge.args = args;
    }

    fn terminate(&mut self, terminator: WfTerminator) {
        let block = self.current.take().expect("terminator outside a block");
        self.out.blocks[block.0 as usize].terminator = Some(terminator);
    }

    /// Inline `target` at op `index` of the function being translated. On
    /// success the builder continues in the call's continuation block and the
    /// result value is returned; on failure the caller restores its checkpoint.
    fn inline_call(&mut self, target: &InlineTarget, index: usize, live: &RegSet) -> Result<WfValue, String> {
        let source = self.source;
        let Some(data) = source.callees.get(&target.callee) else {
            return Err(format!("op {index}: callee not available for inlining"));
        };
        let callee = data.function;
        if self.inline_depth >= MAX_INLINE_DEPTH {
            return Err("inlining too deep".into());
        }
        if callee.is_async || callee.is_concurrent
            || callee.capture_count != 0 || !callee.exception_handlers.is_empty()
            || callee.return_kind != RegisterKind::Value
            || target.args.len() > callee.arity as usize
        {
            return Err(format!("op {index}: callee is not inlinable"));
        }
        let reachable_ops = data.reachable.iter().filter(|&&live| live).count();
        if reachable_ops > MAX_INLINE_CALLEE_OPS || self.inlined_ops + reachable_ops > MAX_INLINE_TOTAL_OPS {
            return Err(format!("op {index}: callee too large to inline"));
        }
        let kinds = classify_ops(data, source)?;
        if kinds.iter().flatten().any(|(kind, _)| matches!(kind, OpKind::SelfCall { .. })) {
            return Err(format!("op {index}: recursive callee"));
        }
        // An auto-memoized callee is pure, so inlining it is semantically
        // invisible; only a call-free one is inlined, where the memo table
        // costs more than the body. Memoized callees that call out keep the
        // table, which can save exponential work.
        if callee.memoization.enabled()
            && kinds.iter().flatten().any(|(kind, _)| matches!(kind, OpKind::Runtime { .. }))
        {
            return Err(format!("op {index}: memoized callee with calls"));
        }

        // Every guard of the inlined body (including the dispatch guard)
        // replays this call: the caller's state at its call op.
        let replay = self.deopt(index, true, live)?;
        let receiver = match target.receiver {
            Some(reg) => { let value = self.read(reg, index)?; Some(self.tagged(value)?) }
            None => None,
        };
        if let (Some(class), Some(receiver)) = (target.class_guard, receiver) {
            self.emit(WfTy::Bool, WfInst::GuardClass { value: receiver, class, deopt: replay });
        }
        let mut args = Vec::with_capacity(target.args.len());
        for &reg in &target.args { args.push(self.read(reg, index)?); }

        // Translate the callee with its own GPR state.
        let caller_state = std::mem::replace(&mut self.state, vec![None; 256]);
        for argument in 0..callee.arity as usize {
            let value = match args.get(argument) {
                Some(&value) => value,
                None => self.emit(WfTy::Tagged, WfInst::ConstTagged(TAG_NULL)),
            };
            let destination = callee.entry_arg_dsts[argument];
            if destination != u16::MAX { self.state[destination as usize] = Some(value); }
        }
        if callee.uses_this && callee.entry_this_dst != u16::MAX {
            let this = receiver.ok_or_else(|| format!("op {index}: method inlined without a receiver"))?;
            self.state[callee.entry_this_dst as usize] = Some(this);
        }
        let outer = self.inline.replace(InlineFrame { deopt: replay, effect_done: false, returns: Vec::new() });
        self.inline_depth += 1;
        self.inlined_ops += reachable_ops;
        let translated = translate(self, data, &kinds);
        self.inline_depth -= 1;
        let frame = std::mem::replace(&mut self.inline, outer).expect("inline frame");
        self.state = caller_state;
        translated?;
        // A store inside a nested inlined body is also an effect of the body
        // that inlined it: its later guards must not replay the call either.
        if frame.effect_done {
            if let Some(outer) = self.inline.as_mut() { outer.effect_done = true; }
        }
        if frame.returns.is_empty() {
            return Err(format!("op {index}: inlined callee never returns"));
        }
        // Continuation: one parameter typed by the meet of the returned values.
        let first = self.out.ty(frame.returns[0].1);
        let ty = if frame.returns.iter().all(|(_, value)| self.out.ty(*value) == first) { first } else { WfTy::Tagged };
        let continuation = self.new_block();
        let result = self.new_value(ty);
        self.out.blocks[continuation.0 as usize].params.push(result);
        for (from, value) in frame.returns {
            let value = self.convert_in(from, value, ty)?;
            self.set_edge(from, 0, continuation, vec![value]);
        }
        self.current = Some(continuation);
        Ok(result)
    }
}

/// Build the v2 IR, or explain why the function is not supported.
pub(super) fn build(source: &WfSource<'_>) -> Result<WfFunction, String> {
    let function = source.root.function;
    if function.capture_count != 0 {
        return Err("closures are not supported by WFJIT v2 yet".into());
    }
    let kinds = classify_ops(source.root, source)?;
    // Stores are not replay-safe: a nested bailout replays the outermost
    // native self-call, which would repeat them.
    let stores = kinds.iter().flatten().any(|(kind, _)| matches!(kind, OpKind::FieldSet { .. }));
    let self_calls = kinds.iter().flatten().any(|(kind, _)| matches!(kind, OpKind::SelfCall { .. }));
    if stores && self_calls {
        return Err("field stores in a natively recursive function".into());
    }
    // Optimistically type native self-call results as Numbers, then verify that
    // every Return returned an unboxed Number. Sound by induction over completed
    // native executions: a callee whose guard fails unwinds the whole chain and
    // never returns a value.
    // Parameter specialization: a parameter becomes F64 when every self-call
    // passes an unboxed Number for it. Iterate to a fixpoint (the arguments a
    // body passes depend on its own parameter types).
    let arity = function.arity as usize;
    let mut param_types = vec![WfTy::Tagged; arity];
    let mut best = None;
    for _ in 0..4 {
        let optimistic = build_with(source, &kinds, true, &param_types)?;
        let built = if optimistic.returns_number { optimistic } else {
            build_with(source, &kinds, false, &param_types)?
        };
        let observed = built_self_call_f64(&built);
        let next: Vec<WfTy> = (0..arity).map(|i| if observed.get(i).copied().unwrap_or(false) { WfTy::F64 } else { WfTy::Tagged }).collect();
        let stable = next == param_types;
        best = Some(built);
        if stable { break; }
        param_types = next;
    }
    best.ok_or_else(|| "no IR built".to_string())
}

/// Which parameters received an F64 at every native self-call (false when
/// there is no self-call).
fn built_self_call_f64(function: &WfFunction) -> Vec<bool> {
    let mut all_f64 = vec![true; function.arity];
    let mut any_call = false;
    for block in &function.blocks {
        for (_, inst) in &block.insts {
            if let WfInst::SelfCall { args, .. } = inst {
                any_call = true;
                for (index, arg) in args.iter().enumerate().take(function.arity) {
                    let source = match &block.insts.iter().find(|(value, _)| value == arg).map(|(_, inst)| inst) {
                        Some(WfInst::BoxF64(_)) => WfTy::F64,
                        _ => function.ty(*arg),
                    };
                    if source != WfTy::F64 { all_f64[index] = false; }
                }
            }
        }
    }
    if !any_call { return vec![false; function.arity]; }
    all_f64
}

fn build_with(
    source: &WfSource<'_>,
    kinds: &[Option<(OpKind, usize)>],
    assume_self_call_numbers: bool,
    param_types: &[WfTy],
) -> Result<WfFunction, String> {
    let root = source.root;
    let function = root.function;
    let root_self_calls = kinds.iter().flatten().any(|(kind, _)| matches!(kind, OpKind::SelfCall { .. }));
    let mut builder = Builder {
        source,
        out: WfFunction {
            function_index: root.function_index,
            arity: function.arity as usize,
            // `this` only needs passing when the method keeps it in a GPR.
            has_this: function.uses_this && function.entry_this_dst != u16::MAX,
            blocks: Vec::new(),
            value_types: Vec::new(),
            deopts: Vec::new(),
            entry_epoch_guard: None,
            returns_number: true,
            param_types: param_types.to_vec(),
            inlined_calls: 0,
        },
        state: vec![None; 256],
        current: None,
        assume_self_call_numbers,
        returns_number: true,
        root_self_calls,
        inline: None,
        inline_depth: 0,
        inlined_ops: 0,
    };

    let entry = builder.new_block();
    builder.current = Some(entry);
    for argument in 0..function.arity as usize {
        let value = builder.emit(param_types[argument], WfInst::Param(argument as u16));
        let destination = function.entry_arg_dsts[argument];
        if destination != u16::MAX {
            builder.state[destination as usize] = Some(value);
        }
    }
    if builder.out.has_this {
        let this = builder.emit(WfTy::Tagged, WfInst::Param(function.arity as u16));
        builder.state[function.entry_this_dst as usize] = Some(this);
    }
    // Speculative dispatch (numeric, identity, PICs, inlined methods and the
    // ops of inlined bodies) is valid for one method epoch, guarded on entry.
    let needs_epoch = kinds.iter().flatten().any(|(kind, _)| matches!(kind,
        OpKind::Numeric { .. } | OpKind::Identity { .. } | OpKind::MapGet { .. }
        | OpKind::Runtime { inline: Some(_), .. }));
    if needs_epoch {
        let live_in = liveness(kinds);
        let deopt = builder.deopt(0, true, &live_in[0])?;
        builder.out.entry_epoch_guard = Some((source.expected_method_epoch, deopt));
    }

    translate(&mut builder, root, kinds)?;

    // Every terminator must point at a real block now.
    for data in &builder.out.blocks {
        let targets: Vec<WfBlock> = match &data.terminator {
            Some(WfTerminator::Jump(edge)) => vec![edge.target],
            Some(WfTerminator::Branch { then, otherwise, .. }) => vec![then.target, otherwise.target],
            Some(WfTerminator::Return(_)) => Vec::new(),
            None => return Err("block without terminator".into()),
        };
        if targets.iter().any(|target| target.0 as usize >= builder.out.blocks.len()) {
            return Err("edge to an unbuilt block".into());
        }
    }
    let returns_number = builder.returns_number;
    let mut out = builder.out;
    out.returns_number = returns_number;
    Ok(out)
}

/// Translate one function body (the root, or an inlined callee when
/// `builder.inline` is set) starting in `builder.current` with `builder.state`.
fn translate(builder: &mut Builder<'_>, data: &WfFunctionData<'_>, kinds: &[Option<(OpKind, usize)>]) -> Result<(), String> {
    let count = data.function.ops.len();
    let live_in = liveness(kinds);

    // Ops that start a block: entry, jump targets and branch fall-throughs.
    let mut block_start = vec![false; count];
    for (kind, next) in kinds.iter().flatten() {
        match kind {
            OpKind::Jump { target } => block_start[*target] = true,
            OpKind::JumpIf { target, .. } => { block_start[*target] = true; block_start[*next] = true; }
            _ => {}
        }
    }

    let mut pending: Vec<Vec<PendingEdge>> = (0..count).map(|_| Vec::new()).collect();
    // Placeholder targets until the real block exists (patched by set_edge).
    let placeholder = WfBlock(u32::MAX);
    for index in 0..count {
        if index != 0 && block_start[index] {
            // Close the fall-through block into this one.
            if let Some(current) = builder.current {
                builder.terminate(WfTerminator::Jump(WfEdge { target: placeholder, args: Vec::new() }));
                pending[index].push(PendingEdge { from: current, slot: 0, state: builder.state.clone() });
            }
            let incoming = std::mem::take(&mut pending[index]);
            if incoming.is_empty() {
                builder.current = None;
                continue;
            }
            let block = builder.new_block();
            // Merge: a live GPR with the same value on every edge stays; with
            // different values it becomes a block parameter; missing on any
            // edge means it is never read before being written again.
            let live = live_in[index];
            let mut merged = vec![None; 256];
            let mut params: Vec<(u8, WfTy)> = Vec::new();
            for reg in 0..=255u8 {
                if !set_has(&live, reg) { continue; }
                let values: Option<Vec<WfValue>> = incoming.iter().map(|edge| edge.state[reg as usize]).collect();
                let Some(values) = values else { continue; };
                if values.iter().all(|value| *value == values[0]) {
                    merged[reg as usize] = Some(values[0]);
                } else {
                    let first = builder.out.ty(values[0]);
                    let ty = if values.iter().all(|value| builder.out.ty(*value) == first) { first } else { WfTy::Tagged };
                    params.push((reg, ty));
                }
            }
            for (reg, ty) in &params {
                let param = builder.new_value(*ty);
                builder.out.blocks[block.0 as usize].params.push(param);
                merged[*reg as usize] = Some(param);
            }
            for edge in incoming {
                let mut args = Vec::with_capacity(params.len());
                for (reg, ty) in &params {
                    let value = edge.state[*reg as usize].expect("merged parameter missing on an edge");
                    args.push(builder.convert_in(edge.from, value, *ty)?);
                }
                builder.set_edge(edge.from, edge.slot, block, args);
            }
            builder.state = merged;
            builder.current = Some(block);
        }
        if builder.current.is_none() { continue; }
        let Some((kind, next)) = kinds[index].clone() else {
            return Err(format!("op {index} reached but not classified"));
        };
        let live = live_in[index];
        match kind {
            OpKind::Nop => {}
            OpKind::Move { dst, src } => {
                let value = builder.read(src, index)?;
                builder.state[dst as usize] = Some(value);
            }
            OpKind::LoadConst { dst, word } => {
                let value = if word & TAG_MASK != TAG_BASE {
                    builder.emit(WfTy::F64, WfInst::ConstF64(word))
                } else {
                    builder.emit(WfTy::Tagged, WfInst::ConstTagged(word))
                };
                builder.state[dst as usize] = Some(value);
            }
            OpKind::LoadAbsolute { dst, address } => {
                let value = builder.emit(WfTy::Tagged, WfInst::LoadAbsolute(address));
                builder.state[dst as usize] = Some(value);
            }
            OpKind::Numeric { dst, lhs, rhs, quick } => {
                let mut deopt = None;
                let left = builder.number(lhs, index, &mut deopt, &live)?;
                let right = match rhs { Some(rhs) => Some(builder.number(rhs, index, &mut deopt, &live)?), None => None };
                let value = match quick {
                    NumericQuickOp::Neg => builder.emit(WfTy::F64, WfInst::FloatNeg(left)),
                    NumericQuickOp::Add | NumericQuickOp::Sub | NumericQuickOp::Mul | NumericQuickOp::Div => {
                        let op = match quick {
                            NumericQuickOp::Add => WfFloatBin::Add,
                            NumericQuickOp::Sub => WfFloatBin::Sub,
                            NumericQuickOp::Mul => WfFloatBin::Mul,
                            _ => WfFloatBin::Div,
                        };
                        builder.emit(WfTy::F64, WfInst::FloatBin { op, lhs: left, rhs: right.unwrap() })
                    }
                    NumericQuickOp::Lt | NumericQuickOp::Le | NumericQuickOp::Gt
                    | NumericQuickOp::Ge | NumericQuickOp::Eq | NumericQuickOp::Ne => {
                        let cc = match quick {
                            NumericQuickOp::Lt => WfFloatCmp::Lt,
                            NumericQuickOp::Le => WfFloatCmp::Le,
                            NumericQuickOp::Gt => WfFloatCmp::Gt,
                            NumericQuickOp::Ge => WfFloatCmp::Ge,
                            NumericQuickOp::Eq => WfFloatCmp::Eq,
                            _ => WfFloatCmp::Ne,
                        };
                        builder.emit(WfTy::Bool, WfInst::FloatCmp { cc, lhs: left, rhs: right.unwrap() })
                    }
                    other => return Err(format!("op {index}: numeric {other:?} not supported")),
                };
                builder.state[dst as usize] = Some(value);
            }
            OpKind::Identity { dst, lhs, rhs, classes, negate } => {
                let deopt = builder.deopt(index, true, &live)?;
                let left = builder.read(lhs, index)?;
                let left = builder.tagged(left)?;
                let right = builder.read(rhs, index)?;
                let right = builder.tagged(right)?;
                let value = builder.emit(WfTy::Bool, WfInst::IdentityCompare { lhs: left, rhs: right, classes, negate, deopt });
                builder.state[dst as usize] = Some(value);
            }
            OpKind::MapGet { dst, object, entries } => {
                let deopt = builder.deopt(index, true, &live)?;
                let object = builder.read(object, index)?;
                let object = builder.tagged(object)?;
                let value = builder.emit(WfTy::Tagged, WfInst::MapPropertyGet { object, entries, deopt });
                builder.state[dst as usize] = Some(value);
            }
            OpKind::SelfCall { dst, args } => {
                if builder.inline.is_some() {
                    return Err(format!("op {index}: self-call inside an inlined body"));
                }
                let deopt = builder.deopt(index, false, &live)?;
                let mut values = Vec::with_capacity(args.len());
                let mut guard = None;
                for (position, arg) in args.into_iter().enumerate() {
                    let want = builder.out.param_types[position];
                    values.push(match (arg, want) {
                        (Some(reg), WfTy::F64) => builder.number(reg, index, &mut guard, &live)?,
                        (Some(reg), _) => { let value = builder.read(reg, index)?; builder.tagged(value)? }
                        // A missing argument is null: never an unboxed Number.
                        (None, WfTy::F64) => return Err(format!("op {index}: missing F64 argument")),
                        (None, _) => builder.emit(WfTy::Tagged, WfInst::ConstTagged(TAG_NULL)),
                    });
                }
                let result = builder.emit(WfTy::Tagged, WfInst::SelfCall { args: values, deopt });
                let result = if builder.assume_self_call_numbers {
                    builder.emit(WfTy::F64, WfInst::AssumeNumber(result))
                } else { result };
                builder.state[dst as usize] = Some(result);
            }
            OpKind::FieldGet { dst, object, slot, nullable } => {
                let deopt = builder.deopt(index, true, &live)?;
                let object = builder.read(object, index)?;
                let object = builder.tagged(object)?;
                let value = builder.emit(WfTy::Tagged, WfInst::InstanceFieldGet { object, slot, nullable, deopt });
                builder.state[dst as usize] = Some(value);
            }
            OpKind::FieldSet { object, value, slot, class_key } => {
                if builder.inline.is_some() && builder.root_self_calls {
                    return Err(format!("op {index}: store inlined into a natively recursive function"));
                }
                let deopt = builder.deopt(index, true, &live)?;
                let object = builder.read(object, index)?;
                let object = builder.tagged(object)?;
                let stored = builder.read(value, index)?;
                // Unboxed Numbers and Booleans can never be heap references.
                let value_immediate = builder.out.ty(stored) != WfTy::Tagged;
                let stored = builder.tagged(stored)?;
                builder.emit(WfTy::Bool, WfInst::InstanceFieldSet {
                    object, value: stored, slot, class_key, value_immediate, deopt,
                });
                if let Some(frame) = builder.inline.as_mut() { frame.effect_done = true; }
            }
            OpKind::Runtime { helper, write, inline, .. } => {
                // Prefer inlining the call; on failure restore and call out.
                let mut inlined = None;
                if let Some(target) = &inline {
                    let checkpoint = (builder.out.clone(), builder.state.clone(), builder.current, builder.inlined_ops);
                    match builder.inline_call(target, index, &live) {
                        Ok(value) => {
                            builder.out.inlined_calls += 1;
                            if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
                                eprintln!("[napalm-wfjit] v2 inlined {:?} at op {index} of {:?}",
                                    builder.source.callees.get(&target.callee).map(|data| data.function.id), data.function.id);
                            }
                            inlined = Some(value);
                        }
                        Err(reason) => {
                            if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
                                eprintln!("[napalm-wfjit] v2 inline declined at op {index} of {:?}: {reason}", data.function.id);
                            }
                            (builder.out, builder.state, builder.current, builder.inlined_ops) = checkpoint;
                        }
                    }
                }
                if let Some(value) = inlined {
                    if let Some(dst) = write { builder.state[dst as usize] = Some(value); }
                } else {
                    if builder.inline.is_some() {
                        return Err(format!("op {index}: host helper call inside an inlined body"));
                    }
                    if builder.root_self_calls {
                        // A nested native frame has no bank for a host helper.
                        return Err(format!("op {index}: host helper call in a natively recursive function"));
                    }
                    let state = builder.deopt(index, false, &live)?;
                    let ty = if write.is_some() { WfTy::Tagged } else { WfTy::Bool };
                    let result = builder.emit(ty, WfInst::RuntimeOp { helper, op_index: index as u32, state, dest: write });
                    if let Some(dst) = write { builder.state[dst as usize] = Some(result); }
                    // The helper may have run code that rebinds methods.
                    if let Some((expected, _)) = builder.out.entry_epoch_guard {
                        let deopt = builder.deopt(next, true, &live_in[next])?;
                        builder.emit(WfTy::Bool, WfInst::GuardEpoch { expected, deopt });
                    }
                }
            }
            OpKind::Jump { target } => {
                let from = builder.current.unwrap();
                let state = builder.state.clone();
                builder.terminate(WfTerminator::Jump(WfEdge { target: placeholder, args: Vec::new() }));
                pending[target].push(PendingEdge { from, slot: 0, state });
                continue;
            }
            OpKind::JumpIf { cond, target, when_true } => {
                let value = builder.read(cond, index)?;
                let condition = match builder.out.ty(value) {
                    WfTy::Bool => value,
                    WfTy::Tagged => builder.emit(WfTy::Bool, WfInst::Truthy(value)),
                    // Numbers are always truthy in Napalm.
                    WfTy::F64 => builder.emit(WfTy::Bool, WfInst::ConstBool(true)),
                };
                let from = builder.current.unwrap();
                let state = builder.state.clone();
                let (then_target, else_target) = if when_true { (target, next) } else { (next, target) };
                builder.terminate(WfTerminator::Branch {
                    condition,
                    then: WfEdge { target: placeholder, args: Vec::new() },
                    otherwise: WfEdge { target: placeholder, args: Vec::new() },
                });
                pending[then_target].push(PendingEdge { from, slot: 0, state: state.clone() });
                pending[else_target].push(PendingEdge { from, slot: 1, state });
                continue;
            }
            OpKind::Return { src } => {
                let value = builder.read(src, index)?;
                if builder.inline.is_some() {
                    // Leave the inlined body: jump to the call's continuation
                    // (wired by inline_call once every return is known).
                    let from = builder.current.unwrap();
                    builder.terminate(WfTerminator::Jump(WfEdge { target: placeholder, args: Vec::new() }));
                    builder.inline.as_mut().unwrap().returns.push((from, value));
                    continue;
                }
                let value = if builder.assume_self_call_numbers && builder.out.ty(value) == WfTy::F64 {
                    // Raw bits; kept only if every Return is a Number.
                    value
                } else {
                    builder.returns_number = false;
                    builder.tagged(value)?
                };
                builder.terminate(WfTerminator::Return(value));
                continue;
            }
        }
        // Straight-line op: control continues into `next`; if that op starts a
        // block, the next iteration closes this one into it.
    }
    if builder.current.is_some() {
        return Err("control falls off the end of the function".into());
    }
    if pending.iter().any(|edges| !edges.is_empty()) {
        return Err("unresolved control-flow edge".into());
    }
    Ok(())
}
