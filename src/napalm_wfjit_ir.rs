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
use smallvec::{SmallVec, smallvec};

type RegList = SmallVec<[u8; 4]>;

/// Pass scheduling, CFG/dominator analysis and the SSA verifier.
#[path = "napalm_wf_ir_optimizer.rs"]
mod wf_optimizer;
/// Optimization passes over completed IR (run by `build`).
#[path = "napalm_wf_ir_optimizing.rs"]
mod wf_optimizing;

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

/// Builtin methods lowered natively behind a receiver class guard (see
/// `WfInst::Primitive`). None of them collects, runs code or throws: the
/// rare cases (out of range, non-integer index...) deoptimize so the
/// interpreter re-executes the op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WfPrimitive {
    /// Pure Number methods with an unboxed F64 receiver and result.
    NumAbs,
    NumFloor,
    NumCeil,
    NumSqrt,
    /// `list.length` / `list.count` (F64).
    ListCount,
    /// `list[i]` for an in-bounds non-negative integer index (Tagged).
    ListSubscript,
    /// List iteration protocol: `iterate(cursor)`, `iteratorValue(cursor)`.
    ListIterate,
    ListIteratorValue,
    /// `!bool` (Bool).
    BoolNot,
    /// `string == other` (Bool), pure leaf helper.
    StringEq,
    /// `string.codePointAt(i)` (Tagged: Number or null), pure leaf helper.
    StringCodePointAt,
    /// `bytes[i]` (Tagged Number), pure leaf helper.
    BytesIndex,
}

impl WfPrimitive {
    /// Arguments after the receiver.
    pub(super) fn arity(self) -> usize {
        match self {
            WfPrimitive::ListCount | WfPrimitive::BoolNot
            | WfPrimitive::NumAbs | WfPrimitive::NumFloor | WfPrimitive::NumCeil | WfPrimitive::NumSqrt => 0,
            _ => 1,
        }
    }
    pub(super) fn numeric_receiver(self) -> bool {
        matches!(self, WfPrimitive::NumAbs | WfPrimitive::NumFloor | WfPrimitive::NumCeil | WfPrimitive::NumSqrt)
    }
    /// The argument is used as an unboxed Number.
    pub(super) fn numeric_arg(self) -> bool {
        matches!(self, WfPrimitive::ListSubscript | WfPrimitive::ListIteratorValue
            | WfPrimitive::StringCodePointAt | WfPrimitive::BytesIndex)
    }
    pub(super) fn result_ty(self) -> WfTy {
        match self {
            WfPrimitive::ListCount | WfPrimitive::NumAbs | WfPrimitive::NumFloor
            | WfPrimitive::NumCeil | WfPrimitive::NumSqrt => WfTy::F64,
            WfPrimitive::BoolNot | WfPrimitive::StringEq => WfTy::Bool,
            _ => WfTy::Tagged,
        }
    }
}

/// Allocation ops lowered natively (see `WfInst::AllocObject`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WfAllocKind {
    /// AllocMap fused with its property stores (small Map literal).
    SmallMapLiteral,
    /// NewObjectEmpty/Shape/WithProperties, NewInterfaceOrTypeObject.
    SpecializedObject,
    /// DefineArrayWithGenerics (typed array, optional capacity).
    TypedArray,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WfFloatCmp { Eq, Ne, Lt, Le, Gt, Ge }

/// Host helper that executes one bytecode op against the top frame's bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WfRuntimeHelper {
    /// `napalm_wfjit_runtime_op`: the op's own handler.
    Op,
    /// `napalm_wfjit_direct_call`: a direct call to another function.
    DirectCall,
    /// `napalm_wfjit_direct_method`: InvokeMethodDirect whose receiver class
    /// was guarded natively.
    DirectMethod,
    /// `napalm_wfjit_profiled_method`: a profiled monomorphic Invoke to the
    /// given target, receiver class guarded natively.
    ProfiledMethod(u32),
    /// `napalm_wfjit_call_to_return`: any call op, run to completion through
    /// the interpreter's generic call path.
    CallToReturn,
}

/// Back edges handed to the tracing tier after this many native iterations
/// of one loop (cumulative): hot loops then belong to the loop tracer.
pub(super) const WF_LOOP_HANDOFF_ITERATIONS: u32 = 1000;

/// What a `CallExplicit` calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WfCallTarget {
    /// A statically known bytecode function (prepared index).
    Static(u32),
    /// The method an `Invoke` of inlined function `function` dispatches to
    /// for the receiver's class, resolved at run time through the op's own
    /// inline cache `ic_slot` (any target kind: bytecode or host method).
    Dynamic { function: u32, ic_slot: u32 },
}

#[derive(Clone, Debug, PartialEq)]
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
    /// callee chain bailed out). With `nested_banks`, the caller first
    /// publishes `deopt`'s registers and root map for the GC.
    SelfCall { args: Vec<WfValue>, op_index: u32, deopt: WfDeopt },
    /// Tagged: allocate the object of op `op_index` (a literal or typed
    /// array) from operands materialized in the current frame's bank, which
    /// may be a nested frame's shadow bank. Inline TLAB path when profiled,
    /// else an explicit-bank helper; failure deoptimizes at `state` before
    /// the object is exposed. Allocation is replay-safe.
    /// `function` owns op `op_index` (the root, or an inlined callee: then
    /// `state` holds the inlined frames, published as GC shadow frames whose
    /// innermost bank the allocation reads its operands from).
    AllocObject { op_index: u32, kind: WfAllocKind, state: WfDeopt, function: u32 },
    /// Tagged: class instance field `slot` (absolute). Deoptimizes unless the
    /// object is a live Instance with that slot, reactive tracking is off, and
    /// (unless `nullable`) the field is initialized (non-null).
    InstanceFieldGet { object: WfValue, slot: u32, nullable: bool, deopt: WfDeopt },
    /// Stores an immediate (non-heap) Tagged value into field `slot` of an
    /// instance of exactly `class_key`; deoptimizes on any other case
    /// (heap value needing a barrier, reactivity, other class). The result
    /// value is a meaningless Bool.
    InstanceFieldSet { object: WfValue, value: WfValue, slot: u32, class_key: u64, value_immediate: bool, deopt: WfDeopt },
    /// Bool, meaningless: a class field store the runtime validates against
    /// the field's type (contract, generic field), with the write barrier, in
    /// a non-allocating leaf helper. A rejected store (or any receiver it does
    /// not handle) deoptimizes at `deopt` before the store; the interpreter
    /// then reports the error. Never emitted with reactivity enabled.
    InstanceFieldStoreChecked { object: WfValue, value: WfValue, slot: u32, deopt: WfDeopt },
    /// Tagged: `left + right` of two Strings (a new String), inside an
    /// inlined body. An allocation: the frames of `state` are published as at
    /// an inlined `AllocObject`; non-String operands deoptimize at `state`.
    StringConcat { left: WfValue, right: WfValue, state: WfDeopt },
    /// Run op `op_index` through a host helper (top frame only; a nested
    /// frame deoptimizes instead). `state` lists the live GPRs, which are
    /// materialized first because the helper reads operands from the bank and
    /// may collect or run code. The helper may move the bank. The result is
    /// the reloaded `dest` GPR (Tagged), or a meaningless Bool.
    /// `next_op` is where the interpreter must stand afterwards; in functions
    /// with exception handlers anything else (a caught exception) bails out.
    /// `clobbers`: the op is generic and may have written any GPR.
    RuntimeOp { helper: WfRuntimeHelper, op_index: u32, next_op: u32, state: WfDeopt, dest: Option<u8>, clobbers: bool },
    /// Tagged GPR `reg` read from the top frame's bank. Only valid where the
    /// bank is known to exist and be current: right after a RuntimeOp, or at
    /// entry of a body that never runs as a nested native frame.
    FrameLoad(u8),
    /// Re-check the method epoch after code may have run (Bool, meaningless).
    GuardEpoch { expected: u64, deopt: WfDeopt },
    /// Deoptimize unless the Tagged `value` is a heap object of exactly
    /// receiver class `class` (inlined method dispatch). Bool, meaningless.
    GuardClass { value: WfValue, class: u64, deopt: WfDeopt },
    /// Bool from a native self-call result that the body returned raw.
    AssumeBool(WfValue),
    /// A call made from inside an inlined body, whose own frame does not exist:
    /// the runtime enters `target` with explicit arguments as if the root
    /// frame were executing its call op `state.op_index` (the root's
    /// destination register of that op, dead until the inlined call returns,
    /// receives the result). While it runs, the root's live GPRs
    /// (`state.registers`) are in its bank and every inlined frame
    /// (`state.frames`) is published as a GC shadow frame. Tagged result; on
    /// failure (exception, the interpreter took over) the body bails out
    /// without touching the bank.
    /// `scratch` is that destination register (the only root GPR the call
    /// writes).
    CallExplicit { target: WfCallTarget, receiver: Option<WfValue>, args: Vec<WfValue>, state: WfDeopt, scratch: u8 },
    /// A builtin method on a receiver whose class was already guarded
    /// (F64 receiver when `kind.numeric_receiver()`, otherwise Tagged;
    /// the argument is F64 when `kind.numeric_arg()`).
    /// Deoptimizes at `deopt` on any case it does not handle.
    Primitive { kind: WfPrimitive, receiver: WfValue, arg: Option<WfValue>, deopt: WfDeopt },
    /// Loop back edge bookkeeping (Bool, meaningless): counts iterations in
    /// the cell at `counter` and deoptimizes to the loop header (`deopt`)
    /// once the loop is hot enough for the tracer, when a loop trace exists
    /// for the header (`trace_slot` holds its dense TraceId or `u32::MAX`),
    /// or when the heap entered shared mode (byte at offset `shared_flag`
    /// of the Interpreter).
    LoopBackedge { counter: usize, trace_slot: usize, shared_flag: usize, deopt: WfDeopt },
    /// Tagged: a complete new object (small literal or class-instance shell)
    /// with SSA slot values (`kind.slots()` of them, Tagged), allocated by the
    /// TLAB-only helper: it never reaches a safepoint, so no GC can happen.
    /// When the TLAB is exhausted it deoptimizes at `deopt`, which replays
    /// the op in the interpreter. Allocation sinking may remove it.
    NewObject { kind: crate::napalm_trace_ir::TraceObjectKind, fields: Vec<WfValue>, deopt: WfDeopt },
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct WfDeoptState {
    /// Prepared op index the interpreter resumes at.
    pub(super) op_index: u32,
    /// Counts against speculation (a failed type/shape/epoch guard), as
    /// opposed to depth limits and child bailouts.
    pub(super) speculative: bool,
    /// GPRs live at `op_index` that hold SSA values, with those values.
    pub(super) registers: Vec<(u8, WfValue)>,
    /// Inlined frames to rebuild on top of the resumed one (outermost
    /// first). Empty: the interpreter resumes the top frame at `op_index`.
    /// Otherwise `op_index` is the root's call op, and each entry pushes the
    /// callee frame of a call that is resumed at `resume_op` with its own
    /// registers: a guard after an effect in an inlined body continues
    /// exactly where it failed instead of replaying the call.
    pub(super) frames: Vec<WfInlineDeoptFrame>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct WfInlineDeoptFrame {
    pub(super) callee: u32,
    /// The call op in the parent frame (the root or the previous entry).
    pub(super) call_op: u32,
    pub(super) resume_op: u32,
    pub(super) registers: Vec<(u8, WfValue)>,
    /// The callee's GPR count and its GC root map at `resume_op`, used when
    /// the frame is published as a native shadow frame (`CallExplicit`).
    pub(super) reg_count: u16,
    pub(super) root_map: u32,
    /// The GPRs that GC map marks as references (all the collector scans in
    /// a shadow frame): those without a value in `registers` are nulled.
    pub(super) gc_refs: Vec<u8>,
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
    /// Type of every Return. `F64`: the body returns raw `f64` bits (the host
    /// entry canonicalizes NaN); `Bool`: raw 0/1 (the host entry boxes it);
    /// `Tagged`: tagged words. Raw returns serve native self-call results.
    pub(super) return_ty: WfTy,
    /// The function has exception handlers: after each host helper the
    /// interpreter must stand at the op's successor, otherwise a handler in
    /// this frame caught an exception and the body bails out.
    pub(super) check_resume_pc: bool,
    /// Nested native frames get a register bank on the native stack, linked
    /// as a GC shadow frame (needed when the native recursion allocates).
    pub(super) nested_banks: bool,
    /// Type of each declared parameter: `F64` when every native self-call
    /// passes an unboxed Number (the host entry unboxes it with a guard),
    /// otherwise `Tagged`.
    pub(super) param_types: Vec<WfTy>,
    /// Calls inlined into this body (statistics).
    pub(super) inlined_calls: u32,
    /// Frame layout of the function, so code generation (possibly on a
    /// background thread) never reads live VM metadata.
    pub(super) frame: WfFrameInfo,
    /// Allocation sites of the `AllocObject` ops, captured with the IR.
    /// Allocation sites by (function, op).
    pub(super) alloc_sites: Vec<((u32, u32), WfAllocSite)>,
}

/// The parts of a prepared function's frame layout the lowering needs.
#[derive(Clone, Debug, Default)]
pub(super) struct WfFrameInfo {
    pub(super) reg_count: u16,
    pub(super) entry_arg_dsts: Vec<u16>,
    pub(super) entry_this_dst: u16,
    pub(super) entry_gc_map: u32,
    pub(super) gc_map_by_op: Vec<u32>,
}

impl WfFrameInfo {
    fn of(function: &PreparedFunction) -> Self {
        Self {
            reg_count: function.frame.general_register_count,
            entry_arg_dsts: function.entry_arg_dsts.to_vec(),
            entry_this_dst: function.entry_this_dst,
            entry_gc_map: function.entry_gc_map,
            gc_map_by_op: function.gc_map_by_op.clone(),
        }
    }
}

/// Profiled shape of an allocation op, taken when the IR was built.
#[derive(Clone, Debug)]
pub(super) enum WfAllocSite {
    /// Small literal or specialized object: the profiled inline path, if any.
    Object(Option<NativeProfiledSmallMapSite>),
    TypedArray(TypedArraySite),
}

impl WfFunction {
    #[inline]
    pub(super) fn ty(&self, value: WfValue) -> WfTy { self.value_types[value.0 as usize] }

    /// One-line description for `NAPALM_JIT_EVENTS`.
    pub(super) fn summary(&self) -> String {
        let mut self_calls = 0;
        let mut helpers = 0;
        let mut loops = 0;
        let mut allocs = 0;
        for block in &self.blocks {
            for (_, inst) in &block.insts {
                match inst {
                    WfInst::SelfCall { .. } => self_calls += 1,
                    WfInst::RuntimeOp { .. } => helpers += 1,
                    WfInst::LoopBackedge { .. } => loops += 1,
                    WfInst::AllocObject { .. } => allocs += 1,
                    _ => {}
                }
            }
        }
        format!("ret={:?} params={:?} native_self_calls={self_calls} helpers={helpers} allocs={allocs} back_edges={loops} banks={} inlined={}",
            self.return_ty, self.param_types, self.nested_banks, self.inlined_calls)
    }
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
    /// Object-creating ops the body may allocate itself (`WfInst::NewObject`).
    pub(super) object_layouts: crate::napalm_collections::HashMap<usize, crate::napalm_tracing_jit::TraceObjectLayout>,
    /// Invoke ops whose runtime inline cache saw exactly one bytecode target:
    /// (receiver class, target), the dispatch profile when the op was never
    /// quickened.
    pub(super) cached_invokes: crate::napalm_collections::HashMap<usize, (u64, u32)>,
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
    /// Address of the root's dense loop-trace table (`u32` per op,
    /// `u32::MAX` when no trace is installed at that loop header).
    pub(super) trace_headers: usize,
    /// Offset of the heap's `shared_mode` flag (one byte) in the Interpreter.
    pub(super) shared_flag: usize,
    /// Iteration counter cell per loop header of the root (stable addresses,
    /// kept across recompilations of the same function).
    pub(super) loop_counters: &'a std::cell::RefCell<crate::napalm_collections::HashMap<usize, usize>>,
    /// Builtin primitive identities (native builtin method lowering).
    pub(super) manifest: Option<&'a crate::napalm_builtin::BuiltinManifest>,
    /// False compiles without any speculative guard (every profiled site runs
    /// through its host helper): `NAPALM_WFJIT_SPECULATE=0`.
    pub(super) speculate: bool,
    /// Root ops whose speculative guards failed in earlier compilations: they
    /// compile without speculation. `u32::MAX` marks the entry's parameter
    /// types (no native self-call specialization).
    pub(super) failed_sites: &'a [u32],
    /// Reactivity is enabled (it never turns off again): stores must notify.
    pub(super) reactive: bool,
}

/// A monomorphic Invoke site whose profiled target is a builtin the v2 tier
/// lowers natively: (receiver class, primitive).
fn native_primitive_site(
    function: &PreparedFunction,
    index: usize,
    source: &WfSource<'_>,
) -> Option<(u64, WfPrimitive)> {
    use crate::napalm_builtin::BuiltinPrimitive as P;
    let prepared = function.ops[index];
    if prepared.flags & QUICK_GENERIC_DISABLED != 0 { return None; }
    let site = function.quick_invokes.get(index)?;
    if site.class.0 == 0 { return None; }
    let kind = match site.target {
        CallTarget::Primitive(id) => match source.manifest?.primitive_kind(id)? {
            P::NumAbs => WfPrimitive::NumAbs,
            P::NumFloor => WfPrimitive::NumFloor,
            P::NumCeil => WfPrimitive::NumCeil,
            P::NumSqrt => WfPrimitive::NumSqrt,
            P::ListCount => WfPrimitive::ListCount,
            P::ListSubscript => WfPrimitive::ListSubscript,
            P::ListIterate => WfPrimitive::ListIterate,
            P::ListIteratorValue => WfPrimitive::ListIteratorValue,
            P::BoolNot => WfPrimitive::BoolNot,
            P::StringEq => WfPrimitive::StringEq,
            P::StringCodePointAt => WfPrimitive::StringCodePointAt,
            _ => return None,
        },
        CallTarget::NativeBuiltin(NativeBuiltin::Bytes(BytesNative::BytesIndex)) => WfPrimitive::BytesIndex,
        _ => return None,
    };
    (prepared.c as usize == kind.arity()).then_some((site.class.0, kind))
}

/// Inlining limits: nesting depth, reachable ops per callee, total per root.
const MAX_INLINE_DEPTH: u32 = 3;
const MAX_INLINE_CALLEE_OPS: usize = 96;
const MAX_INLINE_TOTAL_OPS: usize = 480;
/// Dispatched calls (polymorphic `Invoke`) allowed in inlined bodies per root.
const MAX_INLINED_DISPATCHED_CALLS: u32 = 8;

/// A call the builder may inline: its prepared target, receiver, arguments
/// and the receiver class to guard (method dispatch).
#[derive(Clone, Debug)]
struct InlineTarget {
    callee: u32,
    receiver: Option<u8>,
    args: RegList,
    class_guard: Option<u64>,
}

/// What an op does, as far as the v2 tier is concerned.
#[derive(Clone, Debug)]
enum OpKind {
    Nop,
    Move { dst: u8, src: u8 },
    Swap { a: u8, b: u8 },
    LoadConst { dst: u8, word: u64 },
    /// An opaque raw word (typed-register immediates such as LoadI32).
    LoadRaw { dst: u8, word: u64 },
    LoadAbsolute { dst: u8, address: usize },
    Jump { target: usize },
    JumpIf { cond: u8, target: usize, when_true: bool },
    Return { src: u8 },
    Numeric { dst: u8, lhs: u8, rhs: Option<u8>, quick: NumericQuickOp },
    Identity { dst: u8, lhs: u8, rhs: u8, classes: Vec<u64>, negate: bool },
    MapGet { dst: u8, object: u8, entries: Vec<(u32, u32)> },
    /// Native self-call; `receiver` for methods: register and the exact class
    /// a dynamic dispatch must still select (None for a static call).
    SelfCall { dst: u8, args: Vec<Option<u8>>, receiver: Option<(u8, Option<u64>)> },
    FieldGet { dst: u8, object: u8, slot: u32, nullable: bool },
    FieldSet { object: u8, value: u8, slot: u32, class_key: u64 },
    /// A field store the runtime validates (`InstanceFieldStoreChecked`).
    FieldStoreChecked { object: u8, value: u8, slot: u32 },
    /// Native allocation; operands are read from the bank (all GPRs).
    Alloc { dst: u8, kind: WfAllocKind, reads: RegList },
    /// An object allocated from SSA operands (`WfInst::NewObject`): `fields`
    /// maps each operand register to its dense slot. A class instance may run
    /// its constructor (inlined, or the whole op through the generic path).
    NewObject { dst: u8, kind: crate::napalm_trace_ir::TraceObjectKind, fields: SmallVec<[(u8, u8); 4]>, constructor: Option<u32>, args: RegList },
    /// Monomorphic builtin method lowered natively.
    Primitive { dst: u8, receiver: u8, arg: Option<u8>, class: u64, kind: WfPrimitive },
    /// Executed by a host helper unless `inline` is set and inlining succeeds.
    /// `guard`: receiver register and exact class the helper relies on.
    /// `generic`: the op's operands are not modelled; it reads every GPR and
    /// every GPR live afterwards is reloaded from the bank.
    Runtime {
        helper: WfRuntimeHelper,
        reads: RegList,
        write: Option<u8>,
        inline: Option<InlineTarget>,
        guard: Option<(u8, u64)>,
        generic: bool,
    },
}

impl OpKind {
    fn visit_reads(&self, mut visit: impl FnMut(u8)) {
        match self {
            OpKind::Move { src, .. } | OpKind::Return { src } => visit(*src),
            OpKind::Swap { a, b } => { visit(*a); visit(*b); }
            OpKind::JumpIf { cond, .. } => visit(*cond),
            OpKind::Numeric { lhs, rhs, .. } => std::iter::once(*lhs).chain(*rhs).for_each(visit),
            OpKind::Identity { lhs, rhs, .. } => { visit(*lhs); visit(*rhs); }
            OpKind::MapGet { object, .. } => visit(*object),
            OpKind::SelfCall { args, receiver, .. } =>
                args.iter().flatten().copied().chain(receiver.map(|(reg, _)| reg)).for_each(visit),
            OpKind::FieldGet { object, .. } => visit(*object),
            OpKind::FieldSet { object, value, .. } | OpKind::FieldStoreChecked { object, value, .. } => { visit(*object); visit(*value); }
            OpKind::Runtime { reads, .. } | OpKind::Alloc { reads, .. } => reads.iter().copied().for_each(visit),
            OpKind::NewObject { fields, args, .. } => fields.iter().map(|(_, reg)| *reg).chain(args.iter().copied()).for_each(visit),
            OpKind::Primitive { receiver, arg, .. } => std::iter::once(*receiver).chain(*arg).for_each(visit),
            _ => {}
        }
    }

    /// The single GPR the op defines. Swap and generic runtime ops define
    /// several; liveness treats them as reading (keeping live) instead.
    fn writes(&self) -> Option<u8> {
        match self {
            OpKind::Move { dst, .. } | OpKind::LoadConst { dst, .. } | OpKind::LoadRaw { dst, .. }
            | OpKind::LoadAbsolute { dst, .. }
            | OpKind::Numeric { dst, .. } | OpKind::Identity { dst, .. } | OpKind::MapGet { dst, .. }
            | OpKind::SelfCall { dst, .. } | OpKind::FieldGet { dst, .. }
            | OpKind::Alloc { dst, .. } | OpKind::Primitive { dst, .. } | OpKind::NewObject { dst, .. } => Some(*dst),
            OpKind::Runtime { write, .. } => *write,
            _ => None,
        }
    }
}

/// GPRs of a call-like op's arguments.
fn call_args(function: &PreparedFunction, prepared: &PreparedOp) -> Result<RegList, String> {
    (0..prepared.c as usize)
        .map(|argument| unsafe { wfjit_call_source_reg(function, prepared, argument) }
            .ok_or_else(|| "call argument register out of range".to_string()))
        .collect()
}

/// Call-like ops the v2 tier runs to completion through the interpreter's
/// generic call path (operands not modelled).
fn generic_call_opcode(opcode: Opcode) -> bool {
    matches!(opcode, Opcode::CallClosure | Opcode::InvokeSuper | Opcode::CallPrimitive
        | Opcode::InvokeSpread | Opcode::NewClassInstance)
}

/// Register operands of ops executed by their handler whose operand use is
/// known (checked against the interpreter handlers): (written GPR, read
/// GPRs). Anything else runs in the conservative generic mode.
fn precise_operands(opcode: Opcode, op: &PreparedOp) -> Option<(Option<u8>, RegList)> {
    use Opcode::*;
    Some(match opcode {
        ListGet | ListRemoveAt | MapGet | LoadProperty | IsInstance | StringCodePointAt | MakeRange
        | IAdd | ISub | IMul | IDivS | IRemS | IBitAnd | IBitOr | IBitXor | IShl | IShrU | ISar
        | FAdd | FSub | FMul | FDiv | FRem | ICmpEq | ICmpNe | ICmpLt | ICmpLe | ICmpGt | ICmpGe
        | FCmpEq | FCmpNe | FCmpLt | FCmpLe | FCmpGt | FCmpGe | RefEq => (Some(op.a), smallvec![op.b, op.c]),
        ListLen | LoadCell | AllocCell | AllocListCap
        | INeg | IBitNot | FNeg | NumToInt32 | NumToF64 | Int32ToNum | F64ToNum => (Some(op.a), smallvec![op.b]),
        StringConcat | StringSlice if op.x < 256 => (Some(op.a), smallvec![op.b, op.c, op.x as u8]),
        ListSet | MapSet | StoreProperty => (None, smallvec![op.a, op.b, op.c]),
        ListAppend | StoreCell => (None, smallvec![op.a, op.b]),
        AllocList => (Some(op.a), SmallVec::new()),
        _ => return None,
    })
}

/// `OpKind::NewObject` for an object-creating op whose layout the VM
/// described (see `WfFunctionData::object_layouts`).
fn native_new_object(data: &WfFunctionData<'_>, index: usize, opcode: Opcode, prepared: &PreparedOp) -> Option<OpKind> {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("NAPALM_WFJIT_NEW_OBJECT").ok().as_deref() != Some("0")) {
        return None;
    }
    let layout = data.object_layouts.get(&index)?;
    let operands = match opcode {
        Opcode::NewObjectEmpty | Opcode::NewObjectShape => RegList::new(),
        _ => call_args(data.function, prepared).ok()?,
    };
    if opcode == Opcode::NewClassInstance {
        let constructor = (prepared.cache != NO_INDEX).then_some(prepared.cache);
        // The object is written before the constructor reads its arguments.
        if constructor.is_some() && operands.contains(&prepared.a) {
            return None;
        }
        return Some(OpKind::NewObject { dst: prepared.a, kind: layout.kind, fields: SmallVec::new(), constructor, args: operands });
    }
    if layout.operand_slots.len() != operands.len() {
        return None;
    }
    let fields = layout.operand_slots.iter().copied().zip(operands).collect();
    Some(OpKind::NewObject { dst: prepared.a, kind: layout.kind, fields, constructor: None, args: RegList::new() })
}

/// Every GPR of the frame (the operand set of a generic runtime op).
fn all_registers(function: &PreparedFunction) -> RegList {
    let count = (function.frame.general_register_count as usize).min(256);
    (0..count).map(|reg| reg as u8).collect()
}

/// Classify every reachable op of `data`, or explain why the function is not
/// (yet) supported by the v2 tier. With `native_self_calls` the function's
/// calls to itself become native self-calls; otherwise ordinary calls.
fn classify_ops(
    data: &WfFunctionData<'_>,
    source: &WfSource<'_>,
    native_self_calls: bool,
) -> Result<Vec<Option<(OpKind, usize)>>, String> {
    let function = data.function;
    let count = function.ops.len();
    let mut kinds = vec![None; count];
    let generic = |helper| OpKind::Runtime {
        helper, reads: all_registers(function), write: None, inline: None, guard: None, generic: true,
    };
    // A call of a method to itself through `this` (same receiver class).
    let method_self_call = |index: usize, prepared: &PreparedOp, receiver: u8, class: u64| -> Result<OpKind, String> {
        if prepared.c as usize > function.arity as usize {
            return Err(format!("op {index}: self-call with extra arguments"));
        }
        let mut args = Vec::with_capacity(function.arity as usize);
        for argument in 0..function.arity as usize {
            args.push(if argument < prepared.c as usize {
                Some(unsafe { wfjit_call_source_reg(function, prepared, argument) }
                    .ok_or_else(|| format!("op {index}: unknown argument register"))?)
            } else { None });
        }
        Ok(OpKind::SelfCall { dst: prepared.a, args, receiver: Some((receiver, Some(class))) })
    };
    // Methods take the receiver as the first source of a CallDirect.
    let callee_uses_this = |callee: u32| if callee == data.function_index {
        function.uses_this
    } else {
        source.callees.get(&callee).is_some_and(|callee| callee.function.uses_this)
    };
    for index in 0..count {
        if !data.reachable[index] { continue; }
        let op = function.original_ops[index];
        let prepared = function.ops[index];
        let opcode = data.bytecode.chunk.code.get(function.raw_pcs[index] as usize)
            .and_then(|word| word.opcode())
            .ok_or_else(|| format!("invalid opcode at op {index}"))?;
        // A fused small Map literal consumes its property stores: the
        // runtime helper resumes after them (see `reachable_ops`).
        let fused_map = opcode == Opcode::AllocMap && prepared.c > 0 && prepared.c <= 4
            && prepared.next > index as u32 + 1;
        let next = if fused_map { prepared.next as usize } else { op.next as usize };
        let new_object = native_new_object(data, index, opcode, &prepared);
        let kind = match opcode {
            Opcode::NewObjectEmpty | Opcode::NewObjectShape | Opcode::NewObjectWithProperties
            | Opcode::NewInterfaceOrTypeObject | Opcode::NewClassInstance if new_object.is_some() =>
                new_object.expect("checked above"),
            Opcode::Nop => OpKind::Nop,
            Opcode::Move => OpKind::Move { dst: op.a, src: op.b },
            Opcode::Swap => OpKind::Swap { a: op.a, b: op.b },
            Opcode::LoadI8 => OpKind::LoadRaw { dst: op.a, word: op.b as i8 as i32 as u32 as u64 },
            Opcode::LoadI32 => OpKind::LoadRaw { dst: op.a, word: op.x as u32 as u64 },
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
                } else if let Some((class, kind)) = native_primitive_site(function, index, source) {
                    let arg = if kind.arity() == 1 { Some(call_args(function, &prepared)?[0]) } else { None };
                    OpKind::Primitive { dst: op.a, receiver: op.b, arg, class, kind }
                } else if let Some((class, target)) = profiled_direct_invoke_site(function, index)
                    .map(|(class, target)| (class.0, target))
                    // Never quickened (e.g. run by a trace first): the runtime
                    // inline cache is the profile.
                    .or_else(|| data.cached_invokes.get(&index).copied())
                {
                    if native_self_calls && target == data.function_index {
                        method_self_call(index, &prepared, op.b, class)?
                    } else {
                        // Monomorphic profiled call: inline it behind a receiver
                        // class guard, or call the target behind the same guard.
                        let args = call_args(function, &prepared)?;
                        let reads = std::iter::once(op.b).chain(args.iter().copied()).collect();
                        OpKind::Runtime {
                            helper: WfRuntimeHelper::ProfiledMethod(target), reads, write: Some(op.a),
                            inline: Some(InlineTarget { callee: target, receiver: Some(op.b), args, class_guard: Some(class) }),
                            guard: Some((op.b, class)), generic: false,
                        }
                    }
                } else {
                    // Polymorphic or unprofiled: the interpreter dispatches it.
                    let args = call_args(function, &prepared)?;
                    let reads = std::iter::once(op.b).chain(args.iter().copied()).collect();
                    OpKind::Runtime {
                        helper: WfRuntimeHelper::CallToReturn, reads, write: Some(op.a),
                        inline: None, guard: None, generic: false,
                    }
                }
            }
            Opcode::CallDirect if native_self_calls && prepared.x == data.function_index => {
                // A method receives `this` as the first source (statically
                // bound: no class guard needed).
                let first = function.uses_this as usize;
                let passed = (prepared.c as usize).checked_sub(first)
                    .ok_or_else(|| format!("op {index}: method self-call without receiver"))?;
                if passed > function.arity as usize {
                    return Err(format!("op {index}: self-call with extra arguments"));
                }
                let source = |position: usize| unsafe { wfjit_call_source_reg(function, &prepared, position) }
                    .ok_or_else(|| format!("op {index}: unknown argument register"));
                let mut args = Vec::with_capacity(function.arity as usize);
                for argument in 0..function.arity as usize {
                    args.push(if argument < passed { Some(source(first + argument)?) } else { None });
                }
                let receiver = if function.uses_this { Some((source(0)?, None)) } else { None };
                OpKind::SelfCall { dst: op.a, args, receiver }
            }
            Opcode::GetFieldClass => OpKind::FieldGet {
                dst: prepared.a, object: prepared.b, slot: prepared.x,
                nullable: prepared.y & crate::napalm_bytecode::FIELD_CLASS_NULLABLE != 0,
            },
            Opcode::SetFieldClass => match data.plain_store_classes.get(&index) {
                Some(&class_key) => OpKind::FieldSet {
                    object: prepared.a, value: prepared.b, slot: prepared.x, class_key,
                },
                // Contracted/generic store: validated (and barriered) by a leaf
                // helper; with reactivity on, the op itself runs so it notifies.
                None if !source.reactive => OpKind::FieldStoreChecked { object: prepared.a, value: prepared.b, slot: prepared.x },
                None => OpKind::Runtime { helper: WfRuntimeHelper::Op, reads: smallvec![prepared.a, prepared.b], write: None, inline: None, guard: None, generic: false },
            },
            Opcode::SetFieldShape => OpKind::Runtime { helper: WfRuntimeHelper::Op, reads: smallvec![prepared.a, prepared.b, prepared.c], write: None, inline: None, guard: None, generic: false },
            Opcode::GetFieldShape => OpKind::Runtime { helper: WfRuntimeHelper::Op, reads: smallvec![prepared.b], write: Some(prepared.a), inline: None, guard: None, generic: false },
            Opcode::StoreModuleVar => OpKind::Runtime { helper: WfRuntimeHelper::Op, reads: smallvec![prepared.a], write: None, inline: None, guard: None, generic: false },
            Opcode::InvokeMethodDirect => {
                let args = call_args(function, &prepared)?;
                let reads: RegList = std::iter::once(op.b).chain(args.iter().copied()).collect();
                // The prepared op names the bytecode target; it is only valid for
                // the descriptor's owner class while that class still binds it.
                let class_key = match data.bytecode.chunk.specializations.get(prepared.x as usize) {
                    Some(crate::napalm_bytecode::SpecializedDescriptor::DirectMethod { owner, symbol, .. }) => {
                        source.class_slots.get(owner).and_then(|&slot| {
                            let key = ReceiverClass(CLASS_KIND_INSTANCE | slot as u64);
                            (source.method_targets.get(&(key, *symbol)) == Some(&CallTarget::Bytecode(prepared.y)))
                                .then_some(key.0)
                        })
                    }
                    _ => None,
                };
                match class_key {
                    Some(class) if native_self_calls && prepared.y == data.function_index =>
                        method_self_call(index, &prepared, op.b, class)?,
                    Some(class) => OpKind::Runtime {
                        helper: WfRuntimeHelper::DirectMethod, reads, write: Some(op.a),
                        inline: Some(InlineTarget { callee: prepared.y, receiver: Some(op.b), args, class_guard: Some(class) }),
                        guard: Some((op.b, class)), generic: false,
                    },
                    // Unresolved owner: the op's own dispatch decides.
                    None => OpKind::Runtime {
                        helper: WfRuntimeHelper::CallToReturn, reads, write: Some(op.a),
                        inline: None, guard: None, generic: false,
                    },
                }
            }
            Opcode::CallDirect => {
                let args = call_args(function, &prepared)?;
                OpKind::Runtime {
                    helper: WfRuntimeHelper::DirectCall, reads: args.clone(), write: Some(op.a),
                    inline: Some(if callee_uses_this(prepared.x) && !args.is_empty() {
                        InlineTarget { callee: prepared.x, receiver: Some(args[0]), args: SmallVec::from_slice(&args[1..]), class_guard: None }
                    } else {
                        InlineTarget { callee: prepared.x, receiver: None, args, class_guard: None }
                    }),
                    guard: None, generic: false,
                }
            }
            Opcode::AllocMap if fused_map =>
                OpKind::Alloc { dst: op.a, kind: WfAllocKind::SmallMapLiteral, reads: all_registers(function) },
            // Only the profiled inline path works from a nested frame's
            // shadow bank; otherwise the object factory needs the top frame,
            // which native recursion cannot guarantee.
            Opcode::NewObjectEmpty | Opcode::NewObjectShape
            | Opcode::NewObjectWithProperties | Opcode::NewInterfaceOrTypeObject
                if !native_self_calls || native_profiled_small_map_site(function, data.bytecode, index).is_some() =>
                OpKind::Alloc { dst: op.a, kind: WfAllocKind::SpecializedObject, reads: all_registers(function) },
            Opcode::DefineArrayWithGenerics if !native_self_calls =>
                OpKind::Alloc { dst: op.a, kind: WfAllocKind::TypedArray, reads: all_registers(function) },
            other if generic_call_opcode(other) => generic(WfRuntimeHelper::CallToReturn),
            // A closure captures register values and cells of this frame (the
            // op's `call_args` list) and allocates: its handler reads exactly
            // those registers from the bank and writes the closure to `a`.
            Opcode::MakeClosure => {
                let count = (op.b as usize) | ((op.c as usize) << 8);
                let start = op.args_offset as usize;
                let captures: Option<RegList> = (0..count)
                    .map(|index| function.call_args.get(start + index)
                        .and_then(|reg| u8::try_from(reg.0).ok()))
                    .collect();
                match captures {
                    Some(reads) => OpKind::Runtime {
                        helper: WfRuntimeHelper::Op, reads, write: Some(op.a),
                        inline: None, guard: None, generic: false,
                    },
                    None => generic(WfRuntimeHelper::Op),
                }
            }
            // `throw`: the handler unwinds; the helper reports no successor and
            // the body bails out (or a handler of this frame took over).
            Opcode::FiberAbort => generic(WfRuntimeHelper::Op),
            other if precise_operands(other, &op).is_some() => {
                let (write, reads) = precise_operands(other, &op).unwrap();
                OpKind::Runtime { helper: WfRuntimeHelper::Op, reads, write, inline: None, guard: None, generic: false }
            }
            // Anything else the interpreter can execute as one op against the
            // top frame (allocation, collections, strings, typed registers,
            // cells, properties...) runs through its handler.
            other if eligible_opcode(other)
                && !matches!(other, Opcode::Switch | Opcode::Return2 | Opcode::Launch
                    | Opcode::ImportModule | Opcode::ImportVariable | Opcode::Trap) => {
                if std::env::var_os("NAPALM_WFJIT_GENERIC").is_some() {
                    eprintln!("[napalm-wfjit] generic {other:?}");
                }
                generic(WfRuntimeHelper::Op)
            }
            other => return Err(format!("op {index}: {other:?} is not supported by WFJIT v2")),
        };
        if matches!(kind, OpKind::Jump { .. } | OpKind::JumpIf { .. }) {
            let target = match kind { OpKind::Jump { target } | OpKind::JumpIf { target, .. } => target, _ => 0 };
            if target >= count {
                return Err(format!("op {index}: jump out of the function"));
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

/// Backward liveness over the op graph (iterated to a fixpoint when loops
/// exist): GPRs live on entry to each op.
fn liveness(kinds: &[Option<(OpKind, usize)>]) -> Vec<RegSet> {
    let mut live_in = vec![[0u64; 4]; kinds.len()];
    loop {
        let mut changed = false;
        for index in (0..kinds.len()).rev() {
            let Some((kind, next)) = &kinds[index] else { continue; };
            let mut live = [0u64; 4];
            for successor in successors(kind, *next) {
                for (word, other) in live.iter_mut().zip(live_in[successor]) { *word |= other; }
            }
            if let Some(dst) = kind.writes() { set_remove(&mut live, dst); }
            kind.visit_reads(|reg| set_add(&mut live, reg));
            if live != live_in[index] { live_in[index] = live; changed = true; }
        }
        if !changed { return live_in; }
    }
}

/// Live GPR sets used for translation. With exception handlers every GPR is
/// live everywhere: handler blocks are reached by exception edges liveness
/// does not see, from any helper or from the op a deopt re-executes.
fn live_sets(function: &PreparedFunction, kinds: &[Option<(OpKind, usize)>]) -> Vec<RegSet> {
    if function.exception_handlers.is_empty() {
        return liveness(kinds);
    }
    let mut all = [0u64; 4];
    for reg in all_registers(function) { set_add(&mut all, reg); }
    vec![all; kinds.len()]
}

/// Basic blocks of the op graph in reverse postorder.
struct BlockOrder {
    /// (first op, last op) per block, in reverse postorder from op 0.
    blocks: Vec<(usize, usize)>,
    /// RPO position of the block starting at each op (`usize::MAX` if none).
    position: Vec<usize>,
    /// Loop headers: block starts targeted by an edge from a block at the
    /// same or a later RPO position (back edges, including rotated loops
    /// whose body is entered through a condition placed below it).
    header: Vec<bool>,
}

fn block_order(kinds: &[Option<(OpKind, usize)>]) -> Result<BlockOrder, String> {
    let count = kinds.len();
    let terminates = |kind: &OpKind| matches!(kind, OpKind::Jump { .. } | OpKind::JumpIf { .. } | OpKind::Return { .. });
    let mut leader = vec![false; count];
    if count > 0 { leader[0] = true; }
    for (index, entry) in kinds.iter().enumerate() {
        let Some((kind, next)) = entry else { continue; };
        if terminates(kind) || *next != index + 1 {
            for successor in successors(kind, *next) { leader[successor] = true; }
        }
    }
    // Block extents and successors.
    let mut end_of = vec![usize::MAX; count];
    for start in 0..count {
        if !leader[start] || kinds[start].is_none() { continue; }
        let mut end = start;
        loop {
            let (kind, next) = kinds[end].as_ref().unwrap();
            if terminates(kind) || *next != end + 1 || leader[*next] { break; }
            if kinds[*next].is_none() { return Err(format!("op {end} falls into an unreachable op")); }
            end = *next;
        }
        end_of[start] = end;
    }
    let block_successors = |start: usize| -> smallvec::SmallVec<[usize; 2]> {
        let (kind, next) = kinds[end_of[start]].as_ref().unwrap();
        successors(kind, *next)
    };
    // Iterative DFS postorder from op 0.
    let mut visited = vec![false; count];
    let mut postorder: Vec<usize> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    if count > 0 { visited[0] = true; }
    while let Some(top) = stack.len().checked_sub(1) {
        let (start, child) = stack[top];
        let succ = block_successors(start);
        if child < succ.len() {
            let target = succ[child];
            stack[top].1 += 1;
            if end_of[target] == usize::MAX { return Err(format!("edge to unclassified op {target}")); }
            if !visited[target] { visited[target] = true; stack.push((target, 0)); }
        } else {
            postorder.push(start);
            stack.pop();
        }
    }
    postorder.reverse();
    let blocks: Vec<(usize, usize)> = postorder.into_iter().map(|start| (start, end_of[start])).collect();
    let mut position = vec![usize::MAX; count];
    for (rpo, &(start, _)) in blocks.iter().enumerate() { position[start] = rpo; }
    let mut header = vec![false; count];
    for (rpo, &(start, _)) in blocks.iter().enumerate() {
        for target in block_successors(start) {
            if position[target] <= rpo { header[target] = true; }
        }
    }
    Ok(BlockOrder { blocks, position, header })
}

/// An edge whose target block is not built yet: it carries the full GPR
/// state at the branch; its arguments are filled when the target is built.
struct PendingEdge {
    from: WfBlock,
    /// 0 = Jump / Branch.then, 1 = Branch.otherwise.
    slot: u8,
    state: Vec<Option<WfValue>>,
    /// A helper ran since the method epoch was last checked on this path.
    epoch_dirty: bool,
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
    /// Optimistic type of native self-call results (F64 or Bool).
    assume_self_call: Option<WfTy>,
    /// Returns of this type stay raw; `raw_ok` turns false at any other.
    raw_return: Option<WfTy>,
    raw_ok: bool,
    /// The root contains native self-calls: inlined bodies must then be free
    /// of effects (an outer replay would repeat them).
    root_self_calls: bool,
    /// Innermost inlined body being translated, if any.
    inline: Option<InlineFrame>,
    inline_depth: u32,
    inlined_ops: usize,
    /// Functions being translated (root first): no recursive inlining.
    inline_stack: SmallVec<[u32; 4]>,
    /// Per active inlined call (outermost first): callee, call op in the
    /// parent, and the parent's GPR state and live set at that call.
    inline_chain: SmallVec<[(u32, usize, Vec<Option<WfValue>>, RegSet); 3]>,
    /// Loop-header registers forced to Tagged (a back edge disagreed with
    /// the type seen on entry). Input of this build.
    loop_tagged: &'a crate::napalm_collections::SmallSet16<(usize, u8)>,
    /// Registers this build found disagreeing: rebuild with them Tagged.
    loop_retry: Vec<(usize, u8)>,
    /// A host helper ran (it may have rebound methods) since the method
    /// epoch was last checked on the current path; epoch-dependent ops check
    /// it again first.
    epoch_dirty: bool,
    /// Dispatched calls (`WfCallTarget::Dynamic`) emitted in inlined bodies.
    dispatched_calls: u32,
    /// `NewObject` results of this build. Until one escapes, a store into it
    /// is invisible (an op replay simply allocates a new object), so it is no
    /// effect for the inliner's replay rule.
    fresh_objects: Vec<WfValue>,
}

/// `build_with` failure that asks for a rebuild with `loop_retry` applied.
const LOOP_RETYPE: &str = "loop header types changed";

/// Inlining only appends to the open block and creates new IR data. Keep
/// lengths rather than cloning all previously compiled blocks and snapshots.
struct InlineCheckpoint {
    block: WfBlock,
    insts: usize,
    blocks: usize,
    values: usize,
    deopts: usize,
    state: Vec<Option<WfValue>>,
    inlined_ops: usize,
    inlined_calls: u32,
    epoch_dirty: bool,
    dispatched_calls: u32,
    effect_done: bool,
    fresh_objects: usize,
}

impl InlineCheckpoint {
    fn save(builder: &Builder<'_>) -> Self {
        let block = builder.current.expect("inline call outside a block");
        debug_assert!(builder.out.blocks[block.0 as usize].terminator.is_none());
        Self {
            block,
            insts: builder.out.blocks[block.0 as usize].insts.len(),
            blocks: builder.out.blocks.len(),
            values: builder.out.value_types.len(),
            deopts: builder.out.deopts.len(),
            state: builder.state.clone(),
            inlined_ops: builder.inlined_ops,
            inlined_calls: builder.out.inlined_calls,
            epoch_dirty: builder.epoch_dirty,
            dispatched_calls: builder.dispatched_calls,
            effect_done: builder.inline.as_ref().is_some_and(|frame| frame.effect_done),
            fresh_objects: builder.fresh_objects.len(),
        }
    }

    fn restore(self, builder: &mut Builder<'_>) {
        builder.out.blocks.truncate(self.blocks);
        let block = &mut builder.out.blocks[self.block.0 as usize];
        block.insts.truncate(self.insts);
        block.terminator = None;
        builder.out.value_types.truncate(self.values);
        builder.out.deopts.truncate(self.deopts);
        builder.out.inlined_calls = self.inlined_calls;
        builder.state = self.state;
        builder.current = Some(self.block);
        builder.inlined_ops = self.inlined_ops;
        builder.epoch_dirty = self.epoch_dirty;
        builder.dispatched_calls = self.dispatched_calls;
        if let Some(frame) = &mut builder.inline { frame.effect_done = self.effect_done; }
        builder.fresh_objects.truncate(self.fresh_objects);
    }
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
        let registers_of = |state: &[Option<WfValue>], live: &RegSet| -> Vec<(u8, WfValue)> {
            (0..=255u8)
                .filter(|&reg| set_has(live, reg))
                .filter_map(|reg| state[reg as usize].map(|value| (reg, value)))
                .collect()
        };
        if let Some(frame) = &self.inline {
            if !frame.effect_done {
                return Ok(frame.deopt);
            }
            // After an effect the call cannot be replayed: rebuild the
            // inlined frames and resume at this very op of the innermost one.
            return self.chain_state(op_index, speculative, live);
        }
        let registers = registers_of(&self.state, live);
        self.out.deopts.push(WfDeoptState { op_index: op_index as u32, speculative, registers, frames: Vec::new() });
        Ok(WfDeopt(self.out.deopts.len() as u32 - 1))
    }

    /// Inside an inlined body: the root's state at its outermost call op plus
    /// every inlined frame (outermost first) stopped at its current op, the
    /// innermost at `op_index` with `live`. Used to rebuild the frames on a
    /// deopt and to publish them as GC shadow frames around `CallExplicit`.
    fn chain_state(&mut self, op_index: usize, speculative: bool, live: &RegSet) -> Result<WfDeopt, String> {
        let registers_of = |state: &[Option<WfValue>], live: &RegSet| -> Vec<(u8, WfValue)> {
            (0..=255u8)
                .filter(|&reg| set_has(live, reg))
                .filter_map(|reg| state[reg as usize].map(|value| (reg, value)))
                .collect()
        };
        let chain = &self.inline_chain;
        let mut frames = Vec::with_capacity(chain.len());
        for (level, (callee, call_op, _, _)) in chain.iter().enumerate() {
            // The interpreter cannot re-enter a constructor frame through its
            // `new C()` op (that op allocates): such a body must stay replayable.
            let parent = if level == 0 { Some(self.source.root) } else { self.source.callees.get(&chain[level - 1].0) };
            let parent_opcode = parent.and_then(|parent| {
                let raw_pc = *parent.function.raw_pcs.get(*call_op)?;
                parent.bytecode.chunk.code.get(raw_pc as usize)?.opcode()
            });
            if parent_opcode == Some(Opcode::NewClassInstance) {
                return Err(format!("op {op_index}: an inlined constructor frame cannot be rebuilt"));
            }
            let (resume_op, registers) = match chain.get(level + 1) {
                Some((_, inner_call, parent_state, parent_live)) =>
                    (*inner_call as u32, registers_of(parent_state, parent_live)),
                None => (op_index as u32, registers_of(&self.state, live)),
            };
            let data = self.source.callees.get(callee)
                .ok_or_else(|| format!("op {op_index}: inlined callee without data"))?;
            let reg_count = data.function.frame.general_register_count;
            let root_map = *data.function.gc_map_by_op.get(resume_op as usize)
                .ok_or_else(|| format!("op {op_index}: no GC map at inlined op {resume_op}"))?;
            let gc_refs = data.function.gc_maps.get(root_map as usize).map_or_else(Vec::new, |map| {
                (0..reg_count.min(256))
                    .filter(|&reg| map.general_is_reference(Reg(reg as u8)))
                    .map(|reg| reg as u8)
                    .collect()
            });
            frames.push(WfInlineDeoptFrame {
                callee: *callee, call_op: *call_op as u32, resume_op, registers, reg_count, root_map, gc_refs,
            });
        }
        let (_, root_call, root_state, root_live) = &chain[0];
        let registers = registers_of(root_state, root_live);
        self.out.deopts.push(WfDeoptState { op_index: *root_call as u32, speculative, registers, frames });
        Ok(WfDeopt(self.out.deopts.len() as u32 - 1))
    }

    /// Before an op whose speculation depends on the method epoch: re-check
    /// it if a helper ran since the last check on this path.
    fn ensure_epoch(&mut self, index: usize, live: &RegSet) -> Result<(), String> {
        if self.epoch_dirty {
            if let Some((expected, _)) = self.out.entry_epoch_guard {
                let deopt = self.deopt(index, true, live)?;
                self.emit(WfTy::Bool, WfInst::GuardEpoch { expected, deopt });
            }
            self.epoch_dirty = false;
        }
        Ok(())
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
        // A Boolean is never a Number: box it so the guard below deopts and
        // the interpreter reports the error (or dispatches an operator).
        let value = if self.out.ty(value) == WfTy::Bool { self.tagged(value)? } else { value };
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
        if self.inline_stack.contains(&target.callee) {
            return Err(format!("op {index}: recursive inlining"));
        }
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
        let kinds = classify_ops(data, source, false)?;
        if block_order(&kinds)?.header.iter().any(|&header| header) {
            return Err(format!("op {index}: callee contains a loop"));
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
        let mut args = SmallVec::<[WfValue; 4]>::with_capacity(target.args.len());
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
        self.inline_stack.push(target.callee);
        self.inline_chain.push((target.callee, index, caller_state, *live));
        let translated = translate(self, data, &kinds);
        let (_, _, caller_state, _) = self.inline_chain.pop().expect("inline chain");
        self.inline_stack.pop();
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

/// Build and optimize the v2 IR, or explain why the function is not
/// supported. `NAPALM_WFJIT_OPT=0` skips the optimization passes.
pub(super) fn build(source: &WfSource<'_>) -> Result<WfFunction, String> {
    let mut ir = build_unoptimized(source)?;
    static OPTIMIZE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OPTIMIZE.get_or_init(|| std::env::var("NAPALM_WFJIT_OPT").ok().as_deref() != Some("0")) {
        wf_optimizing::optimize(&mut ir).map_err(|error| format!("WF IR optimization: {error}"))?;
    }
    // Everything code generation reads from the VM is captured now, on the
    // VM thread: the lowering itself may run on the background compiler.
    let root = source.root;
    static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DUMP.get_or_init(|| std::env::var_os("NAPALM_WFJIT_DUMP").is_some()) {
        eprintln!("[napalm-wfjit] IR dump f={}:", root.function_index);
        for (index, block) in ir.blocks.iter().enumerate() {
            eprintln!("  block{index}:");
            for (value, inst) in &block.insts {
                eprintln!("    v{} = {inst:?}", value.0);
            }
            eprintln!("    {:?}", block.terminator);
        }
        for (index, deopt) in ir.deopts.iter().enumerate() {
            eprintln!("  deopt{index}: op {} spec={} regs={:?} frames={:?}",
                deopt.op_index, deopt.speculative, deopt.registers,
                deopt.frames.iter().map(|frame| (frame.callee, frame.call_op, frame.resume_op, &frame.registers)).collect::<Vec<_>>());
        }
    }
    ir.frame = WfFrameInfo::of(root.function);
    for block in &ir.blocks {
        for (_, inst) in &block.insts {
            if let WfInst::AllocObject { op_index, kind, function, .. } = inst {
                let index = *op_index as usize;
                let data = if *function == root.function_index { root } else {
                    source.callees.get(function).ok_or("allocation in a callee without data")?
                };
                let site = match kind {
                    WfAllocKind::TypedArray => WfAllocSite::TypedArray(typed_array_site(data.function, data.bytecode, index)),
                    _ => WfAllocSite::Object(native_profiled_small_map_site(data.function, data.bytecode, index)),
                };
                ir.alloc_sites.push(((*function, *op_index), site));
            }
        }
    }
    Ok(ir)
}

fn build_unoptimized(source: &WfSource<'_>) -> Result<WfFunction, String> {
    let function = source.root.function;
    let arity = function.arity as usize;
    if !source.speculate || source.failed_sites.contains(&u32::MAX) {
        // No native self-call specialization: its parameter types are a guess.
        let mut kinds = classify_ops(source.root, source, false)?;
        strip_failed_sites(source, &mut kinds);
        return build_retyped(source, &kinds, None, None, &vec![WfTy::Tagged; arity]);
    }
    // Native self-calls run nested frames without a register bank: they need
    // a replay-safe body (no stores, no host helpers, no captures read from
    // the bank). Otherwise calls to itself go through the interpreter's call
    // path, which gives every level a real frame.
    let mut native = classify_ops(source.root, source, true)?;
    strip_failed_sites(source, &mut native);
    let self_calls = native.iter().flatten().any(|(kind, _)| matches!(kind, OpKind::SelfCall { .. }));
    if self_calls && function.capture_count == 0 {
        match build_specialized(source, &native) {
            Ok(built) => return Ok(built),
            Err(reason) => {
                if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
                    eprintln!("[napalm-wfjit] v2 native self-calls declined for {:?}: {reason}", function.id);
                }
            }
        }
    }
    let kinds = if self_calls {
        let mut kinds = classify_ops(source.root, source, false)?;
        strip_failed_sites(source, &mut kinds);
        kinds
    } else { native };
    build_retyped(source, &kinds, None, None, &vec![WfTy::Tagged; arity])
}

/// Turn the root ops named in `failed_sites` (all of them without
/// speculation) into their non-speculative form.
fn strip_failed_sites(source: &WfSource<'_>, kinds: &mut [Option<(OpKind, usize)>]) {
    let function = source.root.function;
    for (index, entry) in kinds.iter_mut().enumerate() {
        if let Some((kind, _)) = entry {
            if !source.speculate || source.failed_sites.contains(&(index as u32)) {
                strip_speculation(function, kind);
            }
        }
    }
}

/// Non-speculative form of one op: when its native form relies on a profile
/// (numeric/identity/PIC/primitive dispatch, exact-class field access,
/// guarded or inlined calls, native self-calls) it runs through its host
/// helper instead, so no speculative guard is left for it. Register traffic,
/// control flow and allocation stay native.
fn strip_speculation(function: &PreparedFunction, kind: &mut OpKind) {
    let replacement = match kind {
        OpKind::Numeric { .. } | OpKind::Identity { .. } | OpKind::MapGet { .. }
        | OpKind::Primitive { .. } | OpKind::SelfCall { .. }
        | OpKind::Runtime { guard: Some(_), .. } => OpKind::Runtime {
            helper: WfRuntimeHelper::CallToReturn, reads: all_registers(function), write: None,
            inline: None, guard: None, generic: true,
        },
        OpKind::FieldGet { dst, object, .. } => OpKind::Runtime {
            helper: WfRuntimeHelper::Op, reads: smallvec![*object], write: Some(*dst),
            inline: None, guard: None, generic: false,
        },
        OpKind::FieldSet { object, value, .. } => OpKind::Runtime {
            helper: WfRuntimeHelper::Op, reads: smallvec![*object, *value], write: None,
            inline: None, guard: None, generic: false,
        },
        OpKind::Runtime { inline, .. } => { *inline = None; return; }
        _ => return,
    };
    *kind = replacement;
}

/// Native self-call specialization. Optimistically type self-call results as
/// Numbers (then Booleans) and verify that every Return returned that raw
/// type: sound by induction over completed native executions, because a
/// callee whose guard fails unwinds the whole chain and never returns a value.
/// A parameter becomes F64 when every self-call passes an unboxed Number for
/// it; iterate to a fixpoint (the arguments depend on the parameter types).
fn build_specialized(source: &WfSource<'_>, kinds: &[Option<(OpKind, usize)>]) -> Result<WfFunction, String> {
    let arity = source.root.function.arity as usize;
    let mut param_types = vec![WfTy::Tagged; arity];
    let mut best = None;
    for _ in 0..4 {
        let mut built = None;
        for raw in [WfTy::F64, WfTy::Bool] {
            let candidate = build_retyped(source, kinds, Some(raw), Some(raw), &param_types)?;
            if candidate.return_ty == raw { built = Some(candidate); break; }
        }
        let built = match built {
            Some(built) => built,
            None => build_retyped(source, kinds, None, None, &param_types)?,
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

/// `build_with` until the loop-header types reach a fixpoint.
fn build_retyped(
    source: &WfSource<'_>,
    kinds: &[Option<(OpKind, usize)>],
    assume_self_call: Option<WfTy>,
    raw_return: Option<WfTy>,
    param_types: &[WfTy],
) -> Result<WfFunction, String> {
    let mut loop_tagged = crate::napalm_collections::SmallSet16::default();
    for _ in 0..64 {
        match build_with(source, kinds, assume_self_call, raw_return, param_types, &loop_tagged) {
            Ok(built) => return Ok(built),
            Err(BuildError::Retype(registers)) => loop_tagged.extend(registers),
            Err(BuildError::Unsupported(reason)) => return Err(reason),
        }
    }
    Err("loop header types do not converge".into())
}

enum BuildError {
    Retype(Vec<(usize, u8)>),
    Unsupported(String),
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
    assume_self_call: Option<WfTy>,
    raw_return: Option<WfTy>,
    param_types: &[WfTy],
    loop_tagged: &crate::napalm_collections::SmallSet16<(usize, u8)>,
) -> Result<WfFunction, BuildError> {
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
            return_ty: WfTy::Tagged,
            check_resume_pc: !function.exception_handlers.is_empty(),
            nested_banks: root_self_calls
                && kinds.iter().flatten().any(|(kind, _)| matches!(kind, OpKind::Alloc { .. })),
            param_types: param_types.to_vec(),
            inlined_calls: 0,
            frame: WfFrameInfo::default(),
            alloc_sites: Vec::new(),
        },
        state: vec![None; 256],
        current: None,
        assume_self_call,
        raw_return,
        raw_ok: true,
        root_self_calls,
        inline: None,
        inline_depth: 0,
        inlined_ops: 0,
        inline_stack: smallvec![root.function_index],
        inline_chain: SmallVec::new(),
        loop_tagged,
        loop_retry: Vec::new(),
        epoch_dirty: false,
        dispatched_calls: 0,
        fresh_objects: Vec::new(),
    };
    let unsupported = BuildError::Unsupported;

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
    // Closure captures are copied into their registers by frame entry; a
    // closure body never runs as a nested native frame, so its bank exists.
    for binding in &function.frame.captures {
        let reg = binding.destination.index();
        if reg < 256 {
            let value = builder.emit(WfTy::Tagged, WfInst::FrameLoad(reg as u8));
            builder.state[reg] = Some(value);
        }
    }
    // Speculative dispatch (numeric, identity, PICs, inlined methods and the
    // ops of inlined bodies) is valid for one method epoch, guarded on entry.
    let needs_epoch = kinds.iter().flatten().any(|(kind, _)| matches!(kind,
        OpKind::Numeric { .. } | OpKind::Identity { .. } | OpKind::MapGet { .. }
        | OpKind::Runtime { inline: Some(_), .. } | OpKind::Runtime { guard: Some(_), .. }
        | OpKind::SelfCall { receiver: Some((_, Some(_))), .. } | OpKind::Primitive { .. }));
    if needs_epoch {
        let live_in = live_sets(function, kinds);
        let deopt = builder.deopt(0, true, &live_in[0]).map_err(unsupported)?;
        builder.out.entry_epoch_guard = Some((source.expected_method_epoch, deopt));
    }

    if let Err(reason) = translate(&mut builder, root, kinds) {
        if reason == LOOP_RETYPE {
            return Err(BuildError::Retype(std::mem::take(&mut builder.loop_retry)));
        }
        return Err(unsupported(reason));
    }
    let unsupported = |reason: &str| BuildError::Unsupported(reason.to_string());

    // Every terminator must point at a real block now.
    for data in &builder.out.blocks {
        let targets: Vec<WfBlock> = match &data.terminator {
            Some(WfTerminator::Jump(edge)) => vec![edge.target],
            Some(WfTerminator::Branch { then, otherwise, .. }) => vec![then.target, otherwise.target],
            Some(WfTerminator::Return(_)) => Vec::new(),
            None => return Err(unsupported("block without terminator")),
        };
        if targets.iter().any(|target| target.0 as usize >= builder.out.blocks.len()) {
            return Err(unsupported("edge to an unbuilt block"));
        }
    }
    let return_ty = match builder.raw_return { Some(raw) if builder.raw_ok => raw, _ => WfTy::Tagged };
    let mut out = builder.out;
    out.return_ty = return_ty;
    Ok(out)
}

/// Header block of a loop and its parameters (register, type).
struct LoopHeader {
    block: WfBlock,
    params: Vec<(u8, WfTy)>,
}

/// Wire a back edge `from`/`slot` → loop header `header` with `state`. The
/// edge gets its own block, which counts the iteration (and may hand the
/// loop to the tracer) before jumping to the header with the loop-carried
/// values.
fn back_edge(
    builder: &mut Builder<'_>,
    headers: &crate::napalm_collections::SmallMap16<usize, LoopHeader>,
    from: WfBlock,
    slot: u8,
    header: usize,
    state: Vec<Option<WfValue>>,
    live_header: &RegSet,
) -> Result<(), String> {
    let Some(target) = headers.get(&header) else {
        return Err(format!("back edge to unbuilt loop header {header}"));
    };
    let edge = builder.new_block();
    builder.set_edge(from, slot, edge, Vec::new());
    let saved = std::mem::replace(&mut builder.state, state);
    builder.current = Some(edge);
    let deopt = builder.deopt(header, false, live_header)?;
    let counter = *builder.source.loop_counters.borrow_mut().entry(header)
        .or_insert_with(|| Box::leak(Box::new(0u32)) as *mut u32 as usize);
    builder.emit(WfTy::Bool, WfInst::LoopBackedge {
        counter,
        trace_slot: builder.source.trace_headers + header * std::mem::size_of::<u32>(),
        shared_flag: builder.source.shared_flag,
        deopt,
    });
    let mut args = Vec::with_capacity(target.params.len());
    for &(reg, ty) in &target.params {
        let Some(value) = builder.state[reg as usize] else {
            return Err(format!("loop-carried r{reg} undefined on the back edge to {header}"));
        };
        match builder.convert_in(edge, value, ty) {
            Ok(value) => args.push(value),
            Err(_) => {
                builder.loop_retry.push((header, reg));
                return Err(LOOP_RETYPE.into());
            }
        }
    }
    builder.terminate(WfTerminator::Jump(WfEdge { target: target.block, args }));
    builder.state = saved;
    Ok(())
}

/// Translate one function body (the root, or an inlined callee when
/// `builder.inline` is set) starting in `builder.current` with `builder.state`.
fn translate(builder: &mut Builder<'_>, data: &WfFunctionData<'_>, kinds: &[Option<(OpKind, usize)>]) -> Result<(), String> {
    let count = data.function.ops.len();
    let live_in = live_sets(data.function, kinds);
    let order = block_order(kinds)?;
    let is_header = &order.header;
    let mut headers: crate::napalm_collections::SmallMap16<usize, LoopHeader> = Default::default();

    let mut pending: Vec<Vec<PendingEdge>> = (0..count).map(|_| Vec::new()).collect();
    // Placeholder targets until the real block exists (patched by set_edge).
    let placeholder = WfBlock(u32::MAX);
    // Basic blocks are translated in reverse postorder, so every block but a
    // loop header has all its predecessors built. Send an edge from a
    // just-terminated block: forward edges wait in `pending`, back edges (to
    // a block at the same or an earlier RPO position) are wired to their
    // already built header.
    let mut rpo = 0usize;
    macro_rules! edge_to {
        ($from:expr, $slot:expr, $target:expr, $state:expr, $index:expr) => {{
            let (from, slot, target, state) = ($from, $slot, $target, $state);
            let _ = $index;
            if order.position[target] <= rpo {
                back_edge(builder, &headers, from, slot, target, state, &live_in[target])?;
            } else {
                pending[target].push(PendingEdge { from, slot, state, epoch_dirty: builder.epoch_dirty });
            }
        }};
    }
    for (block_rpo, &(start, end)) in order.blocks.iter().enumerate() {
        rpo = block_rpo;
        let index = start;
        let open_entry = start == 0 && !is_header[0];
        if !open_entry {
            if start == 0 {
                // The entry op heads a loop: the entry block jumps into it.
                let current = builder.current.expect("entry block");
                builder.terminate(WfTerminator::Jump(WfEdge { target: placeholder, args: Vec::new() }));
                pending[0].push(PendingEdge { from: current, slot: 0, state: builder.state.clone(), epoch_dirty: builder.epoch_dirty });
            }
            let incoming = std::mem::take(&mut pending[index]);
            if incoming.is_empty() {
                builder.current = None;
                continue;
            }
            let block = builder.new_block();
            // Back edges are not known yet: a loop header assumes a helper ran.
            let dirty = is_header[index] || incoming.iter().any(|edge| edge.epoch_dirty);
            // Merge: a live GPR with the same value on every edge stays; with
            // different values it becomes a block parameter; missing on any
            // edge means it is never read before being written again. At a
            // loop header every live GPR is a parameter (back edges change
            // them), typed as on entry unless a back edge disagreed.
            let live = live_in[index];
            let loop_header = is_header[index];
            let mut merged = vec![None; 256];
            let mut params: Vec<(u8, WfTy)> = Vec::new();
            for reg in 0..=255u8 {
                if !set_has(&live, reg) { continue; }
                let values: Option<smallvec::SmallVec<[WfValue; 2]>> = incoming.iter().map(|edge| edge.state[reg as usize]).collect();
                let Some(values) = values else { continue; };
                if !loop_header && values.iter().all(|value| *value == values[0]) {
                    merged[reg as usize] = Some(values[0]);
                } else {
                    let first = builder.out.ty(values[0]);
                    let mut ty = if values.iter().all(|value| builder.out.ty(*value) == first) { first } else { WfTy::Tagged };
                    if loop_header && builder.loop_tagged.contains(&(index, reg)) { ty = WfTy::Tagged; }
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
            if loop_header {
                if builder.inline.is_some() {
                    return Err(format!("op {index}: loop inside an inlined body"));
                }
                headers.insert(index, LoopHeader { block, params });
            }
            builder.state = merged;
            builder.current = Some(block);
            builder.epoch_dirty = dirty;
        }
        let mut index = start;
        loop {
        if builder.current.is_none() { break; }
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
            OpKind::Swap { a, b } => {
                let (left, right) = (builder.read(a, index)?, builder.read(b, index)?);
                builder.state[a as usize] = Some(right);
                builder.state[b as usize] = Some(left);
            }
            OpKind::LoadConst { dst, word } => {
                let value = if word & TAG_MASK != TAG_BASE {
                    builder.emit(WfTy::F64, WfInst::ConstF64(word))
                } else if word == TAG_TRUE || word == TAG_FALSE {
                    // Booleans stay unboxed (raw Bool returns, branch conditions).
                    builder.emit(WfTy::Bool, WfInst::ConstBool(word == TAG_TRUE))
                } else {
                    builder.emit(WfTy::Tagged, WfInst::ConstTagged(word))
                };
                builder.state[dst as usize] = Some(value);
            }
            OpKind::LoadRaw { dst, word } => {
                let value = builder.emit(WfTy::Tagged, WfInst::ConstTagged(word));
                builder.state[dst as usize] = Some(value);
            }
            OpKind::LoadAbsolute { dst, address } => {
                let value = builder.emit(WfTy::Tagged, WfInst::LoadAbsolute(address));
                builder.state[dst as usize] = Some(value);
            }
            OpKind::Numeric { dst, lhs, rhs, quick } => {
                builder.ensure_epoch(index, &live)?;
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
                builder.ensure_epoch(index, &live)?;
                let deopt = builder.deopt(index, true, &live)?;
                let left = builder.read(lhs, index)?;
                let left = builder.tagged(left)?;
                let right = builder.read(rhs, index)?;
                let right = builder.tagged(right)?;
                let value = builder.emit(WfTy::Bool, WfInst::IdentityCompare { lhs: left, rhs: right, classes, negate, deopt });
                builder.state[dst as usize] = Some(value);
            }
            OpKind::MapGet { dst, object, entries } => {
                builder.ensure_epoch(index, &live)?;
                let deopt = builder.deopt(index, true, &live)?;
                let object = builder.read(object, index)?;
                let object = builder.tagged(object)?;
                let value = builder.emit(WfTy::Tagged, WfInst::MapPropertyGet { object, entries, deopt });
                builder.state[dst as usize] = Some(value);
            }
            OpKind::SelfCall { dst, args, receiver } => {
                if builder.inline.is_some() {
                    return Err(format!("op {index}: self-call inside an inlined body"));
                }
                let deopt = builder.deopt(index, false, &live)?;
                if matches!(receiver, Some((_, Some(_)))) { builder.ensure_epoch(index, &live)?; }
                let receiver_value = match receiver {
                    Some((reg, class)) => {
                        let value = builder.read(reg, index)?;
                        let value = builder.tagged(value)?;
                        if let Some(class) = class {
                            // A dispatched receiver must keep selecting this method.
                            let guard = builder.deopt(index, true, &live)?;
                            builder.emit(WfTy::Bool, WfInst::GuardClass { value, class, deopt: guard });
                        }
                        Some(value)
                    }
                    None => None,
                };
                let mut values = Vec::with_capacity(args.len() + 1);
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
                if builder.out.has_this {
                    values.push(receiver_value.ok_or_else(|| format!("op {index}: method self-call without receiver"))?);
                }
                let result = builder.emit(WfTy::Tagged, WfInst::SelfCall { args: values, op_index: index as u32, deopt });
                let result = match builder.assume_self_call {
                    Some(WfTy::F64) => builder.emit(WfTy::F64, WfInst::AssumeNumber(result)),
                    Some(WfTy::Bool) => builder.emit(WfTy::Bool, WfInst::AssumeBool(result)),
                    _ => result,
                };
                builder.state[dst as usize] = Some(result);
            }
            OpKind::FieldGet { dst, object, slot, nullable } => {
                let deopt = builder.deopt(index, true, &live)?;
                let object = builder.read(object, index)?;
                let object = builder.tagged(object)?;
                let value = builder.emit(WfTy::Tagged, WfInst::InstanceFieldGet { object, slot, nullable, deopt });
                builder.state[dst as usize] = Some(value);
            }
            OpKind::FieldStoreChecked { object, value, slot } => {
                // A store into an unpublished object of this build is replay-safe.
                let fresh = builder.fresh_objects.contains(&builder.read(object, index)?);
                if builder.root_self_calls && !fresh {
                    return Err(format!("op {index}: field store in a natively recursive function"));
                }
                // A rejected store is an error the interpreter reports, not a
                // failed speculation: nothing to recompile for.
                let deopt = builder.deopt(index, false, &live)?;
                let object = builder.read(object, index)?;
                let object = builder.tagged(object)?;
                let stored = builder.read(value, index)?;
                let stored = builder.tagged(stored)?;
                builder.emit(WfTy::Bool, WfInst::InstanceFieldStoreChecked { object, value: stored, slot, deopt });
                if !fresh {
                    if let Some(frame) = builder.inline.as_mut() { frame.effect_done = true; }
                }
            }
            OpKind::FieldSet { object, value, slot, class_key } => {
                let fresh = builder.fresh_objects.contains(&builder.read(object, index)?);
                if builder.inline.is_some() && builder.root_self_calls && !fresh {
                    return Err(format!("op {index}: store inlined into a natively recursive function"));
                }
                if builder.inline.is_none() && builder.root_self_calls && !fresh {
                    return Err(format!("op {index}: field store in a natively recursive function"));
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
                if !fresh {
                    if let Some(frame) = builder.inline.as_mut() { frame.effect_done = true; }
                }
            }
            OpKind::Primitive { dst, receiver, arg, class, kind } => {
                builder.ensure_epoch(index, &live)?;
                let deopt = builder.deopt(index, true, &live)?;
                let receiver = if kind.numeric_receiver() {
                    builder.number(receiver, index, &mut Some(deopt), &live)?
                } else {
                    let value = builder.read(receiver, index)?;
                    builder.tagged(value)?
                };
                // Booleans are immediates: BoolNot checks the tag itself.
                if kind != WfPrimitive::BoolNot && !kind.numeric_receiver() {
                    builder.emit(WfTy::Bool, WfInst::GuardClass { value: receiver, class, deopt });
                }
                let arg = match arg {
                    Some(reg) if kind.numeric_arg() => { let mut guard = Some(deopt); Some(builder.number(reg, index, &mut guard, &live)?) }
                    Some(reg) => { let value = builder.read(reg, index)?; Some(builder.tagged(value)?) }
                    None => None,
                };
                let value = builder.emit(kind.result_ty(), WfInst::Primitive { kind, receiver, arg, deopt });
                builder.state[dst as usize] = Some(value);
            }
            OpKind::Alloc { dst, kind, .. } => {
                let state = if builder.inline.is_some() {
                    // The operands are the callee's registers: the inlined frames
                    // are published as shadow frames and the allocation reads
                    // the innermost bank. Only the paths that never need the
                    // interpreter's own registers qualify: small Map literals
                    // and objects with a profiled shape (TLAB path).
                    let qualifies = !builder.root_self_calls && match kind {
                        WfAllocKind::SmallMapLiteral => true,
                        WfAllocKind::SpecializedObject =>
                            native_profiled_small_map_site(data.function, data.bytecode, index).is_some(),
                        WfAllocKind::TypedArray => false,
                    };
                    if !qualifies {
                        return Err(format!("op {index}: allocation inside an inlined body"));
                    }
                    let state = builder.chain_state(index, false, &live)?;
                    if builder.out.deopts[state.0 as usize].frames.iter().any(|frame| frame.root_map == u32::MAX) {
                        return Err(format!("op {index}: inlined frame without a GC map"));
                    }
                    state
                } else {
                    builder.deopt(index, false, &live)?
                };
                let value = builder.emit(WfTy::Tagged, WfInst::AllocObject {
                    op_index: index as u32, kind, state, function: data.function_index,
                });
                builder.state[dst as usize] = Some(value);
            }
            OpKind::NewObject { dst, kind, fields, constructor, args } => {
                // An inlined constructor relies on the method epoch.
                if constructor.is_some() { builder.ensure_epoch(index, &live)?; }
                let checkpoint = constructor.map(|_| InlineCheckpoint::save(builder));
                // Nothing is published before the object is complete: an
                // exhausted TLAB replays the whole op in the interpreter.
                let deopt = builder.deopt(index, false, &live)?;
                let null = builder.emit(WfTy::Tagged, WfInst::ConstTagged(TAG_NULL));
                let mut values = vec![null; kind.slots()];
                for &(slot, reg) in &fields {
                    let value = builder.read(reg, index)?;
                    *values.get_mut(slot as usize).ok_or_else(|| format!("op {index}: object slot out of range"))? = builder.tagged(value)?;
                }
                let object = builder.emit(WfTy::Tagged, WfInst::NewObject { kind, fields: values, deopt });
                builder.fresh_objects.push(object);
                builder.state[dst as usize] = Some(object);
                if let Some(callee) = constructor {
                    let target = InlineTarget { callee, receiver: Some(dst), args: args.clone(), class_guard: None };
                    match builder.inline_call(&target, index, &live) {
                        Ok(_) => {
                            // The constructor's result is discarded.
                            builder.out.inlined_calls += 1;
                            builder.state[dst as usize] = Some(object);
                            if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
                                eprintln!("[napalm-wfjit] v2 inlined constructor {:?} at op {index} of {:?}",
                                    builder.source.callees.get(&callee).map(|data| data.function.id), data.function.id);
                            }
                        }
                        Err(reason) => {
                            if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
                                eprintln!("[napalm-wfjit] v2 constructor inline declined at op {index} of {:?}: {reason}", data.function.id);
                            }
                            checkpoint.expect("constructor checkpoint").restore(builder);
                            if builder.inline.is_some() || builder.root_self_calls {
                                return Err(format!("op {index}: constructor call needs the interpreter"));
                            }
                            // The whole op (allocation and constructor) through
                            // the generic call path.
                            let state = builder.deopt(index, false, &live)?;
                            builder.emit(WfTy::Bool, WfInst::RuntimeOp {
                                helper: WfRuntimeHelper::CallToReturn, op_index: index as u32, next_op: next as u32,
                                state, dest: None, clobbers: true,
                            });
                            let live_next = live_in[next];
                            let mut reloaded = vec![None; 256];
                            for reg in all_registers(data.function) {
                                if set_has(&live_next, reg) {
                                    reloaded[reg as usize] = Some(builder.emit(WfTy::Tagged, WfInst::FrameLoad(reg)));
                                }
                            }
                            builder.state = reloaded;
                            builder.epoch_dirty = true;
                        }
                    }
                }
            }
            OpKind::Runtime { helper, write, inline, guard, generic, reads } => {
                // Inlined bodies and guarded direct calls rely on the epoch.
                if inline.is_some() || guard.is_some() {
                    builder.ensure_epoch(index, &live)?;
                }
                // Prefer inlining the call; on failure restore and call out.
                let mut inlined = None;
                if let Some(target) = &inline {
                    let checkpoint = InlineCheckpoint::save(builder);
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
                            checkpoint.restore(builder);
                        }
                    }
                }
                if let Some(value) = inlined {
                    if let Some(dst) = write { builder.state[dst as usize] = Some(value); }
                } else {
                    let mut continue_after_runtime = false;
                    if builder.inline.is_some() {
                        // A call with a statically known target runs from the
                        // inlined body with explicit arguments; the inlined
                        // frames are published as GC shadow frames meanwhile.
                        let opcode = data.bytecode.chunk.code.get(data.function.raw_pcs[index] as usize)
                            .and_then(|word| word.opcode());
                        // String concatenation allocates: run it with the
                        // inlined frames published as GC shadow frames.
                        if matches!((helper, generic, opcode), (WfRuntimeHelper::Op, false, Some(Opcode::StringConcat)))
                            && !builder.root_self_calls && reads.len() >= 2
                        {
                            let left = builder.read(reads[0], index)?;
                            let left = builder.tagged(left)?;
                            let right = builder.read(reads[1], index)?;
                            let right = builder.tagged(right)?;
                            let state = builder.chain_state(index, false, &live)?;
                            if builder.out.deopts[state.0 as usize].frames.iter().any(|frame| frame.root_map == u32::MAX) {
                                return Err(format!("op {index}: inlined frame without a GC map"));
                            }
                            let value = builder.emit(WfTy::Tagged, WfInst::StringConcat { left, right, state });
                            if let Some(dst) = write { builder.state[dst as usize] = Some(value); }
                            continue_after_runtime = true;
                        } else {
                        let explicit = matches!(helper, WfRuntimeHelper::DirectCall
                            | WfRuntimeHelper::ProfiledMethod(_) | WfRuntimeHelper::DirectMethod)
                            && !builder.root_self_calls;
                        // A polymorphic or unprofiled method call dispatches at
                        // run time through the op's own inline cache.
                        let original = data.function.original_ops[index];
                        let dynamic_slot = match (helper, generic, opcode) {
                            _ if builder.root_self_calls => None,
                            (WfRuntimeHelper::CallToReturn, false, Some(Opcode::Invoke)) => Some(original.x),
                            (WfRuntimeHelper::CallToReturn, false, Some(Opcode::InvokeMethodDirect)) => Some(original.cache),
                            _ => None,
                        };
                        let (call_target, receiver_reg, arg_regs): (WfCallTarget, Option<u8>, RegList) =
                            match (inline.as_ref().filter(|_| explicit), dynamic_slot) {
                                (Some(target), _) =>
                                    (WfCallTarget::Static(target.callee), target.receiver, target.args.clone()),
                                // Each one costs a shadow-frame call sequence and frame-rebuilding
                                // deopts for every later guard, for the saving of one callee
                                // frame: past the budget the callee is called, not inlined.
                                (None, Some(_)) if builder.dispatched_calls >= MAX_INLINED_DISPATCHED_CALLS =>
                                    return Err(format!("op {index}: dispatched call budget of inlined bodies exhausted")),
                                (None, Some(ic_slot)) if !reads.is_empty() => (
                                    WfCallTarget::Dynamic { function: data.function_index, ic_slot },
                                    Some(reads[0]), reads[1..].iter().copied().collect(),
                                ),
                                _ => return Err(format!("op {index}: host helper call ({helper:?} {opcode:?}) inside an inlined body")),
                            };
                        let receiver = match receiver_reg {
                            Some(reg) => { let value = builder.read(reg, index)?; Some(builder.tagged(value)?) }
                            None => None,
                        };
                        if let (Some((_, class)), Some(value)) = (guard, receiver) {
                            let deopt = builder.deopt(index, true, &live)?;
                            builder.emit(WfTy::Bool, WfInst::GuardClass { value, class, deopt });
                        }
                        let mut args = Vec::with_capacity(arg_regs.len());
                        for &reg in &arg_regs {
                            let value = builder.read(reg, index)?;
                            args.push(builder.tagged(value)?);
                        }
                        let state = builder.chain_state(index, false, &live)?;
                        let snapshot = &builder.out.deopts[state.0 as usize];
                        if snapshot.frames.iter().any(|frame| frame.root_map == u32::MAX) {
                            return Err(format!("op {index}: inlined frame without a GC map"));
                        }
                        if matches!(call_target, WfCallTarget::Dynamic { .. }) { builder.dispatched_calls += 1; }
                        let root_call = builder.out.deopts[state.0 as usize].op_index as usize;
                        let scratch = builder.source.root.function.ops.get(root_call)
                            .map(|op| op.a)
                            .ok_or_else(|| format!("op {index}: root call op out of range"))?;
                        let result = builder.emit(WfTy::Tagged, WfInst::CallExplicit { target: call_target, receiver, args, state, scratch });
                        if let Some(dst) = write { builder.state[dst as usize] = Some(result); }
                        // A call is an effect: later guards rebuild the frames.
                        if let Some(frame) = builder.inline.as_mut() { frame.effect_done = true; }
                        builder.epoch_dirty = true;
                        continue_after_runtime = true;
                        }
                    }
                    if continue_after_runtime {
                        // Handled above.
                    } else {
                    if builder.root_self_calls {
                        // A nested native frame has no bank for a host helper.
                        return Err(format!("op {index}: host helper call in a natively recursive function"));
                    }
                    if let Some((reg, class)) = guard {
                        // The helper calls the profiled target directly.
                        let deopt = builder.deopt(index, true, &live)?;
                        let value = builder.read(reg, index)?;
                        let value = builder.tagged(value)?;
                        builder.emit(WfTy::Bool, WfInst::GuardClass { value, class, deopt });
                    }
                    let state = builder.deopt(index, false, &live)?;
                    let dest = if generic { None } else { write };
                    let ty = if dest.is_some() { WfTy::Tagged } else { WfTy::Bool };
                    let result = builder.emit(ty, WfInst::RuntimeOp {
                        helper, op_index: index as u32, next_op: next as u32, state, dest, clobbers: generic,
                    });
                    if generic {
                        // Unmodelled operands: every GPR live afterwards comes
                        // from the bank the interpreter just updated.
                        let live_next = live_in[next];
                        let mut reloaded = vec![None; 256];
                        for reg in all_registers(data.function) {
                            if set_has(&live_next, reg) {
                                reloaded[reg as usize] = Some(builder.emit(WfTy::Tagged, WfInst::FrameLoad(reg)));
                            }
                        }
                        builder.state = reloaded;
                    } else if let Some(dst) = write {
                        builder.state[dst as usize] = Some(result);
                    }
                    // The helper may have run code that rebinds methods: the
                    // next epoch-dependent op checks again.
                    builder.epoch_dirty = true;
                    }
                }
            }
            OpKind::Jump { target } => {
                let from = builder.current.unwrap();
                let state = builder.state.clone();
                builder.terminate(WfTerminator::Jump(WfEdge { target: placeholder, args: Vec::new() }));
                edge_to!(from, 0, target, state, index);
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
                edge_to!(from, 0, then_target, state.clone(), index);
                edge_to!(from, 1, else_target, state, index);
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
                let value = if builder.raw_return.is_some_and(|raw| builder.out.ty(value) == raw) {
                    // Raw bits; kept only if every Return has this type.
                    value
                } else {
                    builder.raw_ok = false;
                    builder.tagged(value)?
                };
                builder.terminate(WfTerminator::Return(value));
                continue;
            }
        }
        // Straight-line op: the block's last one falls through into the
        // block at `next` (possibly skipping ahead, e.g. a fused literal).
        if index == end {
            if builder.current.is_some() {
                let from = builder.current.unwrap();
                let state = builder.state.clone();
                builder.terminate(WfTerminator::Jump(WfEdge { target: placeholder, args: Vec::new() }));
                edge_to!(from, 0, next, state, index);
            }
            break;
        }
        index = next;
        }
    }
    if builder.current.is_some() {
        return Err("control falls off the end of the function".into());
    }
    if pending.iter().any(|edges| !edges.is_empty()) {
        return Err("unresolved control-flow edge".into());
    }
    Ok(())
}

