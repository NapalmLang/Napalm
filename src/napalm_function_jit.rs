//! Whole-function Cranelift tier for hot synchronous bytecode functions.
//!
//! Top-level entries execute against the interpreter's existing frame/register
//! stack. Eligible non-memoized self-recursion uses a lightweight native shadow
//! frame instead of constructing another Napalm `Frame`; the VM root provider
//! scans those native registers with the ordinary bytecode `ReferenceMap`.
//! Calls that cannot use that ABI still flow through `enter_direct_bytecode_call`,
//! preserving auto-memoization, async semantics and the normal frame ABI.

use super::*;
use cranelift::codegen::ir::{BlockArg, StackSlotData, StackSlotKind, UserFuncName};

#[path = "napalm_wfjit_ir.rs"]
mod wf_ir;
#[path = "napalm_wfjit_lower.rs"]
mod wf_lower;
use cranelift::codegen::settings::{self, Configurable};
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{default_libcall_names, Linkage, Module};
use cranelift::prelude::{AbiParam, FloatCC, InstBuilder, IntCC, MemFlagsData, types};
use std::ffi::c_void;
use std::mem::offset_of;

use crate::napalm_gc::FastAllocationInfo as GcFastAllocationInfo;
use crate::napalm_tracing_jit::{
    TRACE_HEAP_DATA_RELATIVE_MASK, TRACE_HEAP_KIND_MASK, TRACE_HEAP_METADATA_SHIFT,
};

const MIN_WHOLE_FUNCTION_HOTNESS: u32 = 64;
const MAX_WHOLE_FUNCTION_OPS: usize = 8192;
const MAX_NATIVE_CALL_DEPTH: u32 = 512;
const WFJIT_BAILOUT: u64 = TAG_BASE;

pub(super) type WholeFunctionEntry = unsafe extern "C" fn(*mut c_void, *mut RegisterWord) -> u64;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WholeFunctionJitStats {
    pub hot_calls: u64,
    pub compile_attempts: u64,
    pub compiled_functions: u64,
    pub rejected_functions: u64,
    pub profile_deferrals: u64,
    pub native_entries: u64,
    pub native_direct_calls: u64,
    /// Self-recursive calls executed through the lightweight GC-visible native ABI.
    pub native_recursive_calls: u64,
    pub baseline_direct_calls: u64,
    /// Native entries that handed control back to the interpreter mid-body.
    pub bailouts: u64,
    /// Compiled functions dropped because they kept bailing out.
    pub deoptimized_functions: u64,
    /// Functions compiled by the typed-SSA tier (all compiled functions).
    pub v2_compiled_functions: u64,
    /// Calls inlined by the v2 tier.
    pub v2_inlined_calls: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FunctionState {
    Cold,
    /// v2 code generation is in flight on the background compiler.
    Compiling,
    Compiled,
    Rejected,
}

struct FunctionSlot {
    hotness: u32,
    profile_retries: u8,
    state: FunctionState,
    entry: Option<WholeFunctionEntry>,
    /// Native entries / bailouts since compilation (bailout feedback).
    native_entries: u32,
    bailouts: u32,
    /// The compiled body has speculative guards (a failed guard bails out
    /// instead of calling the generic helper).
    speculative: bool,
    /// Failed speculative guards since compilation (other bailouts, such as
    /// the native depth limit, do not count against speculation).
    speculation_failures: u32,
    /// Prepared ops whose speculative guards failed (`ENTRY_SPECULATION_SITE`
    /// for the entry's parameter types). Kept across recompilations: those
    /// sites compile without speculation from then on.
    failed_sites: Vec<u32>,
    /// Speculative recompilations so far (each re-reads the profiles).
    recompiles: u8,
}

/// Speculative recompilations before the function is left to the
/// interpreter and the tracer.
const MAX_SPECULATIVE_RECOMPILES: u8 = 3;

/// `failed_sites` marker for the entry trampoline's parameter type checks.
const ENTRY_SPECULATION_SITE: u32 = u32::MAX;

impl Default for FunctionSlot {
    fn default() -> Self {
        Self {
            hotness: 0, profile_retries: 0, state: FunctionState::Cold, entry: None,
            native_entries: 0, bailouts: 0, speculative: false,
            speculation_failures: 0, failed_sites: Vec::new(), recompiles: 0,
        }
    }
}

pub(super) struct WholeFunctionJit {
    slots: Vec<FunctionSlot>,
    threshold: u32,
    active_depth: u32,
    stats: WholeFunctionJitStats,
    /// v2 loop iteration counter cells per function (stable across
    /// recompilations, so the tracer handoff happens once per loop).
    wf_loop_counters: crate::napalm_collections::HashMap<u32,
        std::rc::Rc<std::cell::RefCell<crate::napalm_collections::HashMap<usize, usize>>>>,
    /// v2 background compiler (started on the first v2 compilation).
    background: Option<WfBackground>,
}

/// Every runtime helper declared in a whole-function `JITModule`.
#[derive(Clone, Copy)]
struct JitHelpers {
    runtime_op: cranelift::module::FuncId,
    direct_call: cranelift::module::FuncId,
    direct_method: cranelift::module::FuncId,
    profiled_method: cranelift::module::FuncId,
    call_to_return: cranelift::module::FuncId,
    reconstruct_frames: cranelift::module::FuncId,
    string_eq: cranelift::module::FuncId,
    string_code_point_at: cranelift::module::FuncId,
    bytes_index: cranelift::module::FuncId,
    call_explicit: cranelift::module::FuncId,
    invoke_explicit: cranelift::module::FuncId,
    small_map_literal_regs: cranelift::module::FuncId,
    profiled_small_map: cranelift::module::FuncId,
    profiled_small_map_commit: cranelift::module::FuncId,
    specialized_factory: cranelift::module::FuncId,
    bind_factory_contract: cranelift::module::FuncId,
    reserved_typed_array: cranelift::module::FuncId,
    speculation_failed: cranelift::module::FuncId,
    write_barrier: cranelift::module::FuncId,
    store_field_checked: cranelift::module::FuncId,
    string_concat: cranelift::module::FuncId,
}

/// A Cranelift module for whole-function code with every runtime helper
/// registered and declared, owned by the v2 background compiler (worker thread).
fn new_jit_module() -> Result<(JITModule, JitHelpers), String> {
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").map_err(|e| e.to_string())?;
        flags.set("enable_alias_analysis", "true").map_err(|e| e.to_string())?;
        flags.set("use_colocated_libcalls", "false").map_err(|e| e.to_string())?;
        flags.set("is_pic", if cfg!(target_arch = "x86_64") { "true" } else { "false" })
            .map_err(|e| e.to_string())?;
        let isa = cranelift::native::builder()
            .map_err(|e| format!("Cranelift does not support this host: {e}"))?
            .finish(settings::Flags::new(flags))
            .map_err(|e| format!("cannot build whole-function ISA: {e}"))?;

        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
        builder.symbol("napalm_wfjit_runtime_op", napalm_wfjit_runtime_op as *const u8);
        builder.symbol("napalm_wfjit_direct_call", napalm_wfjit_direct_call as *const u8);
        builder.symbol("napalm_wfjit_direct_method", napalm_wfjit_direct_method as *const u8);
        builder.symbol("napalm_wfjit_profiled_method", napalm_wfjit_profiled_method as *const u8);
        builder.symbol("napalm_wfjit_call_to_return", napalm_wfjit_call_to_return as *const u8);
        builder.symbol("napalm_wfjit_reconstruct_frames", napalm_wfjit_reconstruct_frames as *const u8);
        builder.symbol("napalm_wfjit_string_eq", napalm_wfjit_string_eq as *const u8);
        builder.symbol("napalm_wfjit_string_code_point_at", napalm_wfjit_string_code_point_at as *const u8);
        builder.symbol("napalm_wfjit_bytes_index", napalm_wfjit_bytes_index as *const u8);
        builder.symbol("napalm_wfjit_call_explicit", napalm_wfjit_call_explicit as *const u8);
        builder.symbol("napalm_wfjit_invoke_explicit", napalm_wfjit_invoke_explicit as *const u8);
        builder.symbol(
            "napalm_wfjit_small_map_literal_regs",
            napalm_wfjit_small_map_literal_regs as *const u8,
        );
        builder.symbol(
            "napalm_wfjit_profiled_small_map",
            napalm_wfjit_profiled_small_map as *const u8,
        );
        builder.symbol(
            "napalm_wfjit_profiled_small_map_commit",
            napalm_wfjit_profiled_small_map_commit as *const u8,
        );
        builder.symbol(
            "napalm_wfjit_specialized_factory",
            napalm_wfjit_specialized_factory as *const u8,
        );
        builder.symbol(
            "napalm_wfjit_bind_factory_contract",
            napalm_wfjit_bind_factory_contract as *const u8,
        );
        builder.symbol(
            "napalm_wfjit_commit_reserved_typed_array",
            napalm_wfjit_commit_reserved_typed_array as *const u8,
        );
        builder.symbol("napalm_wfjit_speculation_failed", napalm_wfjit_speculation_failed as *const u8);
        builder.symbol("napalm_wfjit_write_barrier", trace_heap_write_barrier as *const u8);
        builder.symbol("napalm_wfjit_store_field_checked", napalm_wfjit_store_field_checked as *const u8);
        builder.symbol("napalm_wfjit_string_concat", napalm_wfjit_string_concat as *const u8);
        let mut module = JITModule::new(builder);
        if module.target_config().pointer_type() != types::I64 {
            return Err("Napalm whole-function JIT requires a 64-bit host".to_owned());
        }
        let pointer = module.target_config().pointer_type();

        let mut helper_sig = module.make_signature();
        helper_sig.params.push(AbiParam::new(pointer));
        helper_sig.params.push(AbiParam::new(types::I32));
        helper_sig.returns.push(AbiParam::new(pointer));
        let runtime_id = module
            .declare_function("napalm_wfjit_runtime_op", Linkage::Import, &helper_sig)
            .map_err(|e| format!("cannot declare whole-function runtime helper: {e}"))?;
        let direct_id = module
            .declare_function("napalm_wfjit_direct_call", Linkage::Import, &helper_sig)
            .map_err(|e| format!("cannot declare whole-function direct-call helper: {e}"))?;
        let direct_method_id = module
            .declare_function("napalm_wfjit_direct_method", Linkage::Import, &helper_sig)
            .map_err(|e| format!("cannot declare whole-function direct method helper: {e}"))?;
        let mut profiled_method_sig = module.make_signature();
        profiled_method_sig.params.push(AbiParam::new(pointer)); // vm
        profiled_method_sig.params.push(AbiParam::new(types::I32)); // op index
        profiled_method_sig.params.push(AbiParam::new(types::I32)); // prepared target
        profiled_method_sig.returns.push(AbiParam::new(pointer));
        let profiled_method_id = module
            .declare_function("napalm_wfjit_profiled_method", Linkage::Import, &profiled_method_sig)
            .map_err(|e| format!("cannot declare whole-function profiled method helper: {e}"))?;
        let call_to_return_id = module
            .declare_function("napalm_wfjit_call_to_return", Linkage::Import, &helper_sig)
            .map_err(|e| format!("cannot declare whole-function generic call helper: {e}"))?;
        let mut reconstruct_sig = module.make_signature();
        for _ in 0..3 { reconstruct_sig.params.push(AbiParam::new(pointer)); } // vm, descriptor, values
        let reconstruct_id = module
            .declare_function("napalm_wfjit_reconstruct_frames", Linkage::Import, &reconstruct_sig)
            .map_err(|e| format!("cannot declare whole-function frame reconstruction helper: {e}"))?;
        // Pure leaf helpers: (vm, receiver word, argument word) -> word.
        let mut leaf_sig = module.make_signature();
        leaf_sig.params.push(AbiParam::new(pointer));
        leaf_sig.params.push(AbiParam::new(types::I64));
        leaf_sig.params.push(AbiParam::new(types::I64));
        leaf_sig.returns.push(AbiParam::new(types::I64));
        let mut declare_leaf = |name: &str| module.declare_function(name, Linkage::Import, &leaf_sig)
            .map_err(|e| format!("cannot declare whole-function leaf helper {name}: {e}"));
        let string_eq_id = declare_leaf("napalm_wfjit_string_eq")?;
        let string_code_point_id = declare_leaf("napalm_wfjit_string_code_point_at")?;
        let bytes_index_id = declare_leaf("napalm_wfjit_bytes_index")?;
        let mut explicit_sig = module.make_signature();
        explicit_sig.params.push(AbiParam::new(pointer)); // vm
        explicit_sig.params.push(AbiParam::new(types::I32)); // root call op
        explicit_sig.params.push(AbiParam::new(types::I32)); // target
        explicit_sig.params.push(AbiParam::new(types::I64)); // receiver
        explicit_sig.params.push(AbiParam::new(types::I32)); // has receiver
        explicit_sig.params.push(AbiParam::new(pointer)); // arguments
        explicit_sig.params.push(AbiParam::new(types::I32)); // argument count
        explicit_sig.params.push(AbiParam::new(pointer)); // result slot
        explicit_sig.returns.push(AbiParam::new(types::I32));
        let call_explicit_id = module
            .declare_function("napalm_wfjit_call_explicit", Linkage::Import, &explicit_sig)
            .map_err(|e| format!("cannot declare whole-function explicit call helper: {e}"))?;
        let mut invoke_sig = module.make_signature();
        invoke_sig.params.push(AbiParam::new(pointer)); // vm
        invoke_sig.params.push(AbiParam::new(types::I32)); // root call op
        invoke_sig.params.push(AbiParam::new(types::I32)); // inlined function
        invoke_sig.params.push(AbiParam::new(types::I32)); // its inline cache slot
        invoke_sig.params.push(AbiParam::new(types::I64)); // receiver
        invoke_sig.params.push(AbiParam::new(pointer)); // arguments
        invoke_sig.params.push(AbiParam::new(types::I32)); // argument count
        invoke_sig.params.push(AbiParam::new(pointer)); // result slot
        invoke_sig.returns.push(AbiParam::new(types::I32));
        let invoke_explicit_id = module
            .declare_function("napalm_wfjit_invoke_explicit", Linkage::Import, &invoke_sig)
            .map_err(|e| format!("cannot declare whole-function dispatched call helper: {e}"))?;

        // Lightweight native-recursion helpers receive the owning function/op
        // identity plus the explicit register base. They never rely on hot-frame
        // register pointers, which is what lets nested calls avoid a Napalm Frame.
        let mut native_call_sig = module.make_signature();
        native_call_sig.params.push(AbiParam::new(pointer));
        native_call_sig.params.push(AbiParam::new(types::I32));
        native_call_sig.params.push(AbiParam::new(types::I32));
        native_call_sig.params.push(AbiParam::new(pointer));
        native_call_sig.returns.push(AbiParam::new(types::I64));
        let small_map_regs_id = module
            .declare_function(
                "napalm_wfjit_small_map_literal_regs",
                Linkage::Import,
                &native_call_sig,
            )
            .map_err(|e| format!("cannot declare register-based small-map helper: {e}"))?;

        let specialized_factory_id = module
            .declare_function(
                "napalm_wfjit_specialized_factory",
                Linkage::Import,
                &native_call_sig,
            )
            .map_err(|e| format!("cannot declare native specialized factory: {e}"))?;

        let mut bind_contract_sig = module.make_signature();
        bind_contract_sig.params.push(AbiParam::new(pointer));
        bind_contract_sig.params.push(AbiParam::new(types::I32));
        bind_contract_sig.params.push(AbiParam::new(types::I32));
        bind_contract_sig.params.push(AbiParam::new(types::I64));
        bind_contract_sig.returns.push(AbiParam::new(types::I64));
        let bind_factory_contract_id = module
            .declare_function(
                "napalm_wfjit_bind_factory_contract",
                Linkage::Import,
                &bind_contract_sig,
            )
            .map_err(|e| format!("cannot declare specialized contract binder: {e}"))?;

        let mut profiled_map_sig = module.make_signature();
        profiled_map_sig.params.push(AbiParam::new(pointer)); // vm
        profiled_map_sig.params.push(AbiParam::new(types::I32)); // shape
        profiled_map_sig.params.push(AbiParam::new(pointer)); // initialized InlineMapValues
        profiled_map_sig.returns.push(AbiParam::new(types::I64));
        let profiled_small_map_id = module
            .declare_function(
                "napalm_wfjit_profiled_small_map",
                Linkage::Import,
                &profiled_map_sig,
            )
            .map_err(|e| format!("cannot declare profiled small-map helper: {e}"))?;

        let mut profiled_map_commit_sig = module.make_signature();
        profiled_map_commit_sig.params.push(AbiParam::new(pointer)); // vm
        profiled_map_commit_sig.params.push(AbiParam::new(types::I64)); // reserved object address
        profiled_map_commit_sig.params.push(AbiParam::new(types::I32)); // shape
        profiled_map_commit_sig.params.push(AbiParam::new(pointer)); // initialized InlineMapValues
        profiled_map_commit_sig.returns.push(AbiParam::new(types::I64));
        let profiled_small_map_commit_id = module
            .declare_function(
                "napalm_wfjit_profiled_small_map_commit",
                Linkage::Import,
                &profiled_map_commit_sig,
            )
            .map_err(|e| format!("cannot declare profiled small-map commit helper: {e}"))?;

        let mut typed_array_commit_sig = module.make_signature();
        typed_array_commit_sig.params.push(AbiParam::new(pointer)); // interpreter
        typed_array_commit_sig.params.push(AbiParam::new(types::I32)); // function
        typed_array_commit_sig.params.push(AbiParam::new(types::I32)); // op
        typed_array_commit_sig.params.push(AbiParam::new(types::I64)); // reserved object address
        typed_array_commit_sig.returns.push(AbiParam::new(types::I64));
        let reserved_typed_array_id = module
            .declare_function(
                "napalm_wfjit_commit_reserved_typed_array",
                Linkage::Import,
                &typed_array_commit_sig,
            )
            .map_err(|e| format!("cannot declare typed array Immix commit helper: {e}"))?;

        let mut speculation_sig = module.make_signature();
        speculation_sig.params.extend([AbiParam::new(pointer), AbiParam::new(types::I32), AbiParam::new(types::I32)]);
        let speculation_failed_id = module
            .declare_function("napalm_wfjit_speculation_failed", Linkage::Import, &speculation_sig)
            .map_err(|e| format!("cannot declare speculation feedback helper: {e}"))?;

        let mut barrier_sig = module.make_signature();
        barrier_sig.params.extend([AbiParam::new(pointer), AbiParam::new(pointer), AbiParam::new(types::I64)]);
        let write_barrier_id = module
            .declare_function("napalm_wfjit_write_barrier", Linkage::Import, &barrier_sig)
            .map_err(|e| format!("cannot declare write barrier helper: {e}"))?;
        let mut store_sig = module.make_signature();
        store_sig.params.extend([AbiParam::new(pointer), AbiParam::new(types::I64), AbiParam::new(types::I64), AbiParam::new(types::I32)]);
        store_sig.returns.push(AbiParam::new(types::I32));
        let store_field_checked_id = module
            .declare_function("napalm_wfjit_store_field_checked", Linkage::Import, &store_sig)
            .map_err(|e| format!("cannot declare checked field store helper: {e}"))?;
        let concat_id = module
            .declare_function("napalm_wfjit_string_concat", Linkage::Import, &leaf_sig)
            .map_err(|e| format!("cannot declare string concatenation helper: {e}"))?;

        Ok((module, JitHelpers {
            runtime_op: runtime_id,
            direct_call: direct_id,
            direct_method: direct_method_id,
            profiled_method: profiled_method_id,
            call_to_return: call_to_return_id,
            reconstruct_frames: reconstruct_id,
            string_eq: string_eq_id,
            string_code_point_at: string_code_point_id,
            bytes_index: bytes_index_id,
            call_explicit: call_explicit_id,
            invoke_explicit: invoke_explicit_id,
            small_map_literal_regs: small_map_regs_id,
            profiled_small_map: profiled_small_map_id,
            profiled_small_map_commit: profiled_small_map_commit_id,
            specialized_factory: specialized_factory_id,
            bind_factory_contract: bind_factory_contract_id,
            reserved_typed_array: reserved_typed_array_id,
            speculation_failed: speculation_failed_id,
            write_barrier: write_barrier_id,
            store_field_checked: store_field_checked_id,
            string_concat: concat_id,
        }))
}

impl WholeFunctionJit {
    pub(super) fn new(function_count: usize, entry_threshold: u32) -> Result<Self, String> {
        let result = Self {
            background: None,
            slots: (0..function_count).map(|_| FunctionSlot::default()).collect(),
            // Direct-call tiering runs before the callee's entry hook. Matching
            // its threshold avoids compiling an entry trace just before a whole
            // function supersedes it; rejected/deferred functions can still trace.
            threshold: if entry_threshold == 0 { u32::MAX } else { entry_threshold.max(MIN_WHOLE_FUNCTION_HOTNESS) },
            active_depth: 0,
            stats: WholeFunctionJitStats::default(),
            wf_loop_counters: Default::default(),
        };
        Ok(result)
    }

    #[inline]
    pub(super) fn stats(&self) -> WholeFunctionJitStats { self.stats }

    #[inline]
    fn entry(&self, function: u32) -> Option<WholeFunctionEntry> {
        self.slots.get(function as usize).and_then(|slot| slot.entry)
    }

    /// Cheap pre-check for interpreter call hooks: a rejected function never
    /// runs natively, so its callers can skip the depth bookkeeping entirely.
    #[inline(always)]
    pub(super) fn may_enter(&self, function: u32) -> bool {
        self.slots.get(function as usize).is_some_and(|slot| slot.state != FunctionState::Rejected)
    }

    #[inline]
    fn note_call(&mut self, function: u32, ops: usize) -> bool {
        self.stats.hot_calls = self.stats.hot_calls.saturating_add(1);
        let Some(slot) = self.slots.get_mut(function as usize) else { return false; };
        if slot.state != FunctionState::Cold { return false; }
        slot.hotness = slot.hotness.saturating_add(1);
        // Compile cost grows with the body: larger functions must be called
        // proportionally more often before compiling them pays off (their
        // loops are the loop tracer's in the meantime).
        let factor = (ops / 32).clamp(1, 32) as u32;
        slot.hotness >= self.threshold.saturating_mul(factor)
    }

    /// Bailout feedback for one native entry of `function`.
    #[inline]
    fn note_native_result(&mut self, function: u32, bailed_out: bool) {
        if bailed_out { self.stats.bailouts = self.stats.bailouts.saturating_add(1); }
        let Some(slot) = self.slots.get_mut(function as usize) else { return; };
        slot.native_entries = slot.native_entries.saturating_add(1);
        if bailed_out { slot.bailouts = slot.bailouts.saturating_add(1); }
        // A speculative body whose guards keep failing replays work in the
        // interpreter on every bailout: drop it and recompile once it is hot
        // again, with the failed sites running through their helpers and the
        // rest speculating anew on the current profiles. A function that
        // still fails after that is left to the interpreter and the tracer.
        if bailed_out && slot.speculative && slot.speculation_failures >= 8
            && u64::from(slot.speculation_failures) * 64 >= u64::from(slot.native_entries)
        {
            slot.recompiles = slot.recompiles.saturating_add(1);
            slot.speculative = false;
            slot.entry = None;
            self.stats.deoptimized_functions = self.stats.deoptimized_functions.saturating_add(1);
            if slot.recompiles > MAX_SPECULATIVE_RECOMPILES {
                slot.state = FunctionState::Rejected;
                self.stats.rejected_functions = self.stats.rejected_functions.saturating_add(1);
                return;
            }
            slot.state = FunctionState::Cold;
            slot.hotness = 0;
            slot.native_entries = 0;
            slot.bailouts = 0;
            slot.speculation_failures = 0;
        } else if bailed_out && slot.bailouts >= 256
            && u64::from(slot.bailouts) * 2 >= u64::from(slot.native_entries)
        {
            // Native code that leaves on most entries for other reasons
            // (helpers declining, exceptions caught in the frame) costs more
            // than it saves: leave the function to the interpreter and tracer.
            slot.state = FunctionState::Rejected;
            slot.entry = None;
            self.stats.rejected_functions = self.stats.rejected_functions.saturating_add(1);
        }
    }

    /// Feedback from a failed speculative guard (cold path of compiled code).
    fn note_speculation_failure(&mut self, function: u32, site: u32) {
        if let Some(slot) = self.slots.get_mut(function as usize) {
            slot.speculation_failures = slot.speculation_failures.saturating_add(1);
            if !slot.failed_sites.contains(&site) && slot.failed_sites.len() < 64 {
                slot.failed_sites.push(site);
            }
        }
    }

    fn reject(&mut self, function: u32) {
        if let Some(slot) = self.slots.get_mut(function as usize) {
            slot.state = FunctionState::Rejected;
            slot.entry = None;
        }
        self.stats.rejected_functions = self.stats.rejected_functions.saturating_add(1);
    }

    fn compile(
        &mut self,
        function_index: u32,
        function: &PreparedFunction,
        bytecode: &BytecodeFunction,
        constants: &[RegisterWord],
        expected_method_epoch: u64,
        identity_sites: &[Option<NativeIdentitySite>],
        small_map_fast_alloc: Option<GcFastAllocationInfo>,
        native_direct_method_class_slots: &crate::napalm_collections::HashMap<ClassId, u32>,
        native_direct_method_targets: &crate::napalm_collections::HashMap<(ReceiverClass, MethodSymbol), CallTarget>,
        native_plain_store_classes: &crate::napalm_collections::HashMap<usize, u64>,
        inline_callees: &crate::napalm_collections::HashMap<u32, wf_ir::WfFunctionData<'_>>,
        manifest: Option<&crate::napalm_builtin::BuiltinManifest>,
        reactive_enabled: bool,
    ) -> Result<(), String> {
        self.stats.compile_attempts = self.stats.compile_attempts.saturating_add(1);
        // Methods and constructors (`uses_this`) are eligible: their frame is
        // built by the interpreter before entry, so `this` already sits in its
        // GPR (entry_this_dst) and in SpecialReg::This for GC and bailouts.
        if function.memoization.enabled()
            || function.is_async
            || function.is_concurrent
            || function.return_kind == RegisterKind::RawPtr
            || function.ops.is_empty()
            || function.ops.len() > MAX_WHOLE_FUNCTION_OPS
        {
            if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
                eprintln!(
                    "[napalm-wfjit] rejected {:?}: memo={} async={} concurrent={} captures={} handlers={} ops={}",
                    function.id, function.memoization.enabled(), function.is_async, function.is_concurrent,
                    function.capture_count, function.exception_handlers.len(), function.ops.len(),
                );
            }
            self.reject(function_index);
            return Ok(());
        }

        let reachable = reachable_ops(function, bytecode);
        let unsupported_invoke = |index: usize, opcode: Opcode| opcode == Opcode::Invoke
            && native_numeric_quick(function, index).is_none()
            && native_map_property_pic(function, index).is_none()
            && identity_sites[index].is_none()
            // Runtime-profiled InvokeMethodDirect (promoted by the VM).
            && profiled_direct_invoke_site(function, index).is_none();
        for index in 0..function.ops.len() {
            if !reachable[index] { continue; }
            let opcode = bytecode.chunk.code
                .get(function.raw_pcs[index] as usize)
                .and_then(|word| word.opcode())
                .ok_or_else(|| format!("invalid opcode at prepared op {index}"))?;
            // A recursive descent can become hot before it has returned even
            // once, leaving the unwind path unprofiled. Retry after more calls
            // instead of settling for a generic call on its first visit, but
            // only while the site can still be promoted to a direct call (not
            // disabled as polymorphic, no host-primitive target).
            let promotable = function.ops[index].flags & QUICK_GENERIC_DISABLED == 0
                && function.quick_invokes.get(index)
                    .is_none_or(|site| site.class.0 == 0 || matches!(site.target, CallTarget::Bytecode(_)));
            if unsupported_invoke(index, opcode) && promotable
                && self.slots[function_index as usize].profile_retries < 4
            {
                let slot = &mut self.slots[function_index as usize];
                slot.hotness = 0;
                slot.profile_retries += 1;
                self.stats.profile_deferrals += 1;
                return Ok(());
            }
        }

        // Typed-SSA IR built on the VM thread; Cranelift runs on the background
        // compiler and the entry is installed later. Sites whose guards failed
        // in earlier compilations run through their helpers.
        let failed_sites = self.slots[function_index as usize].failed_sites.clone();
        let counters = self.wf_loop_counters.entry(function_index).or_default().clone();
        let root = wf_ir::WfFunctionData {
            function_index, function, bytecode, constants,
            reachable,
            identity_sites: identity_sites.to_vec(),
            plain_store_classes: native_plain_store_classes.clone(),
        };
        let source = wf_ir::WfSource {
            root: &root,
            callees: inline_callees,
            expected_method_epoch,
            class_slots: native_direct_method_class_slots,
            method_targets: native_direct_method_targets,
            trace_headers: function.jit_trace_headers.as_ptr() as usize,
            shared_flag: offset_of!(Interpreter<'static>, heap) + offset_of!(Heap, shared_mode),
            loop_counters: &counters,
            manifest,
            speculate: wfjit_speculation_enabled(),
            reactive: reactive_enabled,
            failed_sites: &failed_sites,
        };
        let events = std::env::var_os("NAPALM_JIT_EVENTS").is_some();
        let started = std::time::Instant::now();
        match wf_ir::build(&source) {
            Ok(ir) => {
                let summary = if events { format!("{:?}: {} ops={} insts={} build={:?}", function.id, ir.summary(),
                    function.ops.len(), ir.blocks.iter().map(|block| block.insts.len()).sum::<usize>(),
                    started.elapsed()) } else { String::new() };
                match self.submit_v2(function_index, ir, small_map_fast_alloc, summary) {
                    Ok(()) => return Ok(()),
                    Err(reason) => {
                        if events { eprintln!("[napalm-wfjit] v2 background compiler unavailable: {reason}"); }
                    }
                }
            }
            Err(reason) => {
                if events { eprintln!("[napalm-wfjit] v2 declined {:?}: {reason}", function.id); }
            }
        }
        self.reject(function_index);
        Ok(())
    }
}



/// v2 code generation: a `JITModule` of its own with every helper declared.
/// It lives on the background compiler thread and reads nothing but its job
/// (the IR carries the frame layout and allocation sites it needs).
struct WfCodegen {
    module: JITModule,
    context: cranelift::codegen::Context,
    function_context: FunctionBuilderContext,
    helpers: JitHelpers,
    /// Cranelift instructions of the last body (diagnostics).
    last_clif_insts: usize,
}

impl WfCodegen {
    fn new() -> Result<Self, String> {
        let (module, helpers) = new_jit_module()?;
        let context = module.make_context();
        Ok(Self { module, context, function_context: FunctionBuilderContext::new(), helpers, last_clif_insts: 0 })
    }

    fn compile(&mut self, ir: &wf_ir::WfFunction, fast_alloc: Option<GcFastAllocationInfo>) -> Result<WholeFunctionEntry, String> {
        let pointer = self.module.target_config().pointer_type();
        let mut entry_signature = self.module.make_signature();
        entry_signature.params.push(AbiParam::new(pointer));
        entry_signature.params.push(AbiParam::new(pointer));
        entry_signature.returns.push(AbiParam::new(types::I64));
        let mut body_signature = entry_signature.clone();
        body_signature.call_conv = cranelift::codegen::isa::CallConv::Tail;
        body_signature.params.push(AbiParam::new(types::I32));
        for ty in &ir.param_types {
            body_signature.params.push(AbiParam::new(if *ty == wf_ir::WfTy::F64 { types::F64 } else { types::I64 }));
        }
        if ir.has_this { body_signature.params.push(AbiParam::new(types::I64)); }
        let body_id = self.module.declare_anonymous_function(&body_signature)
            .map_err(|e| format!("cannot declare v2 body: {e}"))?;

        self.context.clear();
        self.context.func.signature = body_signature;
        self.context.func.name = UserFuncName::user(2, body_id.as_u32());
        self.function_context = FunctionBuilderContext::new();
        {
            let target_config = self.module.target_config();
            let func = &mut self.context.func;
            let helpers = wf_lower::WfHelperRefs {
                self_ref: self.module.declare_func_in_func(body_id, func),
                speculation_failed: self.module.declare_func_in_func(self.helpers.speculation_failed, func),
                runtime_op: self.module.declare_func_in_func(self.helpers.runtime_op, func),
                direct_call: self.module.declare_func_in_func(self.helpers.direct_call, func),
                direct_method: self.module.declare_func_in_func(self.helpers.direct_method, func),
                profiled_method: self.module.declare_func_in_func(self.helpers.profiled_method, func),
                call_to_return: self.module.declare_func_in_func(self.helpers.call_to_return, func),
                profiled_small_map: self.module.declare_func_in_func(self.helpers.profiled_small_map, func),
                profiled_small_map_commit: self.module.declare_func_in_func(self.helpers.profiled_small_map_commit, func),
                bind_factory_contract: self.module.declare_func_in_func(self.helpers.bind_factory_contract, func),
                small_map_literal_regs: self.module.declare_func_in_func(self.helpers.small_map_literal_regs, func),
                specialized_factory: self.module.declare_func_in_func(self.helpers.specialized_factory, func),
                reserved_typed_array: self.module.declare_func_in_func(self.helpers.reserved_typed_array, func),
                reconstruct: self.module.declare_func_in_func(self.helpers.reconstruct_frames, func),
                string_eq: self.module.declare_func_in_func(self.helpers.string_eq, func),
                string_code_point_at: self.module.declare_func_in_func(self.helpers.string_code_point_at, func),
                bytes_index: self.module.declare_func_in_func(self.helpers.bytes_index, func),
                call_explicit: self.module.declare_func_in_func(self.helpers.call_explicit, func),
                invoke_explicit: self.module.declare_func_in_func(self.helpers.invoke_explicit, func),
                write_barrier: self.module.declare_func_in_func(self.helpers.write_barrier, func),
                store_field_checked: self.module.declare_func_in_func(self.helpers.store_field_checked, func),
                string_concat: self.module.declare_func_in_func(self.helpers.string_concat, func),
            };
            let mut b = FunctionBuilder::new(&mut self.context.func, &mut self.function_context);
            wf_lower::lower_body(&mut b, ir, fast_alloc, helpers);
            b.finalize(target_config);
        }
        self.last_clif_insts = self.context.func.dfg.num_insts();
        let disasm = std::env::var_os("NAPALM_CRANELIFT_DISASM").is_some();
        self.context.set_disasm(disasm);
        self.module.define_function(body_id, &mut self.context)
            .map_err(|e| format!("cannot define v2 body {:?}: {e:?}", ir.function_index))?;
        if disasm {
            if let Some(vcode) = self.context.compiled_code().and_then(|code| code.vcode.as_ref()) {
                eprintln!("[napalm-wfjit] v2 disassembly {:?}:\n{vcode}", ir.function_index);
            }
        }

        // Host-ABI trampoline: the interpreter-built frame holds the arguments
        // at their entry registers; the chain depth starts at active_depth.
        let entry_id = self.module.declare_anonymous_function(&entry_signature)
            .map_err(|e| format!("cannot declare v2 entry: {e}"))?;
        self.context.clear();
        self.context.func.signature = entry_signature;
        self.context.func.name = UserFuncName::user(2, entry_id.as_u32());
        self.function_context = FunctionBuilderContext::new();
        {
            let target_config = self.module.target_config();
            let body_ref = self.module.declare_func_in_func(body_id, &mut self.context.func);
            let mut b = FunctionBuilder::new(&mut self.context.func, &mut self.function_context);
            let block = b.create_block();
            b.append_block_params_for_function_params(block);
            b.switch_to_block(block);
            let (vm, regs) = (b.block_params(block)[0], b.block_params(block)[1]);
            let flags = MemFlagsData::trusted();
            let jit = b.ins().load(types::I64, flags, vm,
                offset_of!(Interpreter<'static>, whole_function_jit) as i32);
            let depth = b.ins().load(types::I32, flags, jit,
                offset_of!(WholeFunctionJit, active_depth) as i32);
            // Entry guards (method epoch, unboxed parameters) fail before any
            // effect: resume the interpreter at op 0 with the frame untouched.
            let deopt = b.create_block();
            b.set_cold_block(deopt);
            let mut arguments = vec![vm, regs, depth];
            if let Some((expected, _)) = ir.entry_epoch_guard {
                let epoch = b.ins().load(types::I64, flags, vm,
                    offset_of!(Interpreter<'static>, method_epoch) as i32);
                let valid = b.ins().icmp_imm(IntCC::Equal, epoch, expected as i64);
                let next = b.create_block();
                b.ins().brif(valid, next, &[], deopt, &[]);
                b.switch_to_block(next);
            }
            for argument in 0..ir.arity {
                let destination = ir.frame.entry_arg_dsts[argument];
                let word = if destination == u16::MAX {
                    b.ins().iconst(types::I64, TAG_NULL as i64)
                } else {
                    b.ins().load(types::I64, flags, regs, destination as i32 * 8)
                };
                if ir.param_types[argument] == wf_ir::WfTy::F64 {
                    let tag = b.ins().band_imm(word, TAG_MASK as i64);
                    let is_number = b.ins().icmp_imm(IntCC::NotEqual, tag, TAG_BASE as i64);
                    let next = b.create_block();
                    b.ins().brif(is_number, next, &[], deopt, &[]);
                    b.switch_to_block(next);
                    arguments.push(b.ins().bitcast(types::F64, MemFlagsData::new(), word));
                } else {
                    arguments.push(word);
                }
            }
            if ir.has_this {
                arguments.push(b.ins().load(types::I64, flags, regs, ir.frame.entry_this_dst as i32 * 8));
            }
            let call = b.ins().call(body_ref, &arguments);
            let mut result = b.inst_results(call)[0];
            match ir.return_ty {
                wf_ir::WfTy::F64 => {
                    // Raw f64 bits from the body: canonicalize NaN for the
                    // interpreter, but pass the bailout sentinel through.
                    let bits = b.ins().bitcast(types::F64, MemFlagsData::new(), result);
                    let is_nan = b.ins().fcmp(FloatCC::Unordered, bits, bits);
                    let bailout = b.ins().icmp_imm(IntCC::Equal, result, WFJIT_BAILOUT as i64);
                    let not_bailout = b.ins().bxor_imm(bailout, 1);
                    let canonicalize = b.ins().band(is_nan, not_bailout);
                    let canonical = b.ins().iconst(types::I64, CANONICAL_NAN as i64);
                    result = b.ins().select(canonicalize, canonical, result);
                }
                wf_ir::WfTy::Bool => {
                    // Raw 0/1 from the body: box it, passing the sentinel through.
                    let bailout = b.ins().icmp_imm(IntCC::Equal, result, WFJIT_BAILOUT as i64);
                    let boxed = b.ins().iadd_imm(result, TAG_FALSE as i64);
                    result = b.ins().select(bailout, result, boxed);
                }
                wf_ir::WfTy::Tagged => {}
            }
            b.ins().return_(&[result]);
            b.switch_to_block(deopt);
            let speculation_ref = self.module.declare_func_in_func(self.helpers.speculation_failed, b.func);
            let function_arg = b.ins().iconst(types::I32, ir.function_index as i64);
            let site = b.ins().iconst(types::I32, ENTRY_SPECULATION_SITE as i64);
            b.ins().call(speculation_ref, &[vm, function_arg, site]);
            let hot_frame = b.ins().load(types::I64, flags, vm, offset_of!(Interpreter<'static>, hot_frame) as i32);
            let root_map = b.ins().iconst(types::I32, ir.frame.gc_map_by_op[0] as i64);
            b.ins().store(flags, root_map, hot_frame, offset_of!(Frame, root_map) as i32);
            let pc = b.ins().iconst(types::I32, 0);
            b.ins().store(flags, pc, vm, offset_of!(Interpreter<'static>, hot_pc) as i32);
            let sentinel = b.ins().iconst(types::I64, WFJIT_BAILOUT as i64);
            b.ins().return_(&[sentinel]);
            b.seal_all_blocks();
            b.finalize(target_config);
        }
        self.context.set_disasm(false);
        self.module.define_function(entry_id, &mut self.context)
            .map_err(|e| format!("cannot define v2 entry {:?}: {e:?}", ir.function_index))?;
        self.module.finalize_definitions()
            .map_err(|e| format!("cannot finalize v2 function {:?}: {e}", ir.function_index))?;
        let code = self.module.get_finalized_function(entry_id);
        Ok(unsafe { std::mem::transmute::<*const u8, WholeFunctionEntry>(code) })
    }
}

/// A v2 compilation handed to the background compiler.
struct WfJob {
    function_index: u32,
    ir: wf_ir::WfFunction,
    fast_alloc: Option<GcFastAllocationInfo>,
}

// SAFETY: the IR is owned data; the TLAB description only holds addresses
// that generated code embeds as constants (never dereferenced by the worker).
unsafe impl Send for WfJob {}

/// Result of a background compilation: entry address and Cranelift size.
struct WfDone {
    function_index: u32,
    result: Result<(usize, usize), String>,
}

/// The v2 background compiler: a small pool of worker threads, each owning a
/// `WfCodegen` (its own `JITModule`), taking jobs from one shared queue. A
/// function has at most one job in flight (it stays `Compiling` until
/// installed), so results may complete in any order. Generated code lives in
/// its worker's module, which outlives every entry handed out: workers only
/// exit once the VM side drops the job sender.
struct WfBackground {
    jobs: Option<std::sync::mpsc::Sender<WfJob>>,
    done: std::sync::mpsc::Receiver<WfDone>,
    threads: Vec<std::thread::JoinHandle<()>>,
    /// Submitted, not yet installed: function index -> (summary, inlined calls).
    pending: crate::napalm_collections::HashMap<u32, (String, u32)>,
}

/// Background compiler threads: `NAPALM_WFJIT_WORKERS`, or up to 3 while
/// leaving a core to the interpreter.
fn wfjit_worker_count() -> usize {
    std::env::var("NAPALM_WFJIT_WORKERS").ok().and_then(|value| value.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism()
            .map_or(1, |cores| cores.get().saturating_sub(1).clamp(1, 3)))
        .clamp(1, 16)
}

impl WfBackground {
    fn start() -> Result<Self, String> {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<WfJob>();
        let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));
        let (done_tx, done_rx) = std::sync::mpsc::channel::<WfDone>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let workers = wfjit_worker_count();
        let mut threads = Vec::with_capacity(workers);
        for worker in 0..workers {
            let (job_rx, done_tx, ready_tx) = (job_rx.clone(), done_tx.clone(), ready_tx.clone());
            let thread = std::thread::Builder::new()
                .name(format!("napalm-wfjit-{worker}"))
                .spawn(move || {
                    let mut codegen = match WfCodegen::new() {
                        Ok(codegen) => { let _ = ready_tx.send(Ok(())); codegen }
                        Err(error) => { let _ = ready_tx.send(Err(error)); return; }
                    };
                    loop {
                        // Hold the queue only while taking a job.
                        let job = match job_rx.lock() {
                            Ok(queue) => queue.recv(),
                            Err(_) => break,
                        };
                        let Ok(job) = job else { break };
                        let result = codegen.compile(&job.ir, job.fast_alloc)
                            .map(|entry| (entry as usize, codegen.last_clif_insts));
                        if done_tx.send(WfDone { function_index: job.function_index, result }).is_err() { break; }
                    }
                })
                .map_err(|error| format!("cannot start the v2 background compiler: {error}"))?;
            threads.push(thread);
        }
        for _ in 0..workers {
            ready_rx.recv().map_err(|error| error.to_string())??;
        }
        Ok(Self { jobs: Some(job_tx), done: done_rx, threads, pending: Default::default() })
    }
}

impl Drop for WfBackground {
    fn drop(&mut self) {
        // Closing the job channel ends every worker after its current job.
        self.jobs.take();
        for thread in self.threads.drain(..) { let _ = thread.join(); }
    }
}

/// `NAPALM_WFJIT_SPECULATE=0` compiles every function in the non-speculative
/// mode that is otherwise reserved for repeatedly deoptimized functions.
fn wfjit_speculation_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NAPALM_WFJIT_SPECULATE").ok().as_deref() != Some("0"))
}

/// Wait for each v2 compilation to finish before running on (deterministic
/// tiering for tests): `cfg(test)` or `NAPALM_WFJIT_SYNC=1`.
fn wfjit_blocking() -> bool {
    static BLOCKING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *BLOCKING.get_or_init(|| cfg!(test) || std::env::var_os("NAPALM_WFJIT_SYNC").is_some())
}

impl WholeFunctionJit {
    /// Hand a built v2 IR to the background compiler; the function keeps
    /// running in baseline until the entry is installed.
    fn submit_v2(
        &mut self,
        function_index: u32,
        ir: wf_ir::WfFunction,
        fast_alloc: Option<GcFastAllocationInfo>,
        summary: String,
    ) -> Result<(), String> {
        if self.background.is_none() {
            self.background = Some(WfBackground::start()?);
        }
        let background = self.background.as_mut().expect("background compiler");
        let inlined = ir.inlined_calls;
        background.jobs.as_ref().ok_or("background compiler stopped")?
            .send(WfJob { function_index, ir, fast_alloc })
            .map_err(|_| "background compiler stopped".to_string())?;
        background.pending.insert(function_index, (summary, inlined));
        self.slots[function_index as usize].state = FunctionState::Compiling;
        if wfjit_blocking() {
            while self.slots[function_index as usize].state == FunctionState::Compiling {
                if !self.poll_background(true) { break; }
            }
        }
        Ok(())
    }

    /// Install finished background compilations (all available ones, or the
    /// next one when `block`). False when nothing is pending.
    #[inline]
    fn poll_background(&mut self, block: bool) -> bool {
        let Some(background) = self.background.as_mut() else { return false; };
        if background.pending.is_empty() { return false; }
        let mut finished = Vec::new();
        if block {
            match background.done.recv() {
                Ok(done) => finished.push(done),
                Err(_) => return false,
            }
        }
        while let Ok(done) = background.done.try_recv() { finished.push(done); }
        for done in finished {
            let Some((summary, inlined)) = self.background.as_mut().unwrap().pending.remove(&done.function_index) else { continue; };
            self.install_v2(done, summary, inlined);
        }
        true
    }

    fn install_v2(&mut self, done: WfDone, summary: String, inlined: u32) {
        let events = std::env::var_os("NAPALM_JIT_EVENTS").is_some();
        let index = done.function_index as usize;
        let Some(slot) = self.slots.get_mut(index) else { return; };
        if slot.state != FunctionState::Compiling { return; }
        match done.result {
            Ok((entry, clif)) => {
                slot.state = FunctionState::Compiled;
                slot.entry = Some(unsafe { std::mem::transmute::<usize, WholeFunctionEntry>(entry) });
                slot.speculative = true;
                slot.native_entries = 0;
                slot.bailouts = 0;
                slot.speculation_failures = 0;
                self.stats.compiled_functions = self.stats.compiled_functions.saturating_add(1);
                self.stats.v2_compiled_functions = self.stats.v2_compiled_functions.saturating_add(1);
                self.stats.v2_inlined_calls = self.stats.v2_inlined_calls.saturating_add(u64::from(inlined));
                if events { eprintln!("[napalm-wfjit] v2 compiled {summary} clif={clif}"); }
            }
            Err(reason) => {
                // Code generation is deterministic for this IR: leave the
                // function to the interpreter and the tracer.
                slot.state = FunctionState::Rejected;
                slot.entry = None;
                self.stats.rejected_functions = self.stats.rejected_functions.saturating_add(1);
                if events { eprintln!("[napalm-wfjit] v2 code generation failed for function {index}: {reason}"); }
            }
        }
    }
}
#[derive(Clone)]
struct NativeIdentitySite {
    classes: smallvec::SmallVec<[ReceiverClass; 4]>,
    negate: bool,
}

/// Snapshot only implementations the host explicitly permits us to inline.
/// A polymorphic site (e.g. Map | Null) can still compare in native code.
fn native_identity_sites(vm: &Interpreter<'_>, function: u32) -> Vec<Option<NativeIdentitySite>> {
    let bytecode = &vm.program.functions[function as usize];
    let prepared = &vm.prepared.functions[function as usize];
    prepared.original_ops.iter().zip(&prepared.raw_pcs).map(|(op, &raw_pc)| {
        if bytecode.chunk.code[raw_pc as usize].opcode() != Some(Opcode::Invoke) || op.c != 1 {
            return None;
        }
        let descriptor = bytecode.chunk.inline_caches.get(op.x as usize)?;
        let signature = vm.program.metadata.method(descriptor.method)?;
        let name = vm.program.metadata.string(signature.name)?;
        if name != "==" && name != "!=" { return None; }
        let cache = vm.inline_caches[function as usize].get(op.x as usize)?;
        let mut classes = smallvec::SmallVec::new();
        for entry in &cache.entries[..cache.len as usize] {
            if let CallTarget::Primitive(primitive) = entry.target {
                if vm.host.standard_identity_method(primitive, name, signature.arity) {
                    classes.push(entry.class);
                }
            }
        }
        (!classes.is_empty()).then_some(NativeIdentitySite { classes, negate: name == "!=" })
    }).collect()
}

#[inline]
fn native_numeric_quick(function: &PreparedFunction, index: usize) -> Option<NumericQuickOp> {
    let flags = function.ops.get(index)?.flags;
    if (flags & QUICK_GENERIC_DISABLED) != 0 {
        return None;
    }
    let quick = NumericQuickOp::from_flag(flags)?;
    // Bitwise Number operators require full ToUint32 coercion. Keep those in the
    // already-fast specialized interpreter handler until we add range feedback.
    if matches!(
        quick,
        NumericQuickOp::BitNot
            | NumericQuickOp::Rem
            | NumericQuickOp::BitAnd
            | NumericQuickOp::BitOr
            | NumericQuickOp::BitXor
            | NumericQuickOp::Shl
            | NumericQuickOp::ShrU
    ) {
        return None;
    }
    Some(quick)
}

#[derive(Clone, Debug)]
struct NativeMapPropertyPic {
    entries: smallvec::SmallVec<[MapIcEntry; 4]>,
}

#[inline]
fn native_map_property_pic(function: &PreparedFunction, index: usize) -> Option<NativeMapPropertyPic> {
    let site = *function.quick_invokes.get(index)?;
    if !matches!(site.target, CallTarget::NativeBuiltin(NativeBuiltin::MapPropertyGet(_))) {
        return None;
    }
    let cache = &function.quick_map_properties.get(index)?.cache;
    if cache.len == 0 || cache.megamorphic {
        return None;
    }
    let mut entries = smallvec::SmallVec::<[MapIcEntry; 4]>::new();
    entries.extend_from_slice(&cache.entries[..cache.len as usize]);
    Some(NativeMapPropertyPic { entries })
}


/// Dense value base (Instance fields or shaped-Map values) from a handle
/// sidecar: inline storage publishes a tagged owner-relative offset (it moves
/// with the GC object), boxed/spilled storage an absolute pointer (bit 0
/// clear). Branch-free decode against the live owner `address` already loaded
/// from the same entry.
fn emit_heap_data_base(
    b: &mut FunctionBuilder<'_>,
    entry: cranelift::prelude::Value,
    address: cranelift::prelude::Value,
) -> cranelift::prelude::Value {
    let encoded = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, data) as i32);
    let relative = b.ins().band_imm(encoded, TRACE_HEAP_DATA_RELATIVE_MASK as i64);
    let owner_mask = b.ins().ineg(relative);
    let base = b.ins().band(address, owner_mask);
    let offset = b.ins().band_imm(encoded, !(TRACE_HEAP_DATA_RELATIVE_MASK as u64) as i64);
    b.ins().iadd(base, offset)
}

/// Whether a compiled function may recurse through the lightweight native ABI.
///
/// A bailout from a nested native call replays the outermost self-call in the
/// ordinary interpreter frame. Restrict this path to replay-safe operations so a
/// speculative guard failure cannot duplicate an externally visible store or
/// host effect. Small object-literal allocation is allowed because an abandoned
/// object is unreachable and therefore semantically invisible.
fn reachable_ops(function: &PreparedFunction, bytecode: &BytecodeFunction) -> Vec<bool> {
    let mut seen = vec![false; function.ops.len()];
    let mut work = vec![0usize];
    while let Some(index) = work.pop() {
        if index >= seen.len() || seen[index] { continue; }
        seen[index] = true;
        let prepared = function.ops[index];
        let original = function.original_ops[index];
        match bytecode.chunk.code[function.raw_pcs[index] as usize].opcode() {
            Some(Opcode::Return | Opcode::Return2 | Opcode::Trap) => {},
            Some(Opcode::Jump) => work.push(prepared.x as usize),
            Some(Opcode::JumpIfFalse | Opcode::JumpIfTrue) => {
                work.push(prepared.x as usize);
                work.push(original.next as usize);
            }
            Some(Opcode::AllocMap) if prepared.c > 0 && prepared.c <= 4
                && prepared.next > index as u32 + 1 => work.push(prepared.next as usize),
            _ => work.push(original.next as usize),
        }
    }
    seen
}

#[inline]
fn eligible_opcode(opcode: Opcode) -> bool {
    !matches!(opcode,
        Opcode::Trap | Opcode::Switch | Opcode::Return2 | Opcode::Bailout
        | Opcode::GuardType | Opcode::GuardClass | Opcode::GuardBounds
        | Opcode::GuardNotNull | Opcode::GuardInt32 | Opcode::GuardF64
        | Opcode::InvokeSuper | Opcode::CallClosure | Opcode::Launch
        | Opcode::CallPrimitive | Opcode::CallForeign
        | Opcode::CreateClass | Opcode::BindMethod | Opcode::BindForeignMethod
        | Opcode::AllocForeign | Opcode::MakeClosure
        | Opcode::ImportModule | Opcode::ImportVariable
        | Opcode::MakeFiber | Opcode::FiberCall | Opcode::FiberTry
        | Opcode::FiberTransfer | Opcode::FiberTransferError | Opcode::FiberYield
        | Opcode::FiberSuspend | Opcode::FiberAbort | Opcode::Await | Opcode::Catch
        | Opcode::InvokeSpread
    )
}

fn wfjit_call_counters_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| cfg!(test) || std::env::var_os("NAPALM_JIT_STATS").is_some())
}

/// Memory flags for frame GPR `reg`. Every GPR gets its own Cranelift alias
/// region, so the egraph alias analysis can forward a register store to later
/// loads and drop redundant reloads across stores to *other* registers and
/// across untagged VM-field loads (an untagged store is still a full fence).
///
/// Soundness: generated code touches frame-register memory only through
/// `load_reg`/`store_reg`, so a given address is always accessed with its
/// register's region. Register banks of different frames (the caller's base,
/// inlined/native child stack slots) share a region per index, which is merely
/// conservative. Runtime helpers access registers only inside calls, and calls
/// are fences that clear all alias state.
#[inline]
fn reg_flags(b: &mut FunctionBuilder<'_>, reg: u8) -> MemFlagsData {
    let region = b.func.dfg.alias_regions.insert(cranelift::codegen::ir::AliasRegionData {
        user_id: reg as u32,
        description: std::borrow::Cow::Borrowed("napalm gpr"),
    });
    MemFlagsData::trusted().with_alias_region(Some(region))
}
#[inline]
fn load_reg(b: &mut FunctionBuilder<'_>, regs: cranelift::prelude::Value, reg: u8) -> cranelift::prelude::Value {
    let flags = reg_flags(b, reg);
    b.ins().load(types::I64, flags, regs, reg as i32 * 8)
}
#[inline]
fn store_reg(b: &mut FunctionBuilder<'_>, regs: cranelift::prelude::Value, reg: u8, value: cranelift::prelude::Value) {
    let flags = reg_flags(b, reg);
    b.ins().store(flags, value, regs, reg as i32 * 8);
}
#[inline]
fn store_i32(b: &mut FunctionBuilder<'_>, regs: cranelift::prelude::Value, reg: u8, value: cranelift::prelude::Value) {
    let wide = b.ins().uextend(types::I64, value);
    store_reg(b, regs, reg, wide);
}

#[derive(Clone, Copy, Debug)]
struct NativeProfiledSmallMapSite {
    shape: u32,
    value_count: u8,
    field_count: u8,
    key_regs: [u8; SMALL_MAP_INLINE_FIELDS],
    value_regs: [u8; SMALL_MAP_INLINE_FIELDS],
    key_words: [u64; SMALL_MAP_INLINE_FIELDS],
    slots: [u16; SMALL_MAP_INLINE_FIELDS],
    /// Legacy AllocMap+MapSet needs dynamic string-key guards.
    guard_keys: bool,
    /// ObjectShape and NewObjectEmpty create only null-initialized slots.
    null_fields: bool,
    /// A structural contract must be installed before the new object escapes.
    bind_contract: bool,
}

fn native_profiled_small_map_site(
    function: &PreparedFunction,
    bytecode: &BytecodeFunction,
    index: usize,
) -> Option<NativeProfiledSmallMapSite> {
    let op = *function.ops.get(index)?;
    if function.gc_map_by_op.get(index).copied()? == NO_INDEX {
        return None;
    }
    let opcode = bytecode.chunk.code.get(*function.raw_pcs.get(index)? as usize)?.opcode()?;
    let make = |shape: u32, field_count: u8, value_count: u8, slots: [u16; SMALL_MAP_INLINE_FIELDS],
                guard_keys: bool, null_fields: bool, bind_contract: bool| {
        NativeProfiledSmallMapSite {
            shape, field_count, value_count, slots, guard_keys, null_fields, bind_contract,
            key_regs: [0; SMALL_MAP_INLINE_FIELDS],
            value_regs: [0; SMALL_MAP_INLINE_FIELDS],
            key_words: [0; SMALL_MAP_INLINE_FIELDS],
        }
    };
    if opcode == Opcode::NewObjectEmpty {
        return Some(make(0, 0, 0, [0; SMALL_MAP_INLINE_FIELDS], false, true, false));
    }
    let site = function.small_map_literal_sites.get(op.cache as usize)?;
    if !site.initialized || site.disabled || !site.stable_program_keys
        || site.value_count as usize > SMALL_MAP_INLINE_FIELDS
        || site.field_count as usize > SMALL_MAP_INLINE_FIELDS
    {
        return None;
    }
    let (field_count, guard_keys, null_fields, bind_contract) = match opcode {
        Opcode::AllocMap if op.c > 0 && op.c <= 4 => (op.c, true, false, false),
        Opcode::NewObjectWithProperties if op.c > 0 && op.c <= 4 =>
            (op.c, false, false, false),
        Opcode::NewInterfaceOrTypeObject if op.c > 0 && op.c <= 4 =>
            (op.c, false, false, true),
        Opcode::NewObjectShape => (site.field_count, false, true, true),
        _ => return None,
    };
    if site.field_count != field_count {
        return None;
    }
    let needs_contract = bind_contract && matches!(
        bytecode.chunk.specializations.get(op.x as usize),
        Some(crate::napalm_bytecode::SpecializedDescriptor::ObjectShape {
            contract: Some(_), ..
        })
    );
    let mut result = make(site.shape, field_count, site.value_count,
        site.slots, guard_keys, null_fields, needs_contract);
    for i in 0..field_count as usize {
        if guard_keys {
            let offset = op.args_offset as usize;
            result.key_regs[i] = function.call_args.get(offset + i * 2)?.0;
            result.value_regs[i] = function.call_args.get(offset + i * 2 + 1)?.0;
            result.key_words[i] = Value::from_heap(site.key_handles[i]).raw_bits();
        } else if !null_fields {
            result.value_regs[i] = unsafe { wfjit_call_source_reg(function, &op, i) }?;
        }
        if result.slots[i] as usize >= site.value_count as usize {
            return None;
        }
    }
    Some(result)
}

fn emit_native_root_map_update(
    b: &mut FunctionBuilder<'_>,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    root_map: u32,
    op_index: u32,
) {
    let flags = MemFlagsData::trusted();
    let top = b.create_block();
    let nested = b.create_block();
    let ready = b.create_block();
    let hot_regs = b.ins().load(
        types::I64,
        flags,
        vm,
        offset_of!(Interpreter<'static>, hot_registers) as i32,
    );
    let is_top = b.ins().icmp(IntCC::Equal, regs, hot_regs);
    b.ins().brif(is_top, top, &[], nested, &[]);

    b.switch_to_block(top);
    let frame = b.ins().load(
        types::I64,
        flags,
        vm,
        offset_of!(Interpreter<'static>, hot_frame) as i32,
    );
    let root = b.ins().iconst(types::I32, root_map as i64);
    b.ins().store(flags, root, frame, offset_of!(Frame, root_map) as i32);
    let pc = b.ins().iconst(types::I32, op_index as i64);
    b.ins().store(flags, pc, vm, offset_of!(Interpreter<'static>, hot_pc) as i32);
    b.ins().jump(ready, &[]);

    b.switch_to_block(nested);
    let shadow = b.ins().load(
        types::I64,
        flags,
        vm,
        offset_of!(Interpreter<'static>, native_shadow_top) as i32,
    );
    let root = b.ins().iconst(types::I32, root_map as i64);
    b.ins().store(flags, root, shadow, offset_of!(NativeShadowFrame, root_map) as i32);
    b.ins().jump(ready, &[]);

    b.switch_to_block(ready);
    b.seal_block(top);
    b.seal_block(nested);
    b.seal_block(ready);
}

/// What `emit_native_typed_array_alloc` needs to know about its op, captured
/// up front so code generation never reads live VM metadata (it can run on a
/// background thread).
#[derive(Clone, Copy, Debug)]
pub(super) struct TypedArraySite {
    capacity_reg: u8,
    /// `ArrayType::with_capacity`; None when the descriptor is not an array
    /// type (the helper then handles the op).
    with_capacity: Option<bool>,
    root_map: u32,
}

fn typed_array_site(function: &PreparedFunction, bytecode: &BytecodeFunction, index: usize) -> TypedArraySite {
    let op = function.ops[index];
    let with_capacity = match bytecode.chunk.specializations.get(op.x as usize) {
        Some(crate::napalm_bytecode::SpecializedDescriptor::ArrayType { with_capacity, .. }) => Some(*with_capacity),
        _ => None,
    };
    TypedArraySite { capacity_reg: op.b, with_capacity, root_map: function.gc_map_by_op[index] }
}

fn emit_native_typed_array_alloc(
    b: &mut FunctionBuilder<'_>,
    slow_helper: cranelift::codegen::ir::FuncRef,
    commit_helper: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    site: TypedArraySite,
    function_index: u32,
    index: usize,
    fast: Option<GcFastAllocationInfo>,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let Some(fast) = fast else {
        return call_native_value_helper(b, slow_helper, vm, function_index, index as u32, regs, abort);
    };
    let Some(with_capacity) = site.with_capacity else {
        return call_native_value_helper(b, slow_helper, vm, function_index, index as u32, regs, abort);
    };
    if !fast.alignment.is_power_of_two() || fast.bytes == 0 {
        return call_native_value_helper(b, slow_helper, vm, function_index, index as u32, regs, abort);
    }

    emit_native_root_map_update(b, vm, regs, site.root_map, index as u32);
    let flags = MemFlagsData::trusted();
    let try_fast = b.create_block();
    let slow = b.create_block();
    let commit = b.create_block();
    let done = b.create_block();
    b.append_block_param(commit, types::I64);
    b.append_block_param(done, types::I64);
    if with_capacity {
        let capacity = load_reg(b, regs, site.capacity_reg);
        let empty = b.ins().icmp_imm(IntCC::Equal, capacity, 0);
        b.ins().brif(empty, try_fast, &[], slow, &[]);
    } else {
        b.ins().jump(try_fast, &[]);
    }

    b.switch_to_block(try_fast);
    let cursor_ptr = b.ins().iconst(types::I64, fast.cursor as usize as i64);
    let limit_ptr = b.ins().iconst(types::I64, fast.limit as usize as i64);
    let cursor = b.ins().load(types::I64, flags, cursor_ptr, 0);
    let limit = b.ins().load(types::I64, flags, limit_ptr, 0);
    let biased = b.ins().iadd_imm(cursor, (fast.alignment - 1) as i64);
    let mask = b.ins().iconst(types::I64, -(fast.alignment as i64));
    let aligned = b.ins().band(biased, mask);
    let end = b.ins().iadd_imm(aligned, fast.bytes as i64);
    let fits = b.ins().icmp(IntCC::UnsignedLessThanOrEqual, end, limit);
    let no_overflow = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, end, aligned);
    let alignment_ok = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, biased, cursor);
    let valid = b.ins().band(fits, no_overflow);
    let valid = b.ins().band(valid, alignment_ok);
    b.ins().brif(valid, commit, &[BlockArg::Value(aligned)], slow, &[]);

    b.switch_to_block(commit);
    let address = b.block_params(commit)[0];
    let end = b.ins().iadd_imm(address, fast.bytes as i64);
    b.ins().store(flags, end, cursor_ptr, 0);
    let func = b.ins().iconst(types::I32, function_index as i64);
    let op_index = b.ins().iconst(types::I32, index as i64);
    let call = b.ins().call(commit_helper, &[vm, func, op_index, address]);
    let result = b.inst_results(call)[0];
    b.ins().jump(done, &[BlockArg::Value(result)]);

    b.switch_to_block(slow);
    let func = b.ins().iconst(types::I32, function_index as i64);
    let op_index = b.ins().iconst(types::I32, index as i64);
    let call = b.ins().call(slow_helper, &[vm, func, op_index, regs]);
    let result = b.inst_results(call)[0];
    b.ins().jump(done, &[BlockArg::Value(result)]);

    b.seal_block(try_fast);
    b.seal_block(commit);
    b.seal_block(slow);
    b.seal_block(done);
    b.switch_to_block(done);
    let value = b.block_params(done)[0];
    let failed = b.ins().icmp_imm(IntCC::Equal, value, WFJIT_BAILOUT as i64);
    let success = b.create_block();
    b.append_block_param(success, types::I64);
    b.ins().brif(failed, abort, &[], success, &[BlockArg::Value(value)]);
    b.seal_block(success);
    b.switch_to_block(success);
    b.block_params(success)[0]
}

fn emit_profiled_small_map_alloc(
    b: &mut FunctionBuilder<'_>,
    slow_helper: cranelift::codegen::ir::FuncRef,
    commit_helper: cranelift::codegen::ir::FuncRef,
    bind_contract_helper: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    root_map: u32,
    function_index: u32,
    index: usize,
    site: &NativeProfiledSmallMapSite,
    fast_alloc: Option<GcFastAllocationInfo>,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let field_count = site.field_count as usize;

    // Program-string handles cannot be recycled. Guard their exact tagged words
    // so a polymorphic/dynamic-key fused site safely bails to the baseline path.
    for i in 0..field_count {
        if !site.guard_keys { break; }
        let actual = load_reg(b, regs, site.key_regs[i]);
        let expected = b.ins().iconst(types::I64, site.key_words[i] as i64);
        let matches = b.ins().icmp(IntCC::Equal, actual, expected);
        let ok = b.create_block();
        b.ins().brif(matches, ok, &[], abort, &[]);
        b.switch_to_block(ok);
        b.seal_block(ok);
    }

    emit_native_root_map_update(
        b,
        vm,
        regs,
        root_map,
        index as u32,
    );

    let storage = b.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        std::mem::size_of::<InlineMapValues>() as u32,
        3,
    ));
    let values = b.ins().stack_addr(types::I64, storage, 0);
    let flags = MemFlagsData::trusted();
    // Materialize the final slot order once in generated code. The site has
    // already validated every slot; neither helper needs to unpack/check it
    // again for every node. Initialize padding and unused fields as well.
    let zero = b.ins().iconst(types::I64, 0);
    b.ins().store(flags, zero, values, offset_of!(InlineMapValues, len) as i32);
    let len = b.ins().iconst(types::I8, site.value_count as i64);
    b.ins().store(flags, len, values, offset_of!(InlineMapValues, len) as i32);
    let null = b.ins().iconst(types::I64, TAG_NULL as i64);
    for slot in 0..SMALL_MAP_INLINE_FIELDS {
        // Duplicate literal keys retain their last value without duplicate
        // runtime stores. All initializer expressions have already executed.
        let value = if site.null_fields {
            null
        } else {
            (0..field_count).rfind(|&i| site.slots[i] as usize == slot)
                .map_or(null, |i| load_reg(b, regs, site.value_regs[i]))
        };
        b.ins().store(flags, value, values,
            (offset_of!(InlineMapValues, values) + slot * 8) as i32);
    }

    let shape = b.ins().iconst(types::I32, site.shape as i64);

    let value = if let Some(fast) = fast_alloc {
        debug_assert!(fast.alignment.is_power_of_two());
        debug_assert!(fast.bytes > 0);

        let fast_block = b.create_block();
        let slow_block = b.create_block();
        let commit_block = b.create_block();
        let done_block = b.create_block();
        b.append_block_param(commit_block, types::I64); // reserved object address
        b.append_block_param(done_block, types::I64); // tagged Napalm result

        // Load the current Immix TLAB state directly. MMTk's precise-stress mode
        // deliberately makes the limit fail this check, so it naturally reaches
        // the ordinary allocation helper. A concurrent collection request may
        // wait until this finite TLAB is exhausted; no GC can inspect the mutator
        // until it subsequently enters a safepoint/slow path.
        b.ins().jump(fast_block, &[]);
        b.switch_to_block(fast_block);
        let cursor_ptr = b.ins().iconst(types::I64, fast.cursor as usize as i64);
        let limit_ptr = b.ins().iconst(types::I64, fast.limit as usize as i64);
        let cursor = b.ins().load(types::I64, flags, cursor_ptr, 0);
        let limit = b.ins().load(types::I64, flags, limit_ptr, 0);
        let biased = b.ins().iadd_imm(cursor, (fast.alignment - 1) as i64);
        let align_mask = b.ins().iconst(types::I64, -(fast.alignment as i64));
        let aligned = b.ins().band(biased, align_mask);
        let new_cursor = b.ins().iadd_imm(aligned, fast.bytes as i64);
        let fits = b
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, new_cursor, limit);
        let without_end_overflow = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, new_cursor, aligned);
        let without_alignment_overflow = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, biased, cursor);
        let fits = b.ins().band(fits, without_end_overflow);
        let fits = b.ins().band(fits, without_alignment_overflow);
        b.ins().brif(
            fits,
            commit_block,
            &[BlockArg::Value(aligned)],
            slow_block,
            &[],
        );

        b.switch_to_block(commit_block);
        let reserved = b.block_params(commit_block)[0];
        let committed_cursor = b.ins().iadd_imm(reserved, fast.bytes as i64);
        b.ins().store(flags, committed_cursor, cursor_ptr, 0);
        let call = b.ins().call(
            commit_helper,
            &[vm, reserved, shape, values],
        );
        let committed = b.inst_results(call)[0];
        b.ins().jump(done_block, &[BlockArg::Value(committed)]);

        b.switch_to_block(slow_block);
        let call = b
            .ins()
            .call(slow_helper, &[vm, shape, values]);
        let allocated = b.inst_results(call)[0];
        b.ins().jump(done_block, &[BlockArg::Value(allocated)]);

        b.seal_block(fast_block);
        b.seal_block(commit_block);
        b.seal_block(slow_block);
        b.seal_block(done_block);
        b.switch_to_block(done_block);
        b.block_params(done_block)[0]
    } else {
        let call = b
            .ins()
            .call(slow_helper, &[vm, shape, values]);
        b.inst_results(call)[0]
    };

    let value = if site.bind_contract {
        // The complete object is published by the map helper before contract
        // validation. On failure the helper returns BAILOUT; baseline replays
        // the allocation and reports the original structural-type error.
        let function = b.ins().iconst(types::I32, function_index as i64);
        let op = b.ins().iconst(types::I32, index as i64);
        let call = b.ins().call(bind_contract_helper, &[vm, function, op, value]);
        b.inst_results(call)[0]
    } else {
        value
    };

    let failed = b.ins().icmp_imm(IntCC::Equal, value, WFJIT_BAILOUT as i64);
    let ok = b.create_block();
    b.append_block_param(ok, types::I64);
    b.ins().brif(failed, abort, &[], ok, &[BlockArg::Value(value)]);
    b.seal_block(ok);
    b.switch_to_block(ok);
    b.block_params(ok)[0]
}

fn call_native_value_helper(
    b: &mut FunctionBuilder<'_>,
    helper: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    function_index: u32,
    op_index: u32,
    regs: cranelift::prelude::Value,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let function = b.ins().iconst(types::I32, function_index as i64);
    let op = b.ins().iconst(types::I32, op_index as i64);
    let call = b.ins().call(helper, &[vm, function, op, regs]);
    let value = b.inst_results(call)[0];
    let failed = b.ins().icmp_imm(IntCC::Equal, value, WFJIT_BAILOUT as i64);
    let ok = b.create_block();
    b.append_block_param(ok, types::I64);
    b.ins().brif(failed, abort, &[], ok, &[BlockArg::Value(value)]);
    b.seal_block(ok);
    b.switch_to_block(ok);
    b.block_params(ok)[0]
}

/// Compile a target after it has become hot and execute it immediately if a
/// native entry is available. Auto-memoized targets are never compiled: their
/// normal call path remains authoritative.
pub(super) unsafe fn try_direct_call(
    vm: &mut Interpreter<'_>,
    op: &PreparedOp,
    sources: &[RegisterWord],
) -> bool {
    let Some(entry) = (unsafe { compiled_entry(vm, op.x) }) else { return false; };
    unsafe { execute_native_direct_call(vm, *op, sources, op.x, entry) };
    true
}

/// The interpreter has just pushed the frame of `target` through an ordinary
/// entry: a bytecode method dispatched by Invoke/InvokeMethodDirect, or the
/// constructor of `new C(...)` (receiver = new instance, no result register).
/// Run that frame natively when the target is compiled (or has just become
/// hot). Nothing happens if no frame was pushed (async, memo hit, error). On a
/// bailout the frame stays in place and baseline dispatch resumes it.
pub(super) unsafe fn try_run_pushed_call(
    vm: &mut Interpreter<'_>,
    target: u32,
    caller_fiber: u32,
    caller_depth: usize,
) {
    if vm.run_state != RunState::Running
        || vm.current_fiber != caller_fiber
        || vm.fiber(caller_fiber).frames.len() != caller_depth + 1
        || vm.hot_function_index != target
    {
        return;
    }
    let Some(entry) = (unsafe { compiled_entry(vm, target) }) else { return; };
    unsafe { run_pushed_frame(vm, caller_fiber, caller_depth, target, entry) };
}

/// Exact-class SetFieldClass sites of `target` whose store needs no contract
/// check (untyped field, or proven by the compiler).
fn plain_store_classes_for(vm: &Interpreter<'_>, target: u32) -> crate::napalm_collections::HashMap<usize, u64> {
    let mut classes = crate::napalm_collections::HashMap::default();
    // Reactivity never turns off again: with it on, every store must notify,
    // so a native store would only deoptimize. Use the runtime op instead.
    if vm.reactive_enabled { return classes; }
    let prepared = &vm.prepared.functions[target as usize];
    let bytecode = &vm.program.functions[target as usize];
    for (index, op) in prepared.ops.iter().enumerate() {
        if bytecode.chunk.code[prepared.raw_pcs[index] as usize].opcode() != Some(Opcode::SetFieldClass) { continue; }
        let Some(crate::napalm_bytecode::SpecializedDescriptor::FieldClass { class, unchecked_store, .. }) =
            bytecode.chunk.specializations.get(prepared.original_ops[index].x as usize)
        else { continue; };
        let Some(&class_slot) = vm.prepared.class_by_id.get(class) else { continue; };
        let Some(info) = vm.classes.get(class_slot as usize) else { continue; };
        if !*unchecked_store
            && (info.instance_field_types.get(op.x as usize) != Some(&Type::Top)
                || info.generic_fields.get(op.x as usize) != Some(&false))
        { continue; }
        classes.insert(index, CLASS_KIND_INSTANCE | class_slot as u64);
    }
    classes
}

/// Builder view of one function (borrowed from the VM for one compilation).
///
/// # Safety
/// The VM's prepared functions, programs and constants must not change while
/// the result is alive (true for the duration of a synchronous compile).
unsafe fn wf_function_data(vm: &Interpreter<'_>, index: u32) -> wf_ir::WfFunctionData<'static> {
    let function: &'static PreparedFunction = unsafe { &*vm.prepared.functions.as_ptr().add(index as usize) };
    let bytecode: &'static BytecodeFunction = unsafe { &*(&vm.program.functions[index as usize] as *const BytecodeFunction) };
    let constants: &'static [RegisterWord] = unsafe { &*(vm.resolved_constants[index as usize].as_slice() as *const [RegisterWord]) };
    wf_ir::WfFunctionData {
        function_index: index,
        function,
        bytecode,
        constants,
        reachable: reachable_ops(function, bytecode),
        identity_sites: native_identity_sites(vm, index),
        plain_store_classes: plain_store_classes_for(vm, index),
    }
}

/// Functions `target` may inline: direct calls, resolvable direct methods and
/// profiled monomorphic Invokes, transitively up to the inlining depth.
///
/// # Safety
/// Same contract as `wf_function_data`.
unsafe fn collect_inline_callees(
    vm: &Interpreter<'_>,
    target: u32,
    class_slots: &crate::napalm_collections::HashMap<ClassId, u32>,
    method_targets: &crate::napalm_collections::HashMap<(ReceiverClass, MethodSymbol), CallTarget>,
) -> crate::napalm_collections::HashMap<u32, wf_ir::WfFunctionData<'static>> {
    let mut callees = crate::napalm_collections::HashMap::default();
    let mut frontier = vec![target];
    for _ in 0..3 {
        let mut next = Vec::new();
        for function_index in frontier {
            let prepared = &vm.prepared.functions[function_index as usize];
            let bytecode = &vm.program.functions[function_index as usize];
            for (index, op) in prepared.ops.iter().enumerate() {
                let candidate = match bytecode.chunk.code[prepared.raw_pcs[index] as usize].opcode() {
                    Some(Opcode::CallDirect) => Some(op.x),
                    Some(Opcode::InvokeMethodDirect) => match bytecode.chunk.specializations.get(op.x as usize) {
                        Some(crate::napalm_bytecode::SpecializedDescriptor::DirectMethod { owner, symbol, .. }) =>
                            class_slots.get(owner).and_then(|&slot| {
                                let key = ReceiverClass(CLASS_KIND_INSTANCE | slot as u64);
                                (method_targets.get(&(key, *symbol)) == Some(&CallTarget::Bytecode(op.y))).then_some(op.y)
                            }),
                        _ => None,
                    },
                    Some(Opcode::Invoke) => profiled_direct_invoke_site(prepared, index).map(|(_, callee)| callee),
                    _ => None,
                };
                let Some(callee) = candidate else { continue; };
                if callee == target || callees.contains_key(&callee) || callees.len() >= 24 { continue; }
                if (callee as usize) >= vm.prepared.functions.len() { continue; }
                callees.insert(callee, unsafe { wf_function_data(vm, callee) });
                next.push(callee);
            }
        }
        frontier = next;
    }
    callees
}

/// Count a call to `target`, compiling it once it becomes hot, and return its
/// native entry if one is installed. Auto-memoized targets never compile.
unsafe fn compiled_entry(vm: &mut Interpreter<'_>, target: u32) -> Option<WholeFunctionEntry> {
    let jit_ptr = vm.whole_function_jit.as_mut().map(|jit| &mut **jit as *mut WholeFunctionJit)?;
    // Install whatever the background compiler finished (no-op when idle).
    unsafe { (*jit_ptr).poll_background(false); }
    // Settled functions (the common case on hot paths) need one state check:
    // rejected ones never compile again, compiled ones just return their entry.
    let settled = unsafe { (*jit_ptr).slots.get(target as usize).map(|slot| (slot.state, slot.entry)) };
    match settled {
        None | Some((FunctionState::Rejected, _)) | Some((FunctionState::Compiling, _)) => return None,
        Some((FunctionState::Compiled, entry)) => {
            return if unsafe { (*jit_ptr).active_depth } < MAX_NATIVE_CALL_DEPTH { entry } else { None };
        }
        Some((FunctionState::Cold, _)) => {}
    }
    if vm.prepared.functions[target as usize].memoization.enabled() {
        // Settle the decision so later calls skip this path entirely.
        unsafe { (*jit_ptr).reject(target) };
        return None;
    }
    if unsafe { (*jit_ptr).active_depth >= MAX_NATIVE_CALL_DEPTH } {
        return None;
    }

    let ops = vm.prepared.functions[target as usize].ops.len();
    if unsafe { (*jit_ptr).note_call(target, ops) } {
        let native_direct_method_class_slots: crate::napalm_collections::HashMap<ClassId, u32> =
            vm.prepared.class_by_id.iter().map(|(id, slot)| (*id, *slot)).collect();
        let mut native_direct_method_targets = crate::napalm_collections::HashMap::default();
        for (&owner, &class_slot) in &native_direct_method_class_slots {
            let class = ReceiverClass(CLASS_KIND_INSTANCE | class_slot as u64);
            for (symbol, method) in vm.classes[class_slot as usize].instance_methods.iter().enumerate() {
                if let Some(target) = method {
                    native_direct_method_targets.insert((class, MethodSymbol(symbol as u32)), *target);
                }
            }
        }
        let native_plain_store_classes = plain_store_classes_for(vm, target);
        let small_map_fast_alloc = vm.heap.inline_map_fast_allocation_info();
        let identity_sites = native_identity_sites(vm, target);
        let inline_callees = unsafe { collect_inline_callees(vm, target,
            &native_direct_method_class_slots, &native_direct_method_targets) };
        let manifest = vm.concurrent_manifest.as_ref().map(|manifest| manifest as *const crate::napalm_builtin::BuiltinManifest);
        let prepared = vm.prepared.functions.as_ptr().add(target as usize);
        let bytecode = &vm.program.functions[target as usize] as *const BytecodeFunction;
        let constants = &vm.resolved_constants[target as usize] as *const Vec<RegisterWord>;
        if let Err(error) = unsafe {
            (*jit_ptr).compile(
                target,
                &*prepared,
                &*bytecode,
                &*constants,
                vm.method_epoch,
                &identity_sites,
                small_map_fast_alloc,
                &native_direct_method_class_slots,
                &native_direct_method_targets,
                &native_plain_store_classes,
                &inline_callees,
                manifest.map(|manifest| unsafe { &*manifest }),
                vm.reactive_enabled,
            )
        } {
            unsafe { (*jit_ptr).reject(target); }
            if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
                eprintln!("[napalm-wfjit] rejected {:?}: {error}", (&*prepared).id);
            }
        } else if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
            if unsafe { (*jit_ptr).entry(target) }.is_some() {
                eprintln!("[napalm-wfjit] compiled {:?}", unsafe { (&*prepared).id });
            }
        }
    }

    unsafe { (*jit_ptr).entry(target) }
}

unsafe fn execute_native_direct_call(
    vm: &mut Interpreter<'_>,
    op: PreparedOp,
    sources: &[RegisterWord],
    target: u32,
    entry: WholeFunctionEntry,
) {
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    // Method-like targets take `sources[0]` as their receiver: the ordinary
    // entry stores it in SpecialReg::This and in its GPR (entry_this_dst).
    vm.enter_direct_bytecode_call(&op, target, sources, None);
    if vm.run_state != RunState::Running { return; }

    // A memo hit or async call completes without pushing a frame. This is kept
    // here even though compiled targets themselves are never memoized, because
    // it makes the call helper safe for future cross-target specialization.
    if vm.current_fiber != caller_fiber || vm.fiber(caller_fiber).frames.len() == caller_depth {
        return;
    }
    if vm.hot_function_index != target { return; }
    unsafe { run_pushed_frame(vm, caller_fiber, caller_depth, target, entry) };
}

/// Execute the already-pushed top frame of `target` natively. On success the
/// frame returns through the ordinary path (honouring its return register, or
/// none for constructors); on a bailout it is left for baseline dispatch.
unsafe fn run_pushed_frame(
    vm: &mut Interpreter<'_>,
    caller_fiber: u32,
    caller_depth: usize,
    target: u32,
    entry: WholeFunctionEntry,
) {
    {
        let jit = vm.whole_function_jit.as_mut().unwrap();
        jit.stats.native_entries = jit.stats.native_entries.saturating_add(1);
        jit.active_depth = jit.active_depth.saturating_add(1);
    }
    let result = unsafe { entry(vm as *mut _ as *mut c_void, vm.hot_registers) };
    if let Some(jit) = vm.whole_function_jit.as_mut() {
        jit.active_depth = jit.active_depth.saturating_sub(1);
        jit.note_native_result(target, result == WFJIT_BAILOUT);
    }
    if result == WFJIT_BAILOUT { trace_bailout(vm, target); }
    if result == WFJIT_BAILOUT {
        // The native body deliberately handed control back to the interpreter at
        // the current bytecode PC. Leave the callee frame intact; the outer
        // token-threaded loop resumes it normally.
        return;
    }
    if vm.run_state == RunState::Running
        && vm.current_fiber == caller_fiber
        && vm.fiber(caller_fiber).frames.len() == caller_depth + 1
        && vm.hot_function_index == target
    {
        vm.return_from_frame(RegisterWord(result), None);
    }
}


#[inline(always)]
unsafe fn wfjit_set_native_root_map(
    vm: &mut Interpreter<'_>,
    function_index: u32,
    op_index: u32,
    registers: *mut RegisterWord,
) -> bool {
    let Some(function) = vm.prepared.functions.get(function_index as usize) else {
        return false;
    };
    let Some(&root_map) = function.gc_map_by_op.get(op_index as usize) else {
        return false;
    };
    if root_map == NO_INDEX {
        return false;
    }

    if registers == vm.hot_registers {
        if vm.hot_frame.is_null() || vm.hot_function_index != function_index {
            return false;
        }
        // Keep the ordinary Fiber frame precise while the top-level native body
        // is active. hot_pc is also the exact replay point if a nested native
        // speculation bails out.
        unsafe { (*vm.hot_frame).root_map = root_map; }
        vm.hot_pc = op_index;
        return true;
    }

    let shadow = vm.native_shadow_top;
    if shadow.is_null() {
        return false;
    }
    let frame = unsafe { &mut *shadow };
    if frame.registers != registers || frame.function != function_index {
        return false;
    }
    frame.root_map = root_map;
    true
}

#[inline(always)]
unsafe fn wfjit_call_source_reg(
    function: &PreparedFunction,
    op: &PreparedOp,
    index: usize,
) -> Option<u8> {
    if index >= op.c as usize {
        return None;
    }
    if op.c <= 4 {
        Some(((op.args_offset >> (index * 8)) & 0xff) as u8)
    } else {
        function
            .call_args
            .get(op.args_offset as usize + index)
            .map(|reg| reg.0)
    }
}

fn allocate_prepared_small_map_literal_from_regs<const N: usize>(
    vm: &mut Interpreter<'_>,
    function_index: u32,
    op: &PreparedOp,
    registers: *mut RegisterWord,
) -> Result<Value, RuntimeError> {
    debug_assert_eq!(op.c as usize, N);
    let function_ptr = unsafe {
        vm.prepared
            .functions
            .as_mut_ptr()
            .add(function_index as usize)
    };
    let offset = op.args_offset as usize;
    let mut pairs = [(Value::NULL, Value::NULL); N];
    let args = unsafe { &(*function_ptr).call_args };

    for (index, pair) in pairs.iter_mut().enumerate() {
        let key_reg = unsafe { args.get_unchecked(offset + index * 2).0 };
        let value_reg = unsafe { args.get_unchecked(offset + index * 2 + 1).0 };
        *pair = (
            Value::from_word(unsafe { *registers.add(key_reg as usize) }),
            Value::from_word(unsafe { *registers.add(value_reg as usize) }),
        );
    }

    let site_index = op.cache as usize;
    if site_index >= unsafe { (*function_ptr).small_map_literal_sites.len() } {
        return Err(RuntimeError::new("invalid whole-function small Map literal site"));
    }
    let site = unsafe {
        (*function_ptr)
            .small_map_literal_sites
            .as_mut_ptr()
            .add(site_index)
    };
    if unsafe { !(*site).initialized } {
        let stable = pairs.iter().all(|(key, _)| {
            vm.strings
                .iter()
                .any(|candidate| candidate.raw_bits() == key.raw_bits())
        });
        unsafe { (*site).stable_program_keys = stable; }
    }

    vm.heap.alloc_map_from_site(pairs, unsafe { &mut *site })
}

unsafe extern "C" fn napalm_wfjit_profiled_small_map_commit(
    vm: *mut c_void,
    reserved_address: u64,
    shape: u32,
    source_values: *const InlineMapValues,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running
        || reserved_address == 0
        || source_values.is_null()
    {
        return WFJIT_BAILOUT;
    }

    // The generated caller initializes the complete repr(C), fixed-capacity
    // value array using a compile-time validated allocation-site layout.
    let values = unsafe { source_values.read() };
    debug_assert!(values.len() <= SMALL_MAP_INLINE_FIELDS);

    unsafe {
        vm.heap
            .alloc_inline_map_reserved(reserved_address as usize, shape, values)
            .raw_bits()
    }
}

unsafe extern "C" fn napalm_wfjit_profiled_small_map(
    vm: *mut c_void,
    shape: u32,
    source_values: *const InlineMapValues,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running
        || source_values.is_null()
    {
        return WFJIT_BAILOUT;
    }

    let values = unsafe { source_values.read() };
    debug_assert!(values.len() <= SMALL_MAP_INLINE_FIELDS);

    vm.heap.alloc_inline_map(shape, values).raw_bits()
}

unsafe extern "C" fn napalm_wfjit_small_map_literal_regs(
    vm: *mut c_void,
    function_index: u32,
    op_index: u32,
    registers: *mut RegisterWord,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running
        || registers.is_null()
        || !unsafe { wfjit_set_native_root_map(vm, function_index, op_index, registers) }
    {
        return WFJIT_BAILOUT;
    }

    // Heap allocation polls GC after protecting the copied operands. These are
    // stable handles, so a second poll before copying them is unnecessary.

    let Some(function) = vm.prepared.functions.get(function_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(&op) = function.ops.get(op_index as usize) else {
        return WFJIT_BAILOUT;
    };

    let result = match op.c {
        1 => allocate_prepared_small_map_literal_from_regs::<1>(
            vm, function_index, &op, registers,
        ),
        2 => allocate_prepared_small_map_literal_from_regs::<2>(
            vm, function_index, &op, registers,
        ),
        3 => allocate_prepared_small_map_literal_from_regs::<3>(
            vm, function_index, &op, registers,
        ),
        4 => allocate_prepared_small_map_literal_from_regs::<4>(
            vm, function_index, &op, registers,
        ),
        _ => return WFJIT_BAILOUT,
    };

    result.map_or(WFJIT_BAILOUT, Value::raw_bits)
}

/// Install the structural contract after a fast native TLAB map allocation.
/// Contract checks may reject a runtime value; the native caller bails out and
/// executes the authoritative baseline error path without exposing the object.
unsafe extern "C" fn napalm_wfjit_bind_factory_contract(
    vm: *mut c_void,
    function_index: u32,
    op_index: u32,
    raw: u64,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || raw == WFJIT_BAILOUT {
        return WFJIT_BAILOUT;
    }
    let Some(function) = vm.prepared.functions.get(function_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(op) = function.ops.get(op_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let descriptor = vm.program.functions.get(function_index as usize)
        .and_then(|function| function.chunk.specializations.get(op.x as usize));
    let Some(crate::napalm_bytecode::SpecializedDescriptor::ObjectShape {
        contract: Some(contract), ..
    }) = descriptor else {
        return WFJIT_BAILOUT;
    };
    if vm.heap.bind_contract(unsafe { Value::from_raw_bits_unchecked(raw) }, contract).is_err() {
        return WFJIT_BAILOUT;
    }
    raw
}

/// Publish a zero-capacity typed array in the Immix range reserved by
/// generated WFJIT code. This cannot trigger a managed GC collection.
unsafe extern "C" fn napalm_wfjit_commit_reserved_typed_array(
    vm: *mut c_void,
    function_index: u32,
    op_index: u32,
    reserved: u64,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if reserved == 0 || vm.run_state != RunState::Running {
        return WFJIT_BAILOUT;
    }
    let Some(function) = vm.prepared.functions.get(function_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(&op) = function.ops.get(op_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(crate::napalm_bytecode::SpecializedDescriptor::ArrayType { element, .. }) = vm
        .program.functions.get(function_index as usize)
        .and_then(|f| f.chunk.specializations.get(op.x as usize))
    else { return WFJIT_BAILOUT; };
    let contract = Type::List(Some(Box::new(element.clone())));
    let value = unsafe {
        vm.heap.alloc_reserved_managed(reserved as usize, HeapObject::List(Vec::new()))
    };
    if vm.heap.bind_contract(value, &contract).is_err() {
        return WFJIT_BAILOUT;
    }
    value.raw_bits()
}

unsafe extern "C" fn napalm_wfjit_specialized_factory(
    vm: *mut c_void,
    function_index: u32,
    op_index: u32,
    registers: *mut RegisterWord,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || registers.is_null()
        || !unsafe { wfjit_set_native_root_map(vm, function_index, op_index, registers) }
    {
        return WFJIT_BAILOUT;
    }
    let Some(function) = vm.prepared.functions.get(function_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(&op) = function.ops.get(op_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(opcode) = vm.program.functions.get(function_index as usize)
        .and_then(|f| f.chunk.code.get(function.raw_pcs[op_index as usize] as usize))
        .and_then(|word| word.opcode()) else {
        return WFJIT_BAILOUT;
    };
    // Native-recursive shadow frames use a separate GPR bank. The object
    // factory reads the normal VM registers, so require the top-level frame.
    if registers != vm.hot_registers { return WFJIT_BAILOUT; }
    let result = match opcode {
        Opcode::NewObjectEmpty | Opcode::NewObjectShape
        | Opcode::NewObjectWithProperties | Opcode::NewInterfaceOrTypeObject =>
            allocate_specialized_object(vm, &op, opcode),
        Opcode::DefineArrayWithGenerics => {
            let descriptor = vm.program.functions[function_index as usize]
                .chunk.specializations.get(op.x as usize);
            let Some(crate::napalm_bytecode::SpecializedDescriptor::ArrayType {
                element, with_capacity,
            }) = descriptor else { return WFJIT_BAILOUT; };
            let capacity = if *with_capacity {
                let count = word_i32(unsafe { *registers.add(op.b as usize) });
                if count < 0 { return WFJIT_BAILOUT; }
                count as usize
            } else { 0 };
            let contract = Type::List(Some(Box::new(element.clone())));
            let result = vm.heap.alloc(HeapObject::List(Vec::with_capacity(capacity)));
            vm.heap.bind_contract(result, &contract).map(|()| result)
        },
        _ => return WFJIT_BAILOUT,
    };
    result.map_or(WFJIT_BAILOUT, Value::raw_bits)
}

/// Cold-path feedback for a failed speculative guard of `function` at
/// prepared op `op` (`ENTRY_SPECULATION_SITE` for the entry's parameter type
/// checks). Only touches the JIT's bookkeeping; never the frame, registers or
/// heap.
unsafe extern "C" fn napalm_wfjit_speculation_failed(vm: *mut c_void, function: u32, op: u32) {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'_>) };
    if let Some(jit) = vm.whole_function_jit.as_mut() {
        jit.note_speculation_failure(function, op);
    }
}

unsafe extern "C" fn napalm_wfjit_direct_call(vm: *mut c_void, op_index: u32) -> *mut RegisterWord {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    count_helper(vm, "direct", op_index);
    if vm.run_state != RunState::Running || vm.hot_frame.is_null() { return std::ptr::null_mut(); }
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    let caller_function = vm.hot_function_index;
    let op = unsafe { *vm.current_prepared().ops.get_unchecked(op_index as usize) };
    let mut words = [RegisterWord(0); 17];
    let count = if op.c <= 4 {
        // Small-call register IDs are packed directly in PreparedOp::args_offset.
        // Treating args_offset as a side-table offset here corrupts recursive
        // arguments (for example createTree(d)). Reuse the interpreter ABI.
        let mut small = [RegisterWord(0); 4];
        let count = unsafe { vm.gather_call_words_small(&op, &mut small) };
        words[..count].copy_from_slice(&small[..count]);
        count
    } else {
        unsafe { vm.gather_call_words_large(&op, &mut words) }
    };
    let target = op.x;
    unsafe { vm.set_pc(op_index); }

    // Always use the ordinary call entry. This is the memoization firewall: if a
    // future native caller reaches an auto-memoized target, probe/miss bookkeeping
    // and return caching still happen exactly once in the existing runtime path.
    vm.enter_direct_bytecode_call(&op, target, &words[..count], None);
    if vm.run_state != RunState::Running { return std::ptr::null_mut(); }
    if vm.current_fiber != caller_fiber { return std::ptr::null_mut(); }
    if vm.fiber(caller_fiber).frames.len() == caller_depth {
        return vm.hot_registers;
    }

    // Native callers must contribute hotness too. Otherwise compiling the
    // caller can strand a callee just below the compilation threshold forever.
    // The ordinary entry above has already rooted its arguments and receiver.
    let native = unsafe { compiled_entry(vm, target) };
    if let Some(entry) = native {
        if let Some(jit) = vm.whole_function_jit.as_mut() {
            jit.stats.native_direct_calls = jit.stats.native_direct_calls.saturating_add(1);
            jit.stats.native_entries = jit.stats.native_entries.saturating_add(1);
            jit.active_depth = jit.active_depth.saturating_add(1);
        }
        let result = unsafe { entry(vm as *mut _ as *mut c_void, vm.hot_registers) };
        if let Some(jit) = vm.whole_function_jit.as_mut() {
            jit.active_depth = jit.active_depth.saturating_sub(1);
            jit.note_native_result(target, result == WFJIT_BAILOUT);
        }
        if result == WFJIT_BAILOUT { trace_bailout(vm, target); }
        if result == WFJIT_BAILOUT {
            // Finish this callee in baseline so the native caller can resume with
            // the same direct-call ABI and a freshly relocated register base.
            while vm.run_state == RunState::Running
                && vm.current_fiber == caller_fiber
                && vm.fiber(caller_fiber).frames.len() > caller_depth
            {
                let pc = vm.hot_pc as usize;
                let handler = unsafe { *vm.hot_handlers.add(pc) };
                let prepared_op = unsafe { vm.hot_ops.add(pc) };
                unsafe { handler(vm, prepared_op); }
            }
        } else if vm.run_state == RunState::Running
            && vm.current_fiber == caller_fiber
            && vm.fiber(caller_fiber).frames.len() == caller_depth + 1
            && vm.hot_function_index == target
        {
            vm.return_from_frame(RegisterWord(result), None);
        }
    } else {
        if let Some(jit) = vm.whole_function_jit.as_mut() {
            jit.stats.baseline_direct_calls = jit.stats.baseline_direct_calls.saturating_add(1);
        }
        while vm.run_state == RunState::Running
            && vm.current_fiber == caller_fiber
            && vm.fiber(caller_fiber).frames.len() > caller_depth
        {
            let pc = vm.hot_pc as usize;
            let handler = unsafe { *vm.hot_handlers.add(pc) };
            let prepared_op = unsafe { vm.hot_ops.add(pc) };
            unsafe { handler(vm, prepared_op); }
        }
    }

    if vm.run_state == RunState::Running
        && vm.current_fiber == caller_fiber
        && vm.fiber(caller_fiber).frames.len() == caller_depth
        && vm.hot_function_index == caller_function
    {
        vm.hot_registers
    } else {
        std::ptr::null_mut()
    }
}

/// Bytecode InvokeMethodDirect: the prepared op carries the target in Y.
unsafe extern "C" fn napalm_wfjit_direct_method(vm: *mut c_void, op_index: u32) -> *mut RegisterWord {
    let target = {
        let vm = unsafe { &*(vm as *const Interpreter<'static>) };
        if vm.run_state != RunState::Running || vm.hot_frame.is_null() { return std::ptr::null_mut(); }
        unsafe { vm.current_prepared().ops.get_unchecked(op_index as usize).y }
    };
    unsafe { wfjit_direct_method_call(vm, op_index, target) }
}

/// Runtime-profiled InvokeMethodDirect (a promoted Invoke): the target was the
/// profile's at compile time; native code already guarded class and epoch.
unsafe extern "C" fn napalm_wfjit_profiled_method(vm: *mut c_void, op_index: u32, target: u32) -> *mut RegisterWord {
    unsafe { wfjit_direct_method_call(vm, op_index, target) }
}

unsafe fn wfjit_direct_method_call(vm: *mut c_void, op_index: u32, target: u32) -> *mut RegisterWord {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    count_helper(vm, "method", op_index);
    if vm.run_state != RunState::Running || vm.hot_frame.is_null() { return std::ptr::null_mut(); }
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    let caller_function = vm.hot_function_index;
    let op = unsafe { *vm.current_prepared().ops.get_unchecked(op_index as usize) };
    let mut words = [RegisterWord(0); 17];
    let count = if op.c <= 4 {
        // Small-call register IDs are packed directly in PreparedOp::args_offset.
        // Treating args_offset as a side-table offset here corrupts recursive
        // arguments (for example createTree(d)). Reuse the interpreter ABI.
        let mut small = [RegisterWord(0); 4];
        let count = unsafe { vm.gather_call_words_small(&op, &mut small) };
        words[..count].copy_from_slice(&small[..count]);
        count
    } else {
        unsafe { vm.gather_call_words_large(&op, &mut words) }
    };
    let receiver = unsafe { vm.reg(op.b) };
    unsafe { vm.set_pc(op_index); }

    // The native caller already verified the method epoch and receiver class.
    // Always use the ordinary call entry. This is the memoization firewall: if a
    // future native caller reaches an auto-memoized target, probe/miss bookkeeping
    // and return caching still happen exactly once in the existing runtime path.
    vm.enter_bytecode_call(&op, target, Some(receiver), &words[..count], None);
    if vm.run_state != RunState::Running { return std::ptr::null_mut(); }
    if vm.current_fiber != caller_fiber { return std::ptr::null_mut(); }
    if vm.fiber(caller_fiber).frames.len() == caller_depth {
        return vm.hot_registers;
    }

    // Keep tiering methods even when all their callers are already native.
    // compiled_entry retains the memoization and native-depth guards.
    let native = unsafe { compiled_entry(vm, target) };
    if let Some(entry) = native {
        if let Some(jit) = vm.whole_function_jit.as_mut() {
            jit.stats.native_direct_calls = jit.stats.native_direct_calls.saturating_add(1);
            jit.stats.native_entries = jit.stats.native_entries.saturating_add(1);
            jit.active_depth = jit.active_depth.saturating_add(1);
        }
        let result = unsafe { entry(vm as *mut _ as *mut c_void, vm.hot_registers) };
        if let Some(jit) = vm.whole_function_jit.as_mut() {
            jit.active_depth = jit.active_depth.saturating_sub(1);
            jit.note_native_result(target, result == WFJIT_BAILOUT);
        }
        if result == WFJIT_BAILOUT { trace_bailout(vm, target); }
        if result == WFJIT_BAILOUT {
            // Finish this callee in baseline so the native caller can resume with
            // the same direct-call ABI and a freshly relocated register base.
            while vm.run_state == RunState::Running
                && vm.current_fiber == caller_fiber
                && vm.fiber(caller_fiber).frames.len() > caller_depth
            {
                let pc = vm.hot_pc as usize;
                let handler = unsafe { *vm.hot_handlers.add(pc) };
                let prepared_op = unsafe { vm.hot_ops.add(pc) };
                unsafe { handler(vm, prepared_op); }
            }
        } else if vm.run_state == RunState::Running
            && vm.current_fiber == caller_fiber
            && vm.fiber(caller_fiber).frames.len() == caller_depth + 1
            && vm.hot_function_index == target
        {
            vm.return_from_frame(RegisterWord(result), None);
        }
    } else {
        if let Some(jit) = vm.whole_function_jit.as_mut() {
            jit.stats.baseline_direct_calls = jit.stats.baseline_direct_calls.saturating_add(1);
        }
        while vm.run_state == RunState::Running
            && vm.current_fiber == caller_fiber
            && vm.fiber(caller_fiber).frames.len() > caller_depth
        {
            let pc = vm.hot_pc as usize;
            let handler = unsafe { *vm.hot_handlers.add(pc) };
            let prepared_op = unsafe { vm.hot_ops.add(pc) };
            unsafe { handler(vm, prepared_op); }
        }
    }

    if vm.run_state == RunState::Running
        && vm.current_fiber == caller_fiber
        && vm.fiber(caller_fiber).frames.len() == caller_depth
        && vm.hot_function_index == caller_function
    {
        vm.hot_registers
    } else {
        std::ptr::null_mut()
    }
}


/// `NAPALM_WFJIT_HELPERS=1`: count helper executions per (helper, opcode)
/// and print the histogram when the process exits (diagnostics).
#[inline(always)]
fn count_helper(vm: &Interpreter<'_>, helper: &'static str, op_index: u32) {
    if helper_counts_enabled() { count_helper_slow(vm, helper, op_index); }
}

#[cold]
#[inline(never)]
fn count_helper_slow(vm: &Interpreter<'_>, helper: &'static str, op_index: u32) {
    let function = vm.hot_function_index as usize;
    let opcode = vm.prepared.functions.get(function)
        .and_then(|prepared| prepared.raw_pcs.get(op_index as usize).copied())
        .and_then(|raw| vm.program.functions[function].chunk.code.get(raw as usize))
        .and_then(|word| word.opcode());
    let label = match (opcode, vm.prepared.functions.get(function).and_then(|p| p.quick_invokes.get(op_index as usize))) {
        (Some(Opcode::Invoke), Some(site)) => match site.target {
            CallTarget::Primitive(id) => format!("Invoke->{:?}",
                vm.concurrent_manifest.as_ref().and_then(|manifest| manifest.primitive_kind(id))),
            target => format!("Invoke->{target:?}").chars().take(60).collect(),
        },
        (opcode, _) => format!("{opcode:?}"),
    };
    if let Ok(mut guard) = HELPER_COUNTS.lock() {
        *guard.get_or_insert_with(Default::default).entry((helper, label)).or_insert(0) += 1;
    }
}

static HELPER_COUNTS: std::sync::Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> =
    std::sync::Mutex::new(None);

#[inline(always)]
fn helper_counts_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NAPALM_WFJIT_HELPERS").is_some())
}

impl Drop for WholeFunctionJit {
    fn drop(&mut self) {
        if !helper_counts_enabled() { return; }
        if let Ok(guard) = HELPER_COUNTS.lock() {
            if let Some(counts) = guard.as_ref() {
                let mut rows: Vec<_> = counts.iter().collect();
                rows.sort_by(|a, b| b.1.cmp(a.1));
                for ((helper, opcode), count) in rows.into_iter().take(30) {
                    eprintln!("[napalm-wfjit] helper {count:>9} {helper} {opcode}");
                }
            }
        }
    }
}

/// `NAPALM_WFJIT_BAILOUTS=1`: report where each native bailout resumes
/// (diagnostics for deopt storms).
#[cold]
fn trace_bailout(vm: &Interpreter<'_>, target: u32) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("NAPALM_WFJIT_BAILOUTS").is_some()) { return; }
    let function = vm.hot_function_index as usize;
    let pc = vm.hot_pc as usize;
    let opcode = vm.prepared.functions.get(function)
        .and_then(|prepared| prepared.raw_pcs.get(pc).copied())
        .and_then(|raw| vm.program.functions[function].chunk.code.get(raw as usize))
        .and_then(|word| word.opcode());
    eprintln!("[napalm-wfjit] bailout F{target} -> F{function}:{pc} {opcode:?}");
}

/// v2 `CallExplicit`: a call made from an inlined body. The root frame stands
/// at its outermost call op `root_op` (its live GPRs are in the bank, the
/// inlined frames are GC shadow frames). Enter `target` as that op would,
/// with explicit arguments, so the result lands in the op's destination
/// register (dead until the inlined call returns), run it to completion
/// (natively when compiled), and hand the result back through `out`.
/// Status: 0 done, 1 the interpreter took over (exception unwound into or
/// past the root frame...), 2 declined before doing anything.
unsafe extern "C" fn napalm_wfjit_call_explicit(
    vm: *mut c_void,
    root_op: u32,
    target: u32,
    receiver: u64,
    has_receiver: u32,
    args: *const u64,
    argc: u32,
    out: *mut u64,
) -> u32 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || vm.hot_frame.is_null() { return 2; }
    count_helper(vm, "explicit", root_op);
    // Async and concurrent targets take paths that can collect before the
    // arguments are copied into a frame: leave those calls to the interpreter.
    let Some(callee) = vm.prepared.functions.get(target as usize) else { return 2; };
    if callee.is_async || callee.is_concurrent || argc as usize > 16 { return 2; }
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    let caller_function = vm.hot_function_index;
    unsafe { vm.set_pc(root_op); }
    let op = unsafe { *vm.current_prepared().ops.get_unchecked(root_op as usize) };
    let mut words = [RegisterWord(0); 16];
    for index in 0..argc as usize {
        words[index] = RegisterWord(unsafe { *args.add(index) });
    }
    if has_receiver != 0 {
        vm.enter_bytecode_call(&op, target, Some(RegisterWord(receiver)), &words[..argc as usize], None);
    } else {
        // Like CallDirect: a method target takes `this` from the first source.
        vm.enter_direct_bytecode_call(&op, target, &words[..argc as usize], None);
    }
    unsafe { finish_explicit_call(vm, op, Some(target), caller_fiber, caller_depth, caller_function, out) }
}

/// v2 `CallExplicit` with a dispatched target: an `Invoke` of an inlined
/// body (function `function`) whose receiver class is not known statically.
/// The method is looked up through that op's own inline cache (filling it on
/// a miss, like the interpreter), then invoked exactly as the interpreter
/// would (bytecode, host primitive, native builtin or foreign method), with
/// the root frame's outermost call op `root_op` as the carrier: its
/// destination register receives the result. Same status protocol as
/// `napalm_wfjit_call_explicit`; a missing method declines (2), so the
/// interpreter replays the op with its own error or fallback.
unsafe extern "C" fn napalm_wfjit_invoke_explicit(
    vm: *mut c_void,
    root_op: u32,
    function: u32,
    ic_slot: u32,
    receiver: u64,
    args: *const u64,
    argc: u32,
    out: *mut u64,
) -> u32 {
    use crate::napalm_ir::MethodKind;
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || vm.hot_frame.is_null() || argc as usize > 16 { return 2; }
    count_helper(vm, "invoke", root_op);
    let receiver_word = RegisterWord(receiver);
    let Ok(class) = vm.receiver_class(Value::from_word(receiver_word)) else { return 2; };
    let cached = vm.inline_caches.get(function as usize)
        .and_then(|caches| caches.get(ic_slot as usize))
        .and_then(|cache| cache.probe(class));
    let target = match cached {
        Some(target) => target,
        None => match vm.resolve_invoke_miss(function, ic_slot, class) {
            Ok(target) => target,
            Err(_) => return 2,
        },
    };
    let mut words = [RegisterWord(0); 16];
    for index in 0..argc as usize {
        words[index] = RegisterWord(unsafe { *args.add(index) });
    }
    let words = &words[..argc as usize];
    match target {
        CallTarget::Bytecode(callee) => {
            // Same restriction as the static path.
            let Some(callee) = vm.prepared.functions.get(callee as usize) else { return 2; };
            if callee.is_async || callee.is_concurrent { return 2; }
        }
        CallTarget::Foreign(_) => {}
        _ => {
            // `anything + String` falls back to concatenation when no real
            // overload exists: leave that case to the interpreter's own op.
            let is_string = |word: RegisterWord| matches!(vm.heap.get(Value::from_word(word)), Some(HeapObject::String(_)));
            let plus = vm.program.functions.get(function as usize)
                .and_then(|bytecode| bytecode.chunk.inline_caches.get(ic_slot as usize))
                .and_then(|descriptor| vm.program.metadata.method(descriptor.method))
                .is_some_and(|signature| matches!(signature.kind, MethodKind::Method)
                    && vm.program.metadata.string(signature.name) == Some("+"));
            if plus && words.len() == 1 && is_string(words[0]) && !is_string(receiver_word) { return 2; }
        }
    }
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    let caller_function = vm.hot_function_index;
    unsafe { vm.set_pc(root_op); }
    let op = unsafe { *vm.current_prepared().ops.get_unchecked(root_op as usize) };
    // The carrier's own operands belong to the root's call op: no inline
    // cache lookup of the root function may read them as this Invoke's.
    let mut carrier = op;
    carrier.x = u32::MAX;
    vm.invoke_target(&carrier, target, receiver_word, words);
    let entered = match target { CallTarget::Bytecode(callee) => Some(callee), _ => None };
    unsafe { finish_explicit_call(vm, op, entered, caller_fiber, caller_depth, caller_function, out) }
}

/// Second half of an explicit call: run a bytecode callee that was entered
/// (natively when compiled), finish whatever the interpreter has left of it,
/// and report whether control is back at the carrier op's successor in the
/// caller frame (0, result in `out`) or the interpreter took over (1).
unsafe fn finish_explicit_call(
    vm: &mut Interpreter<'static>,
    op: PreparedOp,
    entered: Option<u32>,
    caller_fiber: u32,
    caller_depth: usize,
    caller_function: u32,
    out: *mut u64,
) -> u32 {
    let in_callee = |vm: &Interpreter<'static>| vm.run_state == RunState::Running
        && vm.current_fiber == caller_fiber
        && vm.fiber(caller_fiber).frames.len() > caller_depth;
    if let Some(target) = entered.filter(|&target| in_callee(vm) && vm.hot_function_index == target) {
        match unsafe { compiled_entry(vm, target) } {
            Some(entry) => {
                if let Some(jit) = vm.whole_function_jit.as_mut() {
                    jit.stats.native_entries = jit.stats.native_entries.saturating_add(1);
                    jit.active_depth = jit.active_depth.saturating_add(1);
                }
                let result = unsafe { entry(vm as *mut _ as *mut c_void, vm.hot_registers) };
                if let Some(jit) = vm.whole_function_jit.as_mut() {
                    jit.active_depth = jit.active_depth.saturating_sub(1);
                    jit.note_native_result(target, result == WFJIT_BAILOUT);
                }
                if result != WFJIT_BAILOUT {
                    if vm.run_state == RunState::Running && vm.current_fiber == caller_fiber
                        && vm.fiber(caller_fiber).frames.len() == caller_depth + 1
                        && vm.hot_function_index == target
                    {
                        vm.return_from_frame(RegisterWord(result), None);
                    }
                } else {
                    trace_bailout(vm, target);
                }
            }
            None => {}
        }
    }
    // Finish whatever remains of the callee (or of the frames a host method
    // pushed) in baseline.
    while in_callee(vm) {
        let pc = vm.hot_pc as usize;
        let handler = unsafe { *vm.hot_handlers.add(pc) };
        let prepared_op = unsafe { vm.hot_ops.add(pc) };
        unsafe { handler(vm, prepared_op); }
    }
    if vm.run_state == RunState::Running && vm.current_fiber == caller_fiber
        && vm.fiber(caller_fiber).frames.len() == caller_depth
        && vm.hot_function_index == caller_function
        && vm.hot_pc == op.next
    {
        unsafe { *out = vm.hot_registers.add(op.a as usize).read().0; }
        0
    } else {
        1
    }
}

/// v2 `InstanceFieldStoreChecked`: the class field store of `h_store_field`
/// (field type contract, generic field check, generational barrier) with
/// explicit operands. A leaf: contracts live in the handle sidecar, so
/// nothing allocates in the managed heap, collects or runs Napalm code.
/// Returns 0 when stored; 1 with nothing changed when the interpreter must
/// execute the op instead (it then reports the type error), including a
/// receiver that is not an instance and enabled reactivity (stores notify).
unsafe extern "C" fn napalm_wfjit_store_field_checked(vm: *mut c_void, object: u64, value: u64, slot: u32) -> u32 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || vm.reactive_enabled { return 1; }
    count_helper(vm, "store_field", slot);
    let object = Value::from_word(RegisterWord(object));
    let value = Value::from_word(RegisterWord(value));
    let index = slot as usize;
    let class_slot = match vm.heap.get(object) {
        Some(HeapObject::Instance(fields @ InstanceObject { class_slot, .. })) if index < fields.len() => *class_slot,
        _ => return 1,
    };
    let expected = vm.classes[class_slot as usize].instance_field_types.get(index).cloned().unwrap_or(Type::Top);
    if vm.enforce_runtime_type(value, &expected, "Class field type error").is_err() { return 1; }
    if vm.classes[class_slot as usize].generic_fields.get(index).copied().unwrap_or(false)
        && vm.check_generic_field_store(object, class_slot, index, value).is_err()
    {
        return 1;
    }
    vm.heap.remember_if_reference(object, value);
    let Some(HeapObject::Instance(fields)) = vm.heap.get_mut(object) else { return 1; };
    fields[index] = value;
    0
}

/// v2 `StringConcat` (inside an inlined body): `h_string_concat` with
/// explicit operands. It may collect: the caller published every root (the
/// root frame at its call op, the inlined frames as shadow frames).
/// `WFJIT_BAILOUT` when an operand is not a String (the interpreter then
/// replays the op and reports it).
unsafe extern "C" fn napalm_wfjit_string_concat(vm: *mut c_void, left: u64, right: u64) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running { return WFJIT_BAILOUT; }
    count_helper(vm, "string_concat", 0);
    let (left, right) = (Value::from_word(RegisterWord(left)), Value::from_word(RegisterWord(right)));
    let (a, b) = match (vm.heap.get(left), vm.heap.get(right)) {
        (Some(HeapObject::String(a)), Some(HeapObject::String(b))) => (a.clone(), b.clone()),
        _ => return WFJIT_BAILOUT,
    };
    vm.maybe_collect();
    vm.heap.alloc(HeapObject::String(NapalmString::concat(&a, &b))).raw_bits()
}

/// v2 pure leaf helpers for native builtin methods (`WfInst::Primitive`).
/// They never allocate, collect or run Napalm code, so the caller keeps its
/// values in registers (no materialization). They return the result as a
/// tagged word (0/1 for `string_eq`), or `WFJIT_BAILOUT` for any case they
/// leave to the interpreter (errors), which re-executes the op after a deopt.
unsafe extern "C" fn napalm_wfjit_string_eq(vm: *mut c_void, receiver: u64, other: u64) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    let context = HostContext { heap: &mut vm.heap };
    let equal = context.strings_equal(Value::from_word(RegisterWord(receiver)), Value::from_word(RegisterWord(other)));
    u64::from(equal.unwrap_or(false))
}

unsafe extern "C" fn napalm_wfjit_string_code_point_at(vm: *mut c_void, receiver: u64, index_bits: u64) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    let index = f64::from_bits(index_bits);
    if !index.is_finite() || index.fract() != 0.0 { return WFJIT_BAILOUT; }
    let index = index as i64;
    let receiver = Value::from_word(RegisterWord(receiver));
    let context = HostContext { heap: &mut vm.heap };
    if let Some(code) = context.string_code_point_fast(receiver, index) {
        return code.raw_bits();
    }
    // Same semantics as the interpreter's general path.
    let Some(text) = context.string_utf8(receiver) else { return WFJIT_BAILOUT; };
    let length = text.chars().count() as i64;
    let index = if index < 0 { length.saturating_add(index) } else { index };
    if index < 0 || index >= length { return Value::NULL.raw_bits(); }
    Value::from_num(text.chars().nth(index as usize).map_or(0, |c| c as u32) as f64).raw_bits()
}

unsafe extern "C" fn napalm_wfjit_bytes_index(vm: *mut c_void, receiver: u64, index_bits: u64) -> u64 {
    let vm = unsafe { &*(vm as *const Interpreter<'static>) };
    let Some(HeapObject::Bytes(bytes)) = vm.heap.get(Value::from_word(RegisterWord(receiver))) else {
        return WFJIT_BAILOUT;
    };
    let Ok(index) = crate::napalm_bytes::number_to_length(f64::from_bits(index_bits)) else { return WFJIT_BAILOUT; };
    bytes.at(index).map_or(WFJIT_BAILOUT, |byte| Value::from_num(byte as f64).raw_bits())
}

/// Inlined frames a v2 deopt rebuilds (outermost first); see
/// `WfDeoptState::frames`. Leaked per deopt site of a compiled body.
pub(super) struct WfReconstruct {
    pub(super) frames: Vec<WfReconstructFrame>,
}

pub(super) struct WfReconstructFrame {
    pub(super) callee: u32,
    pub(super) call_op: u32,
    pub(super) resume_op: u32,
    /// GPRs filled from consecutive words of the value buffer.
    pub(super) registers: Vec<u8>,
}

/// v2 deopt after an effect inside inlined code. The caller has materialized
/// the root frame at the outermost call op; enter each inlined call as the
/// interpreter would (pushing the callee frame), fill the callee's live
/// registers and point it at its resume op. The interpreter then continues
/// in the innermost frame; returns flow back through the real frames.
unsafe extern "C" fn napalm_wfjit_reconstruct_frames(
    vm: *mut c_void,
    descriptor: *const WfReconstruct,
    values: *const u64,
) {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    let descriptor = unsafe { &*descriptor };
    let mut offset = 0usize;
    for frame in &descriptor.frames {
        if vm.run_state != RunState::Running || vm.hot_frame.is_null() { return; }
        let fiber = vm.current_fiber;
        let depth = vm.fiber(fiber).frames.len();
        let caller_function = vm.hot_function_index;
        unsafe { vm.set_pc(frame.call_op); }
        let op = unsafe { *vm.current_prepared().ops.get_unchecked(frame.call_op as usize) };
        let raw_pc = unsafe { *vm.current_prepared().raw_pcs.get_unchecked(frame.call_op as usize) };
        let opcode = vm.program.functions[caller_function as usize].chunk.code[raw_pc as usize].opcode();
        let mut words = [RegisterWord(0); 17];
        let count = if op.c <= 4 {
            let mut small = [RegisterWord(0); 4];
            let count = unsafe { vm.gather_call_words_small(&op, &mut small) };
            words[..count].copy_from_slice(&small[..count]);
            count
        } else {
            unsafe { vm.gather_call_words_large(&op, &mut words) }
        };
        match opcode {
            Some(Opcode::CallDirect) => vm.enter_direct_bytecode_call(&op, frame.callee, &words[..count], None),
            Some(Opcode::Invoke | Opcode::InvokeMethodDirect) => {
                let receiver = unsafe { vm.reg(op.b) };
                vm.enter_bytecode_call(&op, frame.callee, Some(receiver), &words[..count], None);
            }
            _ => {}
        }
        if vm.run_state != RunState::Running || vm.current_fiber != fiber
            || vm.fiber(fiber).frames.len() != depth + 1 || vm.hot_function_index != frame.callee
        {
            let op = unsafe { *vm.current_prepared().ops.get_unchecked(frame.call_op as usize) };
            vm.fail_at(&op, "whole-function deoptimization could not rebuild an inlined frame");
            return;
        }
        for &reg in &frame.registers {
            unsafe { *vm.hot_registers.add(reg as usize) = RegisterWord(*values.add(offset)); }
            offset += 1;
        }
        let root_map = vm.current_prepared().gc_map_by_op[frame.resume_op as usize];
        unsafe {
            (*vm.hot_frame).root_map = root_map;
            vm.set_pc(frame.resume_op);
        }
    }
}

/// v2: run any call op of the top frame to completion through the
/// interpreter's generic call path (polymorphic Invoke, InvokeSuper, closures,
/// primitives, constructors). Callees entered this way still tier up through
/// the ordinary entry hooks. Returns the current bank when the interpreter is
/// back in this frame exactly at the op's successor, otherwise null (the
/// interpreter state is authoritative and the native caller bails out).
unsafe extern "C" fn napalm_wfjit_call_to_return(vm: *mut c_void, op_index: u32) -> *mut RegisterWord {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    count_helper(vm, "call", op_index);
    if vm.run_state != RunState::Running || vm.hot_frame.is_null() { return std::ptr::null_mut(); }
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    let caller_function = vm.hot_function_index;
    let expected_next = unsafe { vm.current_prepared().original_ops.get_unchecked(op_index as usize).next };
    unsafe { vm.set_pc(op_index); }
    match vm.trace_execute_call_to_return() {
        Ok(step) if step.running && !step.frame_changed
            && vm.current_fiber == caller_fiber
            && vm.fiber(caller_fiber).frames.len() == caller_depth
            && vm.hot_function_index == caller_function
            && vm.hot_pc == expected_next => vm.hot_registers,
        Ok(_) => std::ptr::null_mut(),
        Err(error) => {
            // Report like the interpreter would at this op.
            let op = unsafe { *vm.current_prepared().ops.get_unchecked(op_index as usize) };
            unsafe { vm.fail_at(&op, error.message); }
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn napalm_wfjit_runtime_op(vm: *mut c_void, op_index: u32) -> *mut RegisterWord {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    count_helper(vm, "op", op_index);
    if vm.run_state != RunState::Running || vm.hot_frame.is_null() { return std::ptr::null_mut(); }
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    let caller_function = vm.hot_function_index;
    unsafe { vm.set_pc(op_index); }

    let raw = unsafe { *vm.current_prepared().original_ops.get_unchecked(op_index as usize) };
    let raw_pc = unsafe { *vm.current_prepared().raw_pcs.get_unchecked(op_index as usize) };
    let opcode = vm.program.functions[caller_function as usize].chunk.code[raw_pc as usize]
        .opcode().unwrap();

    // Shared collections must execute through the locking interpreter quantum.
    // Bail out at this PC before the operation has any effect.
    if vm.heap.shared_mode && unsafe { vm.current_op_touches_shared() }
    {
        return std::ptr::null_mut();
    }

    // Constructors may grow the VM's register stack while the native caller
    // is suspended. Run the constructor to completion and return the *current*
    // GPR base so the caller never retains a dangling register-bank pointer.
    // The tracing tier has a separate inline Immix reservation for zero-field
    // instances; WFJIT currently uses the safe class-constructor slow path.
    if opcode == Opcode::NewClassInstance {
        let expected_next = vm.current_prepared().ops[op_index as usize].next;
        let result = vm.trace_execute_call_to_return();
        return match result {
            Ok(step) if step.running && !step.frame_changed
                && vm.current_fiber == caller_fiber
                && vm.fiber(caller_fiber).frames.len() == caller_depth
                && vm.hot_function_index == caller_function
                && vm.hot_pc == expected_next => vm.hot_registers,
            _ => std::ptr::null_mut(),
        };
    }

    if opcode == Opcode::Invoke {
        let prepared = vm.current_prepared();
        let op = unsafe { *prepared.ops.get_unchecked(op_index as usize) };
        let current = prepared.jit_entry_handlers[op_index as usize]
            .unwrap_or(prepared.handlers[op_index as usize]);
        let numeric_quick = (op.flags & QUICK_GENERIC_DISABLED) == 0
            && NumericQuickOp::from_flag(op.flags).is_some();
        if numeric_quick || is_unfused_monomorphic_invoke_handler(current) {
            unsafe { current(vm, &op); }
        } else {
            // Unsupported/polymorphic Invoke stays authoritative in baseline. The
            // native caller sees a null helper result, returns WFJIT_BAILOUT, and
            // resumes this exact PC without executing the call twice.
            return std::ptr::null_mut();
        }
    } else {
        let mut op = raw;
        let prepared_op = unsafe { *vm.current_prepared().ops.get_unchecked(op_index as usize) };
        if matches!(opcode, Opcode::Switch | Opcode::LoadModuleVar | Opcode::StoreModuleVar
            | Opcode::LoadField | Opcode::StoreField | Opcode::MakeClosure | Opcode::CreateClass
            | Opcode::GetFieldShape | Opcode::SetFieldShape
            | Opcode::GetFieldClass | Opcode::SetFieldClass | Opcode::InvokeMethodDirect
            | Opcode::AllocInstance | Opcode::AllocForeign) {
            op.x = prepared_op.x;
            op.y = prepared_op.y;
        }
        if matches!(opcode, Opcode::MapGet | Opcode::MapSet | Opcode::LoadProperty
            | Opcode::StoreProperty | Opcode::StorePropertyIfPresent
            | Opcode::GetFieldShape | Opcode::SetFieldShape | Opcode::InvokeMethodDirect) {
            op.cache = prepared_op.cache;
        }

        let (handler, op_to_run) = if opcode == Opcode::AllocMap
            && prepared_op.c > 0 && prepared_op.c <= 4 && prepared_op.next > op_index + 1
        {
            let handler = vm.current_prepared().jit_entry_handlers[op_index as usize]
                .unwrap_or(vm.current_prepared().handlers[op_index as usize]);
            (handler, prepared_op)
        } else {
            (handler_for(opcode), op)
        };
        unsafe { handler(vm, &op_to_run); }
    }

    if vm.run_state == RunState::Running
        && vm.current_fiber == caller_fiber
        && vm.fiber(caller_fiber).frames.len() == caller_depth
        && vm.hot_function_index == caller_function
    {
        vm.hot_registers
    } else {
        std::ptr::null_mut()
    }
}
