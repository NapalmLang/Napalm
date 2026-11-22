//! Optimizations on completed whole-function SSA, after ABI specialization.
use super::wf_optimizer::*;
use super::{
    CANONICAL_NAN, TAG_BASE, TAG_FALSE, TAG_MASK, TAG_NULL, TAG_TRUE, WfBlock, WfDeopt, WfFloatBin,
    WfFloatCmp, WfFunction, WfInst, WfTerminator, WfValue,
};
use crate::napalm_collections::HashMap as Table;

pub(super) fn optimize(f: &mut WfFunction) -> Result<(), String> {
    let counts = |ir: &WfFunction| {
        let instructions: usize = ir.blocks.iter().map(|b| b.insts.len()).sum();
        let params: usize = ir.blocks.iter().map(|b| b.params.len()).sum();
        (ir.blocks.len(), instructions, params, ir.deopts.len())
    };
    let event =
        std::env::var_os("NAPALM_JIT_EVENTS").map(|_| (counts(f), std::time::Instant::now()));
    Optimize {
        passes: &[
            &ControlFlow,
            &Sccp,
            &ControlFlow,
            &BlockRepresentation,
            &Values,
            &HeapForward,
            &Values,
            &Predicates,
            &ControlFlow,
            &Parameters,
            &LoopInvariants,
            &MergeBlocks,
            &DeadCode,
            &Snapshots,
        ],
        max_rounds: 8,
        verify_each: cfg!(debug_assertions),
    }
    .run(f)?;
    if let Some(((b0, i0, p0, s0), start)) = event {
        let elapsed = start.elapsed();
        let (b1, i1, p1, s1) = counts(f);
        eprintln!(
            "[napalm-wfjit] IR f={} blocks={b0}->{b1} insts={i0}->{i1} params={p0}->{p1} snapshots={s0}->{s1} opt_us={}",
            f.function_index,
            elapsed.as_micros()
        );
    }
    Ok(())
}

fn pure(i: &WfInst) -> bool {
    use WfInst::*;
    matches!(
        i,
        Param(_)
            | ConstTagged(_)
            | ConstF64(_)
            | ConstBool(_)
            | AssumeNumber(_)
            | AssumeBool(_)
            | BoxF64(_)
            | BoxBool(_)
            | Truthy(_)
            | FloatBin { .. }
            | FloatNeg(_)
            | FloatCmp { .. }
    )
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fact {
    Unknown,
    Word(u64),
    Number(u64),
    Bool(bool),
    BoxedBool(WfValue),
    BoxedNumber,
    Neg(WfValue),
}
impl Fact {
    fn of(i: &WfInst) -> Self {
        match *i {
            WfInst::ConstF64(n) => Self::Number(n),
            WfInst::ConstBool(b) => Self::Bool(b),
            WfInst::ConstTagged(w) => Self::Word(w),
            WfInst::BoxF64(_) => Self::BoxedNumber,
            WfInst::BoxBool(v) => Self::BoxedBool(v),
            WfInst::FloatNeg(v) => Self::Neg(v),
            _ => Self::Unknown,
        }
    }
    fn constant(self) -> Option<WfInst> {
        match self {
            Self::Number(n) => Some(WfInst::ConstF64(n)),
            Self::Word(n) => Some(WfInst::ConstTagged(n)),
            Self::Bool(b) => Some(WfInst::ConstBool(b)),
            _ => None,
        }
    }
}
enum Fold {
    Keep,
    Alias(WfValue),
    Replace(WfInst),
}
fn fold(i: &WfInst, facts: &[Fact]) -> Fold {
    use Fold::*;
    use WfInst::*;
    let fact = |v: WfValue| facts[v.0 as usize];
    match *i {
        BoxBool(v) => {
            if let Fact::Bool(b) = fact(v) {
                return Replace(ConstTagged(if b { TAG_TRUE } else { TAG_FALSE }));
            }
        }
        BoxF64(v) => {
            if let Fact::Number(bits) = fact(v) {
                return Replace(ConstTagged(if f64::from_bits(bits).is_nan() {
                    CANONICAL_NAN
                } else {
                    bits
                }));
            }
        }
        Truthy(v) => match fact(v) {
            Fact::Word(w) => return Replace(ConstBool(w != TAG_FALSE && w != TAG_NULL)),
            Fact::BoxedBool(b) => return Alias(b),
            Fact::BoxedNumber => return Replace(ConstBool(true)),
            _ => {}
        },
        AssumeNumber(v) => {
            if let Fact::Word(w) = fact(v) {
                if w & TAG_MASK != TAG_BASE {
                    return Replace(ConstF64(w));
                }
            }
        }
        UnboxNumber { value: v, .. } => match fact(v) {
            Fact::Word(w) if w & TAG_MASK != TAG_BASE => return Replace(ConstF64(w)),
            // Boxing canonicalizes even a signaling NaN. Keep the bitcast.
            Fact::BoxedNumber => return Replace(AssumeNumber(v)),
            _ => {}
        },
        FloatNeg(v) => match fact(v) {
            Fact::Number(n) => return Replace(ConstF64(n ^ (1 << 63))),
            Fact::Neg(x) => return Alias(x),
            _ => {}
        },
        FloatBin { op, lhs, rhs } => {
            if let (Fact::Number(a), Fact::Number(b)) = (fact(lhs), fact(rhs)) {
                let (a, b) = (f64::from_bits(a), f64::from_bits(b));
                let r = match op {
                    WfFloatBin::Add => a + b,
                    WfFloatBin::Sub => a - b,
                    WfFloatBin::Mul => a * b,
                    WfFloatBin::Div => a / b,
                };
                // Raw NaN payload selection is target dependent until boxing.
                if !r.is_nan() {
                    return Replace(ConstF64(r.to_bits()));
                }
            }
        }
        FloatCmp { cc, lhs, rhs } => {
            use WfFloatCmp::*;
            if let (Fact::Number(a), Fact::Number(b)) = (fact(lhs), fact(rhs)) {
                let (a, b) = (f64::from_bits(a), f64::from_bits(b));
                return Replace(ConstBool(match cc {
                    Eq => a == b,
                    Ne => a != b,
                    Lt => a < b,
                    Le => a <= b,
                    Gt => a > b,
                    Ge => a >= b,
                }));
            }
            if lhs == rhs && matches!(cc, Lt | Gt) {
                return Replace(ConstBool(false));
            }
            if [fact(lhs), fact(rhs)]
                .iter()
                .any(|f| matches!(f,Fact::Number(n) if f64::from_bits(*n).is_nan()))
            {
                return Replace(ConstBool(cc == Ne));
            }
            // Compare canonicalization is safe even when either input is NaN.
            if lhs.0 > rhs.0 {
                return Replace(FloatCmp {
                    cc: match cc {
                        Lt => Gt,
                        Le => Ge,
                        Gt => Lt,
                        Ge => Le,
                        x => x,
                    },
                    lhs: rhs,
                    rhs: lhs,
                });
            }
        }
        _ => {}
    }
    Keep
}

// These identities preserve all non-NaN bits (including signed zero).
// NaN payload and signaling differences are unobservable to comparisons and
// to boxing, but may be observed by a raw self-call ABI.
fn equivalent_after_box(i: &WfInst, facts: &[Fact]) -> Option<WfValue> {
    let WfInst::FloatBin { op, lhs, rhs } = *i else {
        return None;
    };
    let left = facts[lhs.0 as usize];
    let right = facts[rhs.0 as usize];
    let one = Fact::Number(1.0f64.to_bits());
    let neg_zero = Fact::Number((-0.0f64).to_bits());
    match op {
        WfFloatBin::Mul | WfFloatBin::Div if right == one => Some(lhs),
        WfFloatBin::Mul if left == one => Some(rhs),
        WfFloatBin::Add if right == neg_zero => Some(lhs),
        WfFloatBin::Add if left == neg_zero => Some(rhs),
        WfFloatBin::Sub if right == Fact::Number(0) => Some(lhs),
        _ => None,
    }
}

// Structural CSE keys deliberately exclude deopt ids: after a successful
// numeric guard, the same guarded conversion cannot fail a second time.
// Side-effecting instructions are never entered in this table.
#[derive(Clone, PartialEq, Eq, Hash)]
enum ExprKey {
    Param(u16),
    Word(u64),
    Number(u64),
    Bool(bool),
    AssumeNumber(WfValue),
    AssumeBool(WfValue),
    BoxF64(WfValue),
    BoxBool(WfValue),
    Truthy(WfValue),
    Neg(WfValue),
    FloatBin(u8, WfValue, WfValue),
    FloatCmp(u8, WfValue, WfValue),
    UnboxNumber(WfValue),
}

impl ExprKey {
    fn new(i: &WfInst) -> Option<Self> {
        use WfInst::*;
        Some(match *i {
            Param(n) => Self::Param(n),
            ConstTagged(w) => Self::Word(w),
            ConstF64(n) => Self::Number(n),
            ConstBool(b) => Self::Bool(b),
            AssumeNumber(v) => Self::AssumeNumber(v),
            AssumeBool(v) => Self::AssumeBool(v),
            BoxF64(v) => Self::BoxF64(v),
            BoxBool(v) => Self::BoxBool(v),
            Truthy(v) => Self::Truthy(v),
            FloatNeg(v) => Self::Neg(v),
            FloatBin { op, lhs, rhs } => Self::FloatBin(op as u8, lhs, rhs),
            FloatCmp { cc, lhs, rhs } => Self::FloatCmp(cc as u8, lhs, rhs),
            UnboxNumber { value, .. } => Self::UnboxNumber(value),
            _ => return None,
        })
    }
}

struct Values;
impl WfOptimization for Values {
    fn run(&self, f: &mut WfFunction, analyses: &mut Analyses) -> bool {
        let cfg = analyses.cfg(f);
        let mut table = Table::<ExprKey, WfValue>::default();
        let mut facts = vec![Fact::Unknown; f.value_types.len()];
        let mut alias: Vec<_> = (0..facts.len()).map(|i| WfValue(i as u32)).collect();
        let mut boxed = alias.clone();
        let mut changed = false;
        // An undo log makes the available expressions local to a dominator path.
        let mut added = Vec::new();
        let mut stack = vec![(0, None)];
        while let Some((b, mark)) = stack.pop() {
            if let Some(mark) = mark {
                for key in added.drain(mark..) {
                    table.remove(&key);
                }
                continue;
            }
            stack.push((b, Some(added.len())));
            stack.extend(cfg.children[b].iter().map(|&child| (child, None)));
            let mut bank = Table::default();
            let mut guards = Table::default();
            let mut globals = Table::default();
            f.blocks[b].insts.retain_mut(|(result, i)| {
                uses_mut(i, |v| *v = alias[v.0 as usize]);
                if matches!(i, WfInst::BoxF64(_) | WfInst::FloatCmp { .. }) {
                    uses_mut(i, |v| {
                        let next = boxed[v.0 as usize];
                        changed |= next != *v;
                        *v = next;
                    });
                }
                match fold(i, &facts) {
                    Fold::Keep => {}
                    Fold::Replace(new) => {
                        *i = new;
                        changed = true
                    }
                    Fold::Alias(v) => {
                        alias[result.0 as usize] = v;
                        changed = true;
                        return false;
                    }
                }
                let mut existing = None;
                if let Some(key) = ExprKey::new(i) {
                    if let Some(&v) = table.get(&key) {
                        existing = Some(v);
                    } else {
                        added.push(key.clone());
                        table.insert(key, *result);
                    }
                } else {
                    match i {
                        WfInst::FrameLoad(r) => {
                            existing = bank.insert(*r, *result);
                        }
                        WfInst::LoadAbsolute(address) => {
                            existing = globals.insert(*address, *result);
                        }
                        WfInst::GuardClass { value, class, .. } => {
                            existing = guards.insert((*value, *class), *result);
                        }
                        WfInst::RuntimeOp { dest, .. } => {
                            bank.clear();
                            guards.clear();
                            globals.clear();
                            if let Some(r) = dest {
                                bank.insert(*r, *result);
                            }
                        }
                        WfInst::SelfCall { .. } | WfInst::AllocObject { .. } => {
                            bank.clear();
                            guards.clear();
                            globals.clear();
                        }
                        // Guards and native heap reads/stores do not modify
                        // the interpreter GPR bank or module globals.
                        _ => {}
                    }
                }
                if let Some(v) = existing {
                    alias[result.0 as usize] = alias[v.0 as usize];
                    changed = true;
                    false
                } else {
                    facts[result.0 as usize] = Fact::of(i);
                    if let Some(v) = equivalent_after_box(i, &facts) {
                        boxed[result.0 as usize] = boxed[v.0 as usize];
                    }
                    true
                }
            });
        }
        if changed {
            rewrite(f, &alias);
        }
        changed
    }
}

// Sparse conditional constant propagation on executable SSA edges. Values are
// evaluated only in reachable blocks, and phi/block-parameter inputs are
// joined only across executable predecessor edges.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SccpValue {
    Unseen,
    Constant(SccpConstant),
    Overdefined,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SccpConstant {
    Word(u64),
    Number(u64),
    Bool(bool),
}

impl SccpConstant {
    fn inst(self) -> WfInst {
        match self {
            Self::Word(w) => WfInst::ConstTagged(w),
            Self::Number(n) => WfInst::ConstF64(n),
            Self::Bool(b) => WfInst::ConstBool(b),
        }
    }
    fn fact(self) -> Fact {
        match self {
            Self::Word(w) => Fact::Word(w),
            Self::Number(n) => Fact::Number(n),
            Self::Bool(b) => Fact::Bool(b),
        }
    }
}

fn sccp_join(old: SccpValue, incoming: SccpValue) -> SccpValue {
    match (old, incoming) {
        (SccpValue::Overdefined, _) | (_, SccpValue::Overdefined) => SccpValue::Overdefined,
        (SccpValue::Unseen, other) | (other, SccpValue::Unseen) => other,
        (SccpValue::Constant(a), SccpValue::Constant(b)) if a == b => old,
        _ => SccpValue::Overdefined,
    }
}

fn sccp_eval(inst: &WfInst, values: &[SccpValue], facts: &[Fact]) -> SccpValue {
    let constant = match inst {
        WfInst::ConstTagged(w) => Some(SccpConstant::Word(*w)),
        WfInst::ConstF64(n) => Some(SccpConstant::Number(*n)),
        WfInst::ConstBool(b) => Some(SccpConstant::Bool(*b)),
        _ => None,
    };
    if let Some(c) = constant { return SccpValue::Constant(c); }
    match fold(inst, facts) {
        Fold::Replace(WfInst::ConstTagged(w)) => return SccpValue::Constant(SccpConstant::Word(w)),
        Fold::Replace(WfInst::ConstF64(n)) => return SccpValue::Constant(SccpConstant::Number(n)),
        Fold::Replace(WfInst::ConstBool(b)) => return SccpValue::Constant(SccpConstant::Bool(b)),
        Fold::Alias(v) => return values[v.0 as usize],
        _ => {}
    }
    let mut unseen = false;
    uses(inst, |v| unseen |= values[v.0 as usize] == SccpValue::Unseen);
    if unseen { SccpValue::Unseen } else { SccpValue::Overdefined }
}

// A conditional representation pass after SCCP and unreachable-edge pruning.
// The optimistic lattice for tagged block parameters is {Bool, F64}; each
// incoming edge intersects that set. Self edges and cycles can retain their
// candidate until a concrete BoxBool/BoxF64 entry fixes the type. A cycle
// with no typed entry is deliberately left Tagged.
//
// The original Tagged parameter is defined by a Box in the target block;
// deoptimization and any consumer requiring Tagged still use that value.
// For F64, eliminating a re-unbox requires a separate non-NaN proof because
// boxing canonicalizes NaN payloads and the raw self-call ABI can observe them.
struct BlockRepresentation;
impl WfOptimization for BlockRepresentation {
    fn run(&self, f: &mut WfFunction, _analyses: &mut Analyses) -> bool {
        const BOOL: u8 = 1;
        const NUMBER: u8 = 2;
        const BOTH: u8 = BOOL | NUMBER;
        let n = f.value_types.len();
        let mut producer = vec![None; n];
        let mut param_block = vec![None; n];
        for (b, block) in f.blocks.iter().enumerate() {
            for &p in &block.params { param_block[p.0 as usize] = Some(b); }
            for &(v, ref inst) in &block.insts {
                producer[v.0 as usize] = Some(inst.clone());
            }
        }
        let mut incoming = vec![Vec::<(usize, usize)>::new(); f.blocks.len()];
        for (pred, block) in f.blocks.iter().enumerate() {
            for (slot, edge) in edges(block.terminator.as_ref().unwrap()).enumerate() {
                incoming[edge.target.0 as usize].push((pred, slot));
            }
        }
        let mut masks = vec![0u8; n];
        for block in f.blocks.iter().skip(1) {
            for &param in &block.params {
                if f.ty(param) == super::WfTy::Tagged {
                    masks[param.0 as usize] = BOTH;
                }
            }
        }
        // Monotonically shrink each optimistic candidate. SCCP/ControlFlow
        // already removed infeasible predecessors, so they cannot poison it.
        loop {
            let mut changed = false;
            for b in 1..f.blocks.len() {
                for (index, &param) in f.blocks[b].params.iter().enumerate() {
                    let id = param.0 as usize;
                    if masks[id] == 0 { continue; }
                    let mut meet = BOTH;
                    if incoming[b].is_empty() { meet = 0; }
                    for &(pred, slot) in &incoming[b] {
                        let edge = edges(f.blocks[pred].terminator.as_ref().unwrap())
                            .nth(slot).unwrap();
                        let arg = edge.args[index];
                        let candidate = match producer[arg.0 as usize].as_ref() {
                            Some(WfInst::BoxBool(_)) => BOOL,
                            Some(WfInst::BoxF64(_)) => NUMBER,
                            // Folded boxes (SCCP turns BoxBool(const) into a word).
                            Some(WfInst::ConstTagged(word)) if *word == TAG_TRUE || *word == TAG_FALSE => BOOL,
                            Some(WfInst::ConstTagged(word)) if *word & TAG_MASK != TAG_BASE => NUMBER,
                            _ if param_block[arg.0 as usize].is_some() => masks[arg.0 as usize],
                            _ => 0,
                        };
                        meet &= candidate;
                    }
                    let next = masks[id] & meet;
                    if next != masks[id] { masks[id] = next; changed = true; }
                }
            }
            if !changed { break; }
        }

        struct Plan {
            block: usize,
            index: usize,
            tagged: WfValue,
            raw: WfValue,
            ty: super::WfTy,
            non_nan: bool,
        }
        let mut plans = Vec::<Plan>::new();
        let mut raw_of = vec![None; n];
        for b in 1..f.blocks.len() {
            for (index, &param) in f.blocks[b].params.iter().enumerate() {
                let ty = match masks[param.0 as usize] {
                    BOOL => super::WfTy::Bool,
                    NUMBER => super::WfTy::F64,
                    _ => continue,
                };
                let raw = WfValue((n + plans.len()) as u32);
                raw_of[param.0 as usize] = Some(raw);
                plans.push(Plan { block: b, index, tagged: param, raw, ty, non_nan: false });
            }
        }
        if plans.is_empty() { return false; }

        // Determine the actual unboxed SSA source on each predecessor edge.
        // An edge can carry an already promoted parameter (including itself).
        let mut sources = Vec::<Vec<(usize, usize, WfValue)>>::new();
        // Raw constants materialized in a predecessor for a folded box:
        // (block, value, instruction, type). Ids follow the plans' raw ids.
        let mut extra_consts = Vec::<(usize, WfValue, WfInst, super::WfTy)>::new();
        for plan in &plans {
            let mut srcs = Vec::with_capacity(incoming[plan.block].len());
            for &(pred, slot) in &incoming[plan.block] {
                let edge = edges(f.blocks[pred].terminator.as_ref().unwrap())
                    .nth(slot).unwrap();
                let arg = edge.args[plan.index];
                let mut constant = |inst: WfInst, ty: super::WfTy| {
                    let value = WfValue((n + plans.len() + extra_consts.len()) as u32);
                    extra_consts.push((pred, value, inst, ty));
                    Some(value)
                };
                let source = match producer[arg.0 as usize].as_ref() {
                    Some(WfInst::BoxBool(v)) if plan.ty == super::WfTy::Bool => Some(*v),
                    Some(WfInst::BoxF64(v)) if plan.ty == super::WfTy::F64 => Some(*v),
                    Some(WfInst::ConstTagged(word)) if plan.ty == super::WfTy::Bool
                        && (*word == TAG_TRUE || *word == TAG_FALSE) =>
                        constant(WfInst::ConstBool(*word == TAG_TRUE), super::WfTy::Bool),
                    Some(WfInst::ConstTagged(word)) if plan.ty == super::WfTy::F64
                        && *word & TAG_MASK != TAG_BASE =>
                        constant(WfInst::ConstF64(*word), super::WfTy::F64),
                    _ => raw_of[arg.0 as usize],
                };
                let Some(source) = source else { return false; };
                srcs.push((pred, slot, source));
            }
            sources.push(srcs);
        }

        // Optimistic non-NaN fixed point. Only known non-NaN raw producers
        // and already proven raw block parameters can establish the property.
        let mut non_nan = vec![false; n + plans.len() + extra_consts.len()];
        for &(_, value, ref inst, _) in &extra_consts {
            if let WfInst::ConstF64(bits) = inst {
                non_nan[value.0 as usize] = !f64::from_bits(*bits).is_nan();
            }
        }
        loop {
            let mut changed = false;
            for &(v, ref inst) in f.blocks.iter().flat_map(|b| &b.insts) {
                let safe = match inst {
                    WfInst::ConstF64(bits) => !f64::from_bits(*bits).is_nan(),
                    WfInst::FloatNeg(x) => non_nan[x.0 as usize],
                    _ => false,
                };
                if safe && !non_nan[v.0 as usize] {
                    non_nan[v.0 as usize] = true;
                    changed = true;
                }
            }
            if !changed { break; }
        }
        for plan in &plans {
            if plan.ty == super::WfTy::F64 { non_nan[plan.raw.0 as usize] = true; }
        }
        loop {
            let mut changed = false;
            for (plan, srcs) in plans.iter().zip(&sources) {
                if plan.ty != super::WfTy::F64 { continue; }
                let safe = srcs.iter().all(|&(_, _, source)| non_nan[source.0 as usize]);
                if !safe && non_nan[plan.raw.0 as usize] {
                    non_nan[plan.raw.0 as usize] = false;
                    changed = true;
                }
            }
            if !changed { break; }
        }
        for plan in &mut plans {
            plan.non_nan = plan.ty == super::WfTy::F64 && non_nan[plan.raw.0 as usize];
        }

        for plan in &plans {
            f.value_types.push(plan.ty);
            f.blocks[plan.block].params[plan.index] = plan.raw;
            let inst = match plan.ty {
                super::WfTy::Bool => WfInst::BoxBool(plan.raw),
                super::WfTy::F64 => WfInst::BoxF64(plan.raw),
                super::WfTy::Tagged => unreachable!(),
            };
            f.blocks[plan.block].insts.insert(0, (plan.tagged, inst));
        }
        for (block, value, inst, ty) in extra_consts {
            debug_assert_eq!(value.0 as usize, f.value_types.len());
            f.value_types.push(ty);
            f.blocks[block].insts.push((value, inst));
        }
        for (plan, srcs) in plans.iter().zip(&sources) {
            for &(pred, slot, source) in srcs {
                let edge = edges_mut(f.blocks[pred].terminator.as_mut().unwrap())
                    .nth(slot).unwrap();
                edge.args[plan.index] = source;
            }
        }
        let mut forwards = Table::<WfValue, (WfValue, super::WfTy, bool)>::default();
        for plan in &plans {
            forwards.insert(plan.tagged, (plan.raw, plan.ty, plan.non_nan));
        }
        let mut aliases: Vec<_> = (0..f.value_types.len()).map(|v| WfValue(v as u32)).collect();
        let mut removed = false;
        for block in &mut f.blocks {
            block.insts.retain(|(result, inst)| {
                let source = match inst {
                    WfInst::Truthy(value) => forwards.get(value).and_then(|&(raw, ty, _)|
                        (ty == super::WfTy::Bool).then_some(raw)),
                    WfInst::UnboxNumber { value, .. } | WfInst::AssumeNumber(value) =>
                        forwards.get(value).and_then(|&(raw, ty, safe)|
                            (ty == super::WfTy::F64 && safe).then_some(raw)),
                    _ => None,
                };
                if let Some(raw) = source {
                    aliases[result.0 as usize] = raw;
                    removed = true;
                    false
                } else { true }
            });
        }
        if removed { rewrite(f, &aliases); }

        // Push a Tagged return conversion to the actual exit block. A loop
        // header need not box on every iteration merely because its value
        // is returned after the loop. Deopt snapshots can use the raw SSA
        // parameter directly: the lowerer boxes it only on the exit path.
        for plan in &plans {
            let exits: Vec<_> = f.blocks.iter().enumerate().filter_map(|(block, data)| {
                (block != plan.block && matches!(&data.terminator,
                    Some(WfTerminator::Return(value)) if *value == plan.tagged))
                    .then_some(block)
            }).collect();
            for block in exits {
                let boxed = WfValue(f.value_types.len() as u32);
                f.value_types.push(super::WfTy::Tagged);
                let inst = match plan.ty {
                    super::WfTy::Bool => WfInst::BoxBool(plan.raw),
                    super::WfTy::F64 => WfInst::BoxF64(plan.raw),
                    super::WfTy::Tagged => unreachable!(),
                };
                f.blocks[block].insts.push((boxed, inst));
                f.blocks[block].terminator = Some(WfTerminator::Return(boxed));
            }
        }

        let mut used = vec![false; f.value_types.len()];
        for block in &f.blocks {
            for (_, inst) in &block.insts {
                uses(inst, |value| used[value.0 as usize] = true);
            }
            let term = block.terminator.as_ref().unwrap();
            if let Some(value) = control(term) { used[value.0 as usize] = true; }
            for edge in edges(term) {
                for &value in &edge.args { used[value.0 as usize] = true; }
            }
        }
        let mut deferred = vec![false; f.value_types.len()];
        let mut deopt_aliases: Vec<_> = (0..f.value_types.len())
            .map(|v| WfValue(v as u32)).collect();
        for plan in &plans {
            if !used[plan.tagged.0 as usize] {
                deferred[plan.tagged.0 as usize] = true;
                deopt_aliases[plan.tagged.0 as usize] = plan.raw;
            }
        }
        if deferred.iter().any(|&value| value) {
            // No ordinary SSA consumer refers to a deferred definition.
            // Rewriting the deopt values to raw is therefore type-safe.
            rewrite(f, &deopt_aliases);
            for block in &mut f.blocks {
                block.insts.retain(|(result, _)| !deferred[result.0 as usize]);
            }
        }
        true
    }
}

// Heap value numbering is scoped to the dominator tree. Successful native
// accesses establish facts only on the continuation path; no proof is
// transferred across helpers, native calls, or allocations that may run code.
#[derive(Clone, Copy)]
struct KnownField {
    value: WfValue,
    nonnull: bool,
}

#[derive(Clone, Default)]
struct HeapState {
    fields: Table<(WfValue, u32), KnownField>,
    map_reads: Table<(WfValue, Vec<(u32, u32)>), WfValue>,
    map_shapes: Table<WfValue, u32>,
    map_slots: Table<(WfValue, u32, u32), WfValue>,
    classes: Table<WfValue, u64>,
    /// Objects allocated by this body since the last possible escape/safepoint.
    /// Other receivers may be shared and must never be forwarded.
    fresh_local: Vec<WfValue>,
    tracking_off: bool,
    epoch: Option<u64>,
}

impl HeapState {
    fn kill_effects(&mut self) {
        self.fields.clear();
        self.map_reads.clear();
        self.map_shapes.clear();
        self.map_slots.clear();
        self.classes.clear();
        self.fresh_local.clear();
        self.tracking_off = false;
        self.epoch = None;
    }
}

fn is_known_nonnull(inst: Option<&WfInst>) -> bool {
    match inst {
        Some(WfInst::ConstTagged(w)) => *w != TAG_NULL,
        Some(WfInst::BoxF64(_) | WfInst::BoxBool(_)) => true,
        Some(WfInst::InstanceFieldGet { nullable: false, .. }) => true,
        _ => false,
    }
}

struct HeapForward;
impl WfOptimization for HeapForward {
    fn run(&self, f: &mut WfFunction, analyses: &mut Analyses) -> bool {
        let cfg = analyses.cfg(f);
        let mut producer = vec![None; f.value_types.len()];
        for block in &f.blocks {
            for (v, i) in &block.insts {
                producer[v.0 as usize] = Some(i.clone());
            }
        }
        let mut aliases: Vec<_> = (0..f.value_types.len())
            .map(|v| WfValue(v as u32)).collect();
        let mut state = HeapState::default();
        let mut stack = vec![(0, None)];
        let mut changed = false;
        while let Some((b, saved)) = stack.pop() {
            if let Some(previous) = saved {
                state = previous;
                continue;
            }
            let previous = state.clone();
            // A sibling path may have mutated heap or interpreter state.
            // Dominator ancestry alone is not a memory-SSA proof at a join.
            if b != 0 && cfg.pred[b].len() != 1 { state.kill_effects(); }
            f.blocks[b].insts.retain_mut(|(result, inst)| {
                uses_mut(inst, |v| *v = aliases[v.0 as usize]);
                match inst {
                    WfInst::InstanceFieldGet { object, slot, nullable, .. } => {
                        let key = (*object, *slot);
                        if state.tracking_off && state.fresh_local.contains(object) {
                            if let Some(&field) = state.fields.get(&key) {
                                if *nullable || field.nonnull {
                                    aliases[result.0 as usize] = field.value;
                                    changed = true;
                                    return false;
                                }
                            }
                        }
                        // Completing a get proves that tracking was disabled.
                        state.tracking_off = true;
                        if state.fresh_local.contains(object) {
                            state.fields.insert(key, KnownField {
                                value: *result,
                                nonnull: !*nullable,
                            });
                        }
                    }
                    WfInst::InstanceFieldSet { object, value, slot, class_key, .. } => {
                        // Unknown receivers can alias. A fresh, unescaped
                        // allocation cannot alias another receiver; nor can
                        // two objects with proven distinct exact classes.
                        let invalidated: Vec<_> = state.fields.keys().copied()
                            .filter(|&(other, s)| {
                                if s != *slot { return false; }
                                if other == *object { return true; }
                                if state.fresh_local.contains(&other)
                                    || state.fresh_local.contains(object) { return false; }
                                state.classes.get(&other).is_none_or(|&class| class == *class_key)
                            })
                            .collect();
                        for key in invalidated { state.fields.remove(&key); }
                        let stored = aliases[value.0 as usize];
                        if state.fresh_local.contains(object) {
                            state.fields.insert((*object, *slot), KnownField {
                                value: stored,
                                nonnull: is_known_nonnull(producer[stored.0 as usize].as_ref()),
                            });
                        }
                        state.classes.insert(*object, *class_key);
                    }
                    WfInst::MapPropertyGet { object, entries, .. } if state.fresh_local.contains(object) => {
                        let exact_key = (*object, entries.clone());
                        let from_shape = state.map_shapes.get(object).and_then(|shape| {
                            entries.iter().find(|(s, _)| s == shape).and_then(|&(s, slot)| {
                                state.map_slots.get(&(*object, s, slot)).copied()
                            })
                        });
                        if let Some(v) = from_shape.or_else(|| state.map_reads.get(&exact_key).copied()) {
                            aliases[result.0 as usize] = v;
                            changed = true;
                            return false;
                        }
                        state.map_reads.insert(exact_key, *result);
                        if let [(shape, slot)] = entries.as_slice() {
                            state.map_shapes.insert(*object, *shape);
                            state.map_slots.insert((*object, *shape, *slot), *result);
                        }
                    }
                    WfInst::GuardEpoch { expected, .. } => {
                        if state.epoch == Some(*expected) {
                            *inst = WfInst::ConstBool(false);
                            changed = true;
                        } else {
                            state.epoch = Some(*expected);
                        }
                    }
                    WfInst::GuardClass { value, class, .. } => {
                        if state.classes.get(value).copied() == Some(*class) {
                            *inst = WfInst::ConstBool(false);
                            changed = true;
                        } else {
                            // A successful guard proves the receiver is live
                            // and of this exact class until code may run.
                            state.classes.insert(*value, *class);
                        }
                    }
                    WfInst::AllocObject { .. } => {
                        state.kill_effects();
                        state.fresh_local.push(*result);
                    }
                    WfInst::RuntimeOp { .. }
                    | WfInst::SelfCall { .. } => state.kill_effects(),
                    _ => {}
                }
                true
            });
            stack.push((b, Some(previous)));
            stack.extend(cfg.children[b].iter().rev().map(|&child| (child, None)));
        }
        if changed { rewrite(f, &aliases); }
        changed
    }
}

struct Sccp;
impl WfOptimization for Sccp {
    fn invalidates_cfg(&self) -> bool { true }
    fn run(&self, f: &mut WfFunction, _analyses: &mut Analyses) -> bool {
        let nblocks = f.blocks.len();
        let mut incoming = vec![Vec::<(usize, usize)>::new(); nblocks];
        for (pred, block) in f.blocks.iter().enumerate() {
            for (slot, edge) in edges(block.terminator.as_ref().unwrap()).enumerate() {
                incoming[edge.target.0 as usize].push((pred, slot));
            }
        }
        let mut reachable = vec![false; nblocks];
        let mut executable = vec![[false; 2]; nblocks];
        let mut lattice = vec![SccpValue::Unseen; f.value_types.len()];
        let mut facts = vec![Fact::Unknown; lattice.len()];
        let mut producers = vec![None; lattice.len()];
        for block in &f.blocks {
            for &(v, ref inst) in &block.insts {
                producers[v.0 as usize] = Some(inst);
            }
        }
        reachable[0] = true;
        loop {
            let mut changed = false;
            for (idx, producer) in producers.iter().enumerate() {
                facts[idx] = match lattice[idx] {
                    SccpValue::Constant(c) => c.fact(),
                    _ => producer.map_or(Fact::Unknown, |i| Fact::of(i)),
                };
            }
            for b in 0..nblocks {
                if !reachable[b] { continue; }
                for (idx, &p) in f.blocks[b].params.iter().enumerate() {
                    let mut value = SccpValue::Unseen;
                    for &(pred, slot) in &incoming[b] {
                        if !reachable[pred] || !executable[pred][slot] { continue; }
                        let edge = edges(f.blocks[pred].terminator.as_ref().unwrap())
                            .nth(slot).unwrap();
                        let argument = edge.args[idx];
                        if argument != p {
                            value = sccp_join(value, lattice[argument.0 as usize]);
                        }
                    }
                    let id = p.0 as usize;
                    let next = sccp_join(lattice[id], value);
                    if next != lattice[id] { lattice[id] = next; changed = true; }
                }
                for &(v, ref inst) in &f.blocks[b].insts {
                    let id = v.0 as usize;
                    let next = sccp_join(lattice[id], sccp_eval(inst, &lattice, &facts));
                    if lattice[id] != next {
                        lattice[id] = next;
                        if let SccpValue::Constant(c) = next { facts[id] = c.fact(); }
                        changed = true;
                    }
                }
                let term = f.blocks[b].terminator.as_ref().unwrap();
                let mask = match term {
                    WfTerminator::Jump(_) => [true, false],
                    WfTerminator::Return(_) => [false, false],
                    WfTerminator::Branch { condition, .. } => match lattice[condition.0 as usize] {
                        SccpValue::Constant(SccpConstant::Bool(true)) => [true, false],
                        SccpValue::Constant(SccpConstant::Bool(false)) => [false, true],
                        SccpValue::Overdefined => [true, true],
                        _ => [false, false],
                    },
                };
                for (slot, edge) in edges(term).enumerate() {
                    if mask[slot] && !executable[b][slot] {
                        executable[b][slot] = true;
                        reachable[edge.target.0 as usize] = true;
                        changed = true;
                    }
                }
            }
            if changed { continue; }
            // A conservative fallback for cyclic SSA with no constant seed.
            // Never treat an unresolved condition as evidence of dead control flow.
            for b in 0..nblocks {
                if !reachable[b] { continue; }
                if let Some(WfTerminator::Branch { condition, .. }) = &f.blocks[b].terminator {
                    if lattice[condition.0 as usize] == SccpValue::Unseen {
                        for (slot, edge) in edges(f.blocks[b].terminator.as_ref().unwrap()).enumerate() {
                            if !executable[b][slot] {
                                executable[b][slot] = true;
                                reachable[edge.target.0 as usize] = true;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed { break; }
        }
        let mut changed = false;
        for (b, block) in f.blocks.iter_mut().enumerate() {
            if !reachable[b] { continue; }
            for (v, inst) in &mut block.insts {
                let SccpValue::Constant(c) = lattice[v.0 as usize] else { continue };
                // A guard or memory effect must never disappear merely because
                // its output (usually a meaningless Bool) is constant.
                if !(pure(inst) || matches!(inst, WfInst::UnboxNumber { .. })) { continue; }
                let identical = match (c, &*inst) {
                    (SccpConstant::Word(a), WfInst::ConstTagged(b)) => a == *b,
                    (SccpConstant::Number(a), WfInst::ConstF64(b)) => a == *b,
                    (SccpConstant::Bool(a), WfInst::ConstBool(b)) => a == *b,
                    _ => false,
                };
                if !identical { *inst = c.inst(); changed = true; }
            }
            if let Some(WfTerminator::Branch { condition, then, otherwise }) = &block.terminator {
                let take = match lattice[condition.0 as usize] {
                    SccpValue::Constant(SccpConstant::Bool(b)) => Some(b),
                    _ => None,
                };
                if let Some(take) = take {
                    block.terminator = Some(WfTerminator::Jump(if take { then } else { otherwise }.clone()));
                    changed = true;
                }
            }
        }
        changed
    }
}

struct ControlFlow;
impl WfOptimization for ControlFlow {
    fn invalidates_cfg(&self) -> bool { true }
    fn run(&self, f: &mut WfFunction, _analyses: &mut Analyses) -> bool {
        if f.blocks.len() < 2 {
            return false;
        }
        let mut known = vec![None; f.value_types.len()];
        for b in &f.blocks {
            for &(v, ref i) in &b.insts {
                if let WfInst::ConstBool(c) = i {
                    known[v.0 as usize] = Some(*c)
                }
            }
        }
        let mut changed = false;
        for b in &mut f.blocks {
            if let Some(WfTerminator::Branch {
                condition,
                then,
                otherwise,
            }) = &b.terminator
            {
                let same = then.target == otherwise.target && then.args == otherwise.args;
                if let Some(take_then) = known[condition.0 as usize].or(same.then_some(true)) {
                    b.terminator = Some(WfTerminator::Jump(
                        if take_then { then } else { otherwise }.clone(),
                    ));
                    changed = true;
                }
            }
        }
        // Thread one empty, parameterless jump per edge.
        // Bound work per iteration and keep cycles intact.
        let mut forward = vec![None; f.blocks.len()];
        for (b, data) in f.blocks.iter().enumerate().skip(1) {
            if data.insts.is_empty() {
                if let Some(WfTerminator::Jump(e)) = &data.terminator {
                    // A block parameter may also be used directly in descendants
                    // or exit states. Only bypass blocks with no definitions.
                    if data.params.is_empty() && e.target.0 as usize != b {
                        forward[b] = Some(e.clone());
                    }
                }
            }
        }
        for (b, data) in f.blocks.iter_mut().enumerate() {
            for e in edges_mut(data.terminator.as_mut().unwrap()) {
                if let Some(next) = &forward[e.target.0 as usize] {
                    if next.target.0 as usize != b {
                        *e = next.clone();
                        changed = true;
                    }
                }
            }
        }
        // Reverse postorder keeps every strict dominator before its uses in
        // the backend's single linear walk, including newly threaded edges.
        let mut visited = vec![false; f.blocks.len()];
        let mut order = Vec::new();
        let mut stack = vec![(0, false)];
        while let Some((b, exit)) = stack.pop() {
            if exit {
                order.push(b);
                continue;
            }
            if std::mem::replace(&mut visited[b], true) {
                continue;
            }
            stack.push((b, true));
            stack.extend(
                edges(f.blocks[b].terminator.as_ref().unwrap())
                    .map(|e| (e.target.0 as usize, false)),
            );
        }
        order.reverse();
        if !order.iter().copied().eq(0..f.blocks.len()) {
            let mut remap = vec![0; f.blocks.len()];
            for (idx, &b) in order.iter().enumerate() {
                remap[b] = idx as u32;
            }
            let mut old = std::mem::take(&mut f.blocks);
            f.blocks = order
                .into_iter()
                .map(|b| {
                    let mut data = std::mem::take(&mut old[b]);
                    for e in edges_mut(data.terminator.as_mut().unwrap()) {
                        e.target = WfBlock(remap[e.target.0 as usize]);
                    }
                    data
                })
                .collect();
            changed = true;
        }
        changed
    }
}

fn filter_signature(f: &mut WfFunction, keep: impl Fn(WfValue) -> bool) {
    for b in 0..f.blocks.len() {
        let mut term = f.blocks[b].terminator.take().unwrap();
        for e in edges_mut(&mut term) {
            let mut params = f.blocks[e.target.0 as usize].params.iter();
            e.args.retain(|_| keep(*params.next().unwrap()));
        }
        f.blocks[b].terminator = Some(term);
    }
    for b in &mut f.blocks {
        b.params.retain(|&p| keep(p));
    }
}
struct Parameters;
impl WfOptimization for Parameters {
    fn run(&self, f: &mut WfFunction, analyses: &mut Analyses) -> bool {
        if f.blocks.len() < 2 {
            return false;
        }
        let n = f.value_types.len();
        let cfg = analyses.cfg(f);
        let mut def = vec![usize::MAX; n];
        let mut facts = vec![Fact::Unknown; n];
        for (b, data) in f.blocks.iter().enumerate() {
            for p in &data.params {
                def[p.0 as usize] = b;
            }
            for (v, i) in &data.insts {
                def[v.0 as usize] = b;
                facts[v.0 as usize] = Fact::of(i);
            }
        }
        let mut alias: Vec<_> = (0..n).map(|i| WfValue(i as u32)).collect();
        let mut keep = vec![true; n];
        let mut changed = false;
        for b in 1..f.blocks.len() {
            if f.blocks[b].params.is_empty() {
                continue;
            }
            let mut in_edges = Vec::new();
            let mut prev = None;
            for &p in &cfg.pred[b] {
                if prev == Some(p) {
                    continue;
                }
                prev = Some(p);
                in_edges.extend(
                    edges(f.blocks[p].terminator.as_ref().unwrap())
                        .filter(|e| e.target.0 as usize == b),
                );
            }
            let mut local = Vec::new();
            let mut columns = Table::default();
            for (idx, &p) in f.blocks[b].params.iter().enumerate() {
                let mut vals = in_edges
                    .iter()
                    .map(|e| alias[e.args[idx].0 as usize])
                    .filter(|&v| v != p);
                let Some(first) = vals.next() else { continue };
                let mut same = true;
                let mut constant = facts[first.0 as usize].constant();
                for v in vals {
                    same &= v == first;
                    if facts[first.0 as usize] != facts[v.0 as usize] {
                        constant = None;
                    }
                }
                let d = def[first.0 as usize];
                if same && d < b && cfg.dominates(d, b) {
                    alias[p.0 as usize] = first;
                } else if let Some(i) = constant {
                    local.push((p, i));
                } else {
                    // Distinct GPRs at a join can have identical incoming SSA columns.
                    let column: Vec<_> = in_edges
                        .iter()
                        .map(|e| alias[e.args[idx].0 as usize])
                        .collect();
                    if let Some(&other) = columns.get(&column) {
                        alias[p.0 as usize] = other;
                    } else {
                        columns.insert(column, p);
                        continue;
                    }
                }
                keep[p.0 as usize] = false;
                changed = true;
            }
            if !local.is_empty() {
                local.append(&mut f.blocks[b].insts);
                f.blocks[b].insts = local;
            }
        }
        if changed {
            filter_signature(f, |v| keep[v.0 as usize]);
            rewrite(f, &alias);
        }
        changed
    }
}

// Move total, memory-independent expressions from a natural loop header to
// its dedicated preheader. Header parameters may already have been forwarded.
struct LoopInvariants;
impl WfOptimization for LoopInvariants {
    fn run(&self, f: &mut WfFunction, analyses: &mut Analyses) -> bool {
        if f.blocks.len() < 2 {
            return false;
        }
        let cfg = analyses.cfg(f);
        let mut def = vec![usize::MAX; f.value_types.len()];
        for (b, data) in f.blocks.iter().enumerate() {
            for v in data.params.iter().chain(data.insts.iter().map(|(v, _)| v)) {
                def[v.0 as usize] = b;
            }
        }
        let mut changed = false;
        for b in 1..f.blocks.len() {
            if !cfg.pred[b].iter().any(|&p| cfg.dominates(b, p)) {
                continue;
            }
            let mut incoming = cfg.pred[b]
                .iter()
                .copied()
                .filter(|&p| !cfg.dominates(b, p));
            let Some(pre) = incoming.next() else { continue };
            if incoming.next().is_some()
                || pre >= b
                || !cfg.dominates(pre, b)
                || !matches!(f.blocks[pre].terminator, Some(WfTerminator::Jump(_)))
            {
                continue;
            }
            let old = std::mem::take(&mut f.blocks[b].insts);
            let mut kept = Vec::with_capacity(old.len());
            for (v, i) in old {
                let mut invariant = pure(&i) && !matches!(i, WfInst::Param(_));
                if invariant {
                    uses(&i, |arg| {
                        invariant &= cfg.dominates(def[arg.0 as usize], pre)
                    });
                }
                if invariant {
                    def[v.0 as usize] = pre;
                    f.blocks[pre].insts.push((v, i));
                    changed = true;
                } else {
                    kept.push((v, i));
                }
            }
            f.blocks[b].insts = kept;
        }
        changed
    }
}

// A branch outcome is a fact throughout the subtree dominated by its
// single-incoming-edge successor. Forget it on exit from that subtree.
struct Predicates;
impl WfOptimization for Predicates {
    fn invalidates_cfg(&self) -> bool { true }
    fn run(&self, f: &mut WfFunction, analyses: &mut Analyses) -> bool {
        if f.blocks.len() < 2 {
            return false;
        }
        let cfg = analyses.cfg(f);
        let mut known = vec![None; f.value_types.len()];
        let mut work = vec![(0, None)];
        let mut fold = Vec::new();
        while let Some((b, restore)) = work.pop() {
            if let Some((v, old)) = restore {
                known[v] = old;
                continue;
            }
            if let [p] = cfg.pred[b].as_slice() {
                if let Some(WfTerminator::Branch {
                    condition,
                    then,
                    otherwise,
                }) = &f.blocks[*p].terminator
                {
                    if then.target != otherwise.target {
                        let idx = condition.0 as usize;
                        let old = known[idx].replace(then.target.0 as usize == b);
                        work.push((b, Some((idx, old))));
                    }
                }
            }
            if let Some(WfTerminator::Branch { condition, .. }) = &f.blocks[b].terminator {
                if let Some(taken) = known[condition.0 as usize] {
                    fold.push((b, taken));
                }
            }
            work.extend(cfg.children[b].iter().map(|&c| (c, None)));
        }
        let changed = !fold.is_empty();
        for (b, taken) in fold {
            let Some(WfTerminator::Branch {
                then, otherwise, ..
            }) = f.blocks[b].terminator.take()
            else {
                unreachable!()
            };
            f.blocks[b].terminator = Some(WfTerminator::Jump(if taken { then } else { otherwise }));
        }
        changed
    }
}

struct MergeBlocks;
impl WfOptimization for MergeBlocks {
    fn invalidates_cfg(&self) -> bool { true }
    fn run(&self, f: &mut WfFunction, _analyses: &mut Analyses) -> bool {
        if f.blocks.len() < 2 {
            return false;
        }
        let mut incoming = vec![0usize; f.blocks.len()];
        for b in &f.blocks {
            for e in edges(b.terminator.as_ref().unwrap()) {
                incoming[e.target.0 as usize] += 1;
            }
        }
        let mut changed = false;
        for b in 0..f.blocks.len() {
            while let Some(WfTerminator::Jump(e)) = &f.blocks[b].terminator {
                let t = e.target.0 as usize;
                if t <= b || incoming[t] != 1 || !f.blocks[t].params.is_empty() {
                    break;
                }
                let (before, after) = f.blocks.split_at_mut(t);
                before[b].insts.append(&mut after[0].insts);
                before[b].terminator = after[0].terminator.take();
                changed = true;
            }
        }
        if changed {
            let mut next = 0;
            let remap: Vec<_> = f
                .blocks
                .iter()
                .map(|b| {
                    let idx = next;
                    next += u32::from(b.terminator.is_some());
                    idx
                })
                .collect();
            f.blocks.retain(|b| b.terminator.is_some());
            for b in &mut f.blocks {
                for e in edges_mut(b.terminator.as_mut().unwrap()) {
                    e.target = WfBlock(remap[e.target.0 as usize]);
                }
            }
        }
        changed
    }
}

struct DeadCode;
impl WfOptimization for DeadCode {
    fn run(&self, f: &mut WfFunction, _analyses: &mut Analyses) -> bool {
        let n = f.value_types.len();
        let mut producer = vec![None; n];
        let mut incoming = Table::<WfValue, Vec<WfValue>>::default();
        let mut work = Vec::new();
        let mut seen_state = vec![false; f.deopts.len()];
        let mut add_state = |d: WfDeopt, work: &mut Vec<WfValue>| {
            if !std::mem::replace(&mut seen_state[d.0 as usize], true) {
                work.extend(state_values(&f.deopts[d.0 as usize]));
            }
        };
        if let Some((_, d)) = f.entry_epoch_guard {
            add_state(d, &mut work);
        }
        for b in &f.blocks {
            for (v, i) in &b.insts {
                producer[v.0 as usize] = Some(i);
                if !pure(i) && !matches!(i, WfInst::FrameLoad(_) | WfInst::LoadAbsolute(_)) {
                    work.push(*v);
                }
                if let Some(&d) = snapshot(i) {
                    add_state(d, &mut work);
                }
            }
            let t = b.terminator.as_ref().unwrap();
            work.extend(control(t));
            for e in edges(t) {
                for (&p, &v) in f.blocks[e.target.0 as usize].params.iter().zip(&e.args) {
                    incoming.entry(p).or_default().push(v);
                }
            }
        }
        let mut live = vec![false; n];
        while let Some(v) = work.pop() {
            if std::mem::replace(&mut live[v.0 as usize], true) {
                continue;
            }
            if let Some(i) = producer[v.0 as usize] {
                uses(i, |arg| work.push(*arg));
            } else if let Some(args) = incoming.get(&v) {
                work.extend(args);
            }
        }
        let mut changed = false;
        for b in &mut f.blocks {
            changed |= b.params.iter().any(|v| !live[v.0 as usize]);
            let before = b.insts.len();
            b.insts.retain(|(v, _)| live[v.0 as usize]);
            changed |= b.insts.len() != before;
        }
        if changed {
            filter_signature(f, |p| live[p.0 as usize]);
        }
        changed
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct FrameKey {
    callee: u32,
    call_op: u32,
    resume_op: u32,
    registers: Vec<(u8, WfValue)>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct SnapshotKey {
    op_index: u32,
    speculative: bool,
    registers: Vec<(u8, WfValue)>,
    frames: Vec<FrameKey>,
}

impl From<&super::WfDeoptState> for SnapshotKey {
    fn from(s: &super::WfDeoptState) -> Self {
        Self {
            op_index: s.op_index,
            speculative: s.speculative,
            registers: s.registers.clone(),
            frames: s.frames.iter().map(|frame| FrameKey {
                callee: frame.callee,
                call_op: frame.call_op,
                resume_op: frame.resume_op,
                registers: frame.registers.clone(),
            }).collect(),
        }
    }
}

struct Snapshots;
impl WfOptimization for Snapshots {
    fn run(&self, f: &mut WfFunction, _analyses: &mut Analyses) -> bool {
        let mut used = vec![false; f.deopts.len()];
        for b in &f.blocks {
            for (_, i) in &b.insts {
                if let Some(d) = snapshot(i) {
                    used[d.0 as usize] = true;
                }
            }
        }
        if let Some((_, d)) = f.entry_epoch_guard {
            used[d.0 as usize] = true;
        }
        let mut intern = Table::default();
        let mut remap = vec![WfDeopt(0); used.len()];
        for (n, s) in f.deopts.iter().enumerate() {
            if used[n] {
                let next = WfDeopt(intern.len() as u32);
                let key = SnapshotKey::from(s);
                if let Some(&id) = intern.get(&key) {
                    remap[n] = id;
                    used[n] = false;
                } else {
                    intern.insert(key, next);
                    remap[n] = next;
                }
            }
        }
        if intern.len() == f.deopts.len() {
            return false;
        }
        drop(intern);
        let mut it = used.into_iter();
        f.deopts.retain(|_| it.next().unwrap());
        for b in &mut f.blocks {
            for (_, i) in &mut b.insts {
                if let Some(d) = snapshot_mut(i) {
                    *d = remap[d.0 as usize];
                }
            }
        }
        if let Some((_, d)) = &mut f.entry_epoch_guard {
            *d = remap[d.0 as usize];
        }
        true
    }
}

#[cfg(test)]
mod wf_opt_tests {
    use super::optimize;
    use super::super::*;
    use WfInst::*;
    use WfTy::*;

    fn v(n: u32) -> WfValue {
        WfValue(n)
    }
    fn edge(b: u32, args: &[u32]) -> WfEdge {
        WfEdge {
            target: WfBlock(b),
            args: args.iter().map(|&n| v(n)).collect(),
        }
    }
    fn block(params: &[u32], insts: Vec<(u32, WfInst)>, term: WfTerminator) -> WfBlockData {
        WfBlockData {
            params: params.iter().map(|&n| v(n)).collect(),
            insts: insts.into_iter().map(|(n, i)| (v(n), i)).collect(),
            terminator: Some(term),
        }
    }
    fn function(types: &[WfTy], params: &[WfTy], ret: WfTy, blocks: Vec<WfBlockData>) -> WfFunction {
        WfFunction {
            function_index: 0,
            arity: params.len(),
            has_this: false,
            blocks,
            value_types: types.to_vec(),
            deopts: Vec::new(),
            entry_epoch_guard: None,
            return_ty: ret,
            check_resume_pc: false,
            nested_banks: false,
            param_types: params.to_vec(),
            inlined_calls: 0,
        }
    }
    fn state(pc: u32, regs: &[(u8, WfValue)]) -> WfDeoptState {
        WfDeoptState {
            op_index: pc,
            speculative: true,
            registers: regs.to_vec(),
            frames: vec![],
        }
    }
    fn ret(n: u32) -> WfTerminator {
        WfTerminator::Return(v(n))
    }
    fn branch(c: u32, a: u32, b: u32) -> WfTerminator {
        WfTerminator::Branch {
            condition: v(c),
            then: edge(a, &[]),
            otherwise: edge(b, &[]),
        }
    }
    fn all(f: &WfFunction) -> impl Iterator<Item = &WfInst> {
        f.blocks.iter().flat_map(|b| b.insts.iter().map(|(_, i)| i))
    }

    #[test]
    fn sccp_propagates_through_executable_phi_edges() {
        let mut f = function(
            &[Tagged, Bool, Tagged, Tagged, Tagged, Bool, Tagged, Tagged],
            &[Tagged], Tagged,
            vec![
                block(&[], vec![(0, Param(0)), (1, Truthy(v(0)))], branch(1, 1, 2)),
                block(&[], vec![(2, ConstTagged(TAG_TRUE))], WfTerminator::Jump(edge(3, &[2]))),
                block(&[], vec![(3, ConstTagged(TAG_TRUE))], WfTerminator::Jump(edge(3, &[3]))),
                block(&[4], vec![(5, Truthy(v(4)))], branch(5, 4, 5)),
                block(&[], vec![(6, ConstTagged(TAG_TRUE))], ret(6)),
                block(&[], vec![(7, ConstTagged(TAG_FALSE))], ret(7)),
            ],
        );
        optimize(&mut f).unwrap();
        // The join is constant `true`: the second branch folds, and since both
        // arms of the first one feed that same constant, it may fold as well.
        assert!(f.blocks.iter().filter(|b|
            matches!(b.terminator, Some(WfTerminator::Branch { .. }))).count() <= 1);
        assert!(!all(&f).any(|i| matches!(i, ConstTagged(w) if *w == TAG_FALSE)));
        assert!(all(&f).any(|i| matches!(i, ConstTagged(w) if *w == TAG_TRUE)));
    }

    #[test]
    fn shared_or_unknown_receivers_are_not_forwarded() {
        let mut f = function(
            &[Tagged, Tagged, Bool, Tagged, Tagged], &[Tagged], Tagged,
            vec![block(&[], vec![
                (0, Param(0)),
                (1, ConstTagged(TAG_TRUE)),
                (2, InstanceFieldSet { object:v(0), value:v(1), slot:4,
                    class_key:8, value_immediate:true, deopt:WfDeopt(0) }),
                (3, InstanceFieldGet { object:v(0), slot:4, nullable:true, deopt:WfDeopt(0) }),
                (4, InstanceFieldGet { object:v(0), slot:4, nullable:true, deopt:WfDeopt(0) }),
            ], ret(4))],
        );
        f.deopts.push(state(2, &[(0,v(0))]));
        optimize(&mut f).unwrap();
        assert_eq!(all(&f).filter(|i| matches!(i, InstanceFieldGet { .. })).count(), 2);
        assert_eq!(all(&f).filter(|i| matches!(i, InstanceFieldSet { .. })).count(), 1);
    }

    #[test]
    fn fresh_object_fields_forward_only_after_a_tracking_guard() {
        let mut f = function(
            &[Tagged, Tagged, Bool, Tagged, Tagged], &[], Tagged,
            vec![block(&[], vec![
                (0, AllocObject { op_index: 0, kind: WfAllocKind::SpecializedObject,
                    state: WfDeopt(0) }),
                (1, ConstTagged(TAG_TRUE)),
                (2, InstanceFieldSet { object:v(0), value:v(1), slot:4,
                    class_key:8, value_immediate:true, deopt:WfDeopt(0) }),
                (3, InstanceFieldGet { object:v(0), slot:4, nullable:true, deopt:WfDeopt(0) }),
                (4, InstanceFieldGet { object:v(0), slot:4, nullable:true, deopt:WfDeopt(0) }),
            ], ret(4))],
        );
        f.deopts.push(state(0, &[]));
        optimize(&mut f).unwrap();
        // The first read establishes that dependency tracking is disabled.
        assert_eq!(all(&f).filter(|i| matches!(i, InstanceFieldGet { .. })).count(), 1);
    }

    #[test]
    fn snapshots_keep_raw_loop_values_without_hot_path_boxing() {
        use super::super::wf_optimizer::{Analyses, WfOptimization, verify};
        let mut f = function(
            &[Tagged, Bool, Tagged, Tagged, Bool, Bool, Tagged], &[Tagged], Tagged,
            vec![
                block(&[], vec![(0, Param(0)), (1, Truthy(v(0))), (2, BoxBool(v(1)))],
                    WfTerminator::Jump(edge(1, &[2]))),
                block(&[3], vec![
                    (4, Truthy(v(3))),
                    (5, GuardEpoch { expected: 1, deopt: WfDeopt(0) }),
                ], branch(4, 2, 3)),
                block(&[], vec![], WfTerminator::Jump(edge(1, &[3]))),
                block(&[], vec![(6, ConstTagged(TAG_NULL))], ret(6)),
            ],
        );
        f.deopts.push(state(1, &[(0, v(3))]));
        assert!(super::BlockRepresentation.run(&mut f, &mut Analyses::default()));
        verify(&f).unwrap();
        let raw = f.blocks[1].params[0];
        assert_eq!(f.ty(raw), Bool);
        assert_eq!(f.deopts[0].registers[0].1, raw);
        assert!(!f.blocks[1].insts.iter().any(|(_, inst)| matches!(inst, BoxBool(_))));
    }

    #[test]
    fn executable_edge_pruning_precedes_representation_meet() {
        use super::super::wf_optimizer::{Analyses, WfOptimization, verify};
        let mut f = function(
            &[Bool, Tagged, Bool, Tagged, F64, Tagged, Tagged], &[Tagged], Tagged,
            vec![
                block(&[], vec![(0, ConstBool(true)), (1, Param(0)), (2, Truthy(v(1)))],
                    branch(0, 1, 2)),
                block(&[], vec![(3, BoxBool(v(2)))],
                    WfTerminator::Jump(edge(3, &[3]))),
                block(&[], vec![(4, ConstF64(1.0f64.to_bits())), (5, BoxF64(v(4)))],
                    WfTerminator::Jump(edge(3, &[5]))),
                block(&[6], vec![], ret(6)),
            ],
        );
        let mut analyses = Analyses::default();
        assert!(super::Sccp.run(&mut f, &mut analyses));
        assert!(super::ControlFlow.run(&mut f, &mut analyses));
        assert!(super::BlockRepresentation.run(&mut f, &mut analyses));
        verify(&f).unwrap();
        assert!(f.blocks.iter().any(|block| block.params.iter().any(|p| f.ty(*p) == Bool)));
    }

    #[test]
    fn optimistic_representation_handles_a_loop_carried_self_edge() {
        use super::super::wf_optimizer::{Analyses, WfOptimization, verify};
        let mut f = function(
            &[Tagged, Bool, Tagged, Tagged, Bool], &[Tagged], Tagged,
            vec![
                block(&[], vec![(0, Param(0)), (1, Truthy(v(0))), (2, BoxBool(v(1)))],
                    WfTerminator::Jump(edge(1, &[2]))),
                block(&[3], vec![(4, Truthy(v(3)))],
                    WfTerminator::Branch {
                        condition: v(4),
                        then: edge(2, &[]),
                        otherwise: edge(3, &[]),
                    }),
                block(&[], vec![], WfTerminator::Jump(edge(1, &[3]))),
                block(&[], vec![], ret(3)),
            ],
        );
        assert!(super::BlockRepresentation.run(&mut f, &mut Analyses::default()));
        verify(&f).unwrap();
        assert_eq!(f.ty(f.blocks[1].params[0]), Bool);
        let carried = f.blocks[1].params[0];
        let backedge = match f.blocks[2].terminator.as_ref().unwrap() {
            WfTerminator::Jump(edge) => edge,
            _ => unreachable!(),
        };
        assert_eq!(backedge.args, vec![carried]);
    }

    #[test]
    fn heap_facts_do_not_cross_a_join_with_a_mutating_predecessor() {
        let mut f = function(
            &[Tagged, Bool, Tagged, Bool, Tagged], &[Tagged], Tagged,
            vec![
                block(&[], vec![
                    (0, Param(0)),
                    (1, Truthy(v(0))),
                    (2, InstanceFieldGet {object:v(0),slot:4,nullable:true,deopt:WfDeopt(0)}),
                ], branch(1, 1, 2)),
                block(&[], vec![(3, RuntimeOp {
                    helper:WfRuntimeHelper::Op, op_index:2, next_op:3,
                    state:WfDeopt(0), dest:None, clobbers:true,
                })], WfTerminator::Jump(edge(3,&[]))),
                block(&[], vec![], WfTerminator::Jump(edge(3,&[]))),
                block(&[], vec![(4, InstanceFieldGet {
                    object:v(0), slot:4, nullable:true, deopt:WfDeopt(0),
                })], ret(4)),
            ],
        );
        f.deopts.push(state(2, &[(0,v(0))]));
        optimize(&mut f).unwrap();
        assert_eq!(all(&f).filter(|i| matches!(i, InstanceFieldGet { .. })).count(), 2);
    }

    #[test]
    fn boxed_boolean_joins_get_raw_block_parameters() {
        let mut f = function(
            &[Tagged, Bool, Bool, Tagged, Bool, Tagged, Tagged],
            &[Tagged], Tagged,
            vec![
                block(&[], vec![(0,Param(0)),(1,Truthy(v(0)))], branch(1,1,2)),
                block(&[], vec![(2,ConstBool(true)),(3,BoxBool(v(2)))],
                    WfTerminator::Jump(edge(3,&[3]))),
                block(&[], vec![(4,ConstBool(false)),(5,BoxBool(v(4)))],
                    WfTerminator::Jump(edge(3,&[5]))),
                block(&[6], vec![], ret(6)),
            ],
        );
        optimize(&mut f).unwrap();
        assert!(f.blocks.iter().any(|block| block.params.iter().any(|p| f.ty(*p) == Bool)), "{f:#?}");
    }

    #[test]
    fn folded_numbers_match_bits_and_unordered_comparisons() {
        let inputs = [
            0.,
            -0.,
            1.,
            -3.,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0xfff0_0000_0000_0081),
        ];
        for a in inputs {
            for b in inputs {
                for op in [
                    WfFloatBin::Add,
                    WfFloatBin::Sub,
                    WfFloatBin::Mul,
                    WfFloatBin::Div,
                ] {
                    let mut f = function(
                        &[F64, F64, F64],
                        &[],
                        F64,
                        vec![block(
                            &[],
                            vec![
                                (0, ConstF64(a.to_bits())),
                                (1, ConstF64(b.to_bits())),
                                (
                                    2,
                                    FloatBin {
                                        op,
                                        lhs: v(0),
                                        rhs: v(1),
                                    },
                                ),
                            ],
                            ret(2),
                        )],
                    );
                    let expected = match op {
                        WfFloatBin::Add => a + b,
                        WfFloatBin::Sub => a - b,
                        WfFloatBin::Mul => a * b,
                        WfFloatBin::Div => a / b,
                    };
                    optimize(&mut f).unwrap();
                    if expected.is_nan() {
                        assert!(all(&f).any(|i| matches!(i, FloatBin { .. })));
                    } else {
                        assert_eq!(
                            all(&f).collect::<Vec<_>>(),
                            vec![&ConstF64(expected.to_bits())]
                        );
                    }
                }
                for cc in [
                    WfFloatCmp::Eq,
                    WfFloatCmp::Ne,
                    WfFloatCmp::Lt,
                    WfFloatCmp::Le,
                    WfFloatCmp::Gt,
                    WfFloatCmp::Ge,
                ] {
                    let mut f = function(
                        &[F64, F64, Bool],
                        &[],
                        Bool,
                        vec![block(
                            &[],
                            vec![
                                (0, ConstF64(a.to_bits())),
                                (1, ConstF64(b.to_bits())),
                                (
                                    2,
                                    FloatCmp {
                                        cc,
                                        lhs: v(0),
                                        rhs: v(1),
                                    },
                                ),
                            ],
                            ret(2),
                        )],
                    );
                    let expected = match cc {
                        WfFloatCmp::Eq => a == b,
                        WfFloatCmp::Ne => a != b,
                        WfFloatCmp::Lt => a < b,
                        WfFloatCmp::Le => a <= b,
                        WfFloatCmp::Gt => a > b,
                        WfFloatCmp::Ge => a >= b,
                    };
                    optimize(&mut f).unwrap();
                    assert_eq!(all(&f).collect::<Vec<_>>(), vec![&ConstBool(expected)]);
                }
            }
        }
    }

    #[test]
    fn boxing_canonicalizes_nan_before_raw_return() {
        let mut f = function(
            &[F64, Tagged, F64],
            &[],
            F64,
            vec![block(
                &[],
                vec![
                    (0, ConstF64(0xfff8_0000_0000_0001)),
                    (1, BoxF64(v(0))),
                    (
                        2,
                        UnboxNumber {
                            value: v(1),
                            deopt: WfDeopt(0),
                        },
                    ),
                ],
                ret(2),
            )],
        );
        f.deopts.push(state(2, &[]));
        optimize(&mut f).unwrap();
        assert!(f.deopts.is_empty());
        assert_eq!(all(&f).collect::<Vec<_>>(), vec![&ConstF64(CANONICAL_NAN)]);
    }

    #[test]
    fn dangerous_guards_keep_their_exit_and_inlined_frame() {
        let mut f = function(
            &[Tagged, Tagged, F64, F64],
            &[],
            F64,
            vec![block(
                &[],
                vec![
                    (0, ConstTagged(TAG_NULL)),
                    (1, ConstTagged(TAG_TRUE)),
                    (
                        2,
                        UnboxNumber {
                            value: v(0),
                            deopt: WfDeopt(0),
                        },
                    ),
                    (
                        3,
                        UnboxNumber {
                            value: v(0),
                            deopt: WfDeopt(1),
                        },
                    ),
                ],
                ret(3),
            )],
        );
        f.deopts = vec![state(3, &[]), state(4, &[])];
        f.deopts[0].frames.push(WfInlineDeoptFrame {
            callee: 2,
            call_op: 3,
            resume_op: 4,
            registers: vec![(1, v(1))],
        });
        optimize(&mut f).unwrap();
        assert_eq!(f.deopts.len(), 1);
        assert_eq!(f.deopts[0].frames[0].registers, vec![(1, v(1))]);
        assert_eq!(
            all(&f).filter(|i| matches!(i, UnboxNumber { .. })).count(),
            1
        );
        assert!(all(&f).any(|i| *i == ConstTagged(TAG_TRUE)));
    }

    #[test]
    fn join_keeps_distinct_arguments_for_parallel_edges() {
        let mut f = function(
            &[Tagged, Bool, F64, F64, F64],
            &[Tagged],
            F64,
            vec![
                block(
                    &[],
                    vec![
                        (0, Param(0)),
                        (1, Truthy(v(0))),
                        (2, ConstF64(0)),
                        (3, ConstF64(1)),
                    ],
                    WfTerminator::Branch {
                        condition: v(1),
                        then: edge(1, &[2]),
                        otherwise: edge(1, &[3]),
                    },
                ),
                block(&[4], vec![], ret(4)),
            ],
        );
        optimize(&mut f).unwrap();
        assert_eq!(f.blocks.len(), 2);
        assert_eq!(f.blocks[1].params, vec![v(4)]);
        let Some(WfTerminator::Branch {
            then, otherwise, ..
        }) = &f.blocks[0].terminator
        else {
            panic!("branch lost")
        };
        assert_ne!(then.args, otherwise.args);
    }

    #[test]
    fn loop_drops_dead_phi_cycles_and_hoists_invariants() {
        let mut f = function(
            &[F64, Tagged, F64, F64, F64, F64, Bool, Bool, F64],
            &[F64, Tagged],
            F64,
            vec![
                block(
                    &[],
                    vec![(0, Param(0)), (1, Param(1))],
                    WfTerminator::Jump(edge(1, &[0, 0])),
                ),
                block(
                    &[2, 3],
                    vec![(4, FloatNeg(v(2))), (5, FloatNeg(v(3))), (6, Truthy(v(1)))],
                    branch(6, 2, 3),
                ),
                block(
                    &[],
                    vec![
                        (
                            7,
                            LoopBackedge {
                                counter: 1,
                                trace_slot: 2,
                                shared_flag: 3,
                                deopt: WfDeopt(0),
                            },
                        ),
                        (
                            8,
                            FloatBin {
                                op: WfFloatBin::Add,
                                lhs: v(5),
                                rhs: v(3),
                            },
                        ),
                    ],
                    WfTerminator::Jump(edge(1, &[2, 8])),
                ),
                block(&[], vec![], ret(4)),
            ],
        );
        f.deopts.push(state(8, &[(2, v(4))]));
        optimize(&mut f).unwrap();
        assert!(f.blocks.iter().all(|b| b.params.is_empty()));
        assert!(
            f.blocks[0]
                .insts
                .iter()
                .any(|(_, i)| matches!(i, FloatNeg(_)))
        );
        assert_eq!(
            all(&f).filter(|i| matches!(i, LoopBackedge { .. })).count(),
            1
        );
        assert!(!all(&f).any(|i| matches!(i, FloatBin { .. })));
    }

    #[test]
    fn snapshots_intern_only_exact_matches_and_keep_epoch_roots() {
        let mut f = function(
            &[Tagged, Bool, Bool, Bool, Bool],
            &[],
            Tagged,
            vec![block(
                &[],
                vec![
                    (0, ConstTagged(23)),
                    (
                        1,
                        GuardEpoch {
                            expected: 2,
                            deopt: WfDeopt(1),
                        },
                    ),
                    (
                        2,
                        GuardEpoch {
                            expected: 3,
                            deopt: WfDeopt(2),
                        },
                    ),
                    (
                        3,
                        GuardEpoch {
                            expected: 4,
                            deopt: WfDeopt(3),
                        },
                    ),
                    (
                        4,
                        GuardEpoch {
                            expected: 5,
                            deopt: WfDeopt(4),
                        },
                    ),
                ],
                ret(0),
            )],
        );
        f.deopts = vec![
            state(10, &[]),
            state(4, &[(1, v(0))]),
            state(4, &[(1, v(0))]),
            state(4, &[(1, v(0))]),
            state(4, &[(1, v(0))]),
        ];
        f.deopts[3].speculative = false;
        f.deopts[4].frames.push(WfInlineDeoptFrame {
            callee: 5,
            call_op: 4,
            resume_op: 7,
            registers: vec![],
        });
        f.entry_epoch_guard = Some((7, WfDeopt(0)));
        optimize(&mut f).unwrap();
        assert_eq!(f.deopts.len(), 4);
        assert_eq!(
            f.deopts[f.entry_epoch_guard.unwrap().1.0 as usize].op_index,
            10
        );
    }

    #[test]
    fn helper_invalidates_frame_and_class_facts() {
        let mut f = function(
            &[Tagged, Bool, Bool, Bool, Tagged, Bool],
            &[],
            Tagged,
            vec![block(
                &[],
                vec![
                    (0, FrameLoad(1)),
                    (
                        1,
                        GuardClass {
                            value: v(0),
                            class: 2,
                            deopt: WfDeopt(0),
                        },
                    ),
                    (
                        2,
                        GuardClass {
                            value: v(0),
                            class: 2,
                            deopt: WfDeopt(0),
                        },
                    ),
                    (
                        3,
                        RuntimeOp {
                            helper: WfRuntimeHelper::Op,
                            op_index: 2,
                            next_op: 3,
                            state: WfDeopt(0),
                            dest: None,
                            clobbers: true,
                        },
                    ),
                    (4, FrameLoad(1)),
                    (
                        5,
                        GuardClass {
                            value: v(0),
                            class: 2,
                            deopt: WfDeopt(1),
                        },
                    ),
                ],
                ret(4),
            )],
        );
        f.deopts = vec![state(2, &[(1, v(0))]), state(3, &[(1, v(4))])];
        optimize(&mut f).unwrap();
        assert_eq!(all(&f).filter(|i| matches!(i, FrameLoad(_))).count(), 2);
        assert_eq!(
            all(&f).filter(|i| matches!(i, GuardClass { .. })).count(),
            2
        );
    }

    #[test]
    fn verifier_rejects_non_dominating_snapshot() {
        let mut f = function(
            &[Tagged, Bool, Tagged, Bool],
            &[Tagged],
            Tagged,
            vec![
                block(&[], vec![(0, Param(0)), (1, Truthy(v(0)))], branch(1, 1, 2)),
                block(
                    &[],
                    vec![(2, ConstTagged(TAG_TRUE))],
                    WfTerminator::Jump(edge(2, &[])),
                ),
                block(
                    &[],
                    vec![(
                        3,
                        GuardEpoch {
                            expected: 0,
                            deopt: WfDeopt(0),
                        },
                    )],
                    ret(0),
                ),
            ],
        );
        f.deopts = vec![state(4, &[(7, v(2))])];
        assert!(optimize(&mut f).is_err());
    }

    #[test]
    fn signed_zero_identities_require_an_insensitive_use() {
        fn example(ret_ty: WfTy, zero: f64) -> WfFunction {
            function(
                &[F64, F64, F64, Tagged],
                &[F64],
                ret_ty,
                vec![block(
                    &[],
                    vec![
                        (0, Param(0)),
                        (1, ConstF64(zero.to_bits())),
                        (
                            2,
                            FloatBin {
                                op: WfFloatBin::Add,
                                lhs: v(0),
                                rhs: v(1),
                            },
                        ),
                        (3, BoxF64(v(2))),
                    ],
                    ret(if ret_ty == Tagged { 3 } else { 2 }),
                )],
            )
        }
        let mut boxed = example(Tagged, -0.);
        optimize(&mut boxed).unwrap();
        assert_eq!(
            all(&boxed).collect::<Vec<_>>(),
            vec![&Param(0), &BoxF64(v(0))]
        );
        for (ty, z) in [(Tagged, 0.), (F64, -0.)] {
            let mut f = example(ty, z);
            optimize(&mut f).unwrap();
            assert!(all(&f).any(|i| matches!(i, FloatBin { .. })));
        }
    }

    #[test]
    fn gvn_obeys_dominance_and_joins_duplicate_phi_columns() {
        let mut f = function(
            &[Tagged, Bool, F64, F64, F64, F64, F64, F64, F64],
            &[Tagged, F64],
            F64,
            vec![
                block(
                    &[],
                    vec![(0, Param(0)), (1, Truthy(v(0))), (2, Param(1))],
                    branch(1, 1, 2),
                ),
                block(
                    &[],
                    vec![(3, FloatNeg(v(2))), (4, FloatNeg(v(2)))],
                    WfTerminator::Jump(edge(3, &[3, 4])),
                ),
                block(
                    &[],
                    vec![(5, ConstF64(0))],
                    WfTerminator::Jump(edge(3, &[5, 5])),
                ),
                block(
                    &[6, 7],
                    vec![(
                        8,
                        FloatBin {
                            op: WfFloatBin::Mul,
                            lhs: v(6),
                            rhs: v(7),
                        },
                    )],
                    ret(8),
                ),
            ],
        );
        optimize(&mut f).unwrap();
        assert_eq!(f.blocks.iter().map(|b| b.params.len()).sum::<usize>(), 1);
        assert_eq!(all(&f).filter(|i| matches!(i, FloatNeg(_))).count(), 1);
        assert!(all(&f).any(|i| matches!(i,FloatBin{lhs,rhs,..} if lhs==rhs)));
    }

    #[test]
    fn dominated_predicate_removes_infeasible_effect() {
        let mut f = function(
            &[Tagged, Bool, Bool, Tagged, Tagged],
            &[Tagged],
            Tagged,
            vec![
                block(&[], vec![(0, Param(0)), (1, Truthy(v(0)))], branch(1, 1, 2)),
                block(&[], vec![], branch(1, 3, 4)),
                block(&[], vec![(3, ConstTagged(TAG_NULL))], ret(3)),
                block(&[], vec![(4, ConstTagged(TAG_TRUE))], ret(4)),
                block(
                    &[],
                    vec![(
                        2,
                        GuardEpoch {
                            expected: 7,
                            deopt: WfDeopt(0),
                        },
                    )],
                    ret(0),
                ),
            ],
        );
        f.deopts.push(state(9, &[(0, v(0))]));
        optimize(&mut f).unwrap();
        assert!(f.deopts.is_empty());
        assert_eq!(
            f.blocks
                .iter()
                .filter(|b| matches!(b.terminator, Some(WfTerminator::Branch { .. })))
                .count(),
            1
        );
    }
}
