//! Shared traversal, CFG analysis and verified pass scheduling.
use super::{WfDeopt, WfDeoptState, WfEdge, WfFunction, WfInst, WfTerminator, WfTy, WfValue};

/// A pass can use CFG and dominators without rebuilding them if it preserves edges.
pub(super) trait WfOptimization {
    fn run(&self, f: &mut WfFunction, analyses: &mut Analyses) -> bool;
    fn invalidates_cfg(&self) -> bool { false }
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

pub(super) struct Optimize<'a> {
    pub passes: &'a [&'a dyn WfOptimization],
    pub max_rounds: usize,
    pub verify_each: bool,
}

/// Cached structural analyses. Passes that modify the graph must invalidate them.
#[derive(Default)]
pub(super) struct Analyses {
    cfg: Option<Cfg>,
}

impl Analyses {
    pub fn cfg<'a>(&'a mut self, f: &WfFunction) -> &'a Cfg {
        self.cfg.get_or_insert_with(|| Cfg::new(f))
    }

    fn invalidate_cfg(&mut self) { self.cfg = None; }
}

impl Optimize<'_> {
    pub fn run(&self, f: &mut WfFunction) -> Result<(), String> {
        verify(f)?;
        let mut analyses = Analyses::default();
        let profile = std::env::var_os("NAPALM_JIT_PASS_EVENTS").is_some();
        for round in 0..self.max_rounds {
            let mut changed = false;
            for pass in self.passes {
                let start = profile.then(std::time::Instant::now);
                let modified = pass.run(f, &mut analyses);
                if let Some(start) = start {
                    eprintln!("[napalm-wfjit] wf-opt round={round} pass={} changed={modified} us={}",
                        pass.name(), start.elapsed().as_micros());
                }
                if modified && pass.invalidates_cfg() {
                    analyses.invalidate_cfg();
                }
                if self.verify_each && modified {
                    verify_with_cfg(f, analyses.cfg(f))
                        .map_err(|err| format!("{}: {err}", pass.name()))?;
                }
                changed |= modified;
            }
            if !changed { break; }
        }
        verify_with_cfg(f, analyses.cfg(f))
    }
}

// Generate read and write visitors from the same exhaustive operand list.
macro_rules! operands {
    ($name:ident, $($mut:tt)?) => {
        pub(super) fn $name(i: &$($mut)? WfInst, mut visit: impl FnMut(&$($mut)? WfValue)) {
            use WfInst::*;
            match i {
                UnboxNumber {value,..} | GuardClass {value,..} | AssumeNumber(value) |
                AssumeBool(value) | BoxBool(value) | BoxF64(value) | Truthy(value) | FloatNeg(value) |
                InstanceFieldGet {object:value,..} | MapPropertyGet {object:value,..} => visit(value),
                FloatBin {lhs,rhs,..} | FloatCmp {lhs,rhs,..} | IdentityCompare {lhs,rhs,..} |
                InstanceFieldSet {object:lhs,value:rhs,..} | InstanceFieldStoreChecked {object:lhs,value:rhs,..} |
                StringConcat {left:lhs,right:rhs,..} => { visit(lhs); visit(rhs); }
                SelfCall {args,..} | NewObject {fields:args,..} => args.into_iter().for_each(visit),
                Primitive {receiver, arg, ..} => { visit(receiver); if let Some(arg) = arg { visit(arg); } }
                CallExplicit {receiver, args, ..} => {
                    if let Some(receiver) = receiver { visit(receiver); }
                    args.into_iter().for_each(visit);
                }
                Param(_) | ConstTagged(_) | ConstBool(_) | ConstF64(_) | LoadAbsolute(_) | AllocObject {..} |
                RuntimeOp {..} | FrameLoad(_) | GuardEpoch {..} | LoopBackedge {..} => {}
            }
        }
    };
}
operands!(uses,);
operands!(uses_mut, mut);
macro_rules! snapshots {
    ($name:ident, $($mut:tt)?) => {
        pub(super) fn $name(i: &$($mut)? WfInst) -> Option<&$($mut)? WfDeopt> {
            use WfInst::*;
            match i {
                UnboxNumber {deopt,..} | GuardClass {deopt,..} | InstanceFieldGet {deopt,..} |
                MapPropertyGet {deopt,..} | IdentityCompare {deopt,..} | InstanceFieldSet {deopt,..} | InstanceFieldStoreChecked {deopt,..} |
                SelfCall {deopt,..} | GuardEpoch {deopt,..} | LoopBackedge {deopt,..} | Primitive {deopt,..} |
                CallExplicit {state:deopt,..} |
                AllocObject {state:deopt,..} | StringConcat {state:deopt,..} | RuntimeOp {state:deopt,..} |
                NewObject {deopt,..} => Some(deopt),
                Param(_) | ConstTagged(_) | ConstF64(_) | ConstBool(_) | LoadAbsolute(_) |
                AssumeNumber(_) | AssumeBool(_) | BoxF64(_) | BoxBool(_) | Truthy(_) |
                FloatBin {..} | FloatCmp {..} | FloatNeg(_) | FrameLoad(_) => None,
            }
        }
    };
}
snapshots!(snapshot,);
snapshots!(snapshot_mut, mut);
macro_rules! edge_iter {
    ($name:ident, $($mut:tt)?) => {
        pub(super) fn $name(t: &$($mut)? WfTerminator) -> impl Iterator<Item=&$($mut)? WfEdge> {
            let pair = match t {
                WfTerminator::Branch {then,otherwise,..} => [Some(then),Some(otherwise)],
                WfTerminator::Jump(e) => [Some(e),None],
                WfTerminator::Return(_) => [None,None],
            };
            pair.into_iter().flatten()
        }
    };
}
edge_iter!(edges,);
edge_iter!(edges_mut, mut);

pub(super) fn state_values(s: &WfDeoptState) -> impl Iterator<Item = WfValue> + '_ {
    s.registers
        .iter()
        .chain(s.frames.iter().flat_map(|fr| &fr.registers))
        .map(|&(_, v)| v)
}
pub(super) fn control(t: &WfTerminator) -> Option<WfValue> {
    match t {
        WfTerminator::Branch { condition, .. } => Some(*condition),
        WfTerminator::Return(v) => Some(*v),
        _ => None,
    }
}
pub(super) fn rewrite(f: &mut WfFunction, aliases: &[WfValue]) {
    let replace = |v: &mut WfValue| *v = aliases[v.0 as usize];
    for b in &mut f.blocks {
        for (_, i) in &mut b.insts {
            uses_mut(i, replace);
        }
        let t = b.terminator.as_mut().unwrap();
        match t {
            WfTerminator::Branch { condition, .. } | WfTerminator::Return(condition) => {
                replace(condition)
            }
            _ => {}
        }
        for e in edges_mut(t) {
            e.args.iter_mut().for_each(replace);
        }
    }
    for s in &mut f.deopts {
        for (_, v) in s
            .registers
            .iter_mut()
            .chain(s.frames.iter_mut().flat_map(|fr| &mut fr.registers))
        {
            replace(v);
        }
    }
}

pub(super) struct Cfg {
    pub pred: Vec<smallvec::SmallVec<[usize; 2]>>,
    pub children: Vec<smallvec::SmallVec<[usize; 2]>>,
    enter: Vec<usize>,
    leave: Vec<usize>,
}
impl Cfg {
    pub fn new(f: &WfFunction) -> Self {
        let n = f.blocks.len();
        let mut pred = vec![smallvec::SmallVec::new(); n];
        for (b, data) in f.blocks.iter().enumerate() {
            for e in edges(data.terminator.as_ref().unwrap()) {
                pred[e.target.0 as usize].push(b);
            }
        }
        let mut seen = vec![false; n];
        let mut stack = vec![(0, false)];
        let mut rpo = Vec::new();
        while let Some((b, exit)) = stack.pop() {
            if exit {
                rpo.push(b);
                continue;
            }
            if std::mem::replace(&mut seen[b], true) {
                continue;
            }
            stack.push((b, true));
            stack.extend(
                edges(f.blocks[b].terminator.as_ref().unwrap())
                    .map(|e| (e.target.0 as usize, false)),
            );
        }
        rpo.reverse();
        let mut rank = vec![usize::MAX; n];
        for (i, &b) in rpo.iter().enumerate() {
            rank[b] = i;
        }
        let mut parent = vec![usize::MAX; n];
        parent[0] = 0;
        loop {
            let mut changed = false;
            for &b in &rpo[1..] {
                let mut incoming = pred[b].iter().copied().filter(|&p| parent[p] != usize::MAX);
                let mut common = incoming.next().unwrap();
                for mut p in incoming {
                    while common != p {
                        if rank[common] > rank[p] {
                            common = parent[common]
                        } else {
                            p = parent[p]
                        }
                    }
                }
                changed |= parent[b] != common;
                parent[b] = common;
            }
            if !changed {
                break;
            }
        }
        let mut children = vec![smallvec::SmallVec::new(); n];
        for &b in &rpo[1..] {
            children[parent[b]].push(b);
        }
        let mut enter = vec![usize::MAX; n];
        let mut leave = vec![0; n];
        stack.push((0, false));
        let mut clock = 0;
        while let Some((b, exit)) = stack.pop() {
            if exit {
                leave[b] = clock;
            } else {
                enter[b] = clock;
                clock += 1;
                stack.push((b, true));
                stack.extend(children[b].iter().map(|&c| (c, false)));
            }
        }
        Self {
            pred,
            children,
            enter,
            leave,
        }
    }
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        self.enter[a] <= self.enter[b] && self.enter[b] < self.leave[a]
    }
}

pub(super) fn verify(f: &WfFunction) -> Result<(), String> {
    verify_with_cfg(f, &Cfg::new(f))
}

fn verify_with_cfg(f: &WfFunction, cfg: &Cfg) -> Result<(), String> {
    let n = f.value_types.len();
    if f.blocks.is_empty() || !f.blocks[0].params.is_empty() || f.param_types.len() != f.arity {
        return Err("invalid WF entry signature".into());
    }
    let mut defs = vec![None; n];
    for (b, data) in f.blocks.iter().enumerate() {
        for (v, pos) in data
            .params
            .iter()
            .map(|v| (v, 0))
            .chain(data.insts.iter().enumerate().map(|(i, (v, _))| (v, i + 1)))
        {
            if defs
                .get_mut(v.0 as usize)
                .is_none_or(|d| d.replace((b, pos)).is_some())
            {
                return Err(format!("b{b}: invalid or duplicate definition {v:?}"));
            }
        }
        let t = data
            .terminator
            .as_ref()
            .ok_or_else(|| format!("b{b}: missing terminator"))?;
        for e in edges(t) {
            let target = f
                .blocks
                .get(e.target.0 as usize)
                .ok_or("invalid edge target")?;
            if e.target.0 == 0 || e.args.len() != target.params.len() {
                return Err(format!("b{b}: invalid edge signature"));
            }
        }
    }
    for (b, data) in f.blocks.iter().enumerate() {
        let valid = |v: WfValue, pos: usize| {
            defs.get(v.0 as usize)
                .copied()
                .flatten()
                .is_some_and(|(d, p)| {
                    if d == b {
                        p < pos
                    } else {
                        d < b && cfg.dominates(d, b)
                    }
                })
        };
        let state_ok = |d: WfDeopt, pos: usize| {
            f.deopts
                .get(d.0 as usize)
                .is_some_and(|s| state_values(s).all(|v| valid(v, pos)))
        };
        if b == 0
            && f.entry_epoch_guard
                .is_some_and(|(_, d)| !state_ok(d, usize::MAX))
        {
            return Err("invalid entry snapshot".into());
        }
        for (pos, (result, i)) in data.insts.iter().enumerate() {
            let mut ok = true;
            uses(i, |v| ok &= valid(*v, pos + 1));
            if !ok || snapshot(i).is_some_and(|&d| !state_ok(d, pos + 1)) {
                return Err(format!(
                    "b{b} {result:?}: unavailable SSA operand or snapshot"
                ));
            }
            use WfInst::*;
            use WfTy::*;
            let ty = |v: &WfValue| f.ty(*v);
            let (out, operands) = match i {
                Param(p) => {
                    let p = *p as usize;
                    (
                        *f.param_types.get(p).unwrap_or(&Tagged),
                        p < f.arity || (p == f.arity && f.has_this),
                    )
                }
                ConstTagged(_) | FrameLoad(_) | LoadAbsolute(_) | AllocObject { .. } => {
                    (Tagged, true)
                }
                ConstF64(_) => (F64, true),
                ConstBool(_) | LoopBackedge { .. } | GuardEpoch { .. } => (Bool, true),
                BoxF64(v) => (Tagged, ty(v) == F64),
                BoxBool(v) => (Tagged, ty(v) == Bool),
                UnboxNumber { value: v, .. } | AssumeNumber(v) => (F64, ty(v) == Tagged),
                AssumeBool(v) | Truthy(v) | GuardClass { value: v, .. } => (Bool, ty(v) == Tagged),
                FloatNeg(v) => (F64, ty(v) == F64),
                FloatBin { lhs, rhs, .. } => (F64, ty(lhs) == F64 && ty(rhs) == F64),
                FloatCmp { lhs, rhs, .. } => (Bool, ty(lhs) == F64 && ty(rhs) == F64),
                IdentityCompare { lhs, rhs, .. }
                | InstanceFieldSet {
                    object: lhs,
                    value: rhs,
                    ..
                }
                | InstanceFieldStoreChecked { object: lhs, value: rhs, .. } => (Bool, ty(lhs) == Tagged && ty(rhs) == Tagged),
                InstanceFieldGet { object, .. } | MapPropertyGet { object, .. } => {
                    (Tagged, ty(object) == Tagged)
                }
                StringConcat { left, right, .. } => (Tagged, ty(left) == Tagged && ty(right) == Tagged),
                NewObject { kind, fields, .. } => (Tagged, fields.len() == kind.slots() && fields.iter().all(|v| ty(v) == Tagged)),
                SelfCall { args, .. } => (
                    Tagged,
                    args.len() == f.arity + usize::from(f.has_this)
                        && args
                            .iter()
                            .enumerate()
                            .all(|(n, v)| ty(v) == *f.param_types.get(n).unwrap_or(&Tagged)),
                ),
                RuntimeOp { dest, .. } => (if dest.is_some() { Tagged } else { Bool }, true),
                CallExplicit { receiver, args, .. } => (
                    Tagged,
                    receiver.as_ref().is_none_or(|v| ty(v) == Tagged) && args.iter().all(|v| ty(v) == Tagged),
                ),
                Primitive { kind, receiver, arg, .. } => (
                    kind.result_ty(),
                    ty(receiver) == (if kind.numeric_receiver() { F64 } else { Tagged }) && match arg {
                        Some(arg) => kind.arity() == 1
                            && ty(arg) == if kind.numeric_arg() { F64 } else { Tagged },
                        None => kind.arity() == 0,
                    },
                ),
            };
            if out != f.ty(*result) || !operands {
                return Err(format!("b{b} {result:?}: invalid instruction type"));
            }
        }
        let t = data.terminator.as_ref().unwrap();
        if let Some(v) = control(t) {
            let ty = if matches!(t, WfTerminator::Return(_)) {
                f.return_ty
            } else {
                WfTy::Bool
            };
            if !valid(v, usize::MAX) || f.ty(v) != ty {
                return Err(format!("b{b}: invalid terminator operand"));
            }
        }
        for e in edges(t) {
            for (&v, &p) in e.args.iter().zip(&f.blocks[e.target.0 as usize].params) {
                if !valid(v, usize::MAX) || f.ty(v) != f.ty(p) {
                    return Err(format!("b{b}: invalid edge argument"));
                }
            }
        }
    }
    Ok(())
}
