//! Allocation sinking: escape analysis plus scalar replacement (SROA) of the
//! objects a trace allocates itself (`TraceInst::NewObject`).
//!
//! A trace body is linear, so the analysis is a single forward walk per
//! allocation. An object stays *virtual* when every use is one of:
//!
//! * a proof about the object that holds by construction (`GuardHeapObject` of
//!   its kind, `GuardClass` containing its class, `GuardNotNull`,
//!   `GuardNoContract`, a write barrier on it);
//! * a field read (`HeapLoadField`, `HeapMapGetShaped`, `HeapMapGetPic`): it
//!   becomes the current SSA value of that field;
//! * a field write (`HeapStoreField`, `HeapMapSetShaped`): it only updates the
//!   tracked field state;
//! * a side-exit snapshot: the exit records the object's field state at that
//!   point (`SideExit::virtuals`) and materializes it before resuming;
//! * the backedge of one loop phi (an instance carried to the next iteration,
//!   `acc = acc.add(v)`): the phi is replaced by one phi per field, whose
//!   initial values are loaded once in the preheader from the object the loop
//!   was entered with. Exits during the first iteration hand back that very
//!   object (a flag phi tells them apart), so identity is preserved.
//!
//! Anything else (frame commits, runtime boundaries, storing the object into
//! another object, identity tests...) is an escape and leaves the allocation
//! untouched.

use super::*;
use crate::napalm_trace_ir::{TraceObjectKind, TraceVirtualObject};

/// Words of `TraceActivation::exit_virtual_words` (see the native backends).
const EXIT_VIRTUAL_WORDS: usize = crate::napalm_tracing_jit::TRACE_EXIT_VIRTUAL_WORDS;

pub struct AllocationSinking;

impl TraceOptimization for AllocationSinking {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let preheader: HashSet<usize> = trace.preheader.iter().copied().collect();
        let body: Vec<usize> = (0..trace.nodes.len()).filter(|node| !preheader.contains(node)).collect();
        let candidates: Vec<usize> = body
            .iter()
            .copied()
            .filter(|&node| {
                matches!(trace.nodes[node].kind, TraceInst::NewObject { .. }) && trace.nodes[node].results.len() == 1
            })
            .collect();
        let mut changed = false;
        for node in candidates {
            if let Some(plan) = analyze(trace, node, &body) {
                apply(trace, node, plan);
                changed = true;
            }
        }
        changed
    }
}

/// The previous iteration's object, reached through a loop phi.
struct Carried {
    phi: TraceValueId,
    /// The object the loop was entered with (the phi's initial value).
    initial: TraceValueId,
    class: u64,
    /// Field reads of the carried object: (result, slot).
    loads: Vec<(TraceValueId, usize)>,
    removed: Vec<usize>,
    tagged: HashSet<TraceValueId>,
    /// Live exits naming the carried object.
    exits: Vec<usize>,
    dead_exits: Vec<usize>,
}

struct SinkPlan {
    kind: TraceObjectKind,
    /// Field reads to forward: (result, value).
    loads: Vec<(TraceValueId, TraceValueId)>,
    /// Nodes that disappear with the object (proofs, stores, barriers).
    removed: Vec<usize>,
    /// Tagged names of the object (the allocation and value-preserving guards).
    tagged: HashSet<TraceValueId>,
    /// Field state for each exit whose snapshot names the object.
    exits: Vec<(usize, Vec<TraceValueId>)>,
    /// Exits no node can take any more that still name the object.
    dead_exits: Vec<usize>,
    /// Field state at the end of the body (the carried phis' backedges).
    end_state: Vec<TraceValueId>,
    carried: Option<Carried>,
}

/// `field` as a value of representation `rep`, if no conversion is needed
/// beyond looking through a box.
fn forwarded(trace: &Trace, field: TraceValueId, rep: TraceRep) -> Option<TraceValueId> {
    let field = trace.canonical_value(field);
    let info = trace.value(field)?;
    if info.rep == rep {
        return Some(field);
    }
    let TraceValueOrigin::InstResult { node } = info.origin else { return None };
    let inner = match (&trace.nodes.get(node)?.kind, rep) {
        (TraceInst::BoxF64 { value }, TraceRep::F64)
        | (TraceInst::BoxInt32 { value }, TraceRep::Int32)
        | (TraceInst::BoxUInt32 { value }, TraceRep::UInt32) => trace.canonical_value(*value),
        _ => return None,
    };
    (trace.value(inner)?.rep == rep).then_some(inner)
}

fn is_constant(trace: &Trace, value: TraceValueId) -> bool {
    matches!(
        trace.value(trace.canonical_value(value)).map(|value| &value.origin),
        Some(TraceValueOrigin::Constant(_))
    )
}

fn names(trace: &Trace, exit: &SideExit, tagged: &HashSet<TraceValueId>) -> bool {
    exit.frames.iter().flat_map(|frame| &frame.writes).any(|write| {
        matches!(write.value, SnapshotValue::Value(value) if tagged.contains(&trace.canonical_value(value)))
    })
}

/// Exits some remaining node (or the loop itself) can still take.
fn live_exits(trace: &Trace, removed: &[usize]) -> HashSet<usize> {
    let mut live: HashSet<usize> = HashSet::default();
    for (index, node) in trace.nodes.iter().enumerate() {
        if !matches!(node.kind, TraceInst::Nop) && !removed.contains(&index) {
            if let Some(exit) = node.kind.exit() {
                live.insert(exit.0 as usize);
            }
        }
    }
    live.extend(trace.loop_phis.iter().filter_map(|phi| phi.entry_exit).map(|exit| exit.0 as usize));
    live.extend(trace.native_budget_exit.map(|exit| exit.0 as usize));
    if let TraceTerminator::Exit(exit) = trace.terminator {
        live.insert(exit.0 as usize);
    }
    live
}

fn analyze(trace: &Trace, alloc: usize, body: &[usize]) -> Option<SinkPlan> {
    let TraceInst::NewObject { kind, fields, .. } = &trace.nodes[alloc].kind else { return None };
    let kind = *kind;
    let object = trace.canonical_value(trace.nodes[alloc].results[0]);
    let mut state: Vec<TraceValueId> = fields.iter().map(|&field| trace.canonical_value(field)).collect();
    let mut tagged: HashSet<TraceValueId> = HashSet::default();
    tagged.insert(object);
    let mut entries: HashSet<TraceValueId> = HashSet::default();
    let canonical = |value: TraceValueId| trace.canonical_value(value);

    let mut exit_states: HashMap<usize, Vec<TraceValueId>> = HashMap::default();
    let mut record_exit = |exit: SideExitId, state: &[TraceValueId], tagged: &HashSet<TraceValueId>| -> bool {
        let Some(side_exit) = trace.exits.get(exit.0 as usize) else { return false };
        // Another sunk object's field may name this one: no nesting yet.
        let nested = side_exit.virtuals.iter().flat_map(|object| object.values()).any(|field| {
            matches!(*field, SnapshotValue::Value(value) if tagged.contains(&canonical(value)))
        });
        if nested {
            return false;
        }
        if !names(trace, side_exit, tagged) {
            return true;
        }
        match exit_states.get(&(exit.0 as usize)) {
            // Guards sharing an exit replay the same bytecode, hence must see
            // the same object state.
            Some(known) => known.as_slice() == state,
            None => {
                exit_states.insert(exit.0 as usize, state.to_vec());
                true
            }
        }
    };

    let mut loads = Vec::new();
    let mut removed = Vec::new();
    let start = body.iter().position(|&node| node == alloc)?;
    for &index in &body[start + 1..] {
        let node = &trace.nodes[index];
        // An exit replays this node, so it sees the state *before* its effect.
        // Record it for removed nodes too: other guards may share the exit.
        if let Some(exit) = node.kind.exit() {
            if !record_exit(exit, &state, &tagged) {
                return None;
            }
        }
        let mut uses_object = false;
        node.kind.clone().visit_operands(|value| {
            let value = canonical(*value);
            uses_object |= tagged.contains(&value) || entries.contains(&value);
        });
        if !uses_object {
            continue;
        }
        let result = node.results.first().copied();
        let is_tagged = |value: &TraceValueId| tagged.contains(&canonical(*value));
        let is_ref = |value: &TraceValueId| tagged.contains(&canonical(*value)) || entries.contains(&canonical(*value));
        let tagged_value = |value: &TraceValueId| trace.value(canonical(*value)).is_some_and(|value| value.rep == TraceRep::Tagged);
        match (&node.kind, kind) {
            (TraceInst::GuardHeapObject { value, kind: proven, .. }, _)
                if is_tagged(value) && *proven == kind.heap_kind() =>
            {
                entries.insert(canonical(result?));
            }
            (TraceInst::GuardClass { value, classes, .. }, TraceObjectKind::Instance { class, .. })
                if is_tagged(value) && classes.contains(&class) =>
            {
                tagged.insert(canonical(result?));
            }
            (TraceInst::GuardNotNull { value, .. }, _) if is_tagged(value) => {
                if let Some(result) = result {
                    tagged.insert(canonical(result));
                }
            }
            (TraceInst::GuardNoContract { value, .. }, _) if is_ref(value) => {}
            (TraceInst::HeapWriteBarrier { object: owner, value, .. }, _) if is_ref(owner) && !is_ref(value) => {}
            (TraceInst::HeapMapGetShaped { map, key, shape, slot, .. }, TraceObjectKind::ShapedMap { shape: own, len })
                if is_ref(map) && *shape == own && (*slot as usize) < len as usize && is_constant(trace, *key) =>
            {
                let result = result?;
                loads.push((result, forwarded(trace, state[*slot as usize], trace.value(result)?.rep)?));
            }
            (TraceInst::HeapMapGetPic { map, key, entries: pic, .. }, TraceObjectKind::ShapedMap { shape: own, len })
                if is_ref(map) && is_constant(trace, *key) =>
            {
                let slot = pic.iter().find(|entry| entry.shape == own)?.slot as usize;
                if slot >= len as usize {
                    return None;
                }
                let result = result?;
                loads.push((result, forwarded(trace, state[slot], trace.value(result)?.rep)?));
            }
            (TraceInst::HeapMapSetShaped { map, key, value, shape, slot, .. }, TraceObjectKind::ShapedMap { shape: own, len })
                if is_ref(map) && !is_ref(value) && *shape == own && (*slot as usize) < len as usize
                    && is_constant(trace, *key) && tagged_value(value) =>
            {
                state[*slot as usize] = canonical(*value);
            }
            (TraceInst::HeapLoadField { object: owner, slot, .. }, TraceObjectKind::Instance { fields, .. })
                if is_ref(owner) && (*slot as usize) < fields as usize =>
            {
                let result = result?;
                loads.push((result, forwarded(trace, state[*slot as usize], trace.value(result)?.rep)?));
            }
            (TraceInst::HeapStoreField { object: owner, value, slot, .. }, TraceObjectKind::Instance { fields, .. })
                if is_ref(owner) && !is_ref(value) && (*slot as usize) < fields as usize && tagged_value(value) =>
            {
                state[*slot as usize] = canonical(*value);
            }
            _ => return None,
        }
        removed.push(index);
    }

    // Exits taken at the end of the body see the final state.
    let mut end_exits: Vec<SideExitId> = trace.native_budget_exit.into_iter().collect();
    if let TraceTerminator::Exit(exit) = trace.terminator {
        end_exits.push(exit);
    }
    for exit in end_exits {
        if !record_exit(exit, &state, &tagged) {
            return None;
        }
    }

    // A loop phi may carry the object into the next iteration.
    let mut carried_phi = None;
    for phi in &trace.loop_phis {
        if tagged.contains(&canonical(phi.initial)) {
            return None;
        }
        if phi.backedge.is_some_and(|value| tagged.contains(&canonical(value))) {
            if carried_phi.is_some() || phi.rep != TraceRep::Tagged {
                return None;
            }
            carried_phi = Some((phi.value, canonical(phi.initial)));
        }
    }
    let carried = match carried_phi {
        None => None,
        Some((phi, initial)) => {
            let TraceObjectKind::Instance { class, fields } = kind else { return None };
            Some(analyze_carried(trace, body, phi, initial, class, fields as usize, &tagged, &entries)?)
        }
    };

    // Every live exit naming the object must have been reached after it. An
    // exit no remaining node can take (its guards were optimized away) only
    // needs a placeholder.
    let mut all_removed = removed.clone();
    if let Some(carried) = &carried {
        all_removed.extend_from_slice(&carried.removed);
    }
    let live = live_exits(trace, &all_removed);
    let mut dead_exits = Vec::new();
    for (index, exit) in trace.exits.iter().enumerate() {
        if names(trace, exit, &tagged) && !exit_states.contains_key(&index) {
            if live.contains(&index) {
                return None;
            }
            dead_exits.push(index);
        }
    }
    // Materialization buffer budget, per exit.
    for (index, exit) in trace.exits.iter().enumerate() {
        let own = if exit_states.contains_key(&index) { 1 + kind.slots() } else { 0 };
        let previous = carried.as_ref().is_some_and(|carried| carried.exits.contains(&index));
        let extra = own + if previous { 3 + kind.slots() } else { 0 };
        if extra != 0 && crate::napalm_tracing_jit::virtual_word_count(&exit.virtuals) + extra > EXIT_VIRTUAL_WORDS {
            return None;
        }
    }

    Some(SinkPlan {
        kind,
        loads,
        removed,
        tagged,
        exits: exit_states.into_iter().collect(),
        dead_exits,
        end_state: state,
        carried,
    })
}

/// The carried object `phi` (backedge: the sunk allocation) may become per-field
/// phis when it is only guarded, read and snapshotted, and nothing in the body
/// can change the fields of the object the loop was entered with.
#[allow(clippy::too_many_arguments)]
fn analyze_carried(
    trace: &Trace,
    body: &[usize],
    phi: TraceValueId,
    initial: TraceValueId,
    class: u64,
    fields: usize,
    new_tagged: &HashSet<TraceValueId>,
    new_entries: &HashSet<TraceValueId>,
) -> Option<Carried> {
    let canonical = |value: TraceValueId| trace.canonical_value(value);
    if trace.value(initial)?.rep != TraceRep::Tagged {
        return None;
    }
    let mut tagged: HashSet<TraceValueId> = HashSet::default();
    tagged.insert(canonical(phi));
    let mut entries: HashSet<TraceValueId> = HashSet::default();
    let mut loads = Vec::new();
    let mut removed = Vec::new();
    for &index in body {
        let node = &trace.nodes[index];
        match &node.kind {
            // Field values are loaded once, before the loop: nothing may
            // write an instance field (except the sunk object's own, which
            // disappear) or run arbitrary code inside it.
            TraceInst::Runtime { .. } | TraceInst::RuntimeCall { .. } | TraceInst::ReadBarrier { .. } => return None,
            TraceInst::HeapStoreField { object, .. }
                if !new_tagged.contains(&canonical(*object)) && !new_entries.contains(&canonical(*object)) =>
            {
                return None;
            }
            _ => {}
        }
        let mut uses_old = false;
        node.kind.clone().visit_operands(|value| {
            let value = canonical(*value);
            uses_old |= tagged.contains(&value) || entries.contains(&value);
        });
        if !uses_old {
            continue;
        }
        let result = node.results.first().copied();
        let is_tagged = |value: &TraceValueId| tagged.contains(&canonical(*value));
        let is_ref = |value: &TraceValueId| is_tagged(value) || entries.contains(&canonical(*value));
        match &node.kind {
            TraceInst::GuardClass { value, classes, .. } if is_tagged(value) && classes.contains(&class) => {
                tagged.insert(canonical(result?));
            }
            TraceInst::GuardHeapObject { value, kind: TraceHeapObjectKind::Instance, .. } if is_tagged(value) => {
                entries.insert(canonical(result?));
            }
            TraceInst::GuardNotNull { value, .. } if is_tagged(value) => {
                if let Some(result) = result {
                    tagged.insert(canonical(result));
                }
            }
            TraceInst::HeapLoadField { object, slot, .. } if is_ref(object) && (*slot as usize) < fields => {
                let result = result?;
                if trace.value(result)?.rep != TraceRep::Tagged {
                    return None;
                }
                loads.push((result, *slot as usize));
            }
            _ => return None,
        }
        removed.push(index);
    }
    // No other phi may see the carried object.
    for other in &trace.loop_phis {
        if other.value != phi
            && (tagged.contains(&canonical(other.initial))
                || other.backedge.is_some_and(|value| tagged.contains(&canonical(value))))
        {
            return None;
        }
    }
    let live = live_exits(trace, &removed);
    let mut exits = Vec::new();
    let mut dead_exits = Vec::new();
    for (index, exit) in trace.exits.iter().enumerate() {
        let nested = exit.virtuals.iter().flat_map(|object| object.values()).any(|value| {
            matches!(*value, SnapshotValue::Value(value) if tagged.contains(&canonical(value)))
        });
        if nested {
            return None;
        }
        if names(trace, exit, &tagged) {
            if live.contains(&index) { exits.push(index) } else { dead_exits.push(index) }
        }
    }
    Some(Carried { phi, initial, class, loads, removed, tagged, exits, dead_exits })
}

fn snapshot_value(trace: &Trace, value: TraceValueId) -> SnapshotValue {
    match trace.value(value).map(|value| &value.origin) {
        Some(TraceValueOrigin::Constant(constant)) => SnapshotValue::Constant(*constant),
        _ => SnapshotValue::Value(value),
    }
}

/// Point every write naming `tagged` in exit `exit` at `replacement`.
fn replace_writes(trace: &mut Trace, exit: usize, tagged: &HashSet<TraceValueId>, replacement: SnapshotValue) {
    let canonical: Vec<TraceValueId> = (0..trace.values.len())
        .map(|index| trace.canonical_value(TraceValueId(index as u32)))
        .collect();
    for frame in &mut trace.exits[exit].frames {
        for write in &mut frame.writes {
            if matches!(write.value, SnapshotValue::Value(value) if tagged.contains(&canonical[value.0 as usize])) {
                write.value = replacement;
            }
        }
    }
}

fn new_value(trace: &mut Trace, ty: TraceType, rep: TraceRep, origin: TraceValueOrigin) -> TraceValueId {
    let id = TraceValueId(trace.values.len() as u32);
    trace.values.push(TraceValue { id, ty, rep, origin });
    id
}

fn push_preheader(trace: &mut Trace, kind: TraceInst, result: Option<(TraceType, TraceRep)>) -> Option<TraceValueId> {
    let node = trace.nodes.len();
    let value = result.map(|(ty, rep)| new_value(trace, ty, rep, TraceValueOrigin::InstResult { node }));
    trace.nodes.push(TraceNode {
        id: node,
        kind,
        results: value.into_iter().collect(),
        source: trace.anchor.location,
    });
    trace.preheader.push(node);
    value
}

fn apply(trace: &mut Trace, alloc: usize, plan: SinkPlan) {
    for (exit, state) in plan.exits {
        let fields = state.iter().map(|&field| snapshot_value(trace, field)).collect();
        let index = trace.exits[exit].virtuals.len() as u32;
        trace.exits[exit].virtuals.push(TraceVirtualObject { kind: plan.kind, fields, initial: None });
        replace_writes(trace, exit, &plan.tagged, SnapshotValue::Virtual(index));
    }
    for exit in plan.dead_exits {
        replace_writes(trace, exit, &plan.tagged, SnapshotValue::Constant(TraceConstant::Null));
    }
    for &node in &plan.removed {
        trace.nodes[node].kind = TraceInst::Nop;
    }
    trace.nodes[alloc].kind = TraceInst::Nop;
    for (result, value) in plan.loads {
        alias_value(trace, result, value);
    }

    if let Some(carried) = plan.carried {
        // Preheader: prove the entry object's class (an entry guard resuming
        // at the anchor) and load its fields once.
        let anchor = SideExitId(trace.exits.len() as u32);
        trace.exits.push(SideExit {
            id: anchor,
            target: ExitTarget::Interpreter(trace.anchor.location),
            frames: vec![FrameSnapshot { disposition: FrameDisposition::Existing { depth_from_top: 0 }, writes: Vec::new() }],
            linked_trace: None,
            virtuals: Vec::new(),
        });
        let guarded = push_preheader(
            trace,
            TraceInst::GuardClass { value: carried.initial, classes: vec![carried.class], exit: anchor },
            Some((TraceType::Any, TraceRep::Tagged)),
        )
        .expect("GuardClass has a result");
        let mut field_phis = Vec::with_capacity(plan.end_state.len());
        for (slot, &backedge) in plan.end_state.iter().enumerate() {
            let initial = push_preheader(
                trace,
                TraceInst::HeapLoadField { object: guarded, slot: slot as u32, exit: anchor },
                Some((TraceType::Any, TraceRep::Tagged)),
            )
            .expect("HeapLoadField has a result");
            let value = new_value(trace, TraceType::Any, TraceRep::Tagged, TraceValueOrigin::LoopTemporary);
            trace.loop_phis.push(LoopPhi { value, initial, backedge: Some(backedge), rep: TraceRep::Tagged, entry_exit: None });
            field_phis.push(value);
        }
        // True only during the first iteration.
        let first = new_value(trace, TraceType::Bool, TraceRep::Tagged, TraceValueOrigin::Constant(TraceConstant::Bool(true)));
        let later = new_value(trace, TraceType::Bool, TraceRep::Tagged, TraceValueOrigin::Constant(TraceConstant::Bool(false)));
        let flag = new_value(trace, TraceType::Bool, TraceRep::Tagged, TraceValueOrigin::LoopTemporary);
        trace.loop_phis.push(LoopPhi { value: flag, initial: first, backedge: Some(later), rep: TraceRep::Tagged, entry_exit: None });

        for exit in carried.exits {
            let index = trace.exits[exit].virtuals.len() as u32;
            let initial = snapshot_value(trace, carried.initial);
            trace.exits[exit].virtuals.push(TraceVirtualObject {
                kind: plan.kind,
                fields: field_phis.iter().map(|&phi| SnapshotValue::Value(phi)).collect(),
                initial: Some((SnapshotValue::Value(flag), initial)),
            });
            replace_writes(trace, exit, &carried.tagged, SnapshotValue::Virtual(index));
        }
        for exit in carried.dead_exits {
            replace_writes(trace, exit, &carried.tagged, SnapshotValue::Constant(TraceConstant::Null));
        }
        for &node in &carried.removed {
            trace.nodes[node].kind = TraceInst::Nop;
        }
        trace.loop_phis.retain(|phi| phi.value != carried.phi);
        for (result, slot) in carried.loads {
            alias_value(trace, result, field_phis[slot]);
        }
        // A field the loop never changes is loop-invariant: its phi would feed
        // itself back, so it is just the preheader load.
        for &phi in &field_phis {
            let Some(position) = trace.loop_phis.iter().position(|entry| entry.value == phi) else { continue };
            let entry = &trace.loop_phis[position];
            if entry.backedge.is_some_and(|backedge| trace.canonical_value(backedge) == phi) {
                let initial = entry.initial;
                trace.loop_phis.remove(position);
                alias_value(trace, phi, initial);
            }
        }
    }
    trace.stats.sunk_allocations += 1;
}
