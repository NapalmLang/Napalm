// Allocation sinking (escape analysis + SROA) in native traces. Each program
// runs under the interpreter and under the tracing JIT; the JIT run must also
// show that allocations were actually sunk and, where an exit needs the object,
// that it was materialized.

/// Run `source` (whose last variable is the result) with the JIT off and on,
/// compare both results with `expected`, and return the JIT run's stats.
fn run_sinking(source: &str, expected: &str) -> TracingJitStats {
    let parsed = crate::napalm_front_main::parse("test.nap", source);
    assert!(!parsed.has_errors(), "{:#?}", parsed.diagnostics);
    let mut hir = Program::new();
    let manifest = napalm_builtin::install_hir_builtins(&mut hir);
    let output = lower_module(&mut hir, "test.nap", source, "test", &parsed.module);
    assert!(!output.has_errors(), "{:#?}", output.diagnostics);
    let entry = output.lowered.unwrap().entry;
    let program = compile_program(&mut hir).unwrap();
    let mut stats = None;
    for enabled in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(enabled.then(|| TracingJitOptions { hot_loop_threshold: 2, ..Default::default() }));
        vm.run_module(entry).unwrap_or_else(|error| panic!("jit={enabled}: {error:?}"));
        let value = last_source_variable(&vm);
        let context = HostContext { heap: &mut vm.heap };
        assert_eq!(context.format_value(value), expected, "jit={enabled}");
        if enabled {
            stats = vm.jit_stats().cloned();
        }
    }
    stats.unwrap()
}

#[test]
fn napalm_sinking_removes_non_escaping_literals() {
    let stats = run_sinking(r#"
        function f(n) {
            let s = 0;
            for (let i = 0; i < n; i++) { const p = {x: i, y: i * 2, z: 3}; s = s + p.x + p.y + p.z; }
            return s;
        }
        const result = f(5000);
    "#, "37507500");
    assert!(stats.sunk_allocations > 0, "{stats:?}");
}

#[test]
fn napalm_sinking_materializes_objects_on_side_exits() {
    // The hot path never publishes `p`; the cold `push` path needs the real
    // object, with its current field values (including a mutated one).
    let stats = run_sinking(r#"
        function f(n) {
            let out = [];
            let s = 0;
            for (let i = 0; i < n; i++) {
                const v = i < 300 ? i : 'x' + i;
                const p = {a: v, b: i, c: null};
                p.b = p.b + 1;
                s = s + p.b;
                if (i % 37 == 0) { out.push(p); }
                if (i % 100 == 0) { System.gc(); }
            }
            let t = '';
            for (let k = 0; k < out.length; k++) { t = t + out[k].a + ':' + out[k].b + ':' + out[k].c + ','; }
            return [s, out.length, t.length, t.substring(0, 24)];
        }
        const result = f(600);
    "#, "[180300, 17, 221, 0:1:null,37:38:null,74:7]");
    assert!(stats.sunk_allocations > 0, "{stats:?}");
    assert!(stats.materialized_objects > 0, "{stats:?}");
}

#[test]
fn napalm_sinking_handles_instances_with_inlined_constructors() {
    let stats = run_sinking(r#"
        class V { constructor(x, y) { this.x = x; this.y = y; } sum() { return this.x + this.y; } }
        function f(n) {
            let s = 0; let keep = [];
            for (let i = 0; i < n; i++) {
                const v = new V(i, i * 2);
                s = s + v.x + v.y;
                if (i % 50 == 49) { keep.push(v); }
            }
            let t = 0;
            for (let k = 0; k < keep.length; k++) { t = t + keep[k].sum(); }
            return [s, keep.length, t];
        }
        const result = f(1000);
    "#, "[1498500, 20, 31440]");
    assert!(stats.sunk_allocations > 0, "{stats:?}");
    assert!(stats.materialized_objects > 0, "{stats:?}");
}

#[test]
fn napalm_sinking_keeps_escaping_and_loop_carried_objects() {
    // Objects stored into other objects or carried around the loop escape;
    // results must stay exact.
    run_sinking(r#"
        class Node { constructor(v, next) { this.v = v; this.next = next; } }
        function chain(n) {
            let head = null; let s = 0;
            for (let i = 0; i < n; i++) {
                const box = {v: i};
                const node = new Node(box, head);
                if (i % 3 == 0) { head = node; }
                s = s + node.v.v;
            }
            let c = 0; let p = head;
            while (p != null) { c = c + p.v.v; p = p.next; }
            return [s, c];
        }
        function carried(n) {
            let acc = {x: 0, y: 0};
            for (let i = 0; i < n; i++) { acc = {x: acc.x + 1, y: acc.y + i}; }
            return [acc.x, acc.y];
        }
        const result = [chain(3000), carried(3000)];
    "#, "[[4498500, 1498500], [3000, 4498500]]");
}

#[test]
fn napalm_sinking_scalarizes_loop_carried_instances() {
    // `acc = acc.add(..)` becomes per-field loop phis; exits hand back either
    // the entry object (first iteration) or a materialized copy.
    let stats = run_sinking(r#"
        class V { constructor(x, y) { this.x = x; this.y = y; } add(d) { return new V(this.x + d, this.y); } }
        class P { constructor(a, b) { this.a = a; this.b = b; } step(k) { return new P(this.b, this.a + k); } }
        function identity(n) {
            const start = new V(0, 'tag');
            let acc = start;
            const seen = [];
            let same = 0;
            for (let i = 0; i < n; i++) {
                if (i % 7 == 0) { seen.push(acc); if (acc === start) { same = same + 1; } }
                if (i % 200 == 0) { System.gc(); }
                acc = acc.add(1);
            }
            let ok = 0;
            for (let k = 0; k < seen.length; k++) { if (seen[k].x == k * 7 && seen[k].y == 'tag') { ok = ok + 1; } }
            return [acc.x, seen.length, ok, same, start.x];
        }
        function run(n) {
            let p = new P(1, 2);
            let s = 0;
            for (let i = 0; i < n; i++) { p = p.step(i % 3); s = s + p.a; }
            return [p.a, p.b, s];
        }
        let runs = [];
        for (let r = 0; r < 12; r++) { runs.push(run(150 + r)); }
        const result = [identity(1000), runs[0], runs[11]];
    "#, "[[1000, 143, 143, 1, 0], [76, 77, 5800], [81, 82, 6668]]");
    assert!(stats.sunk_allocations > 0, "{stats:?}");
}
