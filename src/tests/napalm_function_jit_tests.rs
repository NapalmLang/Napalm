// Regressions for whole-function native recursion and moving inline Map storage.
fn whole_function_program(source: &str) -> (BytecodeProgram, FuncId, BuiltinManifest) {
    let parsed = crate::napalm_front_main::parse("whole_function.napalm", source);
    assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
    let mut hir = Program::new();
    let manifest = napalm_builtin::install_hir_builtins(&mut hir);
    let lowered = lower_module(&mut hir, "whole_function.napalm", source, "test", &parsed.module);
    assert!(!lowered.has_errors(), "{:?}", lowered.diagnostics);
    let entry = lowered.lowered.unwrap().entry;
    (compile_program(&mut hir).unwrap(), entry, manifest)
}

#[test]
fn napalm_whole_function_tiering_avoids_redundant_recursive_entry_traces() {
    let (program, entry, manifest) = whole_function_program(r#"
        function make(depth) {
            if (depth > 0) return {left: make(depth - 1), right: make(depth - 1)};
            return {left: null, right: null};
        }
        function count(node) {
            if (node.left == null) return 1;
            return 1 + count(node.left) + count(node.right);
        }
        // Keep this a native-recursion test: observing mutation makes the
        // producer ineligible for automatic object memoization.
        make(1).left = null;
        const result = count(make(10));
    "#);
    for threshold in [0, 64, 96] {
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(Some(TracingJitOptions {
            hot_entry_threshold: threshold,
            ..Default::default()
        }));
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 2047.0);
        let stats = vm.whole_function_jit_stats().unwrap();
        assert_eq!(stats.compiled_functions, if threshold == 0 { 0 } else { 2 }, "{stats:?}");
        let traces = vm.jit_stats().unwrap();
        assert_eq!(traces.entry_traces_compiled, 0, "{traces:?}");
        if threshold == 0 {
            assert_eq!(stats.compile_attempts, 0, "{stats:?}");
        } else {
            assert!(stats.native_recursive_calls > 1000, "{stats:?}");
        }
        assert!(vm.native_shadow_top.is_null());
    }
}

#[test]
fn napalm_whole_function_trees_survive_gc_and_shape_bailouts() {
    let (program, entry, manifest) = whole_function_program(r#"
        function make(depth) {
            if (depth > 0) return {left: make(depth - 1), right: make(depth - 1)};
            return {left: null, right: null};
        }
        function count(node) {
            if (node.left == null) return 1;
            return 1 + count(node.left) + count(node.right);
        }
        let sum = 0;
        for (let i = 0; i < 40; i++) {
            const tree = make(10);
            System.gc();
            sum += count(tree);
            // Different shape and slot order inside an already-native walk.
            tree.left = {extra: 7, right: null, left: null};
            System.gc();
            sum += count(tree);
        }
        const result = sum;
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 40.0 * (2047.0 + 1025.0));
        assert!(vm.gc_statistics().collections > 0);
        if native {
            let stats = vm.whole_function_jit_stats().unwrap();
            assert!(stats.compiled_functions >= 2, "{stats:?}");
            assert!(stats.native_recursive_calls > 1000, "{stats:?}");
            // Native recursion with shadow banks for the allocating `make`.
            assert!(stats.v2_compiled_functions >= 2, "{stats:?}");
        }
        assert!(vm.native_shadow_top.is_null());
    }
}

#[test]
fn napalm_whole_function_identity_guards_preserve_other_equality() {
    let (program, entry, manifest) = whole_function_program(r#"
        function equal(left, right) {
            if (left == right) return {value: 1};
            return {value: 0};
        }
        const object = {};
        let sum = 0;
        for (let i = 0; i < 400; i++) {
            sum += equal(object, object).value + equal(null, null).value;
        }
        // Numeric and string equality must fall back from the identity PIC.
        const result = [sum, equal(0, -0).value, equal(0 / 0, 0 / 0).value,
            equal('a' + 'b', 'ab').value, equal(true, false).value, equal(object, {}).value];
    "#);
    let mut vm = source_vm(&program, &manifest);
    vm.run_module(entry).unwrap();
    let value = last_source_variable(&vm);
    assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value), "[800, 1, 0, 1, 0, 0]");
    assert!(vm.whole_function_jit_stats().unwrap().compiled_functions > 0);
}

#[test]
fn napalm_whole_function_speculation_bails_out_and_deoptimizes() {
    // Hot numeric recursion compiles speculative numeric guards. A String
    // operand fails them: in the top frame the op is resumed by baseline, deep
    // in native recursion the outermost self-call is replayed, and repeated
    // failures drop the speculative body for a generic recompilation.
    let (program, entry, manifest) = whole_function_program(r#"
        // Reading a reassigned module variable keeps these out of automatic
        // memoization (memoized functions never reach this tier).
        let one = 0;
        one = 1;
        function sum(n, step) {
            if (n < one) return step;
            return sum(n - one, step) + step;
        }
        function leafy(n, leaf) {
            if (n < one) return leaf;
            return leafy(n - one, leaf) + one;
        }
        let total = 0;
        for (let i = 0; i < 300; i++) total += sum(20, 1.5) + leafy(25, 3);
        const strings = [sum(3, 'ab'), leafy(4, 'x')];
        let mixed = '';
        for (let i = 0; i < 60; i++) mixed = mixed + leafy(6, i % 2 == 0 ? 'q' : 7);
        for (let i = 0; i < 300; i++) total += sum(10, 2);
        const result = [total, strings, mixed.length, sum(2, 0 / 0)];
    "#);
    let mut vm = source_vm(&program, &manifest);
    vm.run_module(entry).unwrap();
    let value = last_source_variable(&vm);
    assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value), "[24450, [abababab, x1111], 270, NaN]");
    let stats = vm.whole_function_jit_stats().unwrap();
    assert!(stats.compiled_functions >= 2, "{stats:?}");
    assert!(stats.bailouts > 0, "{stats:?}");
    assert!(stats.deoptimized_functions > 0, "{stats:?}");
    assert!(vm.native_shadow_top.is_null());
}

#[test]
fn napalm_whole_function_v2_specializes_parameters_and_guards_entry() {
    // `down` is compiled by the typed-SSA tier: `n` becomes an unboxed F64
    // parameter (every self-call passes `n - one`), the result is a raw Number.
    // Non-Number arguments from the interpreter fail the entry guard and run in
    // baseline, which reports the same errors.
    let (program, entry, manifest) = whole_function_program(r#"
        let one = 0;
        one = 1;
        function down(n) {
            if (n < one) return 0;
            return down(n - one) + one;
        }
        let total = 0;
        for (let i = 0; i < 300; i++) total += down(40);
        const fractional = down(2.5);
        const errors = [];
        for (const v of ['7', null, true, {a: 1}, [3]]) {
            try { errors.add(down(v)); } catch (e) { errors.add('err'); }
        }
        for (let i = 0; i < 300; i++) total += down(30);
        const result = [total, fractional, errors, down(-1), down(700)];
    "#);
    let mut vm = source_vm(&program, &manifest);
    vm.run_module(entry).unwrap();
    let value = last_source_variable(&vm);
    assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value),
        "[21000, 2, [err, err, err, err, err], 0, 700]");
    let stats = vm.whole_function_jit_stats().unwrap();
    assert!(stats.v2_compiled_functions >= 1, "{stats:?}");
    assert!(vm.native_shadow_top.is_null());
}

#[test]
fn napalm_whole_function_v2_inlines_guarded_methods_and_replays_on_miss() {
    // `addw` inlines both profiled `weight` targets behind class guards; a
    // swapped receiver pair fails a guard inside the inlined body and replays
    // the whole call in the interpreter. Setter-like bodies (`bump`) must run
    // exactly once per call even when their inlining is declined.
    let (program, entry, manifest) = whole_function_program(r#"
        let one = 0;
        one = 1;
        class A {
            constructor(v) { this.v = v; this.count = 0; }
            weight() { return this.v * 2; }
            bump() { this.count = this.count + one; return this.count; }
            bumpTwice() { this.bump(); return this.bump(); }
            get() { return this.v; }
        }
        class B {
            constructor(v) { this.v = v; this.count = 0; }
            weight() { return this.v + 100; }
            bump() { this.count = this.count + 10; return this.count; }
            bumpTwice() { this.bump(); return this.bump(); }
            get() { return this.v; }
        }
        function score(x, n) {
            if (n < one) return x.weight();
            return x.weight() + score(x, n - one);
        }
        function churn(x) {
            const a = x.bump();
            const b = x.bumpTwice();
            return a + b + x.get() * 2;
        }
        function addw(x, y) { return x.weight() + y.weight(); }
        // Driven from a function so the calls are direct calls that count
        // toward whole-function hotness.
        function main(out, a, b) {
            for (let i = 0; i < 200; i++) out.add(score(a, 10));
            out.add(score(b, 5));
            for (let i = 0; i < 200; i++) out.add(churn(a));
            out.add(churn(b));
            for (let i = 0; i < 200; i++) out.add(addw(a, b));
            out.add(addw(b, a));
            for (const m of [a, b, a, b]) { out.add(score(m, 2)); out.add(churn(m)); }
            try { out.add(score(new A(null), 2)); } catch (e) { out.add('err'); }
            return [out.reduce((s, x) => typeof x == 'Number' ? s + x : s, 0),
                out.slice(-10), a.count, b.count];
        }
        const result = main([], new A(3), new B(4));
    "#);
    let mut vm = source_vm(&program, &manifest);
    // Only the whole-function tier: loop traces would otherwise absorb the
    // calls in `main` before the callees become hot.
    vm.set_jit_options(Some(TracingJitOptions {
        hot_entry_threshold: 2,
        hot_loop_threshold: u32::MAX,
        min_entry_trace_bytecodes: u32::MAX,
        llvm_hot_threshold: 0,
        ..Default::default()
    }));
    vm.run_module(entry).unwrap();
    let value = last_source_variable(&vm);
    assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value),
        "[160744, [110, 18, 1210, 312, 108, 18, 1216, 312, 168, err], 606, 90]");
    let stats = vm.whole_function_jit_stats().unwrap();
    assert!(stats.v2_inlined_calls >= 1, "{stats:?}");
    assert!(vm.native_shadow_top.is_null());
}

/// Run `source` in the interpreter and with only the whole-function tier
/// (loop traces off, entry traces never install), assert equal `result`
/// values and return the WFJIT stats.
fn whole_function_only_run(source: &str) -> WholeFunctionJitStats {
    let (program, entry, manifest) = whole_function_program(source);
    let mut baseline = source_vm(&program, &manifest);
    baseline.set_jit_options(None);
    baseline.run_module(entry).unwrap();
    let value = last_source_variable(&baseline);
    let expected = HostContext { heap: &mut baseline.heap }.format_value(value);
    let mut vm = source_vm(&program, &manifest);
    vm.set_jit_options(Some(TracingJitOptions {
        hot_entry_threshold: 2,
        hot_loop_threshold: u32::MAX,
        min_entry_trace_bytecodes: u32::MAX,
        llvm_hot_threshold: 0,
        ..Default::default()
    }));
    vm.run_module(entry).unwrap();
    let value = last_source_variable(&vm);
    assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value), expected,
        "whole-function JIT changed the result");
    assert!(vm.native_shadow_top.is_null(), "native roots leaked after returning");
    vm.whole_function_jit_stats().unwrap()
}

#[test]
fn napalm_whole_function_v2_compiles_loops() {
    // Counted, nested, early-exit and allocating loops; a loop-carried value
    // that changes type (retyped header parameter); a loop long enough to be
    // handed to the tracer mid-run.
    // `one` is a mutable module variable: it keeps the numeric functions out
    // of automatic memoization.
    let stats = whole_function_only_run(r#"
        let one = 0;
        one = 1;
        function sumTo(n) { let s = 0; for (let i = one; i <= n; i++) s += i; return s; }
        function nested(n) { let s = 0; for (let i = 0; i < n; i += one) { for (let j = 0; j < i; j++) s += j; } return s; }
        function pushLoop(n) { const xs = []; for (let i = 0; i < n; i++) xs.push(i * 2); return xs.length + xs[n - 1]; }
        function retype(n) { let s: any = 0; for (let i = 0; i < n; i++) { s = s + i; if (i == 5) s = 'x'; } return s; }
        function early(xs) { for (const x of xs) { if (x > 3) return x; } return -1; }
        function main() {
            let total = 0;
            for (let i = 0; i < 300; i++) {
                total += sumTo(i % 40) + nested(i % 12) + pushLoop(i % 9 + 1);
                total += early([1, 2, i % 6, 9]);
            }
            const strings = [];
            for (let i = 0; i < 100; i++) strings.push(retype(8 + i % 3));
            return [total, sumTo(5000), strings[99], strings[50]];
        }
        const result = main();
    "#);
    assert!(stats.v2_compiled_functions >= 4, "{stats:?}");
}

#[test]
fn napalm_whole_function_v2_closures_handlers_and_dispatch() {
    // Polymorphic dispatch through the interpreter's call path, method
    // self-calls (static and dispatched), raw Boolean recursion, closures with
    // captures, exceptions thrown by callees and caught in a compiled frame
    // (the handler reads a value defined before `try`), recursion with field
    // stores, and allocation.
    let stats = whole_function_only_run(r#"
        let one = 0;
        one = 1;
        class Cat { speak() { return 1; } }
        class Dog { speak() { return 2; } }
        class Cow { speak() { return 3; } }
        function chorus(a, b) { return a.speak() + b.speak() * 10; }
        class Calc {
            constructor(base) { this.base = base; }
            fact(n) { if (n <= one) return this.base; return n * this.fact(n - one); }
        }
        function even(n) { if (n == 0) return true; if (n == one) return false; return even(n - 2); }
        function makeAdder(k) { return x => x + k; }
        function check(a, b) { if (b == 0) throw 'div0'; return a / b; }
        function safeDiv(a, b) {
            const fallback = a * 1000;
            try { return check(a, b); } catch (e) { return fallback; }
        }
        class Acc { constructor() { this.count = 0; } }
        function walk(depth, acc) {
            acc.count = acc.count + 1;
            if (depth > 0) { walk(depth - 1, acc); walk(depth - 1, acc); }
            return acc.count;
        }
        function point(x, y) { return {x: x, y: y}; }
        function main() {
            const animals = [new Cat(), new Dog(), new Cow()];
            let total = 0;
            for (let i = 0; i < 600; i++) total += chorus(animals[i % 3], animals[(i + 1) % 3]);
            const calc = new Calc(2);
            let facts = 0;
            for (let i = 0; i < 400; i++) facts += calc.fact(10);
            let evens = 0;
            for (let i = 0; i < 400; i++) if (even(i % 50)) evens++;
            const add5 = makeAdder(5);
            let adds = 0;
            for (let i = 0; i < 600; i++) adds += add5(i);
            let divs = 0;
            for (let i = 0; i < 600; i++) divs += safeDiv(i, i % 7);
            let walks = 0;
            for (let i = 0; i < 200; i++) walks += walk(5, new Acc());
            let points = 0;
            for (let i = 0; i < 600; i++) { const p = point(i, i + 1); points += p.x + p.y; }
            return [total, facts, evens, adds, divs, walks, points, even(7), safeDiv(3, 0)];
        }
        const result = main();
    "#);
    assert!(stats.v2_compiled_functions >= 8, "{stats:?}");
}

#[test]
fn napalm_whole_function_v2_inlines_callees_that_make_calls() {
    // `mid` makes calls, yet it is inlined into `top`: its calls run with
    // explicit arguments while its live values sit in a GC shadow frame. The
    // callees allocate (and collect), throw (caught by the root's handler),
    // and a guard fails after a call (the inlined frame is rebuilt).
    let stats = whole_function_only_run(r#"
        let one = 0;
        one = 1;
        class Box { constructor(v) { this.v = v; } get() { return this.v; } }
        function churn(n) { const xs = []; for (let i = 0; i < n; i += one) xs.push({k: i}); return xs.length; }
        function check(x) { if (x == 7) throw 'seven'; return x + one; }
        function mid(b, x) {
            const before = b.get();
            const r = check(x) + churn(40);
            return before + r + b.get();
        }
        function top(b, x) { try { return mid(b, x); } catch (e) { return -1; } }
        function midGuard(b, x) { const r = check(x); return r + b.v; }
        function topGuard(b, x) { try { return midGuard(b, x) * 2; } catch (e) { return -5; } }
        function main() {
            let sum = 0;
            for (let i = 0; i < 600; i++) sum += top(new Box(i), i % 10);
            let guarded = 0;
            for (let i = 0; i < 400; i++) guarded += topGuard(new Box(3), i % 9);
            const shapes = [topGuard({v: 10}, 1), topGuard(new Box(4), 7), topGuard({v: 2}, 3)];
            return [sum, guarded, shapes];
        }
        const result = main();
    "#);
    assert!(stats.v2_inlined_calls >= 2, "{stats:?}");
}

#[test]
fn napalm_whole_function_v2_native_number_methods() {
    let stats = whole_function_only_run(r#"
        let one = 0;
        one = 1;
        function math(n, x, mode) {
            const a = x.abs;
            const b = x.floor;
            const c = x.ceil;
            const d = x.sqrt;
            const value = mode == 0 ? a : mode == 1 ? b : mode == 2 ? c : d;
            if (n < one) return value;
            return value + math(n - one, x, mode);
        }
        let total = 0;
        for (let i = 0; i < 100; i++) total += math(12, 2.25, i % 4);
        const edges = [];
        for (let mode = 0; mode < 4; mode++) {
            edges.push([math(0, -2.25, mode), math(0, -0.2, mode),
                math(0, 0 / 0, mode), math(0, 1 / 0, mode), math(0, -1 / 0, mode),
                1 / math(0, -0.0, mode), 1 / math(0, 0.0, mode)]);
        }
        let bad: any = null;
        try { bad = math(0, 'wrong receiver', 0); } catch (e) { bad = 'err'; }
        const result = [total, edges, bad, math(3, 4, 3)];
    "#);
    assert!(stats.v2_compiled_functions >= 1, "{stats:?}");
    // Native self-calls reject runtime helpers, so this also proves the four
    // Number methods were actually lowered, rather than called via the VM.
    assert!(stats.native_recursive_calls > 100, "{stats:?}");
}

#[test]
fn napalm_whole_function_v2_failed_inline_restores_nested_ir() {
    let stats = whole_function_only_run(r#"
        let bias = 0;
        bias = 1;
        function inner(x) { return x + bias; }
        function allocating(x) {
            const y = inner(x);
            if (x > 0) return {value: y};
            return {value: -y};
        }
        function caller(x) { return allocating(x).value + inner(x); }
        let total = 0;
        for (let i = 0; i < 200; i++) total += caller(i - 100);
        const result = total;
    "#);
    assert!(stats.v2_compiled_functions >= 2, "{stats:?}");
}

#[test]
fn napalm_whole_function_v2_native_builtin_methods() {
    // List length/subscript/iteration, Boolean not, String equality and
    // codePointAt, and Bytes subscript run natively (or through pure leaf
    // helpers) once their sites are monomorphic. Every case they do not handle
    // natively (negative or fractional indices, out of range, non-ASCII text,
    // lists changed while iterating) must deopt and match the interpreter.
    let stats = whole_function_only_run(r#"
        function listWork(xs, i) {
            let s = 0;
            for (const x of xs) s += x;
            return s + xs.length + (xs[i] ?? -1);
        }
        function textWork(text, word, i) {
            let s = 0;
            if (text == word) s += 1000;
            const c = text.codePointAt(i);
            return s + (c == null ? -1 : c);
        }
        function notWork(flag) { return !flag; }
        function bytesWork(bytes, i) { return bytes[i]; }
        function shrinking(xs) { let s = 0; for (const x of xs) { s += x; if (x == 2) xs.pop(); } return s; }
        function main() {
            const xs = [1, 2, 3, 4];
            let total = 0;
            for (let i = 0; i < 400; i++) total += listWork(xs, i % 4);
            let fractional: any = 0;
            try { fractional = listWork(xs, 1.5); } catch (e) { fractional = 'err'; }
            const edges = [listWork(xs, -1), listWork(xs, 9), fractional, listWork([], 0)];
            let text = 0;
            for (let i = 0; i < 400; i++) text += textWork("hello", i % 2 == 0 ? "hello" : "help", i % 5);
            const textEdges = [textWork("héllo", "héllo", 1), textWork("abc", "abc", -1),
                textWork("abc", "x", 7), textWork("😀a", "", 0)];
            let flips = 0;
            for (let i = 0; i < 400; i++) if (notWork(i % 3 == 0)) flips++;
            const bytes = Bytes.fromUtf8("napalm");
            let byteSum = 0;
            for (let i = 0; i < 400; i++) byteSum += bytesWork(bytes, i % 6);
            let shrinkSum = 0;
            for (let i = 0; i < 100; i++) shrinkSum += shrinking([1, 2, 3, 4]);
            return [total, edges, text, textEdges, flips, byteSum, shrinkSum];
        }
        const result = main();
    "#);
    assert!(stats.v2_compiled_functions >= 4, "{stats:?}");
}

#[test]
fn napalm_whole_function_v2_rebuilds_inlined_frames_after_effects() {
    // `touch` stores before a guard that fails for Map arguments. Inlined
    // into `drive` (and through `mid`, two levels deep), the failing guard
    // cannot replay the call; the deopt rebuilds the inlined frames and the
    // interpreter resumes after the store, which must happen exactly once.
    let stats = whole_function_only_run(r#"
        class Counter { constructor() { this.n = 0; this.v = 1; } }
        class Item { constructor(v) { this.v = v; } }
        function touch(c, x) { c.n = c.n + 1; return x.v + c.v; }
        function mid(c, x) { return touch(c, x) * 2; }
        function drive(c, x) { return touch(c, x) + 1; }
        function drive2(c, x) { return mid(c, x) + 1; }
        function main() {
            const c = new Counter();
            const item = new Item(3);
            const map = {v: 10};
            let sum = 0;
            for (let i = 0; i < 400; i++) sum += drive(c, item) + drive2(c, item);
            for (let i = 0; i < 20; i++) sum += drive(c, i % 2 == 0 ? map : item) + drive2(c, map);
            return [sum, c.n];
        }
        const result = main();
    "#);
    assert!(stats.v2_inlined_calls >= 2, "{stats:?}");
}

#[test]
fn napalm_whole_function_v2_stores_references_with_the_write_barrier() {
    // Constructors store heap references (child nodes, strings, lists) into
    // exact-class fields: the native store runs the generational barrier
    // instead of deoptimizing, and the stored graph survives collections.
    let stats = whole_function_only_run(r#"
        class Node { constructor(left, right, tag) { this.left = left; this.right = right; this.tag = tag; } }
        function make(d) {
            if (d == 0) return new Node(null, null, 'leaf');
            return new Node(make(d - 1), make(d - 1), [d]);
        }
        function count(n) { return n == null ? 0 : 1 + count(n.left) + count(n.right); }
        function main() {
            let total = 0;
            for (let i = 0; i < 40; i++) { const tree = make(7); System.gc(); total += count(tree); }
            return total;
        }
        const result = main();
    "#);
    assert!(stats.v2_compiled_functions >= 2, "{stats:?}");
    assert_eq!(stats.deoptimized_functions, 0, "{stats:?}");
}

#[test]
fn napalm_whole_function_v2_recompiles_failed_sites_generically() {
    // `add` speculates on Numbers, then its site starts seeing strings: the
    // deopt feedback names the failed op, and the recompiled body runs that
    // site through its helper, so the function stays compiled.
    let stats = whole_function_only_run(r#"
        let one = 0;
        one = 1;
        function add(a, b) { return a + b + one * 0; }
        function main() {
            let out: any = 0;
            for (let i = 0; i < 300; i++) out = add(i, one);
            let text: any = '';
            for (let i = 0; i < 3000; i++) text = add(i % 2 == 0 ? 'a' : 1, 'b');
            for (let i = 0; i < 300; i++) out = out + add(i, one);
            return [out, text];
        }
        const result = main();
    "#);
    assert!(stats.deoptimized_functions >= 1, "{stats:?}");
    assert!(stats.v2_compiled_functions >= 2, "{stats:?}");
    assert_eq!(stats.rejected_functions, 0, "{stats:?}");
}

#[test]
fn napalm_whole_function_recursion_replays_guards_and_depth_limit() {
    let (program, entry, manifest) = whole_function_program(r#"
        function chain(depth, value) {
            if (depth > 0) return {next: chain(depth - 1, value), value: depth + value};
            return {next: null, value: 0 + value};
        }
        let warm = chain(160, 1);
        let deep = chain(300, 2);
        let text = chain(20, 'x');
        System.gc();
        let count = 0;
        while (deep != null) { count++; deep = deep.next; }
        while (text.next != null) text = text.next;
        const result = [count, warm.value, text.value];
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        let value = last_source_variable(&vm);
        assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value), "[301, 161, 0x]");
        assert!(vm.native_shadow_top.is_null());
        if native { assert!(vm.whole_function_jit_stats().unwrap().native_recursive_calls > 0); }
    }
}

#[test]
fn napalm_inline_maps_grow_promote_and_keep_live_sidecars() {
    // The loop lives in a function: module-scope variables use runtime opcodes
    // and would never let the trace reach native code.
    run_napalm_checked(r#"
        function run() {
            const map = {a: {value: 41}, b: 2};
            let sum = 0;
            for (let i = 0; i < 100; i++) {
                sum += map.a.value;
                if (i == 30) { map.c = 3; map.d = 4; map.e = 5; }
                if (i == 60) map[7] = 'seven';
                if (i % 10 == 0) System.gc();
            }
            return [sum, map.a.value, map.e, map[7]];
        }
        const result = run();
    "#, "[4100, 41, 5, seven]", true);
}

#[test]
fn napalm_whole_function_read_only_recursion_replays_deep_and_changed_shapes() {
    let (program, entry, manifest) = whole_function_program(r#"
        function make(depth) {
            if (depth > 0) return {next: make(depth - 1)};
            return {next: null};
        }
        function count(node) {
            if (node == null) return 0;
            return 1 + count(node.next);
        }
        const short = make(20);
        let total = 0;
        for (let i = 0; i < 100; i++) total += count(short);
        const deep = make(300);
        System.gc();
        const deepCount = count(deep);
        short.next.next = {extra: 42, next: {next: null}};
        System.gc();
        const result = [total, deepCount, count(short)];
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        let value = last_source_variable(&vm);
        assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value), "[2100, 301, 4]");
        assert!(vm.native_shadow_top.is_null());
        if native {
            assert!(vm.whole_function_jit_stats().unwrap().compiled_functions >= 2);
            assert!(vm.whole_function_jit_stats().unwrap().native_recursive_calls > 1000);
            // Each recursive function compiles with native self-calls.
            let stats = vm.whole_function_jit_stats().unwrap();
            assert!(stats.v2_compiled_functions >= 2, "{stats:?}");
        }
    }
}

#[test]
fn napalm_whole_function_literal_slots_preserve_duplicates_and_dynamic_keys() {
    let (program, entry, manifest) = whole_function_program(r#"
        function make(key, value) {
            return {b: value, [key]: value + 1, b: value + 2, c: value + 3};
        }
        let sum = 0;
        for (let i = 0; i < 500; i++) {
            const object = make('a', i);
            if (i % 25 == 0) System.gc();
            sum += object.a + object.b + object.c;
        }
        const changed = make('c', 10);
        System.gc();
        const result = [sum, changed.b, changed.c, changed.a];
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        let value = last_source_variable(&vm);
        assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value), "[377250, 12, 13, null]");
        if native { assert!(vm.whole_function_jit_stats().unwrap().compiled_functions > 0); }
    }
}

#[test]
fn napalm_profiled_invoke_promotes_to_direct_method_and_compiles_natively() {
    // `shape` is untyped, so WIR cannot devirtualize `shape.area()`. The VM's
    // call-site profile promotes the monomorphic site to InvokeMethodDirect and
    // the whole-function JIT compiles `measure` (it used to reject it). A later
    // receiver of another class must fall back to the generic Invoke.
    let (program, entry, manifest) = whole_function_program(r#"
        class Square { side: number; constructor(s: number) { this.side = s; } area(): number { return this.side * this.side; } }
        class Circle { r: number; constructor(r: number) { this.r = r; } area(): number { return 3 * this.r * this.r; } }
        function measure(shape) { return shape.area() + 1; }
        const square = new Square(3);
        let sum = 0;
        for (let i = 0; i < 400; i++) sum += measure(square);
        sum += measure(new Circle(2));
        for (let i = 0; i < 10; i++) sum += measure(square);
        const result = sum;
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        // Isolate the whole-function tier: no loop traces may inline `measure`.
        vm.set_jit_options(native.then(|| TracingJitOptions {
            hot_loop_threshold: u32::MAX,
            hot_entry_threshold: 64,
            ..Default::default()
        }));
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 410.0 * 10.0 + 13.0);
        if native {
            let stats = vm.whole_function_jit_stats().unwrap();
            assert!(stats.compiled_functions >= 1, "{stats:?}");
            assert!(stats.native_entries > 0, "{stats:?}");
        }
    }
}

#[test]
fn inline_instance_fields_do_not_grow_managed_objects() {
    // Up to INSTANCE_INLINE_FIELDS fields live inside the managed object; the
    // fixed managed payload (every object pays it) must not grow for that.
    assert_eq!(std::mem::size_of::<HeapObject>(), std::mem::size_of::<NapalmMap>());
    assert_eq!(std::mem::size_of::<HeapObject>(), 48);
    assert!(InstanceObject::nulls(0, INSTANCE_INLINE_FIELDS).is_inline());
    assert!(!InstanceObject::nulls(0, INSTANCE_INLINE_FIELDS + 1).is_inline());
    assert_eq!(InstanceObject::nulls(0, 3).len(), 3);
    assert_eq!(InstanceObject::nulls(0, 7).len(), 7);
}

#[test]
fn inline_and_boxed_instance_fields_survive_moving_gc_in_native_code() {
    let (program, entry, manifest) = whole_function_program(r#"
        class Small { a: number; b: number; constructor(a: number, b: number) { this.a = a; this.b = b; } }
        class Wide {
            a: number; b: number; c: number; d: number; e: number; f: number;
            constructor(x: number) { this.a = x; this.b = x + 1; this.c = x + 2; this.d = x + 3; this.e = x + 4; this.f = x + 5; }
        }
        function run(n: number): number {
            const keep = [];
            let sum = 0;
            for (let i = 0; i < n; i++) {
                const s = new Small(i, 2);
                const w = new Wide(i);
                if (i % 50 == 0) { keep.push(s); keep.push(w); System.gc(); }
                s.b = s.b + w.f;
                sum = sum + s.a + s.b + w.a + w.f;
            }
            for (let j = 0; j < keep.length; j++) {
                const o = keep[j];
                sum = sum + o.a + o.b;
            }
            return sum;
        }
        const result = run(2000);
    "#);
    let mut expected = 0.0f64;
    let mut kept = 0.0f64;
    for i in 0..2000 {
        let i = i as f64;
        let b = 2.0 + i + 5.0;
        expected += i + b + i + (i + 5.0);
        if (i as i64) % 50 == 0 { kept += (i + b) + (i + (i + 1.0)); }
    }
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), expected + kept, "native={native}");
        assert!(vm.gc_statistics().collections > 0);
    }
}

#[test]
fn inline_instance_sidecar_uses_gc_relative_field_storage() {
    for count in [2usize, 9] {
        let entry = HandleEntry::new(FREE_HANDLE_NONE);
        let managed = ManagedObject {
            handle: 0,
            object: HeapObject::Instance(InstanceObject::nulls(3, count)),
        };
        let base = (&managed as *const ManagedObject) as usize;
        HandleTable::refresh_entry(&entry, base, &managed.object);
        let encoded = entry.data.load(Ordering::Relaxed);
        let fields = match &managed.object { HeapObject::Instance(fields) => fields.as_ptr() as usize, _ => unreachable!() };
        let relative = encoded & crate::napalm_tracing_jit::TRACE_HEAP_DATA_RELATIVE_MASK != 0;
        assert_eq!(relative, count <= INSTANCE_INLINE_FIELDS, "count {count}");
        // The branch-free JIT decode resolves both representations.
        let mask = crate::napalm_tracing_jit::TRACE_HEAP_DATA_RELATIVE_MASK;
        let decoded = (encoded & !mask) + (base & (encoded & mask).wrapping_neg());
        assert_eq!(decoded, fields, "count {count}");
        assert_eq!(entry.len.load(Ordering::Relaxed), count);
    }
}

#[test]
fn napalm_whole_function_small_maps_enable_tlab_allocation() {
    let mut heap = Heap::default();
    let info = heap.inline_map_fast_allocation_info()
        .expect("small managed objects must support direct TLAB allocation");
    assert!(!info.cursor.is_null() && !info.limit.is_null());
    assert!(info.bytes >= std::mem::size_of::<ManagedObject>());
    assert!(info.alignment.is_power_of_two());
}
