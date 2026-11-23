// Each source runs interpreted and with tracing JIT, through both compiler
// pipelines. Test the language boundary, not regexr's engine implementation.
const REGEX_TEST_PRELUDE: &str = r#"
    let assertion = 0;
    function expect(ok) {
        assertion++;
        if (!ok) { throw `regex assertion ${assertion} failed`; }
    }
    function rejects(action) {
        let failed = false;
        try { action(); } catch (error) { failed = true; }
        expect(failed);
    }
"#;

fn run_regex_test(body: &str) {
    run_napalm(
        &format!("{REGEX_TEST_PRELUDE}\n{body}\nconst result = \"ok\";"),
        "ok",
    );
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn napalm_regex_defaults_to_native_compilation() {
    // This pattern selects JitShiftOr in regexr 0.6.2, rather than one of its
    // literal/class fast paths that intentionally bypass native compilation.
    run_regex_test(
        r#"
        const pattern = "[a-z]+\\d";
        const expressions = [
            new Regex(pattern),
            Regex.new(pattern),
            Regex.builder(pattern).build(),
            new RegexBuilder(pattern).build(),
            RegexBuilder.new(pattern).build()
        ];
        for (const re of expressions) {
            expect(re.engineName() === "JitShiftOr");
            expect(re.find("abc7").asStr() === "abc7");
            expect(re.captures("abc7")[0] === "abc7");
        }
        const builder = Regex.builder(pattern);
        const compiled = builder.build();
        const plain = builder.jit(false).build();
        expect(plain.engineName() === "ShiftOr" && plain.isMatch("abc7"));
        expect(compiled.engineName() === "JitShiftOr");
        expect(builder.jit(true).build().engineName() === "JitShiftOr");
    "#,
    );
}

#[test]
fn napalm_regex_search_and_literal_replacement() {
    run_regex_test(
        r#"
        const re: Regex = new Regex("[0-9]+");
        expect(re instanceof Regex && typeof re === "Regex");
        expect(!(re instanceof RegexBuilder));
        expect(re.as_str() === "[0-9]+" && re.asStr() === re.toString());
        expect(re.engine_name() === re.engineName() && re.engineName().length > 0);
        expect(re.is_match("a12") && re.isMatch("a12"));
        expect(re.try_is_match("a12") && re.tryIsMatch("a12"));
        expect(!re.isMatch("abc") && re.find("abc") === null && re.tryFind("abc") === null);
        expect(re.captures("abc") === null && re.tryCaptures("abc") === null);
        expect(re.try_find("a12").asStr() === "12");
        expect(re.try_captures("a12")[0] === "12");
        expect(re.replace("a12b34", "$0") === "a$0b34");
        expect(re.replace_all("a12b34", "${name}") === "a${name}b${name}");
        expect(re.replaceAll("a12b34", "X") === "aXbX");
        expect(re.replace("none", "X") === "none");
        expect(re.replaceAll("none", "X") === "none");
        const literal = "a.b|c [x] #\t\n";
        const escaped = Regex.escape(literal);
        expect(Regex.new("^(?x:" + escaped + ")$").isMatch(literal));
        expect(!Regex.new(Regex.escape("a.b")).isMatch("axb"));
        expect(Regex.new("(?i)hello").isMatch("HELLO"));
        expect(Regex.new("(?<=foo)bar(?=!)").find("foobar!").asStr() === "bar");
        expect(Regex.new("(ab)\\1").isMatch("abab"));
    "#,
    );
}

#[test]
fn napalm_regex_match_utf8_offsets_and_empty_iteration() {
    run_regex_test(
        r#"
        const re = Regex.new(".");
        const matches = re.findIter("é世🎉");
        expect(matches.length === 3 && re.find_iter("é世🎉").length === 3);
        const m: RegexMatch = matches[1];
        expect(m instanceof RegexMatch && typeof m === "RegexMatch");
        expect(m.start() === 2 && m.end() === 5 && m.len() === 3 && m.length === 3);
        expect(m.as_str() === "世" && m.asStr() === "世" && m.toString() === "世");
        expect(m.as_bytes().toUtf8() === "世" && m.asBytes().length === 3);
        expect(m.range().toString() === "2...5");
        expect(!m.is_empty() && !m.isEmpty());
        let text = "";
        for (const found of matches) { text += found.asStr(); }
        expect(text === "é世🎉");
        const empty = Regex.new("").findIter("é世🎉");
        expect(empty.length === 4);
        expect(empty[0].start() === 0 && empty[1].start() === 2 && empty[2].start() === 5 && empty[3].start() === 9);
        expect(empty[3].isEmpty() && empty[3].asBytes().length === 0);
        expect(Regex.new("a*").findIter("aa").length === 1);
        expect(Regex.new("^").findIter("abc").length === 1);
        expect(Regex.new("(?<=a)b").findIter("abab").length === 2);
        expect(Regex.new("z").findIter("abc").length === 0);
        expect(Regex.new("").replaceAll("é", "-") === "-é-");
    "#,
    );
}

#[test]
fn napalm_regex_captures_named_optional_and_subscripts() {
    run_regex_test(
        r#"
        const re = Regex.new("(?P<word>[a-z]+)(?:-(?<n>[0-9]+))?");
        const names = re.captureNames();
        expect(names.length === 2 && re.capture_names().length === 2);
        expect((names[0] === "word" && names[1] === "n") || (names[0] === "n" && names[1] === "word"));
        const c: RegexCaptures = re.captures("abc-12");
        expect(c instanceof RegexCaptures && typeof c === "RegexCaptures");
        expect(c.len() === 3 && c.length === 3 && !c.is_empty() && !c.isEmpty());
        expect(c.get(0).asStr() === "abc-12" && c.get(1).asStr() === "abc");
        expect(c.name("n").asStr() === "12" && c.name("n").start() === 4);
        expect(c[0] === "abc-12" && c["word"] === "abc");
        expect(c.get(99) === null && c.name("missing") === null);
        expect(c.get(4294967296) === null && c.get(9007199254740991) === null);
        rejects(() => c[99]);
        rejects(() => c["missing"]);
        const optional = re.tryCaptures("abc");
        expect(optional.get(2) === null && optional.name("n") === null);
        rejects(() => optional[2]);
        rejects(() => optional["n"]);
        const all = re.capturesIter("abc-12 def");
        expect(all.length === 2 && all[0]["n"] === "12" && all[1].name("n") === null);
        expect(re.captures_iter("abc-12 def").length === 2);
        expect(re.capturesIter("123").length === 0);
        const empty = Regex.new("(?P<x>)").capturesIter("é");
        expect(empty.length === 2 && empty[1].name("x").start() === 2);
        expect(Regex.new("literal").captureNames().length === 0);
    "#,
    );
}

#[test]
fn napalm_regex_builder_limits_and_errors() {
    run_regex_test(
        r#"
        const builder: RegexBuilder = new RegexBuilder("(a+)\\1");
        expect(builder instanceof RegexBuilder && typeof builder === "RegexBuilder");
        expect(builder.jit(false) === builder && builder.optimizePrefixes(false) === builder);
        expect(builder.backtrackLimit(0) === builder);
        const limited = builder.build();
        rejects(() => limited.tryFind("aaaa"));
        rejects(() => limited.try_find("aaaa"));
        rejects(() => limited.tryCaptures("aaaa"));
        rejects(() => limited.try_captures("aaaa"));
        rejects(() => limited.tryIsMatch("aaaa"));
        rejects(() => limited.try_is_match("aaaa"));
        // regexr 0.6.2 infallible searches use its default budget, not the
        // builder's custom budget. Preserve that distinction exactly.
        expect(limited.find("aaaa").asStr() === "aaaa");
        expect(limited.isMatch("aaaa") && limited.captures("aaaa")[0] === "aaaa");
        expect(!limited.tryIsMatch("zz"));
        expect(builder.backtrack_limit(100000).build().tryIsMatch("aaaa"));
        rejects(() => Regex.builder("a{20}").sizeLimit(2).build());
        expect(Regex.builder("a{20}").size_limit(100).build().isMatch("aaaaaaaaaaaaaaaaaaaa"));
        rejects(() => RegexBuilder.new("(((a)))").nestLimit(1).build());
        expect(RegexBuilder.new("(((a)))").nest_limit(10).build().isMatch("a"));
        expect(Regex.builder("the|that|them").optimize_prefixes(true).jit(true).build().isMatch("that"));
        rejects(() => Regex.new("("));
        rejects(() => RegexBuilder.new("[").build());
        rejects(() => Regex.new(42));
        rejects(() => Regex.escape(null));
        rejects(() => RegexBuilder.new(false));
        rejects(() => builder.jit(1));
        rejects(() => builder.optimizePrefixes("true"));
        rejects(() => builder.sizeLimit(-1));
        rejects(() => builder.nestLimit(0.5));
        rejects(() => builder.sizeLimit(4294967296));
        rejects(() => builder.backtrackLimit(9007199254740992));
        rejects(() => builder.backtrackLimit(0 / 0));
        rejects(() => builder.backtrackLimit(1 / 0));
        rejects(() => limited.find(12));
        rejects(() => limited.replace("abab", 12));
        rejects(() => limited.captures("aaaa").get(-1));
        rejects(() => limited.captures("aaaa").name(1));
    "#,
    );
}

#[test]
fn napalm_regex_retained_values_contracts_and_gc() {
    run_regex_test(
        r#"
        function identity(re: Regex): Regex { return re; }
        function retain(): RegexCaptures {
            return Regex.new("(?<letter>.)").captures("é");
        }
        const captures = retain();
        const match = captures.name("letter");
        const matches: RegexMatch[] = Regex.new(".").findIter("世🎉");
        System.gc();
        expect(captures["letter"] === "é" && match.asStr() === "é");
        expect(matches[1].asStr() === "🎉");
        expect(identity(Regex.new("x")).isMatch("x"));
        const re = Regex.new("(?<n>[0-9]+)");
        let sum = 0;
        for (let i = 0; i < 80; i++) {
            const c = re.captures("a12b34");
            sum += c.name("n").len();
            if (i % 10 === 0) System.gc();
        }
        expect(sum === 160 && captures[0] === "é");
    "#,
    );
}

#[test]
fn napalm_regex_rejects_wrong_runtime_types() {
    for name in ["Regex", "RegexBuilder", "RegexMatch", "RegexCaptures"] {
        let source = format!(
            "function typed(value: {name}) {{ return value; }} const result = typed({{}});"
        );
        let (program, entry, manifest) = compile_typescript_module(&source);
        let mut vm = source_vm(&program, &manifest);
        let error = vm.run_module(entry).unwrap_err();
        assert!(error.message.contains(name), "{error:?}");
    }
}

#[test]
fn napalm_regex_burning_roundtrip() {
    let source = r#"
        const re: Regex = Regex.builder("(?<word>[a-z]+)").optimizePrefixes(true).build();
        const captures: RegexCaptures = re.tryCaptures("123 abc");
        System.gc();
        const result = [captures["word"], captures.get(0).start(), re.replaceAll("abc def", "$1")];
    "#;
    let (program, entry, manifest) = compile_typescript_module(source);
    let bootstrap = crate::napalm_burning::Bootstrap {
        entry,
        module_name: "test".into(),
        builtins: manifest,
    };
    let bytes = crate::napalm_burning::encode(&program, &bootstrap).unwrap();
    drop((program, bootstrap));
    let artifact = crate::napalm_burning::decode(&bytes).unwrap();
    let mut vm = source_vm(&artifact.program, &artifact.bootstrap.builtins);
    vm.run_module(artifact.bootstrap.entry).unwrap();
    let value = last_source_variable(&vm);
    assert_eq!(
        HostContext { heap: &mut vm.heap }.format_value(value),
        "[abc, 4, $1 $1]"
    );
}
