# Napalm LSP parser: compiler-compatibility fixes

This delivery replaces the affected `lsp/` sources from milestone 3.
It uses the uploaded `server(1).napalm` (identical to the previous server)
and fixes compilation failures originating in its `./parser` import.

## What changed

- Removed unsupported default parameters from the 11 `Parser` methods that had them.
- Updated 131 method invocations to pass the original default values explicitly.
- Retained `parse(source)` and added `parseLexed(source, lexical)` for clients that have already tokenized the source. The LSP server and smoke test use `parseLexed`.
- Replaced `new Set(iterable)` with `createStringSet` using `new Set()` followed by `.add(value)`, as required by Napalm.
- Replaced the unsupported comma-separated `for` increment with an equivalent increment inside the loop body.
- Added a minimal independent lexical-scope regression script, plus Node static compatibility checks.

## Why the original diagnostics multiplied

The Rust frontend explicitly rejects default parameter initializers
(`default parameters are not implemented yet`). Its syntax recovery then
misinterpreted the remaining class methods, producing dozens of errors
about missing identifiers, statements and duplicate bindings. These are
most plausibly cascading diagnostics, not independent bugs in variable
scoping. The independent scope test helps verify that hypothesis.

If `engine_scope_regression.napalm` still produces `NAPALM-B001` with no
other syntax errors, that would be evidence of a genuine engine scope bug.

## Smoke tests (repository root)

```powershell
.\target\release\napalm.exe .\lsp\scripts\engine_scope_regression.napalm
.\target\release\napalm.exe .\lsp\scripts\parser_smoke.napalm
.\target\release\napalm.exe .\lsp\scripts\parser_selfhost.napalm
node --test .\lsp\tests\*.test.mjs
```

For an LSP test, launch `lsp/main.napalm`, not `lsp/src/server.napalm`.
`server.napalm` is a module that exports handlers, not an executable entrypoint.

## Verification limitations

Node tests were executed in this environment. The parser smoke tests and
self-hosting source checks were also executed using a JavaScript compatibility
shim; this does **not** exercise Napalm's compiler, VM, type system, or GC.
No Napalm or Rust executable is installed in this environment, so the actual
engine compilation and integration tests remain pending.
