# Napalm LSP – Milestone 7: flow-aware editor intelligence

The lexer, parser, semantic analyzer, module resolver, basic inference engine,
and all new LSP features are implemented in Napalm. No Rust frontend API is used.

## Functionality

- Query-local flow narrowing inside `if` / `else` for null guards, truthy local
  identifiers, and `typeof` equality tests; sequential writes and branch joins.
  Terminal branches ending with `return` or `throw` are excluded from joins.
- Cached lexical, parser and semantic snapshots for unchanged modules. Every
  refresh rebuilds links and module diagnostics. Unsaved editor buffers have
  priority, and watched-file notifications invalidate affected disk paths.
- LSP `textDocument/signatureHelp`, `textDocument/prepareRename`,
  `textDocument/rename`, `textDocument/semanticTokens/full`, and
  `textDocument/semanticTokens/range`.
- Fix: `hover` and `completion` now reach their real dispatchers instead of
  returning document symbols.

## Validation

Run from repository root:

```powershell
node --test .\lsp\tests\*.test.mjs
.\target\release\napalm.exe .\lsp\scripts\flow_editor_smoke.napalm
.\target\release\napalm.exe .\lsp\scripts\intelligence_smoke.napalm
.\target\release\napalm.exe --compile .\lsp\main.napalm
```

Node tests run a compatibility harness and exercise the actual Napalm source
under adapted JavaScript semantics. They do not prove that the Napalm engine
compiles or runs these modules. Live integration tests automatically activate
when `NAPALM_BIN` points to a built executable.

## Limitations and safety

- Flow analysis is bounded, non-SSA and query-local. It narrows local names,
  not mutable object properties or arbitrary aliases. Calls, exceptions,
  closures, loops and cross-thread mutation are treated conservatively.
- Signatures currently cover named functions, declared parameters and known
  object methods. Advanced overloads and constructor signatures are not yet
  supported.
- Rename generates client-applied edits; it never writes files. Aliased imports
  are handled separately so renaming the origin does not silently rename the
  local alias. Property renames and strings/reflection are not covered.
- The index uses file-watch notifications as the invalidation source for disk
  changes. A change invisible to the editor's watcher may remain cached until
  a later invalidation or server restart. Re-export links are always refreshed.
- Semantic token kinds are deliberately conservative. A token legend is
  advertised to the LSP client and all offsets use UTF-16 code units.
