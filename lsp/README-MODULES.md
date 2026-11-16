# Milestone 5 — Cross-module resolution

All analysis remains **in Napalm**. No Rust frontend API, Node.js runtime,
external parser, or FFI is used by the language server. Node.js is used only
for external test harnesses.

## Source files

- `src/module_index.napalm`: module graph, import resolution, exports,
  re-exports, URI handling, cross-file definitions and references, and
  dependency diagnostics.
- `src/semantic.napalm`: unchanged local lexical scoping except imports now
  register both a type and value binding where appropriate.
- `src/server.napalm`: rebuilds the graph after document edits and received
  `workspace/didChangeWatchedFiles` notifications; republishes diagnostics for
  open documents whose dependencies may have changed.
- `scripts/modules_smoke.napalm`: native Napalm smoke test.
- `tests/module-index.test.mjs`: cross-file algorithm tests with a JavaScript
  compatibility adapter and temporary filesystem fixtures.
- `tests/server.test.mjs`: binary LSP integration tests, including cross-file
  navigation; run only when a Napalm executable is available.

## Resolution rules

1. For a source import such as `./module`, try `module.napalm`, followed by
   `module/index.napalm`. An explicit extension must be `.napalm`.
2. Bare imports without `@lib/` or `@std/` are importer-relative, mirroring
   Napalm's module resolver.
3. `@lib/name` and `@std/name` search `.napalm/packages` or `.napalm/std`
   beneath the project root, their corresponding environment path variables,
   and `NAPALM_HOME/packages` or `NAPALM_HOME/std`.
4. The workspace roots supplied through `initialize.workspaceFolders`,
   `initialize.rootUri`, or `initialize.rootPath` take precedence over project
   marker discovery (`.git`, `.napalm`, `Napalm.toml`, `napalm.toml`).
5. File URIs and paths are normalized with Napalm's native `Uri` and `Path`
   APIs; Windows paths are compared case-insensitively.
6. Open editor buffers override disk files, even before the user saves.
7. Re-export chains and import aliases resolve to the originating declaration.
   Cycles are guarded so resolution terminates.

## Diagnostics

- `NAPALM-MOD-001`: missing, unreadable, or rejected module path.
- `NAPALM-MOD-002`: imported/re-exported name not found in the target module.
- `NAPALM-MOD-004`: duplicate exported name in the same type/value domain.

When a dependency contains lexical or parser errors, missing-export
notifications for that dependency are suppressed to reduce cascading errors.
A malformed open document still receives lexer and parser diagnostics;
semantic and module diagnostics wait until its syntax is valid.

## Navigation

The supported LSP methods remain `textDocument/definition`,
`textDocument/references`, and `textDocument/documentSymbol`. Their behavior
now extends to reachable imported files. References are collected across all
modules reachable from open documents, not across every unrelated file on disk.
Ranges are sourced from the same UTF-16 AST metadata as the previous milestone.

## Bounds, invalidation and limitations

- A graph includes at most 512 modules, with at most 4 MiB of UTF-8 text per
  source, and a maximum re-export chain depth of 128. These are defensive
  limits, not language restrictions.
- The graph is rebuilt after each open, change, or close notification. This is
  correct for unsaved edits but not yet an incremental dependency cache. Large
  workspaces should eventually reuse unchanged per-module analyses.
- Closed disk files are reread when a graph is rebuilt. Filesystem changes
  without an editor edit can also be processed when the client sends
  `workspace/didChangeWatchedFiles`. Automatic registration of watched-file
  notifications and direct `Fs.watch` wiring are not included yet.
- Local `this.field` resolution is provided by the earlier semantic pass; this
  milestone does not add full class inheritance or type inference.
- Module namespace wildcard imports (`import * as ns`) and `export *` are not
  currently supported by the Napalm parser and are not implemented here.
- Package search does not yet implement every binary-adjacent standard-library
  location of the engine's Rust resolver.
- Non-file LSP document URIs are not indexed as source files.
- Path aliases referring to the same file through different symlinks can still
  result in separate index records. Package-root symlink escapes are checked
  against real filesystem paths.
- Physical project files are read through `Fs.readText` from inside the Napalm
  process. A missing or unreadable dependency produces a diagnostic; the
  server itself does not crash or execute the imported module.

## Validation

From the repository root, with the updated Napalm runtime:

```powershell
.\target\release\napalm.exe .\lsp\scripts\modules_smoke.napalm
.\target\release\napalm.exe .\lsp\scripts\semantic_smoke.napalm
node --test .\lsp\tests\*.test.mjs
```

The JavaScript-compatible tests exercise the actual lexer, parser, semantic
analyzer, and cross-module index logic with an adapter for Napalm primitives;
they do **not** constitute proof that Napalm compiles the code. Running the
smoke test and the live LSP tests under Napalm is required before release.

A production binary can be created using the engine's new AOT build mode:

```powershell
.\target\release\napalm.exe --compile .\lsp\main.napalm
```

The editor should launch the resulting standalone server executable with
stdio pipes. Never use `System.print()` in the server process: `stdout` is
reserved for JSON-RPC messages.
