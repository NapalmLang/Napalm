# Milestone 4 — Semantic symbols and local navigation

The language server now implements **lexical symbol resolution in Napalm**. It
uses the lexer and syntax-tree parser from the previous milestones; it does not
load the engine's Rust frontend, use FFI, or delegate analysis to TypeScript.

## Components

- `src/semantic.napalm`: scopes, declarations, value/type domains, identifier
  references, duplicate and unresolved-name diagnostics, document symbols.
- `src/server.napalm`: caches one analysis per open document version, publishes
  semantic diagnostics, and responds to three new LSP requests.
- `scripts/semantic_smoke.napalm`: executable tests written in Napalm.
- `tests/semantic.test.mjs`: independent algorithm tests with handwritten ASTs.
- `tests/semantic-compat.test.mjs`: tests the lexer, parser, and semantic source
  together with a limited JavaScript compatibility adapter. This is **not** a
  substitute for running the tests in the Napalm VM.
- `tests/server.test.mjs`: live JSON-RPC tests, enabled when `NAPALM_BIN` points
  at a built engine or a binary exists at `target/release/napalm[.exe]`.

## Supported LSP capabilities

| LSP method | Result |
| --- | --- |
| `textDocument/definition` | Local declaration location, or `null` |
| `textDocument/references` | In-document reference locations, optionally including the declaration |
| `textDocument/documentSymbol` | Functions, classes (with members), enums, types, interfaces, variables |
| `textDocument/publishDiagnostics` | Lexical, syntax, and semantic diagnostics |

`initialize` advertises `definitionProvider`, `referencesProvider`, and
`documentSymbolProvider`; no unimplemented capability is claimed.

### Semantic coverage

Scopes: module, block, function, class, `for` loops, `switch`, `catch`, and list
comprehensions. Handles function parameters and defaults, destructuring,
imports/exports, generic type parameters, class members (`this.property`),
accessor pairs, and enum members. Forward declarations are indexed before body
analysis for editor navigation.

Value names and type names have separate lookup domains. Resolution of names
from the same document includes nested lexical scopes and shadowing.

The current semantic errors are:

- `NAPALM-SEM-001`: unresolved expression identifier (engine globals exempt)
- `NAPALM-SEM-002`: duplicate declaration in one lexical scope
- `NAPALM-SEM-003`: excessive semantic nesting

If the lexer or parser reports any error for an incomplete document, semantic
errors are suppressed to prevent cascading false positives. Local navigation
still uses the partial AST.

## Important limits

This is **not yet a full static type checker**. It does not infer value types,
resolve arbitrary object properties, find definitions across modules, or report
missing type declarations. Import names resolve locally and now jump across files through the
separate module graph implemented in milestone 5. The list of built-in globals is based on the core runtime
surface and should be refreshed when new built-ins are added.

Forward references can be navigated even where a runtime temporal-dead-zone
rule might later reject evaluation. Cross-file symbol resolution and richer
semantic diagnostics belong to a subsequent milestone.

Only source files are analyzed; `.burning` remains a compiled binary format.

## Validation

From the repository root:

```powershell
.\target\release\napalm.exe .\lsp\scripts\semantic_smoke.napalm
node --test .\lsp\tests\*.test.mjs
```

To run the live server tests with another engine path:

```powershell
$env:NAPALM_BIN = "C:\\path\\to\\napalm.exe"
node --test .\lsp\tests\server.test.mjs
```

The server entry point is `lsp/main.napalm`; `src/server.napalm` is a module.
For a standalone server binary, use Napalm's AOT packaging workflow for that
entry point (e.g. `napalm --compile lsp/main.napalm`). Do not emit debugging
text on `stdout`, as it carries LSP JSON-RPC frames.

## Update: milestone 5

Intermodule navigation, source resolution, and cross-file references are now
implemented in `src/module_index.napalm`. See [README-MODULES.md](README-MODULES.md)
for current rules and limits.
