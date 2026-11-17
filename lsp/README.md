# Napalm Language Server

An independent language server written entirely in **Napalm**. The project
intentionally reimplements the compiler's frontend in the language itself to
exercise the language runtime; it does not expose or call the Rust frontend.

## Components and milestone status

| Module | Responsibility |
| --- | --- |
| `src/transport.napalm` | Binary-safe JSON-RPC over stdin/stdout |
| `src/documents.napalm` | Open buffers and incremental document edits |
| `src/positions.napalm` | UTF-16 LSP positions and Unicode text editing |
| `src/lexer.napalm` | Loss-tolerant tokens, templates, comments and lex errors |
| `src/parser.napalm` | Napalm AST, Pratt expressions, declarations and error recovery |
| `src/semantic.napalm` | Local scopes, symbols, references and semantic diagnostics |
| **`src/module_index.napalm`** | **Intermodule imports, exports, re-exports, cross-file navigation** |
| `src/server.napalm` | LSP lifecycle, diagnostics and navigation handlers |
| `main.napalm` | Executable entry point |

Syntax coloring is delivered by the separate declarative VS Code extension in
`editors/vscode`; it works without starting the language server.

Milestone details: [Parser](README-PARSER.md),
[Semantic](README-SEMANTIC.md),
[Cross-module](README-MODULES.md).

## Running

```powershell
.\target\release\napalm.exe .\lsp\main.napalm
```

Do not execute `src/server.napalm` as the entry point; it is an imported
module. The server's stdout is exclusively reserved for JSON-RPC frames.

## Verifying

```powershell
.\target\release\napalm.exe .\lsp\scripts\documents_smoke.napalm
.\target\release\napalm.exe .\lsp\scripts\lexer_smoke.napalm
.\target\release\napalm.exe .\lsp\scripts\parser_smoke.napalm
.\target\release\napalm.exe .\lsp\scripts\parser_selfhost.napalm
.\target\release\napalm.exe .\lsp\scripts\semantic_smoke.napalm
.\target\release\napalm.exe .\lsp\scripts\modules_smoke.napalm
node --test .\lsp\tests\*.test.mjs
```

The Node tests are a secondary compatibility and protocol harness. Real
compilation and smoke tests under Napalm remain mandatory for release.

## Building an executable

```powershell
.\target\release\napalm.exe --compile .\lsp\main.napalm
```

The AOT executable includes the Napalm runtime and imported LSP modules. The
binary is platform-specific but does not need separately distributed sources.

## Milestone 6: editor intelligence

See [README-INTELLIGENCE.md](README-INTELLIGENCE.md) for type inference, hover, and code completion, plus tests and known limitations.
