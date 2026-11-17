# Napalm LSP — Milestone 6: completion, hover and gradual type inference

The language server remains fully written in Napalm. JavaScript/Node is used **only** in the external test harness; the server does not load the engine's Rust frontend or a TypeScript language service.

## What changed

- `src/intelligence.napalm` implements bounded, side-effect-free type inference over the Napalm AST. Inference supports literal types, annotations, generic and array annotations, variables, function parameters and explicit/straightforward inferred returns, object literals, interfaces, classes, enums, built-in methods and imports that resolve to other modules (including barrel re-exports). It deliberately returns `unknown` for ambiguous results and recursive cycles.
- `src/semantic.napalm` now exposes lexical scope intervals and links symbol declarations to the AST node that owns them. Existing diagnostics and navigation continue using the same symbol identities.
- `src/server.napalm` advertises `hoverProvider` and `completionProvider`, validates cursor positions and responds to `textDocument/hover` and `textDocument/completion`.
- `tests/intelligence.test.mjs` runs the actual lexer, parser, semantic pass, module index, and inference sources in a compatibility harness. `scripts/intelligence_smoke.napalm` tests the same APIs under the Napalm engine.

## Completion behavior

Completion returns an LSP CompletionList. It includes visible local bindings and parameters, imports, language keywords, and standard globals. In type annotation contexts it also offers visible types and built-in type names. Member completion recognizes simple expressions such as `person.na`, `Process.st`, `Path.no` and `this.fi`, filtered by the typed prefix. Result items include the inferred type in `detail`. The list is capped at 200 items.

More advanced receiver expressions (`getPerson().na`, long chained member expressions, computed keys) are not yet fully supported. Complex type-narrowing, overload resolution and full generic substitution are also future work. Incomplete programs can still yield partial completions when the lexer/parser recover successfully.

## Hover behavior

Hover displays an LSP Markdown `napalm` code fence with a symbol's declaration category and inferred signature/type. Imported symbol hovers follow the module-index link to the original declaration when available. Simple property hovers use object/class member information.

## Inference limitations

This is **gradual inference**, not a sound or complete static type checker. It does not evaluate arbitrary user code or execute imports. Complex control flow, generic specialization, inherited class members, async resolution, polymorphic overloads, all possible built-in APIs, and arbitrary member chains remain incomplete. Recursive inference has a maximum depth of 32 and cycle detection. There are intentionally no new type-mismatch diagnostics until the analysis can prove them reliably.

The module graph is still reconstructed upon every document change (Milestone 5 behavior). Inference is performed on demand rather than holding persistent GC-managed caches of types for unsaved document versions.

## How to validate

From the repository root:

```powershell
node --test .\lsp\tests\*.test.mjs
.\target\release\napalm.exe .\lsp\scripts\intelligence_smoke.napalm
.\target\release\napalm.exe .\lsp\scripts\semantic_smoke.napalm
.\target\release\napalm.exe --compile .\lsp\main.napalm
```

The Node compatibility harness cannot prove native runtime compatibility. The actual Napalm smoke tests and LSP tests requiring a Napalm executable must be run locally.

## VS Code integration

The extension in `editors/vscode` now includes a small JavaScript LanguageClient
launcher (`extension.js`). This is **not** a language frontend; it only connects
VS Code to the compiled Napalm server over standard I/O. To use it, first run
`napalm --compile lsp/main.napalm`, then build/install the VS Code extension:

```powershell
cd .\lsp\editors\vscode
npm install
npx vsce package --allow-missing-repository
code --install-extension .\napalm-language-0.1.0.vsix
```

The extension automatically searches for `lsp/main.exe` (Windows) or
`lsp/main` (Linux/macOS) in the workspace, or you can set
`napalm.languageServerPath` to the standalone executable. Reload the VS Code
window after changing the setting. The executable is not bundled in this ZIP.
