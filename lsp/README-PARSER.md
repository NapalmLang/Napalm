# Napalm LSP — Milestone 3: syntax parser

The parser is implemented entirely in Napalm. There is no Rust frontend API,
FFI bridge, embedded TypeScript parser, or AST conversion from Rust.

## Files

- `src/parser.napalm`: error-tolerant parser and AST constructors.
- `src/lexer.napalm`: unchanged lexer, the authoritative token source.
- `src/server.napalm`: now publishes both lexer and parser diagnostics.
- `scripts/parser_smoke.napalm`: executable AST assertions.
- `scripts/parser_selfhost.napalm`: parses every LSP source module with the Napalm parser.
- `scripts/parse_file.napalm`: debugging CLI to dump an AST and diagnostics as JSON.
- `tests/server.test.mjs`: includes a live invalid-to-valid syntax edit regression.

## AST API

```napalm
import {parse} from "./src/parser";

const result = parse("const answer = 42;");
const first = result.ast.body[0];
System.print(first.kind); // VariableDeclaration
```

`parse(text, lexical?)` optionally accepts the result from `tokenize(text)`;
this avoids lexing the same unsaved document twice. Return shape:

```text
{
  ast: {
    kind: "Module",
    startByte: 0,
    endByte: <UTF-8 byte count>,
    range: {start: {line, character}, end: {line, character}},
    body: [statement nodes...]
  },
  diagnostics: [
    {code, message, severity: 1, startByte, endByte, range}, ...
  ]
}
```

All AST nodes are JSON-serializable Maps. Every node carries half-open byte
spans and a UTF-16 LSP range. The AST is *syntactic*: no symbol resolution,
type checking, macro expansion, or module loading is performed by the parser.

Recovery is designed for incomplete editor buffers: a syntax error yields an
error node or a local diagnostic, then parsing continues at the next separator
where possible. Safeguards limit recursion, AST nodes, and diagnostics.

## Commands (from repository root)

```powershell
.\target\release\napalm.exe .\lsp\scripts\parser_smoke.napalm
.\target\release\napalm.exe .\lsp\scripts\parser_selfhost.napalm
.\target\release\napalm.exe .\lsp\scripts\parse_file.napalm .\lsp\src\parser.napalm
node --test .\lsp\tests\*.test.mjs
```

`parse_file.napalm` is intentionally a standalone command-line debugging tool:
it writes JSON to stdout. The LSP executable must reserve stdout exclusively
for JSON-RPC frames.

## Current implementation boundaries

This milestone covers expressions, assignments, calls, member access, arrays,
objects, list comprehensions, both arrow and bar lambdas, templates, statements,
classes and members, basic decorators, functions, modules, interfaces, enums,
type aliases, union/intersection/array/generic types, and recovery.

It is **not yet a replacement** for every compiler grammar production. The
upcoming milestones will compare actual parse trees against the latest Rust
frontend, add missing advanced language constructs, and implement name
resolution and semantic diagnostics in Napalm. The Rust frontend is a test
reference only, not a runtime dependency.

A development run of parser logic in a JavaScript compatibility harness passed
its own smoke suite and parsed its source modules without syntax diagnostics.
This is **not** evidence of a successful Napalm VM build. Cargo/Napalm runtime
validation must be performed in the user's repository.
