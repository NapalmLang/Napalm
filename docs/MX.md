# Embedded database language

Napalm reserves contiguous `[[[ ... ]]]` for its database eDSL. The initial
implementation parses these blocks into a separate MX AST and lowers that AST
to verified, typed `napalm_mx_ir` logical plans against an existing catalog.
The existing MX optimizer can then select physical operations. The
[SQLite backend](MX_SQLITE.md) accepts both logical and optimized IR and emits
parameterized SQL, retaining the current IR and optimization pipeline.

The VM executes blocks through explicit SQLite connections. A block constructs a
`DatabaseQuery` and captures its `$parameters` from lexical Napalm variables.
`db.execute(query)` resolves the schema, lowers to MX IR, runs the existing
optimizer and executes parameterized SQL. See the [runtime API](MX_SQLITE.md#vm-api)
and [runnable example](../scripts/database.napalm). The syntax examples below
assume a `users` table and host variables `minimumAge` and `id`.

```napalm
const query = [[[
    SELECT name AS label, age + 1 AS next_age
    FROM users
    WHERE active AND age >= $minimumAge
    ORDER BY age DESC
    LIMIT 20 OFFSET 5;
]]];

const writes = [[[
    INSERT INTO users (id, name, active)
    VALUES ($id, 'Ada', true), (2, 'Grace', false);
    UPDATE users SET name = 'Ada Lovelace' WHERE id = $id;
    DELETE FROM users WHERE name IS NULL;
]]];
```

## Initial grammar

- `SELECT * | expression [AS name], ... FROM table [WHERE expression]
  [ORDER BY expression [ASC | DESC], ...] [LIMIT integer] [OFFSET integer]`
- `INSERT INTO table (field, ...) VALUES (expression, ...), ...`
- `UPDATE table SET field = expression, ... [WHERE expression]`
- `DELETE FROM table [WHERE expression]`

Statements in a block are separated by semicolons; the final semicolon is
optional. Empty blocks are errors. Keywords ignore case, while table, field,
alias and parameter names preserve case and use ASCII identifier syntax.
Newlines are whitespace. `--`, `//`, and nested `/* ... */` comments are accepted.
Strings use single or double quotes; repeat the quote to escape it (`'it''s'`).
Backslashes are literal. Strings and comments can contain `]]]`.

Expressions support field names, `$name` parameters, numbers, strings, booleans,
`NULL`, parentheses, unary `-`, `NOT`/`!`, arithmetic `+ - * / %`, comparisons
`= == != <> < <= > >=`, `AND`, `OR`, and `IS [NOT] NULL`.
Precedence from low to high is OR, AND, NOT, comparisons/IS NULL, addition,
multiplication, unary minus. Nullable predicates use the IR's three-valued
logic; filters retain TRUE. Sorting uses NULLS LAST in either direction.
ORDER BY expressions resolve against input fields; projection aliases are not
visible there. LIMIT/OFFSET are literal unsigned 64-bit integers.

INSERT checks duplicate fields, row widths, required fields and assignment
types. Missing nullable fields become NULL. UPDATE expressions read the input
row and check duplicate assignments, types and nullability. An omitted WHERE
targets all rows. No implicit schema creation, conversions, INSERT default-value
expansion, qualified names, joins, aggregates, RETURNING, or transaction syntax
is implemented. Schema statements (`CREATE TABLE`, `CREATE INDEX`, `UPDATE TABLE` /
`ALTER TABLE`, `DROP TABLE` / `DROP INDEX`) and `SET VERSION` are described in
[schema SQL and migrations](MX_SQLITE.md#schema-sql-and-migrations).

## Compiler API

`napalm_front_main::parse` (and the legacy parser) produce
`napalm_front::ExprKind::Mx(Box<Block>)`. The host lexer consumes each block as
one token and delegates its contents to `napalm_mx_front::parse(body, offset)`.
Spans are absolute UTF-8 byte offsets into the original Napalm source.

`napalm_mx_lowerer::lower(&block, &mut program, &parameters)` appends one query
per data statement and returns its `QueryId`s; schema statements use the separate
`napalm_mx_schema::compile` lowerer and are rejected by this relational entry point.
The caller supplies an MX `Program`
whose catalog names refer to its string pool, and typed `Parameter` contracts
(`name`, `ty`, `nullable`). Only referenced parameters are emitted, in first-use
order, with a single `ParamId` per name in each query. `$name` is a binding slot,
not string substitution or a Napalm expression. The SQLite compiler returns
positional bindings; the VM adapter supplies captured host values.

Lowering uses query-local columns, emits snapshot reads or read/write queries,
and calls `verify_query` before appending each result. A failed block rolls back
all newly appended queries and strings. The statements are separate query
plans; the VM executes the entire block in one atomic transaction.

The resulting program can be passed to
`napalm_mx_ir_optimizing::optimization_passes().optimize_program(&mut program)`.
`src/tests/napalm_mx_tests.rs` contains catalog setup and end-to-end examples.

## Bracket compatibility

Every contiguous `[[[` outside host strings/comments opens an MX block.
Double brackets keep their array meaning: `[[1, 2], [3, 4]]` works unchanged.
For three immediately nested arrays, separate the opening brackets with
whitespace: `[ [[1, 2]]]`. Ordinary `]` tokens outside MX blocks retain their
existing meaning. Only three consecutive closing brackets terminate an MX
block; one or two are passed to the MX frontend.

## Validation

```sh
cargo test --no-default-features --offline --bin napalm napalm_mx -- --test-threads=1
cargo test --no-default-features --offline --bin napalm napalm_front -- --test-threads=1
```
