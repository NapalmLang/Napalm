# MX IR to SQLite SQL

`napalm_mx_sqlite` compiles the existing MX IR to SQLite statements. It accepts
both logical plans and the output of the complete MX optimizer. Neither the IR
nor its passes have been changed for this backend.

The implementation includes bundled SQLite through `rusqlite`, SQL generation,
binding metadata, and scalar helper registration. `napalm_mx_vm` connects it to
the VM with explicit connections, schema reflection and captured parameters.
The standalone Rust compiler accepts tables matching an MX catalog in `main`.

## VM API

```napalm
const db = Database.open(":memory:"); // Or a persistent file path.
db.execute([[[
    CREATE TABLE users (id Number PRIMARY KEY, name String NOT NULL, active Bool NOT NULL);
    CREATE INDEX users_name ON users (name);
]]]);
const name = "Ada";
db.execute([[[ INSERT INTO users (id, name, active) VALUES (1, $name, true) ]]]);
const id = 1;
const rows = db.execute([[[ SELECT name FROM users WHERE id = $id ]]]);
System.print(rows[0].name);
db.close();
```

- `Database.open(path)` opens or creates a database; `":memory:"` is private to
  that connection. SQLite is bundled. Foreign keys are enabled.
- Schema creation and changes use SQL inside `[[[ ... ]]]`, through `db.execute`.
  The previous `createTable` and `createIndex` methods are no longer exposed.
- `db.getVersion()` reads the persistent application schema version (initially 0).
  `db.setVersion(n)` sets it to an integer from 0 through 9007199254740991.
- `db.execute(query)` executes the **whole block atomically**. It returns the
  last statement's result: a list of maps for SELECT, or an affected-row count
  for INSERT/UPDATE/DELETE, or zero for schema/version statements.
  Optimized-away writes count as zero. SELECT map keys
  are aliases, field names or `column1`, `column2`, etc. for unnamed expressions;
  duplicate names are errors (use `AS`). Earlier SELECT results are discarded.
- `db.close()` releases the connection and is idempotent; `db.closed` reports
  its state. Unreachable connections are also released by GC.

`$name` captures the current scalar value when the block is constructed, including
inside closures. Re-executing the same query uses those same captured values;
construct another block to bind new values. Null, Bool, Number, UTF-8 String,
Bytes and ByteBuffer are supported; byte buffers are copied. Objects and lists
are rejected. Captures are bound as SQL parameters, never interpolated.

`Database` and `DatabaseQuery` work with annotations, `typeof` and `instanceof`.
Queries survive GC and compile into `.burning` artifacts. Bytecode stores the
validated eDSL source template; the runtime checks table/column types at execution
against the connection's schema. Missing host variables remain compile errors.
Each connection caches its last compiled block by template, parameter types and
SQLite schema version; parameter values are rebound on every execution.

Execution is synchronous. Each block uses a deferred read transaction or an
immediate write transaction, with rollback on any error. There is no public
transaction spanning multiple `execute` calls, raw-SQL escape hatch or connection
transfer between concurrent workers yet. Open connections inside the worker.

Existing SQLite files must have an explicit non-null primary key and compatible declared
types (`REAL`/`INTEGER`/`NUMERIC`, `TEXT`, `BOOLEAN`, `BLOB` and their supported
aliases), binary text collation and matching storage values. Rowid-alias
`INTEGER PRIMARY KEY` columns are implicitly non-null; other primary key fields
must declare `NOT NULL`. Generated/hidden
columns are rejected. Partial, expression and non-binary indexes are omitted
from the MX catalog. Integer result values that cannot be represented exactly
as a Napalm Number are rejected. The catalog currently uses heuristic row counts;
the existing optimizer and its cost model are unchanged.

## Schema SQL and migrations

```napalm
if (db.getVersion() < 2) {
    db.execute([[[
        UPDATE TABLE users ADD COLUMN email String;
        CREATE INDEX users_email ON users (email);
        SET VERSION 2;
    ]]]);
}
```

`UPDATE TABLE` is an eDSL spelling of SQLite `ALTER TABLE`; both spellings are
accepted. Supported schema statements are:

- `CREATE TABLE [IF NOT EXISTS] name (column type [NOT NULL] [PRIMARY KEY]
  [DEFAULT literal], ..., [PRIMARY KEY (column, ...)])`
- `CREATE [UNIQUE] INDEX [IF NOT EXISTS] name ON table (column [ASC|DESC], ...)`
- `UPDATE TABLE name ADD [COLUMN] column type [NOT NULL] [DEFAULT literal]`
- `UPDATE TABLE name DROP [COLUMN] column`
- `UPDATE TABLE name RENAME [COLUMN] old TO new`
- `UPDATE TABLE name RENAME TO new`
- `DROP TABLE|INDEX [IF EXISTS] name`
- `SET VERSION integer`

Types are `Number`, `String`, `Bool`, `Bytes` or the compatible SQLite aliases
listed above. Columns are nullable unless marked `NOT NULL`. Every table needs
an explicit primary key; its fields become non-null, including composite keys.
Defaults must be scalar literals of the column's type; host parameters and
expressions are not supported in DDL defaults. `ADD COLUMN ... DEFAULT ...`
populates existing rows using SQLite's rules. INSERT retains the MX lowerer's
existing rules: supply all required fields explicitly; omitted nullable fields
become NULL, even if the SQLite column has a default.

Opening a database initializes `__napalm_metadata` with a singleton row
`(id = 1, version = 0)` if missing, without resetting an existing version.
The `__napalm_` namespace is reserved; user eDSL cannot read, mutate, drop or
rename these tables. The application version is independent of SQLite's
internal `schema_version`, which invalidates compiled query plans.

Use `SET VERSION` **inside the migration block** to commit schema, data and
version atomically. `db.setVersion(...)` is a separate operation and does not
combine earlier `execute` calls into one transaction. The host `getVersion()`
condition is also outside that transaction; applications with competing migration
runners must coordinate the version check and migration.

Schema statements have a separate AST-to-SQL lowerer (`napalm_mx_schema`), leaving
relational IR and optimizations unchanged. Mixed blocks execute in source order:
a CREATE/ALTER is visible to the next statement's catalog and lowering, and any
failure rolls back the whole block. Compiled plans from migrations are not cached.

## Compiler interface

```rust
use crate::napalm_mx_sqlite::{compile_query, install_functions};
use crate::napalm_mx_ir::QueryId;
use rusqlite::{Connection, params_from_iter, types::Value};

// `program` is an MX Program containing a catalog and lowered queries.
let compiled = compile_query(&program, QueryId(0))?;
let connection = Connection::open_in_memory()?;
install_functions(&connection)?;
// Create/load the application's schema and data before executing.

// ParamIds are query-local and assigned in first-use order by the MX lowerer.
let parameters = vec![Value::Real(42.0)];
for statement in &compiled.statements {
    let values = statement.bind_values(&parameters)?;
    let mut prepared = connection.prepare(&statement.sql)?;
    let mut rows = prepared.query(params_from_iter(values.iter()))?;
    while let Some(row) = rows.next()? {
        // Read cells according to compiled.columns. Convert INTEGER/REAL cells
        // to Napalm Number and 0/1 to Bool using that schema.
    }
}
```

`compile_program` compiles every query. Compilation validates the IR and does
not mutate the input. Each `SqlQuery` includes its query ID, parameter contracts,
result column definitions, transaction mode and ordered statements. Its bindings
contain either a query-local `ParamId` or an owned SQLite value. Identifiers are
quoted and string/blob/numeric constants are bound as values, never pasted into
SQL. Missing parameters and NaN bindings are errors. Parameter type/nullability
contracts remain the responsibility of the caller/VM adapter.

Use a single transaction for **all statements of a writable query**, rolling
back on any error. INSERT/WriteBatch currently produce one statement per row to
preserve sequential conflict behavior and support different field sets. The SQL
does not embed BEGIN/COMMIT, so it can participate in an existing transaction.
`ReadOnlySnapshot` requires a read transaction when an executor runs multiple
statements. Compilation alone does not promise atomicity across separate queries
in an eDSL block.

## Preserved plan semantics

| MX operation | SQL representation |
| --- | --- |
| Scan, filter, projection | SELECT CTEs with query-local column aliases |
| Sort, limit, offset | Explicit order columns and LIMIT/OFFSET |
| Distinct on selected columns | ROW_NUMBER partitioning, retaining one input row |
| Inner/left joins | JOIN / LEFT JOIN |
| Semi/anti joins | Correlated EXISTS / NOT EXISTS |
| Aggregate | GROUP BY and COUNT/SUM/MIN/MAX/AVG, including DISTINCT |
| UnionAll, Values, Empty, Exists | Positional UNION ALL, constant SELECTs, empty SELECT, EXISTS |
| PointGet, primary/index ranges | Key predicates, direction, null ordering, residual and window |
| FetchRows | Join against carried primary-key identities |
| RangeCount | COUNT over the bounded range |
| Buffered Materialize | AS MATERIALIZED CTE |
| Insert / WriteBatch Insert | INSERT, preserving duplicate-key errors |
| WriteBatch Set | Primary-key ON CONFLICT DO UPDATE |
| Update / Delete | Materialized target keys and old RHS values |
| WriteBatch Update / Delete | Per-key UPDATE / DELETE |

Hidden key columns survive pruning and support late fetches and writes. Hidden
order columns survive projection, so an optimizer-eliminated Sort still has a
corresponding final SQL ORDER BY. These columns are excluded from results.
Zero-column relations use one dummy SQL cell per row; `discard_dummy_column`
instructs an executor to discard the cell while preserving row cardinality.

Physical join algorithms, borrowing/visitor delivery, covering hints and memory
cost estimates do not force a SQLite implementation. Their relational semantics
are retained; SQLite chooses actual access paths. Index metadata is translated
to predicates and ordering rather than INDEXED BY. Existing MX optimizations
still run unchanged, including their storage-oriented cost model; that model is
not a SQLite cost estimate. Updating it can be a separate measured change.

For SET, a primary-key conflict updates that row and resets omitted fields to
SQLite defaults/NULL. It does not use INSERT OR REPLACE, which could delete a
different row on an unrelated unique constraint. Batch DELETE takes only primary
key fields; batch UPDATE takes the key plus at least one non-key assignment.

## Scalar semantics and limits

Call `install_functions` once per connection. The generated SQL uses the
`napalm_mx_*` functions for f64 arithmetic, floating remainder, absolute value,
floor, ceiling, round-away-from-zero, ASCII case conversion, Unicode scalar
string length (including embedded NUL), and byte length. These functions are
deterministic, propagate NULL, and do not depend on SQLite's optional math
extension. They avoid SQLite's integer division and integer remainder semantics.

SQLite cannot represent NaN distinctly from NULL. The compiler, binding helper
and numeric functions reject NaN instead of silently substituting NULL.
Infinities are allowed. Aggregate SUM/AVG use SQLite floating-point aggregation;
floating-point reassociation remains subject to the existing optimizer's rules.
Schema storage classes/collations must match the catalog: numbers use numeric
storage, booleans use 0/1, and text comparison uses binary collation. Persistent
type contracts and schema creation are not introduced by this compiler.

SQLite LIMIT/OFFSET are signed 64-bit; larger MX u64 windows are rejected.
Invalid references, unavailable columns, nested writes, and ambiguous/lost target
identities return `CompileError`. Expressions/plans have compilation depth
limits; SQLite's own resource limits still apply when preparing statements.
The caller must use a SQLite version supporting window functions and materialized
CTEs; the bundled build used by the tests does.

## Verification

Tests execute generated SQL against in-memory SQLite, comparing logical and
optimized results and checking CRUD, nullable parameters, joins, aggregates,
ordering after pruning, late fetch, ranges, composite keys, transaction rollback,
identifier escaping and scalar semantics.

```sh
cargo test --no-default-features --offline --bin napalm napalm_mx -- --test-threads=1
```

SQLite references: [CTEs and materialization](https://www.sqlite.org/lang_with.html),
[UPSERT](https://www.sqlite.org/lang_upsert.html),
[expression semantics](https://www.sqlite.org/lang_expr.html).
