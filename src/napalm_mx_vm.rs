//! SQLite connections and captured MX queries owned by the VM. These are GC
//! leaves: captured values, cached plans and SQLite results contain no GC refs.
use super::*;
use crate::napalm_mx_front as ast;
use crate::napalm_mx_ir as mx;
use crate::napalm_mx_lowerer::{self, Parameter};
use crate::napalm_mx_schema as schema;
use crate::napalm_mx_sqlite::{self, SqlQuery};
use rusqlite::{Connection, TransactionBehavior, params_from_iter, types::Value as SqlValue};

#[derive(Debug)]
pub(super) struct NapalmQuery {
    payload: String,
    block: ast::Block,
    captures: Vec<Capture>,
}
#[derive(Clone, Debug)]
struct Capture {
    name: String,
    ty: mx::ScalarType,
    value: SqlValue,
}
impl NapalmQuery {
    pub(super) fn external_bytes(&self) -> usize {
        // Source size also approximates the owned AST (names, strings, nodes).
        self.payload.capacity() * 3
            + self
                .captures
                .iter()
                .map(|c| {
                    c.name.capacity()
                        + match &c.value {
                            SqlValue::Text(v) => v.capacity(),
                            SqlValue::Blob(v) => v.capacity(),
                            _ => 16,
                        }
                })
                .sum::<usize>()
    }
}
#[derive(Debug)]
struct Cached {
    payload: String,
    types: Vec<mx::ScalarType>,
    schema: i64,
    program: mx::Program,
    queries: Vec<SqlQuery>,
}
#[derive(Debug)]
struct DatabaseState {
    connection: Option<Connection>,
    cached: Option<Cached>,
}
#[derive(Debug)]
pub(super) struct NapalmDatabase {
    pub(super) mode: DatabaseMode,
    /// File path or shared-memory name; empty for Memory/Temporary.
    pub(super) path: String,
    state: parking_lot::Mutex<DatabaseState>,
}
impl std::fmt::Display for NapalmDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.mode.takes_path() {
            write!(f, "<Database mode={} path={:?}>", self.mode.name(), self.path)
        } else {
            write!(f, "<Database mode={}>", self.mode.name())
        }
    }
}
impl NapalmDatabase {
    pub(super) fn external_bytes(&self) -> usize {
        let state = self.state.lock();
        self.path.capacity()
            + state.cached.as_ref().map_or(0, |c| {
                c.payload.capacity() * 4
                    + c.queries
                        .iter()
                        .flat_map(|q| &q.statements)
                        .map(|s| s.sql.capacity())
                        .sum::<usize>()
            })
    }
}
enum DatabaseResult {
    Changes(u64),
    Rows(Vec<(String, mx::ScalarType)>, Vec<Vec<SqlValue>>),
}
fn failure(error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::new(format!("Database: {error}"))
}
fn schema_type(declared: &str) -> Result<mx::ScalarType, RuntimeError> {
    schema::scalar_type(declared).map_err(failure)
}

fn get_version(connection: &Connection) -> Result<i64, RuntimeError> {
    // A read-only connection cannot create the metadata table, so a database
    // that Napalm never wrote reports the initial version.
    if connection.is_readonly(rusqlite::MAIN_DB).map_err(failure)? {
        let exists: bool = connection
            .query_row(
                "SELECT EXISTS (SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?1)",
                [schema::METADATA],
                |r| r.get(0),
            )
            .map_err(failure)?;
        if !exists {
            return Ok(0);
        }
    }
    let version: i64 = connection
        .query_row(
            "SELECT version FROM __napalm_metadata WHERE id = 1",
            [],
            |r| r.get(0),
        )
        .map_err(failure)?;
    if version < 0 || version as u64 > schema::MAX_VERSION {
        return Err(failure("invalid database version"));
    }
    Ok(version)
}

/// Opens the SQLite connection for `mode`. Paths are always literal (no
/// `file:` URI interpretation), so a file named `:memory:` needs `./:memory:`.
fn open_connection(mode: DatabaseMode, path: &str) -> Result<Connection, RuntimeError> {
    use rusqlite::OpenFlags as F;
    let base = F::SQLITE_OPEN_NO_MUTEX;
    let read_write = base | F::SQLITE_OPEN_READ_WRITE;
    let opened = match mode {
        DatabaseMode::Memory => Connection::open_in_memory_with_flags(read_write | F::SQLITE_OPEN_CREATE),
        // SQLite's empty filename: a private temporary file deleted on close.
        DatabaseMode::Temporary => Connection::open_with_flags("", read_write | F::SQLITE_OPEN_CREATE),
        // memdb names starting with '/' are shared by all connections of the
        // process. Unlike shared-cache `:memory:`, it keeps normal file locking,
        // so busy_timeout applies between connections.
        DatabaseMode::SharedMemory => Connection::open_with_flags_and_vfs(
            format!("/{path}"),
            read_write | F::SQLITE_OPEN_CREATE,
            "memdb",
        ),
        DatabaseMode::File => Connection::open_with_flags(path, read_write | F::SQLITE_OPEN_CREATE),
        DatabaseMode::ReadWrite => Connection::open_with_flags(path, read_write),
        DatabaseMode::ReadOnly => Connection::open_with_flags(path, base | F::SQLITE_OPEN_READ_ONLY),
    };
    opened.map_err(|error| {
        if mode.takes_path() {
            failure(format!("cannot open {path:?} as {}: {error}", mode.name()))
        } else {
            failure(format!("cannot open {} database: {error}", mode.name()))
        }
    })
}

fn initialize_metadata(connection: &mut Connection) -> Result<(), RuntimeError> {
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(failure)?;
    tx.execute_batch("CREATE TABLE IF NOT EXISTS __napalm_metadata (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL CHECK (typeof(version) = 'integer' AND version >= 0 AND version <= 9007199254740991)
    ); INSERT OR IGNORE INTO __napalm_metadata (id, version) VALUES (1, 0);").map_err(failure)?;
    get_version(&tx)?;
    tx.commit().map_err(failure)
}

fn intern(program: &mut mx::Program, name: &str) -> mx::StringId {
    if let Some(i) = program.strings.iter().position(|n| n == name) {
        return mx::StringId(i as u32);
    }
    let id = mx::StringId(program.strings.len() as u32);
    program.strings.push(name.into());
    id
}

/// Reflect only tables referenced by this block. Parameterized PRAGMA table
/// functions avoid SQL interpolation, including for externally created schemas.
fn catalog(connection: &Connection, block: &ast::Block) -> Result<mx::Program, RuntimeError> {
    let mut program = mx::Program::default();
    let mut table_names = Vec::new();
    for stmt in &block.statements {
        let table = match &stmt.value {
            ast::Statement::Schema(_) => continue,
            ast::Statement::Select { table, .. }
            | ast::Statement::Insert { table, .. }
            | ast::Statement::Update { table, .. }
            | ast::Statement::Delete { table, .. } => &table.value,
        };
        schema::user_name(table).map_err(failure)?;
        if !table_names.contains(table) {
            table_names.push(table.clone());
        }
    }
    let mut next_field = 0;
    for table_name in table_names {
        let table = mx::TableId(program.catalog.tables.len() as u32);
        let fields = connection.prepare("SELECT name, type, \"notnull\", pk, hidden FROM pragma_table_xinfo(?1, 'main') ORDER BY cid").map_err(failure)?
            .query_map([&table_name], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, bool>(2)?, r.get::<_, u32>(3)?, r.get::<_, u32>(4)?))).map_err(failure)?
            .collect::<rusqlite::Result<Vec<_>>>().map_err(failure)?;
        if fields.is_empty() {
            return Err(failure(format!("unknown or empty table {table_name:?}")));
        }
        // SQLite permits NULL in ordinary non-integer primary keys. MX's
        // identity/uniqueness rewrites require non-null keys. INTEGER PRIMARY
        // KEY is implicitly non-null only when it aliases rowid (no PK index).
        let rowid_primary = fields.iter().filter(|f| f.3 != 0).count() == 1
            && fields.iter().any(|f| f.3 == 1 && f.1.eq_ignore_ascii_case("INTEGER"))
            && !connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM pragma_index_list(?1, 'main') WHERE origin = 'pk')",
                [&table_name], |r| r.get::<_, bool>(0),
            ).map_err(failure)?;
        let mut definitions = vec![];
        let mut primary = vec![];
        for (name, declared, required, pk, hidden) in fields {
            if pk != 0 && !required && !rowid_primary {
                return Err(failure(format!(
                    "primary key field {name:?} must be NOT NULL"
                )));
            }
            if hidden != 0 {
                return Err(failure(
                    "generated/hidden columns are not supported by the MX catalog yet",
                ));
            }
            let id = mx::FieldId(next_field);
            next_field += 1;
            if pk != 0 {
                primary.push((pk, id));
            }
            definitions.push(mx::FieldDef {
                id,
                name: intern(&mut program, &name),
                ty: schema_type(&declared)?,
                nullable: !required && pk == 0,
                ordinal: u16::try_from(definitions.len()).map_err(failure)?,
            });
        }
        primary.sort_by_key(|p| p.0);
        if primary.is_empty() {
            return Err(failure(format!(
                "table {table_name:?} requires an explicit primary key"
            )));
        }
        let mut index_ids = vec![];
        let indexes = connection.prepare("SELECT name, \"unique\", origin, partial FROM pragma_index_list(?1, 'main') ORDER BY name").map_err(failure)?
            .query_map([&table_name], |r| Ok((r.get::<_, String>(0)?, r.get::<_, bool>(1)?, r.get::<_, String>(2)?, r.get::<_, bool>(3)?))).map_err(failure)?
            .collect::<rusqlite::Result<Vec<_>>>().map_err(failure)?;
        for (name, unique, origin, partial) in indexes {
            if origin == "pk" || partial {
                continue;
            }
            let keys = connection.prepare("SELECT name, desc, coll FROM pragma_index_xinfo(?1, 'main') WHERE key = 1 ORDER BY seqno").map_err(failure)?
                .query_map([&name], |r| Ok((r.get::<_, Option<String>>(0)?, r.get::<_, bool>(1)?, r.get::<_, String>(2)?))).map_err(failure)?
                .collect::<rusqlite::Result<Vec<_>>>().map_err(failure)?;
            let mut columns = vec![];
            for (field, descending, collation) in keys {
                let Some(field) = field else {
                    columns.clear();
                    break;
                };
                if !collation.eq_ignore_ascii_case("BINARY") {
                    columns.clear();
                    break;
                }
                let def = definitions
                    .iter()
                    .find(|f| program.string(f.name) == Some(field.as_str()))
                    .ok_or_else(|| failure("invalid index field"))?;
                columns.push(mx::IndexColumn {
                    field: def.id,
                    encoding: mx::KeyEncoding::for_type(def.ty)
                        .ok_or_else(|| failure("unorderable index"))?,
                    descending,
                    nulls: if descending {
                        mx::NullOrder::Last
                    } else {
                        mx::NullOrder::First
                    },
                });
            }
            if columns.is_empty() {
                continue;
            }
            let id = mx::IndexId(program.catalog.indexes.len() as u32);
            let mut prefix = vec![2];
            prefix.extend(id.0.to_be_bytes());
            let name = intern(&mut program, &name);
            program.catalog.indexes.push(mx::IndexDef {
                id,
                table,
                name,
                key_prefix: prefix,
                columns,
                included_fields: vec![],
                unique,
            });
            index_ids.push(id);
        }
        let mut prefix = vec![1];
        prefix.extend(table.0.to_be_bytes());
        let name = intern(&mut program, &table_name);
        program.catalog.tables.push(mx::TableDef {
            id: table,
            name,
            key_prefix: prefix,
            fields: definitions,
            primary_key: primary.into_iter().map(|(_, f)| f).collect(),
            indexes: index_ids,
            stats: mx::TableStats {
                rows: 1000,
                ..mx::TableStats::default()
            },
        });
    }
    Ok(program)
}

impl NapalmDatabase {
    fn execute(&self, query: &NapalmQuery) -> Result<DatabaseResult, RuntimeError> {
        let mut state = self.state.lock();
        let DatabaseState { connection, cached } = &mut *state;
        let connection = connection
            .as_mut()
            .ok_or_else(|| failure("connection is closed"))?;
        let writes = query
            .block
            .statements
            .iter()
            .any(|s| !matches!(s.value, ast::Statement::Select { .. }));
        let tx = connection
            .transaction_with_behavior(if writes {
                TransactionBehavior::Immediate
            } else {
                TransactionBehavior::Deferred
            })
            .map_err(failure)?;
        let result = if query
            .block
            .statements
            .iter()
            .any(|s| matches!(s.value, ast::Statement::Schema(_)))
        {
            // DDL changes the catalog seen by later statements in this same
            // transaction. Never publish plans compiled against uncommitted DDL.
            *cached = None;
            let mut result = DatabaseResult::Changes(0);
            for statement in &query.block.statements {
                if let ast::Statement::Schema(command) = &statement.value {
                    let sql = schema::compile(command).map_err(failure)?;
                    tx.execute(&sql, []).map_err(failure)?;
                    result = DatabaseResult::Changes(0);
                } else {
                    let single = NapalmQuery {
                        payload: String::new(),
                        block: ast::Block {
                            statements: vec![statement.clone()],
                            span: statement.span,
                        },
                        captures: query.captures.clone(),
                    };
                    result = execute_data(&tx, &mut None, &single)?;
                }
            }
            result
        } else {
            execute_data(&tx, cached, query)?
        };
        tx.commit().map_err(failure)?;
        Ok(result)
    }
}

fn execute_data(
    tx: &Connection,
    cached: &mut Option<Cached>,
    query: &NapalmQuery,
) -> Result<DatabaseResult, RuntimeError> {
    let schema: i64 = tx
        .query_row("PRAGMA main.schema_version", [], |r| r.get(0))
        .map_err(failure)?;
    let types = query.captures.iter().map(|c| c.ty).collect::<Vec<_>>();
    if !cached
        .as_ref()
        .is_some_and(|c| c.payload == query.payload && c.types == types && c.schema == schema)
    {
        let mut program = catalog(tx, &query.block)?;
        let parameters = query
            .captures
            .iter()
            .map(|c| Parameter {
                name: c.name.clone(),
                ty: c.ty,
                nullable: c.ty == mx::ScalarType::Null,
            })
            .collect::<Vec<_>>();
        napalm_mx_lowerer::lower(&query.block, &mut program, &parameters)
            .map_err(|d| failure(d.message))?;
        crate::napalm_mx_ir_optimizing::optimization_passes()
            .optimize_program(&mut program)
            .map_err(|errors| failure(errors.join("; ")))?;
        let queries = napalm_mx_sqlite::compile_program(&program).map_err(failure)?;
        *cached = Some(Cached {
            payload: query.payload.clone(),
            types,
            schema,
            program,
            queries,
        });
    }
    let cached = cached.as_ref().unwrap();
    let mut result = DatabaseResult::Changes(0);
    for (i, compiled) in cached.queries.iter().enumerate() {
        let select = matches!(
            query.block.statements[i].value,
            ast::Statement::Select { .. }
        );
        let parameters = compiled
            .parameters
            .iter()
            .map(|p| {
                let name = p
                    .name
                    .and_then(|id| cached.program.string(id))
                    .ok_or_else(|| failure("unnamed query parameter"))?;
                query
                    .captures
                    .iter()
                    .find(|c| c.name == name)
                    .map(|c| c.value.clone())
                    .ok_or_else(|| failure(format!("missing parameter ${name}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut columns = vec![];
        for (i, column) in compiled.columns.iter().enumerate() {
            let name = column
                .name
                .and_then(|id| cached.program.string(id))
                .map(str::to_owned)
                .unwrap_or_else(|| format!("column{}", i + 1));
            if columns.iter().any(|(n, _)| n == &name) {
                return Err(failure(format!(
                    "duplicate result name {name:?}; use AS to give each result a unique name"
                )));
            }
            columns.push((name, column.ty));
        }
        let mut rows = vec![];
        let mut changes = 0;
        for statement in &compiled.statements {
            let values = statement.bind_values(&parameters).map_err(failure)?;
            let mut prepared = tx.prepare(&statement.sql).map_err(failure)?;
            if prepared.readonly() {
                let mut cursor = prepared
                    .query(params_from_iter(values.iter()))
                    .map_err(failure)?;
                while let Some(row) = cursor.next().map_err(failure)? {
                    if select {
                        let cells = (0..columns.len())
                            .map(|i| row.get::<_, SqlValue>(i))
                            .collect::<rusqlite::Result<Vec<_>>>()
                            .map_err(failure)?;
                        // Validate before commit, so conversion failures also roll back writes.
                        for ((_, ty), cell) in columns.iter().zip(&cells) {
                            validate_cell(*ty, cell)?;
                        }
                        rows.push(cells);
                    }
                }
            } else {
                changes += prepared
                    .execute(params_from_iter(values.iter()))
                    .map_err(failure)? as u64;
            }
        }
        result = if select {
            DatabaseResult::Rows(columns, rows)
        } else {
            DatabaseResult::Changes(changes)
        };
    }
    Ok(result)
}

fn validate_cell(ty: mx::ScalarType, cell: &SqlValue) -> Result<(), RuntimeError> {
    use mx::ScalarType as T;
    if let SqlValue::Integer(v) = cell {
        if ty == T::Number && (*v as f64) as i128 != *v as i128 {
            return Err(failure(
                "SQLite integer is not exactly representable as a Napalm Number",
            ));
        }
    }
    let valid = matches!(cell, SqlValue::Null)
        || ty == T::Any
        || match ty {
            T::Number => matches!(cell, SqlValue::Integer(_) | SqlValue::Real(_)),
            T::Bool => matches!(cell, SqlValue::Integer(0 | 1)),
            T::String => matches!(cell, SqlValue::Text(_)),
            T::Bytes => matches!(cell, SqlValue::Blob(_)),
            T::Null => false,
            T::Any => true,
        };
    if valid {
        Ok(())
    } else {
        Err(failure(format!("SQLite result does not match {ty:?}")))
    }
}

impl Interpreter<'_> {
    fn mx_text(&self, value: Value) -> Result<String, RuntimeError> {
        match self.heap.get(value) {
            Some(HeapObject::String(v)) => std::str::from_utf8(v.as_ref())
                .map(str::to_owned)
                .map_err(failure),
            _ => Err(failure("expected a String")),
        }
    }
    fn mx_capture_value(&self, value: Value) -> Result<(mx::ScalarType, SqlValue), RuntimeError> {
        use mx::ScalarType as T;
        if value.is_null() {
            return Ok((T::Null, SqlValue::Null));
        }
        if let Some(v) = value.as_bool() {
            return Ok((T::Bool, SqlValue::Integer(i64::from(v))));
        }
        if let Some(v) = value.as_num() {
            if v.is_nan() {
                return Err(failure("NaN cannot be stored in SQLite"));
            }
            return Ok((T::Number, SqlValue::Real(v)));
        }
        match self.heap.get(value) {
            Some(HeapObject::String(_)) => Ok((T::String, SqlValue::Text(self.mx_text(value)?))),
            Some(HeapObject::Bytes(v)) => Ok((T::Bytes, SqlValue::Blob(v.as_slice().to_vec()))),
            Some(HeapObject::ByteBuffer(v)) => {
                Ok((T::Bytes, SqlValue::Blob(v.as_slice().to_vec())))
            }
            _ => Err(failure(
                "query parameters must be Null, Bool, Number, String, Bytes or ByteBuffer",
            )),
        }
    }
    fn mx_database(&self, value: Value) -> Result<Arc<NapalmDatabase>, RuntimeError> {
        match self.heap.get(value) {
            Some(HeapObject::Database(db)) => Ok(db.clone()),
            _ => Err(failure("expected a Database connection")),
        }
    }
    fn mx_operation(
        &mut self,
        native: MxNative,
        receiver: Value,
        args: &[RegisterWord],
    ) -> Result<Value, RuntimeError> {
        let arg = |i: usize| {
            args.get(i)
                .copied()
                .map(Value::from_word)
                .unwrap_or(Value::NULL)
        };
        match native {
            MxNative::Mode(mode) => Ok(Value::from_num(mode as u8 as f64)),
            MxNative::Open => {
                let mode = arg(0).as_num().and_then(DatabaseMode::from_value).ok_or_else(|| {
                    failure("Database.open expects a DatabaseMode member, e.g. Database.open(DatabaseMode.Memory)")
                })?;
                let path = match (mode.takes_path(), args.len() > 1) {
                    (true, true) => self.mx_text(arg(1))?,
                    (false, false) => String::new(),
                    (true, false) => {
                        return Err(failure(format!(
                            "DatabaseMode.{} requires a path: Database.open(DatabaseMode.{0}, path)",
                            mode.name()
                        )));
                    }
                    (false, true) => {
                        return Err(failure(format!(
                            "DatabaseMode.{} does not take a path",
                            mode.name()
                        )));
                    }
                };
                if mode.takes_path() && (path.is_empty() || path.contains('\0')) {
                    return Err(failure("database path must not be empty or contain NUL"));
                }
                let mut connection = open_connection(mode, &path)?;
                // Quoted identifiers must never fall back to string literals
                // (notably for an unknown CREATE INDEX column).
                for option in [
                    rusqlite::config::DbConfig::SQLITE_DBCONFIG_DQS_DDL,
                    rusqlite::config::DbConfig::SQLITE_DBCONFIG_DQS_DML,
                ] {
                    connection.set_db_config(option, false).map_err(failure)?;
                }
                napalm_mx_sqlite::install_functions(&connection).map_err(failure)?;
                connection
                    .pragma_update(None, "foreign_keys", true)
                    .map_err(failure)?;
                connection
                    .busy_timeout(std::time::Duration::from_secs(5))
                    .map_err(failure)?;
                if mode == DatabaseMode::ReadOnly {
                    // Validates existing metadata without writing.
                    get_version(&connection)?;
                } else {
                    initialize_metadata(&mut connection)?;
                }
                Ok(self
                    .heap
                    .alloc(HeapObject::Database(Arc::new(NapalmDatabase {
                        mode,
                        path,
                        state: parking_lot::Mutex::new(DatabaseState {
                            connection: Some(connection),
                            cached: None,
                        }),
                    }))))
            }
            MxNative::Capture => {
                let payload = self.mx_text(arg(0))?;
                let block = ast::parse(&payload, 0).map_err(|d| failure(d.message))?;
                if block.statements.is_empty() {
                    return Err(failure("empty MX block"));
                }
                let Some(HeapObject::List(values)) = self.heap.get(arg(1)) else {
                    return Err(failure("invalid MX capture list"));
                };
                let names = ast::parameters(&block);
                if names.len() != values.len() {
                    return Err(failure("MX capture count mismatch"));
                }
                let captures = names
                    .into_iter()
                    .zip(values)
                    .map(|(name, value)| {
                        let (ty, value) = self.mx_capture_value(*value)?;
                        Ok(Capture {
                            name: name.value.clone(),
                            ty,
                            value,
                        })
                    })
                    .collect::<Result<Vec<_>, RuntimeError>>()?;
                Ok(self
                    .heap
                    .alloc(HeapObject::DatabaseQuery(Arc::new(NapalmQuery {
                        payload,
                        block,
                        captures,
                    }))))
            }
            MxNative::Closed => Ok(Value::from_bool(
                self.mx_database(receiver)?
                    .state
                    .lock()
                    .connection
                    .is_none(),
            )),
            MxNative::Close => {
                let db = self.mx_database(receiver)?;
                let mut state = db.state.lock();
                if let Some(connection) = state.connection.take() {
                    if let Err((connection, error)) = connection.close() {
                        state.connection = Some(connection);
                        return Err(failure(error));
                    }
                }
                state.cached = None;
                Ok(Value::NULL)
            }
            MxNative::CreateTable | MxNative::CreateIndex => {
                Err(failure("use CREATE TABLE / CREATE INDEX in an MX block"))
            }
            MxNative::GetVersion | MxNative::SetVersion => {
                let db = self.mx_database(receiver)?;
                let state = db.state.lock();
                let connection = state
                    .connection
                    .as_ref()
                    .ok_or_else(|| failure("connection is closed"))?;
                if native == MxNative::GetVersion {
                    return Ok(Value::from_num(get_version(connection)? as f64));
                }
                let version = arg(0)
                    .as_num()
                    .ok_or_else(|| failure("version must be a non-negative integer"))?;
                if !version.is_finite()
                    || version < 0.0
                    || version.fract() != 0.0
                    || version > schema::MAX_VERSION as f64
                {
                    return Err(failure(
                        "version must be an exactly representable non-negative integer",
                    ));
                }
                let changed = connection
                    .execute(
                        "UPDATE __napalm_metadata SET version = ?1 WHERE id = 1",
                        [version as i64],
                    )
                    .map_err(failure)?;
                if changed != 1 {
                    return Err(failure("missing database version metadata"));
                }
                Ok(Value::NULL)
            }
            MxNative::Execute => {
                let db = self.mx_database(receiver)?;
                let query = match self.heap.get(arg(0)) {
                    Some(HeapObject::DatabaseQuery(q)) => q.clone(),
                    _ => return Err(failure("execute expects a [[[ ... ]]] query")),
                };
                let result = db.execute(&query);
                self.mx_result(result?)
            }
        }
    }
    fn mx_result(&mut self, result: DatabaseResult) -> Result<Value, RuntimeError> {
        let DatabaseResult::Rows(columns, rows) = result else {
            let DatabaseResult::Changes(n) = result else {
                unreachable!()
            };
            return Ok(Value::from_num(n as f64));
        };
        let list = self
            .heap
            .alloc(HeapObject::List(Vec::with_capacity(rows.len())));
        let list_root = self.add_root(list);
        let populated = (|| {
            for cells in rows {
                let map = self.heap.alloc(HeapObject::Map(NapalmMap::default()));
                let root = self.add_root(map);
                let added = (|| {
                    for ((name, ty), cell) in columns.iter().zip(cells) {
                        let value = match cell {
                            SqlValue::Null => Value::NULL,
                            SqlValue::Integer(v) if *ty == mx::ScalarType::Bool => {
                                Value::from_bool(v != 0)
                            }
                            SqlValue::Integer(v) => Value::from_num(v as f64),
                            SqlValue::Real(v) => Value::from_num(v),
                            SqlValue::Text(v) => self.heap.alloc(HeapObject::String(
                                NapalmString::new(StrBytes::from(v.into_bytes())),
                            )),
                            SqlValue::Blob(v) => self.heap.alloc(HeapObject::Bytes(
                                NapalmBytes::from_inner(bytes::Bytes::from(v)),
                            )),
                        };
                        self.heap.remember_if_reference(map, value);
                        self.heap.map_set_encoded(
                            map,
                            MapKey::String(MapStringKey::new(Arc::<[u8]>::from(name.as_bytes()))),
                            value,
                            None,
                        )?;
                    }
                    Ok::<_, RuntimeError>(())
                })();
                self.release_root(root);
                added?;
                let external = self.heap.get(map).map_or(0, external_bytes_for);
                self.heap.set_external_bytes(map, external);
                self.heap.refresh_jit_view(map);
                self.heap.remember_if_reference(list, map);
                if let Some(HeapObject::List(values)) = self.heap.get_mut(list) {
                    values.push(map);
                }
            }
            Ok::<_, RuntimeError>(())
        })();
        self.release_root(list_root);
        populated?;
        let external = self.heap.get(list).map_or(0, external_bytes_for);
        self.heap.set_external_bytes(list, external);
        self.heap.refresh_jit_view(list);
        Ok(list)
    }
    pub(super) fn execute_mx_native(
        &mut self,
        op: &PreparedOp,
        native: MxNative,
        receiver: RegisterWord,
        args: &[RegisterWord],
    ) {
        self.mark_gc_point(op);
        self.maybe_collect();
        let receiver = Value::from_word(receiver);
        let result = self.mx_operation(native, receiver, args);
        if let Some(HeapObject::Database(db)) = self.heap.get(receiver) {
            let bytes = db.external_bytes();
            self.heap.set_external_bytes(receiver, bytes);
        }
        match result {
            Ok(value) => unsafe {
                self.set_reg(op.a, value.word());
                self.advance(op);
            },
            Err(error) => self.fail_at(op, error.message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(source: &str) -> NapalmQuery {
        NapalmQuery {
            payload: source.to_owned(),
            block: ast::parse(source, 0).unwrap(),
            captures: vec![],
        }
    }

    #[test]
    fn napalm_mx_vm_schema_version_invalidates_cached_columns() {
        let connection = Connection::open_in_memory().unwrap();
        napalm_mx_sqlite::install_functions(&connection).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE items (id REAL NOT NULL PRIMARY KEY); INSERT INTO items VALUES (1)",
            )
            .unwrap();
        let db = NapalmDatabase {
            mode: DatabaseMode::Memory,
            path: String::new(),
            state: parking_lot::Mutex::new(DatabaseState {
                connection: Some(connection),
                cached: None,
            }),
        };
        let template = query("SELECT * FROM items");
        let DatabaseResult::Rows(columns, _) = db.execute(&template).unwrap() else {
            panic!()
        };
        assert_eq!(columns.len(), 1);
        // Bypass the public schema helpers, which explicitly clear the cache.
        db.state
            .lock()
            .connection
            .as_ref()
            .unwrap()
            .execute_batch("ALTER TABLE items ADD COLUMN name TEXT; UPDATE items SET name = 'new'")
            .unwrap();
        let DatabaseResult::Rows(columns, rows) = db.execute(&template).unwrap() else {
            panic!()
        };
        assert_eq!(columns[1].0, "name");
        assert_eq!(rows[0][1], SqlValue::Text("new".into()));
    }

    #[test]
    fn napalm_mx_vm_invalid_external_values_roll_back_before_conversion() {
        let connection = Connection::open_in_memory().unwrap();
        napalm_mx_sqlite::install_functions(&connection).unwrap();
        connection.execute_batch("CREATE TABLE items (id REAL NOT NULL PRIMARY KEY, value BOOLEAN NOT NULL); INSERT INTO items VALUES (1, 7)").unwrap();
        let db = NapalmDatabase {
            mode: DatabaseMode::Memory,
            path: String::new(),
            state: parking_lot::Mutex::new(DatabaseState {
                connection: Some(connection),
                cached: None,
            }),
        };
        let error = db
            .execute(&query(
                "INSERT INTO items (id, value) VALUES (2, true); SELECT * FROM items",
            ))
            .err()
            .unwrap();
        assert!(
            error.message.contains("does not match Bool"),
            "{}",
            error.message
        );
        let state = db.state.lock();
        let count: u32 = state
            .connection
            .as_ref()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM items", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn napalm_mx_vm_catalog_requires_nonnull_primary_keys() {
        let connection = Connection::open_in_memory().unwrap();
        connection.execute_batch("CREATE TABLE nullable_key (id TEXT PRIMARY KEY); CREATE TABLE rowid_key (id INTEGER PRIMARY KEY); CREATE TABLE descending_key (id INTEGER PRIMARY KEY DESC)").unwrap();
        assert!(
            catalog(&connection, &query("SELECT * FROM nullable_key").block)
                .unwrap_err()
                .message
                .contains("NOT NULL")
        );
        assert!(catalog(&connection, &query("SELECT * FROM descending_key").block).is_err());
        assert!(catalog(&connection, &query("SELECT * FROM rowid_key").block).is_ok());
    }
}
