//! MX logical/physical IR -> parameterized SQLite SQL.
//!
//! Physical paths retain their predicates, ordering and windows; delivery and
//! join-algorithm hints remain SQLite's responsibility. No IR is rewritten.
//! Install `install_functions` on each connection before preparing the SQL.
use crate::napalm_collections::HashMap;
use crate::napalm_mx_ir::*;
use rusqlite::{Connection, functions::FunctionFlags, types::Value};
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Binding {
    Parameter(ParamId),
    Constant(Value),
}

#[derive(Clone, Debug)]
pub struct SqlStatement {
    pub sql: String,
    /// Position i binds SQLite parameter ?(i + 1). Never interpolate these values.
    pub bindings: Vec<Binding>,
}
impl SqlStatement {
    /// Resolve query-local ParamIds into a positional rusqlite binding vector.
    pub fn bind_values(&self, parameters: &[Value]) -> Result<Vec<Value>> {
        self.bindings
            .iter()
            .map(|binding| {
                let value = match binding {
                    Binding::Constant(value) => value,
                    Binding::Parameter(id) => parameters
                        .get(id.0 as usize)
                        .ok_or_else(|| err(format!("missing parameter {id:?}")))?,
                };
                if matches!(value, Value::Real(v) if v.is_nan()) {
                    return Err(err("SQLite cannot represent NaN"));
                }
                Ok(value.clone())
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct SqlQuery {
    pub id: QueryId,
    /// Query-local parameter contracts, retained for the future VM adapter.
    pub parameters: Vec<ParamDef>,
    /// Execute in order, in ONE transaction for a writable batch.
    pub statements: Vec<SqlStatement>,
    /// Logical result schema; internal key/order columns never escape.
    pub columns: Vec<ColumnDef>,
    pub transaction: TransactionMode,
    /// SQLite cannot return a zero-column row: discard its one dummy cell.
    pub discard_dummy_column: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileError(pub String);
impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for CompileError {}
type Result<T> = std::result::Result<T, CompileError>;
fn err(message: impl Into<String>) -> CompileError {
    CompileError(message.into())
}

/// Compile every query, accepting either logical or already optimized IR.
pub fn compile_program(program: &Program) -> Result<Vec<SqlQuery>> {
    verify_program(program).map_err(|e| err(format!("invalid MX IR: {e:?}")))?;
    program
        .queries
        .iter()
        .map(|q| Compiler::new(program, q).compile())
        .collect()
}

pub fn compile_query(program: &Program, id: QueryId) -> Result<SqlQuery> {
    let query = program
        .query(id)
        .ok_or_else(|| err(format!("unknown query {id:?}")))?;
    verify_query(query, &program.catalog).map_err(|e| err(format!("invalid MX IR: {e:?}")))?;
    Compiler::new(program, query).compile()
}

/// Quote catalog identifiers; values always travel through bindings instead.
fn quote(s: &str) -> Result<String> {
    if s.contains('\0') {
        return Err(err("SQLite identifiers cannot contain NUL"));
    }
    Ok(format!("\"{}\"", s.replace('"', "\"\"")))
}
fn column(c: ColumnId) -> String {
    format!("c{}", c.0)
}
fn reference(alias: &str, name: &str) -> String {
    format!("{alias}.\"{name}\"")
}

#[derive(Clone)]
struct Identity {
    table: TableId,
    names: Vec<String>,
}
#[derive(Clone)]
struct Order {
    name: String,
    direction: SortDirection,
    nulls: NullOrder,
}
#[derive(Clone, Default)]
struct Relation {
    name: String,
    columns: Vec<ColumnId>,
    identities: Vec<Identity>,
    extra: Vec<String>,
    order: Vec<Order>,
}
impl Relation {
    fn scope(&self, alias: &str) -> HashMap<ColumnId, String> {
        self.columns
            .iter()
            .map(|c| (*c, reference(alias, &column(*c))))
            .collect()
    }
    fn order_sql(&self, alias: &str) -> String {
        if self.order.is_empty() {
            return String::new();
        }
        format!(
            " ORDER BY {}",
            self.order
                .iter()
                .map(|o| format!(
                    "{} {} NULLS {}",
                    reference(alias, &o.name),
                    if o.direction == SortDirection::Asc {
                        "ASC"
                    } else {
                        "DESC"
                    },
                    if o.nulls == NullOrder::First {
                        "FIRST"
                    } else {
                        "LAST"
                    }
                ))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
    fn identity(&self, table: TableId) -> Result<&Identity> {
        let mut found = self.identities.iter().filter(|i| i.table == table);
        let first = found
            .next()
            .ok_or_else(|| err(format!("plan lost row identity for {table:?}")))?;
        if found.next().is_some() {
            return Err(err("ambiguous row identity in self-join mutation/fetch"));
        }
        Ok(first)
    }
}

struct Compiler<'a> {
    program: &'a Program,
    query: &'a Query,
    ctes: Vec<String>,
    relations: HashMap<PlanId, Relation>,
    bindings: Vec<Binding>,
    params: HashMap<ParamId, usize>,
    constants: HashMap<ExprId, usize>,
}
impl<'a> Compiler<'a> {
    fn new(program: &'a Program, query: &'a Query) -> Self {
        Self {
            program,
            query,
            ctes: vec![],
            relations: HashMap::default(),
            bindings: vec![],
            params: HashMap::default(),
            constants: HashMap::default(),
        }
    }
    fn text(&self, id: StringId) -> Result<&str> {
        self.program
            .string(id)
            .ok_or_else(|| err(format!("invalid string {id:?}")))
    }
    fn table(&self, id: TableId) -> Result<&TableDef> {
        self.program
            .catalog
            .table(id)
            .ok_or_else(|| err(format!("invalid table {id:?}")))
    }
    fn table_sql(&self, id: TableId) -> Result<String> {
        // Qualifying the schema also prevents CTE names shadowing physical tables.
        Ok(format!(
            "\"main\".{}",
            quote(self.text(self.table(id)?.name)?)?
        ))
    }
    fn field_sql(&self, table: TableId, field: FieldId, alias: &str) -> Result<String> {
        let def = self
            .table(table)?
            .field(field)
            .ok_or_else(|| err(format!("invalid field {field:?}")))?;
        let name = quote(self.text(def.name)?)?;
        Ok(if alias.is_empty() {
            name
        } else {
            format!("{alias}.{name}")
        })
    }
    fn finish(&mut self, tail: String) -> SqlStatement {
        let sql = if self.ctes.is_empty() {
            tail
        } else {
            format!("WITH {} {tail}", self.ctes.join(",\n"))
        };
        let bindings = std::mem::take(&mut self.bindings);
        self.params.clear();
        self.constants.clear();
        self.ctes.clear();
        self.relations.clear();
        SqlStatement { sql, bindings }
    }
    fn compile(mut self) -> Result<SqlQuery> {
        let root = self.query.plan(self.query.root).unwrap().kind.clone();
        let mut columns = vec![];
        let mut discard_dummy_column = false;
        let statements = match root {
            PlanKind::Insert { table, rows } => self.batch(table, MutationMode::Insert, &rows)?,
            PlanKind::WriteBatch { table, mode, rows } => self.batch(table, mode, &rows)?,
            PlanKind::Update {
                input,
                table,
                assignments,
            } => {
                let sql = self.mutation(input, table, Some(&assignments))?;
                vec![self.finish(sql)]
            }
            PlanKind::Delete { input, table } => {
                let sql = self.mutation(input, table, None)?;
                vec![self.finish(sql)]
            }
            _ => {
                let rel = self.relation(self.query.root, 0)?;
                for c in &rel.columns {
                    columns.push(
                        self.query
                            .column(*c)
                            .ok_or_else(|| err("invalid result column"))?
                            .clone(),
                    );
                }
                discard_dummy_column = columns.is_empty();
                let outputs = rel
                    .columns
                    .iter()
                    .map(|c| reference("r", &column(*c)))
                    .collect::<Vec<_>>();
                let select = if outputs.is_empty() {
                    "1 AS \"_unit\"".into()
                } else {
                    outputs.join(", ")
                };
                vec![self.finish(format!(
                    "SELECT {select} FROM {} AS r{}",
                    rel.name,
                    rel.order_sql("r")
                ))]
            }
        };
        Ok(SqlQuery {
            id: self.query.id,
            parameters: self.query.params.clone(),
            statements,
            columns,
            transaction: self.query.transaction,
            discard_dummy_column,
        })
    }
    fn add_relation(
        &mut self,
        id: PlanId,
        sql: String,
        mut rel: Relation,
        buffered: bool,
    ) -> Relation {
        rel.name = format!("\"mx_p{}\"", id.0);
        self.ctes.push(format!(
            "{} AS {}({sql})",
            rel.name,
            if buffered { "MATERIALIZED " } else { "" }
        ));
        self.relations.insert(id, rel.clone());
        rel
    }
    fn expression(
        &mut self,
        id: ExprId,
        scope: &HashMap<ColumnId, String>,
        depth: usize,
    ) -> Result<String> {
        if depth > 256 {
            return Err(err("MX expression exceeds SQLite compilation depth limit"));
        }
        let expr = self
            .query
            .expr(id)
            .ok_or_else(|| err(format!("invalid expression {id:?}")))?;
        let sql = match expr.kind.clone() {
            ExprKind::Const(value) => {
                let slot = if let Some(slot) = self.constants.get(&id) {
                    *slot
                } else {
                    let value = match value {
                        ScalarValue::Null => Value::Null,
                        ScalarValue::Bool(v) => Value::Integer(i64::from(v)),
                        ScalarValue::Number(v) => {
                            let v = v.to_f64();
                            if v.is_nan() {
                                return Err(err("SQLite cannot represent NaN"));
                            }
                            Value::Real(v)
                        }
                        ScalarValue::String(v) => Value::Text(self.text(v)?.into()),
                        ScalarValue::Bytes(v) => Value::Blob(
                            self.program
                                .bytes(v)
                                .ok_or_else(|| err("invalid byte constant"))?
                                .to_vec(),
                        ),
                    };
                    self.bindings.push(Binding::Constant(value));
                    let slot = self.bindings.len();
                    self.constants.insert(id, slot);
                    slot
                };
                format!("?{slot}")
            }
            ExprKind::Param(param) => {
                let slot = if let Some(slot) = self.params.get(&param) {
                    *slot
                } else {
                    self.bindings.push(Binding::Parameter(param));
                    let slot = self.bindings.len();
                    self.params.insert(param, slot);
                    slot
                };
                format!("?{slot}")
            }
            ExprKind::Column(c) => scope
                .get(&c)
                .cloned()
                .ok_or_else(|| err(format!("column {c:?} is not available in this plan")))?,
            ExprKind::Unary { op, value } => {
                let v = self.expression(value, scope, depth + 1)?;
                format!("({} {v})", if op == UnaryOp::Not { "NOT" } else { "-" })
            }
            ExprKind::Binary { op, left, right } => {
                let l = self.expression(left, scope, depth + 1)?;
                let r = self.expression(right, scope, depth + 1)?;
                let arithmetic = match op {
                    BinaryOp::Add => Some("add"),
                    BinaryOp::Sub => Some("sub"),
                    BinaryOp::Mul => Some("mul"),
                    BinaryOp::Div => Some("div"),
                    BinaryOp::Rem => Some("rem"),
                    _ => None,
                };
                if let Some(f) = arithmetic {
                    format!("napalm_mx_{f}({l}, {r})")
                } else {
                    let op = match op {
                        BinaryOp::Eq => "=",
                        BinaryOp::Ne => "!=",
                        BinaryOp::Lt => "<",
                        BinaryOp::Le => "<=",
                        BinaryOp::Gt => ">",
                        BinaryOp::Ge => ">=",
                        BinaryOp::And => "AND",
                        BinaryOp::Or => "OR",
                        _ => unreachable!(),
                    };
                    format!("({l} {op} {r})")
                }
            }
            ExprKind::IsNull { value, negated } => {
                let v = self.expression(value, scope, depth + 1)?;
                format!("({v} IS {}NULL)", if negated { "NOT " } else { "" })
            }
            ExprKind::Between {
                value,
                low,
                high,
                negated,
            } => {
                let v = self.expression(value, scope, depth + 1)?;
                let l = self.expression(low, scope, depth + 1)?;
                let h = self.expression(high, scope, depth + 1)?;
                format!(
                    "({v} {}BETWEEN {l} AND {h})",
                    if negated { "NOT " } else { "" }
                )
            }
            ExprKind::InList {
                value,
                items,
                negated,
            } => {
                let v = self.expression(value, scope, depth + 1)?;
                let items = items
                    .into_iter()
                    .map(|i| self.expression(i, scope, depth + 1))
                    .collect::<Result<Vec<_>>>()?;
                format!(
                    "({v} {}IN ({}))",
                    if negated { "NOT " } else { "" },
                    items.join(", ")
                )
            }
            ExprKind::Call { function, args } => {
                if args.len() != 1 {
                    return Err(err("MX scalar functions require exactly one argument"));
                }
                let v = self.expression(args[0], scope, depth + 1)?;
                let name = match function {
                    BuiltinFn::Abs => "abs",
                    BuiltinFn::Floor => "floor",
                    BuiltinFn::Ceil => "ceil",
                    BuiltinFn::Round => "round",
                    BuiltinFn::LowerAscii => "lower_ascii",
                    BuiltinFn::UpperAscii => "upper_ascii",
                    BuiltinFn::StringLength => "string_length",
                    BuiltinFn::BytesLength => "bytes_length",
                };
                format!("napalm_mx_{name}({v})")
            }
        };
        Ok(sql)
    }
    fn expr(&mut self, id: ExprId, scope: &HashMap<ColumnId, String>) -> Result<String> {
        self.expression(id, scope, 0)
    }
    fn window(skip: u64, limit: Option<u64>) -> Result<String> {
        if skip > i64::MAX as u64 || limit.is_some_and(|v| v > i64::MAX as u64) {
            return Err(err("SQLite LIMIT/OFFSET must fit a signed 64-bit integer"));
        }
        if skip == 0 && limit.is_none() {
            return Ok(String::new());
        }
        Ok(format!(
            " LIMIT {} OFFSET {skip}",
            limit.map(|v| v.to_string()).unwrap_or_else(|| "-1".into())
        ))
    }
    fn key_fields(
        &self,
        table: TableId,
        index: Option<IndexId>,
    ) -> Result<Vec<(FieldId, SortDirection, NullOrder)>> {
        if let Some(index) = index {
            let index = self
                .program
                .catalog
                .index(index)
                .ok_or_else(|| err("invalid index"))?;
            if index.table != table {
                return Err(err("index/table mismatch"));
            }
            Ok(index
                .columns
                .iter()
                .map(|c| {
                    (
                        c.field,
                        if c.descending {
                            SortDirection::Desc
                        } else {
                            SortDirection::Asc
                        },
                        c.nulls,
                    )
                })
                .collect())
        } else {
            Ok(self
                .table(table)?
                .primary_key
                .iter()
                .map(|f| (*f, SortDirection::Asc, NullOrder::First))
                .collect())
        }
    }
    fn range_predicate(
        &mut self,
        table: TableId,
        fields: &[(FieldId, SortDirection, NullOrder)],
        range: &KeyRange,
    ) -> Result<Vec<String>> {
        let empty = HashMap::default();
        let mut out = vec![];
        for (i, expr) in range.prefix.iter().enumerate() {
            let f = fields
                .get(i)
                .ok_or_else(|| err("key prefix exceeds index width"))?
                .0;
            out.push(format!(
                "{} = {}",
                self.field_sql(table, f, "t")?,
                self.expr(*expr, &empty)?
            ));
        }
        for (bound, lower) in [(&range.lower, true), (&range.upper, false)] {
            let (value, inclusive) = match bound {
                BoundValue::Unbounded => continue,
                BoundValue::Inclusive(v) => (*v, true),
                BoundValue::Exclusive(v) => (*v, false),
            };
            let f = fields
                .get(range.prefix.len())
                .ok_or_else(|| err("range bound exceeds index width"))?
                .0;
            let op = match (lower, inclusive) {
                (true, true) => ">=",
                (true, false) => ">",
                (false, true) => "<=",
                (false, false) => "<",
            };
            out.push(format!(
                "{} {op} {}",
                self.field_sql(table, f, "t")?,
                self.expr(value, &empty)?
            ));
        }
        Ok(out)
    }
    fn scan(
        &mut self,
        id: PlanId,
        table: TableId,
        columns: Vec<ColumnId>,
        fields: Vec<(FieldId, SortDirection, NullOrder)>,
        reverse: bool,
        mut predicates: Vec<String>,
        residual: Option<ExprId>,
        skip: u64,
        limit: Option<u64>,
    ) -> Result<Relation> {
        let mut scope = HashMap::default();
        // Residual predicates can refer to fields pruned from scan outputs.
        for c in &self.query.columns {
            if let ColumnOrigin::Storage { table: t, field } = c.origin {
                if t == table {
                    scope.insert(c.id, self.field_sql(table, field, "t")?);
                }
            }
        }
        let mut selected = vec![];
        for c in &columns {
            selected.push(format!(
                "{} AS \"{}\"",
                scope.get(c).ok_or_else(|| err("invalid scan column"))?,
                column(*c)
            ));
        }
        let mut rel = Relation {
            columns,
            ..Relation::default()
        };
        let pk = self.table(table)?.primary_key.clone();
        let mut identity = Identity {
            table,
            names: vec![],
        };
        for (i, f) in pk.iter().enumerate() {
            let name = format!("k{}_{}", id.0, i);
            selected.push(format!("{} AS \"{name}\"", self.field_sql(table, *f, "t")?));
            identity.names.push(name.clone());
            rel.extra.push(name);
        }
        rel.identities.push(identity);
        for (i, (field, mut direction, mut nulls)) in fields.into_iter().enumerate() {
            if reverse {
                direction = if direction == SortDirection::Asc {
                    SortDirection::Desc
                } else {
                    SortDirection::Asc
                };
                nulls = if nulls == NullOrder::First {
                    NullOrder::Last
                } else {
                    NullOrder::First
                };
            }
            let name = format!("o{}_{}", id.0, i);
            selected.push(format!(
                "{} AS \"{name}\"",
                self.field_sql(table, field, "t")?
            ));
            rel.extra.push(name.clone());
            rel.order.push(Order {
                name,
                direction,
                nulls,
            });
        }
        if let Some(p) = residual {
            predicates.push(self.expr(p, &scope)?);
        }
        let where_sql = if predicates.is_empty() {
            String::new()
        } else {
            format!(
                " WHERE {}",
                predicates
                    .iter()
                    .map(|p| format!("({p})"))
                    .collect::<Vec<_>>()
                    .join(" AND ")
            )
        };
        if selected.is_empty() {
            selected.push("1 AS \"_unit\"".into());
        }
        let inner = format!(
            "SELECT {} FROM {} AS t{where_sql}",
            selected.join(", "),
            self.table_sql(table)?
        );
        let sql = format!(
            "SELECT s.* FROM ({inner}) AS s{}{}",
            rel.order_sql("s"),
            Self::window(skip, limit)?
        );
        Ok(self.add_relation(id, sql, rel, false))
    }
    fn relation(&mut self, id: PlanId, depth: usize) -> Result<Relation> {
        if depth > 128 {
            return Err(err("MX plan exceeds SQLite compilation depth limit"));
        }
        if let Some(rel) = self.relations.get(&id) {
            return Ok(rel.clone());
        }
        let kind = self
            .query
            .plan(id)
            .ok_or_else(|| err("invalid plan"))?
            .kind
            .clone();
        let (sql, rel, buffered) = match kind {
            PlanKind::TableScan { table, columns } => {
                let fields = self.key_fields(table, None)?;
                return self.scan(id, table, columns, fields, false, vec![], None, 0, None);
            }
            PlanKind::PointGet {
                table,
                key,
                columns,
                residual,
                ..
            } => {
                let fields = self.key_fields(table, None)?;
                if key.len() != fields.len() {
                    return Err(err("point lookup must bind the complete primary key"));
                }
                let predicates = self.range_predicate(
                    table,
                    &fields,
                    &KeyRange {
                        prefix: key,
                        ..KeyRange::default()
                    },
                )?;
                return self.scan(
                    id,
                    table,
                    columns,
                    vec![],
                    false,
                    predicates,
                    residual,
                    0,
                    None,
                );
            }
            PlanKind::PrimaryRangeScan {
                table,
                range,
                columns,
                residual,
                direction,
                skip,
                limit,
                ..
            } => {
                let fields = self.key_fields(table, None)?;
                let predicates = self.range_predicate(table, &fields, &range)?;
                return self.scan(
                    id,
                    table,
                    columns,
                    fields,
                    direction == ScanDirection::Reverse,
                    predicates,
                    residual,
                    skip,
                    limit,
                );
            }
            PlanKind::IndexRangeScan {
                table,
                index,
                range,
                columns,
                residual,
                direction,
                skip,
                limit,
                ..
            } => {
                let mut fields = self.key_fields(table, Some(index))?;
                let predicates = self.range_predicate(table, &fields, &range)?;
                // Non-unique index entries append the primary key as a tie-breaker.
                for f in self.table(table)?.primary_key.clone() {
                    if !fields.iter().any(|(k, _, _)| *k == f) {
                        fields.push((f, SortDirection::Asc, NullOrder::First));
                    }
                }
                return self.scan(
                    id,
                    table,
                    columns,
                    fields,
                    direction == ScanDirection::Reverse,
                    predicates,
                    residual,
                    skip,
                    limit,
                );
            }
            PlanKind::Empty { columns } => {
                let select = if columns.is_empty() {
                    "1 AS \"_unit\"".into()
                } else {
                    columns
                        .iter()
                        .map(|c| format!("NULL AS \"{}\"", column(*c)))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                (
                    format!("SELECT {select} WHERE 0"),
                    Relation {
                        columns,
                        ..Relation::default()
                    },
                    false,
                )
            }
            PlanKind::Values { columns, rows } => {
                let mut selects = vec![];
                for row in rows {
                    let mut values = vec![];
                    for (c, e) in columns.iter().zip(row) {
                        values.push(format!(
                            "{} AS \"{}\"",
                            self.expr(e, &HashMap::default())?,
                            column(*c)
                        ));
                    }
                    if values.is_empty() {
                        values.push("1 AS \"_unit\"".into());
                    }
                    selects.push(format!("SELECT {}", values.join(", ")));
                }
                if selects.is_empty() {
                    let names = if columns.is_empty() {
                        "1 AS \"_unit\"".into()
                    } else {
                        columns
                            .iter()
                            .map(|c| format!("NULL AS \"{}\"", column(*c)))
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    selects.push(format!("SELECT {names} WHERE 0"));
                }
                (
                    selects.join(" UNION ALL "),
                    Relation {
                        columns,
                        ..Relation::default()
                    },
                    false,
                )
            }
            PlanKind::Filter { input, predicate } => {
                let r = self.relation(input, depth + 1)?;
                let predicate = self.expr(predicate, &r.scope("a"))?;
                (
                    format!("SELECT a.* FROM {} AS a WHERE {predicate}", r.name),
                    r,
                    false,
                )
            }
            PlanKind::Project { input, expressions } => {
                let mut r = self.relation(input, depth + 1)?;
                let scope = r.scope("a");
                let mut selected = vec![];
                for p in &expressions {
                    selected.push(format!(
                        "{} AS \"{}\"",
                        self.expr(p.expr, &scope)?,
                        column(p.column)
                    ));
                }
                selected.extend(r.extra.iter().map(|e| reference("a", e)));
                if selected.is_empty() {
                    selected.push("1 AS \"_unit\"".into());
                }
                r.columns = expressions.iter().map(|p| p.column).collect();
                (
                    format!("SELECT {} FROM {} AS a", selected.join(", "), r.name),
                    r,
                    false,
                )
            }
            PlanKind::Sort { input, keys } => {
                let mut r = self.relation(input, depth + 1)?;
                let scope = r.scope("a");
                let mut selected = vec!["a.*".into()];
                r.order.clear();
                for (i, key) in keys.iter().enumerate() {
                    let name = format!("o{}_{}", id.0, i);
                    selected.push(format!("{} AS \"{name}\"", self.expr(key.expr, &scope)?));
                    r.extra.push(name.clone());
                    r.order.push(Order {
                        name,
                        direction: key.direction,
                        nulls: key.nulls,
                    });
                }
                (
                    format!("SELECT {} FROM {} AS a", selected.join(", "), r.name),
                    r,
                    false,
                )
            }
            PlanKind::Limit { input, skip, limit } => {
                let r = self.relation(input, depth + 1)?;
                (
                    format!(
                        "SELECT a.* FROM {} AS a{}{}",
                        r.name,
                        r.order_sql("a"),
                        Self::window(skip, limit)?
                    ),
                    r,
                    false,
                )
            }
            PlanKind::Distinct { input, columns } => {
                let r = self.relation(input, depth + 1)?;
                let scope = r.scope("a");
                let keys = columns
                    .iter()
                    .map(|c| {
                        scope
                            .get(c)
                            .cloned()
                            .ok_or_else(|| err("DISTINCT column unavailable"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let partition = if keys.is_empty() {
                    String::new()
                } else {
                    format!("PARTITION BY {}", keys.join(", "))
                };
                let rank = format!("rank{}", id.0);
                let sql = format!(
                    "SELECT d.* FROM (SELECT a.*, ROW_NUMBER() OVER ({partition}{}) AS \"{rank}\" FROM {} AS a) AS d WHERE d.\"{rank}\" = 1",
                    r.order_sql("a"),
                    r.name
                );
                (sql, r, false)
            }
            PlanKind::Materialize { input, mode } => {
                let r = self.relation(input, depth + 1)?;
                (
                    format!("SELECT a.* FROM {} AS a", r.name),
                    r,
                    mode == Materialization::Buffered,
                )
            }
            PlanKind::Join {
                left,
                right,
                kind,
                on,
            }
            | PlanKind::PhysicalJoin {
                left,
                right,
                kind,
                on,
                ..
            } => {
                let mut l = self.relation(left, depth + 1)?;
                let r = self.relation(right, depth + 1)?;
                let mut scope = l.scope("a");
                for (c, s) in r.scope("b") {
                    if scope.insert(c, s).is_some() {
                        return Err(err("join inputs must have distinct query-local columns"));
                    }
                }
                let predicate = self.expr(on, &scope)?;
                if matches!(kind, JoinKind::Semi | JoinKind::Anti) {
                    (
                        format!(
                            "SELECT a.* FROM {} AS a WHERE {}EXISTS (SELECT 1 FROM {} AS b WHERE {predicate})",
                            l.name,
                            if kind == JoinKind::Anti { "NOT " } else { "" },
                            r.name
                        ),
                        l,
                        false,
                    )
                } else {
                    let mut selected = l
                        .columns
                        .iter()
                        .map(|c| reference("a", &column(*c)))
                        .chain(r.columns.iter().map(|c| reference("b", &column(*c))))
                        .collect::<Vec<_>>();
                    selected.extend(l.extra.iter().map(|e| reference("a", e)));
                    selected.extend(r.extra.iter().map(|e| reference("b", e)));
                    if selected.is_empty() {
                        selected.push("1 AS \"_unit\"".into());
                    }
                    let sql = format!(
                        "SELECT {} FROM {} AS a {} JOIN {} AS b ON {predicate}",
                        selected.join(", "),
                        l.name,
                        if kind == JoinKind::Left {
                            "LEFT"
                        } else {
                            "INNER"
                        },
                        r.name
                    );
                    l.columns.extend(r.columns);
                    l.extra.extend(r.extra);
                    l.identities.extend(r.identities);
                    l.order.clear();
                    (sql, l, false)
                }
            }
            PlanKind::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                let r = self.relation(input, depth + 1)?;
                let scope = r.scope("a");
                let mut selected = vec![];
                let mut groups = vec![];
                let mut columns = vec![];
                for p in group_by {
                    let expr = self.expr(p.expr, &scope)?;
                    groups.push(expr.clone());
                    selected.push(format!("{expr} AS \"{}\"", column(p.column)));
                    columns.push(p.column);
                }
                for agg in aggregates {
                    let arg = if let Some(e) = agg.argument {
                        self.expr(e, &scope)?
                    } else {
                        if agg.function != AggregateFn::Count || agg.distinct {
                            return Err(err("only non-distinct COUNT accepts no argument"));
                        }
                        "*".into()
                    };
                    let function = match agg.function {
                        AggregateFn::Count => "COUNT",
                        AggregateFn::Sum => "SUM",
                        AggregateFn::Min => "MIN",
                        AggregateFn::Max => "MAX",
                        AggregateFn::Avg => "AVG",
                    };
                    let arg = if matches!(agg.function, AggregateFn::Sum | AggregateFn::Avg) {
                        format!("CAST({arg} AS REAL)")
                    } else {
                        arg
                    };
                    selected.push(format!(
                        "{function}({}{arg}) AS \"{}\"",
                        if agg.distinct { "DISTINCT " } else { "" },
                        column(agg.column)
                    ));
                    columns.push(agg.column);
                }
                if selected.is_empty() {
                    selected.push("COUNT(*) * 0 + 1 AS \"_unit\"".into());
                }
                let group = if groups.is_empty() {
                    String::new()
                } else {
                    format!(" GROUP BY {}", groups.join(", "))
                };
                (
                    format!("SELECT {} FROM {} AS a{group}", selected.join(", "), r.name),
                    Relation {
                        columns,
                        ..Relation::default()
                    },
                    false,
                )
            }
            PlanKind::UnionAll { inputs, columns } => {
                if inputs.is_empty() {
                    return Err(err("UNION ALL requires at least one input"));
                }
                let inputs = inputs
                    .into_iter()
                    .map(|i| self.relation(i, depth + 1))
                    .collect::<Result<Vec<_>>>()?;
                let mut rel = Relation {
                    columns: columns.clone(),
                    ..Relation::default()
                };
                // Preserve identity for optimizer-produced unions of index probes.
                for (n, identity) in inputs[0].identities.iter().enumerate() {
                    if inputs.iter().all(|r| {
                        r.identities.get(n).is_some_and(|i| {
                            i.table == identity.table && i.names.len() == identity.names.len()
                        })
                    }) {
                        let names = (0..identity.names.len())
                            .map(|i| format!("u{}_{}_{}", id.0, n, i))
                            .collect::<Vec<_>>();
                        rel.extra.extend(names.clone());
                        rel.identities.push(Identity {
                            table: identity.table,
                            names,
                        });
                    } else {
                        break;
                    }
                }
                let mut branches = vec![];
                for input in inputs {
                    if input.columns.len() != columns.len() {
                        return Err(err("UNION ALL output width mismatch"));
                    }
                    let mut selected = input
                        .columns
                        .iter()
                        .zip(&columns)
                        .map(|(src, dst)| {
                            format!("{} AS \"{}\"", reference("a", &column(*src)), column(*dst))
                        })
                        .collect::<Vec<_>>();
                    for (src, dst) in input.identities.iter().zip(&rel.identities) {
                        selected.extend(
                            src.names
                                .iter()
                                .zip(&dst.names)
                                .map(|(s, d)| format!("{} AS \"{d}\"", reference("a", s))),
                        );
                    }
                    if selected.is_empty() {
                        selected.push("1 AS \"_unit\"".into());
                    }
                    branches.push(format!(
                        "SELECT {} FROM {} AS a",
                        selected.join(", "),
                        input.name
                    ));
                }
                (branches.join(" UNION ALL "), rel, false)
            }
            PlanKind::Exists { input, output } => {
                let r = self.relation(input, depth + 1)?;
                (
                    format!(
                        "SELECT EXISTS (SELECT 1 FROM {}) AS \"{}\"",
                        r.name,
                        column(output)
                    ),
                    Relation {
                        columns: vec![output],
                        ..Relation::default()
                    },
                    false,
                )
            }
            PlanKind::FetchRows {
                input,
                table,
                columns,
            } => {
                let r = self.relation(input, depth + 1)?;
                let predicate = self.identity_match(&r, table, "a", "t")?;
                let mut selected = vec![];
                for c in &columns {
                    let (t, f) = self
                        .query
                        .storage_origin(*c)
                        .ok_or_else(|| err("fetch needs storage columns"))?;
                    if t != table {
                        return Err(err("fetch column/table mismatch"));
                    }
                    selected.push(format!(
                        "{} AS \"{}\"",
                        self.field_sql(table, f, "t")?,
                        column(*c)
                    ));
                }
                selected.extend(r.extra.iter().map(|e| reference("a", e)));
                if selected.is_empty() {
                    selected.push("1 AS \"_unit\"".into());
                }
                let sql = format!(
                    "SELECT {} FROM {} AS a INNER JOIN {} AS t ON {predicate}",
                    selected.join(", "),
                    r.name,
                    self.table_sql(table)?
                );
                (sql, Relation { columns, ..r }, false)
            }
            PlanKind::RangeCount {
                output,
                table,
                index,
                range,
                skip,
                limit,
            } => {
                let fields = self.key_fields(table, index)?;
                let predicates = self.range_predicate(table, &fields, &range)?;
                let where_sql = if predicates.is_empty() {
                    String::new()
                } else {
                    format!(" WHERE {}", predicates.join(" AND "))
                };
                let sql = format!(
                    "SELECT COUNT(*) AS \"{}\" FROM (SELECT 1 FROM {} AS t{where_sql}{})",
                    column(output),
                    self.table_sql(table)?,
                    Self::window(skip, limit)?
                );
                (
                    sql,
                    Relation {
                        columns: vec![output],
                        ..Relation::default()
                    },
                    false,
                )
            }
            PlanKind::Insert { .. }
            | PlanKind::Update { .. }
            | PlanKind::Delete { .. }
            | PlanKind::WriteBatch { .. } => return Err(err("writes must be the query root")),
        };
        Ok(self.add_relation(id, sql, rel, buffered))
    }
    fn identity_match(
        &self,
        r: &Relation,
        table: TableId,
        source: &str,
        target: &str,
    ) -> Result<String> {
        let identity = r.identity(table)?;
        let pk = &self.table(table)?.primary_key;
        if pk.is_empty() || pk.len() != identity.names.len() {
            return Err(err("invalid primary-key identity"));
        }
        pk.iter()
            .zip(&identity.names)
            .map(|(f, n)| {
                Ok(format!(
                    "{} = {}",
                    self.field_sql(table, *f, target)?,
                    reference(source, n)
                ))
            })
            .collect::<Result<Vec<_>>>()
            .map(|p| p.join(" AND "))
    }
    fn mutation(
        &mut self,
        input: PlanId,
        table: TableId,
        assignments: Option<&[(FieldId, ExprId)]>,
    ) -> Result<String> {
        let r = self.relation(input, 0)?;
        let scope = r.scope("a");
        let mut selected = r
            .identity(table)?
            .names
            .iter()
            .map(|n| reference("a", n))
            .collect::<Vec<_>>();
        if let Some(assignments) = assignments {
            if assignments.is_empty() {
                return Err(err("UPDATE requires assignments"));
            }
            let mut seen = vec![];
            for (i, (field, value)) in assignments.iter().enumerate() {
                if seen.contains(field) {
                    return Err(err("duplicate UPDATE field"));
                }
                seen.push(*field);
                self.field_sql(table, *field, "")?;
                selected.push(format!("{} AS \"v{i}\"", self.expr(*value, &scope)?));
            }
        }
        // Snapshot all old keys and RHS values before changing any target row.
        self.ctes.push(format!(
            "\"mx_targets\" AS MATERIALIZED (SELECT {} FROM {} AS a)",
            selected.join(", "),
            r.name
        ));
        let predicate = self.identity_match(&r, table, "s", "t")?;
        let exists = format!("EXISTS (SELECT 1 FROM \"mx_targets\" AS s WHERE {predicate})");
        if let Some(assignments) = assignments {
            let fields = assignments
                .iter()
                .map(|(f, _)| self.field_sql(table, *f, ""))
                .collect::<Result<Vec<_>>>()?;
            let values = (0..assignments.len())
                .map(|i| format!("s.\"v{i}\""))
                .collect::<Vec<_>>();
            Ok(format!(
                "UPDATE {} AS t SET ({}) = (SELECT {} FROM \"mx_targets\" AS s WHERE {predicate}) WHERE {exists}",
                self.table_sql(table)?,
                fields.join(", "),
                values.join(", ")
            ))
        } else {
            Ok(format!(
                "DELETE FROM {} AS t WHERE {exists}",
                self.table_sql(table)?
            ))
        }
    }
    fn batch(
        &mut self,
        table: TableId,
        mode: MutationMode,
        rows: &[Vec<(FieldId, ExprId)>],
    ) -> Result<Vec<SqlStatement>> {
        let table_def = self.table(table)?.clone();
        let mut out = vec![];
        for row in rows {
            let mut seen = vec![];
            let mut names = vec![];
            let mut values = vec![];
            for (field, value) in row {
                if mode == MutationMode::Delete && !table_def.primary_key.contains(field) {
                    return Err(err("batch DELETE accepts only primary-key fields"));
                }
                if seen.contains(field) {
                    return Err(err("duplicate write field"));
                }
                seen.push(*field);
                names.push(self.field_sql(table, *field, "")?);
                values.push(self.expr(*value, &HashMap::default())?);
            }
            let sql = match mode {
                MutationMode::Insert | MutationMode::Set => {
                    for f in &table_def.fields {
                        if !f.nullable && !seen.contains(&f.id) {
                            return Err(err("write is missing a required field"));
                        }
                    }
                    let mut sql = if names.is_empty() {
                        format!("INSERT INTO {} DEFAULT VALUES", self.table_sql(table)?)
                    } else {
                        format!(
                            "INSERT INTO {} ({}) VALUES ({})",
                            self.table_sql(table)?,
                            names.join(", "),
                            values.join(", ")
                        )
                    };
                    if mode == MutationMode::Set {
                        let keys = table_def
                            .primary_key
                            .iter()
                            .map(|f| self.field_sql(table, *f, ""))
                            .collect::<Result<Vec<_>>>()?;
                        let updates = table_def
                            .fields
                            .iter()
                            .filter(|f| !table_def.primary_key.contains(&f.id))
                            .map(|f| {
                                let name = self.field_sql(table, f.id, "")?;
                                Ok(format!("{name} = excluded.{name}"))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        sql.push_str(&format!(
                            " ON CONFLICT ({}) DO {}",
                            keys.join(", "),
                            if updates.is_empty() {
                                "NOTHING".into()
                            } else {
                                format!("UPDATE SET {}", updates.join(", "))
                            }
                        ));
                    }
                    sql
                }
                MutationMode::Update | MutationMode::Delete => {
                    let mut predicates = vec![];
                    for f in &table_def.primary_key {
                        let i = seen
                            .iter()
                            .position(|s| s == f)
                            .ok_or_else(|| err("write must supply the complete primary key"))?;
                        predicates.push(format!("{} = {}", names[i], values[i]));
                    }
                    if predicates.is_empty() {
                        return Err(err("write requires a primary key"));
                    }
                    if mode == MutationMode::Delete {
                        format!(
                            "DELETE FROM {} WHERE {}",
                            self.table_sql(table)?,
                            predicates.join(" AND ")
                        )
                    } else {
                        let assignments = seen
                            .iter()
                            .enumerate()
                            .filter(|(_, f)| !table_def.primary_key.contains(f))
                            .map(|(i, _)| format!("{} = {}", names[i], values[i]))
                            .collect::<Vec<_>>();
                        if assignments.is_empty() {
                            return Err(err("batch UPDATE requires a non-key assignment"));
                        }
                        format!(
                            "UPDATE {} SET {} WHERE {}",
                            self.table_sql(table)?,
                            assignments.join(", "),
                            predicates.join(" AND ")
                        )
                    }
                }
            };
            out.push(self.finish(sql));
        }
        Ok(out)
    }
}

/// Arithmetic uses f64, including floating remainder and round-away-from-zero.
/// SQLite's integer division/remainder and implicit text coercions cannot stand
/// in for these operations. NaN is rejected rather than silently stored as NULL.
pub fn install_functions(connection: &Connection) -> rusqlite::Result<()> {
    let flags = FunctionFlags::SQLITE_UTF8
        | FunctionFlags::SQLITE_DETERMINISTIC
        | FunctionFlags::SQLITE_INNOCUOUS;
    fn checked(v: f64) -> rusqlite::Result<f64> {
        if v.is_nan() {
            Err(rusqlite::Error::UserFunctionError(Box::new(err(
                "SQLite cannot represent NaN",
            ))))
        } else {
            Ok(v)
        }
    }
    for op in ["add", "sub", "mul", "div", "rem"] {
        connection.create_scalar_function(
            format!("napalm_mx_{op}").as_str(),
            2,
            flags,
            move |ctx| {
                let (a, b): (Option<f64>, Option<f64>) = (ctx.get(0)?, ctx.get(1)?);
                match (a, b) {
                    (Some(a), Some(b)) => checked(match op {
                        "add" => a + b,
                        "sub" => a - b,
                        "mul" => a * b,
                        "div" => a / b,
                        _ => a % b,
                    })
                    .map(Some),
                    _ => Ok(None),
                }
            },
        )?;
    }
    for op in ["abs", "floor", "ceil", "round"] {
        connection.create_scalar_function(
            format!("napalm_mx_{op}").as_str(),
            1,
            flags,
            move |ctx| {
                let value: Option<f64> = ctx.get(0)?;
                value
                    .map(|v| {
                        checked(match op {
                            "abs" => v.abs(),
                            "floor" => v.floor(),
                            "ceil" => v.ceil(),
                            _ => v.round(),
                        })
                    })
                    .transpose()
            },
        )?;
    }
    for op in ["lower_ascii", "upper_ascii"] {
        connection.create_scalar_function(
            format!("napalm_mx_{op}").as_str(),
            1,
            flags,
            move |ctx| {
                let value: Option<String> = ctx.get(0)?;
                Ok(value.map(|v| {
                    if op == "lower_ascii" {
                        v.to_ascii_lowercase()
                    } else {
                        v.to_ascii_uppercase()
                    }
                }))
            },
        )?;
    }
    connection.create_scalar_function("napalm_mx_string_length", 1, flags, |ctx| {
        let v: Option<String> = ctx.get(0)?;
        Ok(v.map(|v| v.chars().count() as f64))
    })?;
    connection.create_scalar_function("napalm_mx_bytes_length", 1, flags, |ctx| {
        let v: Option<Vec<u8>> = ctx.get(0)?;
        Ok(v.map(|v| v.len() as f64))
    })?;
    Ok(())
}

#[cfg(test)]
#[path = "tests/napalm_mx_sqlite_tests.rs"]
mod tests;
