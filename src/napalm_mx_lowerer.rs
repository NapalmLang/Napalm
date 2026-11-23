//! Catalog-aware MX AST -> typed logical MX IR. Execution is a backend concern.
use crate::napalm_front::{Diagnostic, Span};
use crate::napalm_mx_front::{self as ast, Statement};
use crate::napalm_mx_ir::*;

#[cfg(test)]
#[path = "tests/napalm_mx_tests.rs"]
mod tests;

/// Host-provided parameter contract. Values are bound by the future executor.
#[derive(Clone, Debug)]
pub struct Parameter {
    pub name: String,
    pub ty: ScalarType,
    pub nullable: bool,
}
type Result<T> = std::result::Result<T, Diagnostic>;

/// Append one query per statement. Failure leaves the supplied program unchanged.
/// Catalog names refer to `program.strings`; no schema is invented by lowering.
pub fn lower(
    block: &ast::Block,
    program: &mut Program,
    parameters: &[Parameter],
) -> Result<Vec<QueryId>> {
    let strings = program.strings.len();
    let queries = program.queries.len();
    let result = (|| {
        for (i, p) in parameters.iter().enumerate() {
            if parameters[..i].iter().any(|other| other.name == p.name) {
                return Err(error(
                    block.span,
                    format!("duplicate parameter `${}`", p.name),
                ));
            }
        }
        let mut ids = Vec::new();
        for statement in &block.statements {
            let table_name = match &statement.value {
                Statement::Schema(_) => return Err(error(statement.span, "schema statements use the SQLite schema lowerer")),
                Statement::Select { table, .. }
                | Statement::Insert { table, .. }
                | Statement::Update { table, .. }
                | Statement::Delete { table, .. } => table,
            };
            let table = program
                .catalog
                .tables
                .iter()
                .find(|t| program.string(t.name) == Some(table_name.value.as_str()))
                .cloned()
                .ok_or_else(|| {
                    error(
                        table_name.span,
                        format!("unknown table `{}`", table_name.value),
                    )
                })?;
            let id = QueryId(program.queries.len() as u32);
            let query = Query {
                id,
                name: None,
                root: PlanId(0),
                params: vec![],
                columns: vec![],
                exprs: vec![],
                plans: vec![],
                transaction: TransactionMode::ReadOnlySnapshot,
            };
            let mut l = Lowerer {
                program,
                query,
                table,
                parameters,
                columns: vec![],
            };
            l.query.root = l.statement(&statement.value, statement.span)?;
            verify_query(&l.query, &l.program.catalog)
                .map_err(|errors| error(statement.span, format!("invalid MX plan: {errors:?}")))?;
            l.program.queries.push(l.query);
            ids.push(id);
        }
        Ok(ids)
    })();
    if result.is_err() {
        program.strings.truncate(strings);
        program.queries.truncate(queries);
    }
    result
}
fn error(span: Span, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error("MX-S001", message, span)
}
struct Lowerer<'a> {
    program: &'a mut Program,
    query: Query,
    table: TableDef,
    parameters: &'a [Parameter],
    columns: Vec<(FieldId, ColumnId)>,
}
impl Lowerer<'_> {
    fn intern(&mut self, value: &str) -> StringId {
        if let Some(i) = self.program.strings.iter().position(|s| s == value) {
            return StringId(i as u32);
        }
        let id = StringId(self.program.strings.len() as u32);
        self.program.strings.push(value.into());
        id
    }
    fn field(&self, name: &ast::Name) -> Result<FieldDef> {
        self.table
            .fields
            .iter()
            .find(|f| self.program.string(f.name) == Some(name.value.as_str()))
            .cloned()
            .ok_or_else(|| error(name.span, format!("unknown field `{}`", name.value)))
    }
    fn scan(&mut self, predicate: &Option<ast::Expr>) -> Result<PlanId> {
        for f in &self.table.fields {
            let col = self.query.add_column(
                Some(f.name),
                f.ty,
                f.nullable,
                ColumnOrigin::Storage {
                    table: self.table.id,
                    field: f.id,
                },
            );
            self.columns.push((f.id, col));
        }
        let mut input = self.query.add_plan(PlanKind::TableScan {
            table: self.table.id,
            columns: self.columns.iter().map(|(_, c)| *c).collect(),
        });
        if let Some(expr) = predicate {
            let predicate = self.expr(expr)?;
            self.require(predicate, ScalarType::Bool, expr.span)?;
            input = self.query.add_plan(PlanKind::Filter { input, predicate });
        }
        Ok(input)
    }
    fn statement(&mut self, statement: &Statement, span: Span) -> Result<PlanId> {
        match statement {
            Statement::Select {
                projection,
                predicate,
                order,
                limit,
                offset,
                ..
            } => {
                let mut input = self.scan(predicate)?;
                if !order.is_empty() {
                    let mut keys = Vec::new();
                    for (expr, desc) in order {
                        let id = self.expr(expr)?;
                        let ty = self.query.expr(id).unwrap().ty;
                        if !ty.is_orderable() && ty != ScalarType::Null && ty != ScalarType::Any {
                            return Err(error(expr.span, "ORDER BY requires an orderable value"));
                        }
                        keys.push(SortKey {
                            expr: id,
                            direction: if *desc {
                                SortDirection::Desc
                            } else {
                                SortDirection::Asc
                            },
                            nulls: NullOrder::Last,
                        });
                    }
                    input = self.query.add_plan(PlanKind::Sort { input, keys });
                }
                if limit.is_some() || *offset != 0 {
                    input = self.query.add_plan(PlanKind::Limit {
                        input,
                        skip: *offset,
                        limit: *limit,
                    });
                }
                if let Some(projection) = projection {
                    let mut expressions = Vec::new();
                    for p in projection {
                        let expr = self.expr(&p.expr)?;
                        let name = p
                            .alias
                            .as_ref()
                            .or_else(|| match &p.expr.value {
                                ast::ExprKind::Field(n) => Some(n),
                                _ => None,
                            })
                            .map(|n| self.intern(&n.value));
                        let node = self.query.expr(expr).unwrap();
                        let output = self.query.add_column(
                            name,
                            node.ty,
                            node.nullable,
                            ColumnOrigin::Derived,
                        );
                        expressions.push(ProjectionExpr {
                            column: output,
                            expr,
                        });
                    }
                    input = self
                        .query
                        .add_plan(PlanKind::Project { input, expressions });
                }
                Ok(input)
            }
            Statement::Insert { fields, rows, .. } => {
                self.query.transaction = TransactionMode::ReadWrite;
                let mut defs = Vec::new();
                for name in fields {
                    let field = self.field(name)?;
                    if defs.iter().any(|f: &FieldDef| f.id == field.id) {
                        return Err(error(name.span, "duplicate INSERT field"));
                    }
                    defs.push(field);
                }
                for f in &self.table.fields {
                    if !f.nullable && !defs.iter().any(|d| d.id == f.id) {
                        return Err(error(
                            span,
                            format!(
                                "missing required field `{}`",
                                self.program.string(f.name).unwrap_or("?")
                            ),
                        ));
                    }
                }
                let mut lowered = Vec::new();
                for row in rows {
                    if row.len() != defs.len() {
                        return Err(error(span, "INSERT field/value count mismatch"));
                    }
                    let mut values = Vec::new();
                    for (field, expr) in defs.iter().zip(row) {
                        let id = self.expr(expr)?;
                        self.assignment(field, id, expr.span)?;
                        values.push((field.id, id));
                    }
                    // The initial schema has no defaults. Omitted nullable fields are NULL.
                    for field in &self.table.fields {
                        if !defs.iter().any(|d| d.id == field.id) {
                            let id = self.query.add_expr(
                                ExprKind::Const(ScalarValue::Null),
                                ScalarType::Null,
                                true,
                            );
                            values.push((field.id, id));
                        }
                    }
                    lowered.push(values);
                }
                Ok(self.query.add_plan(PlanKind::Insert {
                    table: self.table.id,
                    rows: lowered,
                }))
            }
            Statement::Update {
                assignments,
                predicate,
                ..
            } => {
                self.query.transaction = TransactionMode::ReadWrite;
                let input = self.scan(predicate)?;
                let mut lowered = Vec::new();
                for (name, expr) in assignments {
                    let field = self.field(name)?;
                    if lowered.iter().any(|(f, _)| *f == field.id) {
                        return Err(error(name.span, "duplicate UPDATE field"));
                    }
                    let id = self.expr(expr)?;
                    self.assignment(&field, id, expr.span)?;
                    lowered.push((field.id, id));
                }
                Ok(self.query.add_plan(PlanKind::Update {
                    input,
                    table: self.table.id,
                    assignments: lowered,
                }))
            }
            Statement::Delete { predicate, .. } => {
                self.query.transaction = TransactionMode::ReadWrite;
                let input = self.scan(predicate)?;
                Ok(self.query.add_plan(PlanKind::Delete {
                    input,
                    table: self.table.id,
                }))
            }
            Statement::Schema(_) => Err(error(span, "schema statements use the SQLite schema lowerer")),
        }
    }
    fn require(&self, id: ExprId, expected: ScalarType, span: Span) -> Result<()> {
        let actual = self.query.expr(id).unwrap().ty;
        if actual == expected
            || actual == ScalarType::Null
            || actual == ScalarType::Any
            || expected == ScalarType::Any
        {
            Ok(())
        } else {
            Err(error(
                span,
                format!("expected {expected:?}, found {actual:?}"),
            ))
        }
    }
    fn assignment(&self, field: &FieldDef, id: ExprId, span: Span) -> Result<()> {
        self.require(id, field.ty, span)?;
        if !field.nullable && self.query.expr(id).unwrap().nullable {
            return Err(error(span, "nullable value assigned to a required field"));
        }
        Ok(())
    }
    fn expr(&mut self, expr: &ast::Expr) -> Result<ExprId> {
        use ast::ExprKind as E;
        let (kind, ty, nullable) = match &expr.value {
            E::Null => (ExprKind::Const(ScalarValue::Null), ScalarType::Null, true),
            E::Bool(v) => (
                ExprKind::Const(ScalarValue::Bool(*v)),
                ScalarType::Bool,
                false,
            ),
            E::Number(v) => (
                ExprKind::Const(ScalarValue::Number(NumberBits::from_f64(*v))),
                ScalarType::Number,
                false,
            ),
            E::String(v) => {
                let id = self.intern(v);
                (
                    ExprKind::Const(ScalarValue::String(id)),
                    ScalarType::String,
                    false,
                )
            }
            E::Field(name) => {
                let field = self.field(name)?;
                let col = self
                    .columns
                    .iter()
                    .find(|(f, _)| *f == field.id)
                    .map(|(_, c)| *c)
                    .ok_or_else(|| {
                        error(
                            name.span,
                            "field references are not allowed in INSERT VALUES",
                        )
                    })?;
                (ExprKind::Column(col), field.ty, field.nullable)
            }
            E::Param(name) => {
                let param = self
                    .parameters
                    .iter()
                    .find(|p| p.name == name.value)
                    .ok_or_else(|| {
                        error(name.span, format!("unknown parameter `${}`", name.value))
                    })?;
                let ty = param.ty;
                let nullable = param.nullable;
                let name_id = self.intern(&name.value);
                let id = if let Some(p) = self.query.params.iter().find(|p| p.name == Some(name_id))
                {
                    p.id
                } else {
                    let id = ParamId(
                        u16::try_from(self.query.params.len())
                            .map_err(|_| error(name.span, "too many MX parameters"))?,
                    );
                    self.query.params.push(ParamDef {
                        id,
                        name: Some(name_id),
                        ty,
                        nullable,
                    });
                    id
                };
                (ExprKind::Param(id), ty, nullable)
            }
            E::Unary(op, value) => {
                let id = self.expr(value)?;
                let ty = if *op == UnaryOp::Not {
                    ScalarType::Bool
                } else {
                    ScalarType::Number
                };
                self.require(id, ty, value.span)?;
                (
                    ExprKind::Unary { op: *op, value: id },
                    ty,
                    self.query.expr(id).unwrap().nullable,
                )
            }
            E::Binary(op, left, right) => {
                let a = self.expr(left)?;
                let b = self.expr(right)?;
                let ty = if op.is_comparison() {
                    let at = self.query.expr(a).unwrap().ty;
                    if at != ScalarType::Null {
                        self.require(b, at, right.span)?;
                    }
                    ScalarType::Bool
                } else {
                    let ty = if matches!(op, BinaryOp::And | BinaryOp::Or) {
                        ScalarType::Bool
                    } else {
                        ScalarType::Number
                    };
                    self.require(a, ty, left.span)?;
                    self.require(b, ty, right.span)?;
                    ty
                };
                let nullable =
                    self.query.expr(a).unwrap().nullable || self.query.expr(b).unwrap().nullable;
                (
                    ExprKind::Binary {
                        op: *op,
                        left: a,
                        right: b,
                    },
                    ty,
                    nullable,
                )
            }
            E::IsNull(value, negated) => {
                let value = self.expr(value)?;
                (
                    ExprKind::IsNull {
                        value,
                        negated: *negated,
                    },
                    ScalarType::Bool,
                    false,
                )
            }
        };
        Ok(self.query.add_expr(kind, ty, nullable))
    }
}
