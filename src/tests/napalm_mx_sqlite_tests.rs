use super::*;
use crate::napalm_mx_front;
use crate::napalm_mx_lowerer::{self, Parameter};
use rusqlite::params_from_iter;

fn catalog() -> Program {
    let mut p = Program::default();
    p.strings = ["users", "id", "name", "active", "age", "by_age"]
        .map(String::from)
        .to_vec();
    p.catalog.tables.push(TableDef {
        id: TableId(0),
        name: StringId(0),
        key_prefix: vec![1],
        fields: [
            (ScalarType::Number, false),
            (ScalarType::String, true),
            (ScalarType::Bool, false),
            (ScalarType::Number, true),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (ty, nullable))| FieldDef {
            id: FieldId(i as u32),
            name: StringId(i as u32 + 1),
            ty,
            nullable,
            ordinal: i as u16,
        })
        .collect(),
        primary_key: vec![FieldId(0)],
        indexes: vec![IndexId(0)],
        stats: TableStats {
            rows: 10000,
            ..TableStats::default()
        },
    });
    p.catalog.indexes.push(IndexDef {
        id: IndexId(0),
        table: TableId(0),
        name: StringId(5),
        key_prefix: vec![2],
        columns: vec![IndexColumn {
            field: FieldId(3),
            encoding: KeyEncoding::NumberOrdered,
            descending: false,
            nulls: NullOrder::Last,
        }],
        included_fields: vec![],
        unique: false,
    });
    p
}
fn db() -> Connection {
    let db = Connection::open_in_memory().unwrap();
    install_functions(&db).unwrap();
    db.execute_batch("CREATE TABLE users(id REAL PRIMARY KEY NOT NULL, name TEXT, active INTEGER NOT NULL, age REAL); CREATE INDEX by_age ON users(age); INSERT INTO users VALUES (1, 'Ada', 1, 30), (2, 'Grace', 1, 40), (3, 'Lin', 0, 20), (4, 'Zed', 0, NULL);").unwrap();
    db
}
fn source(text: &str, optimized: bool) -> Program {
    let mut p = catalog();
    let parameters = vec![Parameter {
        name: "p".into(),
        ty: ScalarType::Number,
        nullable: true,
    }];
    napalm_mx_lowerer::lower(
        &napalm_mx_front::parse(text, 0).unwrap(),
        &mut p,
        &parameters,
    )
    .unwrap();
    if optimized {
        optimize(&mut p);
    }
    p
}
fn optimize(p: &mut Program) {
    crate::napalm_mx_ir_optimizing::optimization_passes()
        .optimize_program(p)
        .unwrap();
}
fn run(db: &Connection, q: &SqlQuery, parameters: &[Value]) -> Vec<Vec<Value>> {
    let mut result = vec![];
    for sql in &q.statements {
        let values = sql.bind_values(parameters).unwrap();
        let mut statement = db
            .prepare(&sql.sql)
            .unwrap_or_else(|e| panic!("{e}\n{}", sql.sql));
        let width = statement.column_count();
        let mut rows = statement
            .query(params_from_iter(values.iter()))
            .unwrap_or_else(|e| panic!("{e}\n{}", sql.sql));
        while let Some(row) = rows.next().unwrap_or_else(|e| panic!("{e}\n{}", sql.sql)) {
            result.push(if q.discard_dummy_column {
                vec![]
            } else {
                (0..width).map(|i| row.get(i).unwrap()).collect()
            });
        }
    }
    result
}
fn evaluate(text: &str, optimized: bool, parameters: &[Value]) -> Vec<Vec<Value>> {
    let p = source(text, optimized);
    run(&db(), &compile_query(&p, QueryId(0)).unwrap(), parameters)
}
fn number(v: f64) -> Value {
    Value::Real(v)
}
fn text(v: &str) -> Value {
    Value::Text(v.into())
}
fn empty_query() -> Query {
    Query {
        id: QueryId(0),
        name: None,
        root: PlanId(0),
        params: vec![],
        columns: vec![],
        exprs: vec![],
        plans: vec![],
        transaction: TransactionMode::Auto,
    }
}
fn constant(q: &mut Query, v: f64) -> ExprId {
    q.add_expr(
        ExprKind::Const(ScalarValue::Number(NumberBits::from_f64(v))),
        ScalarType::Number,
        false,
    )
}
fn col_expr(q: &mut Query, c: ColumnId) -> ExprId {
    let def = q.column(c).unwrap();
    q.add_expr(ExprKind::Column(c), def.ty, def.nullable)
}

#[test]
fn sqlite_logical_and_optimized_selects_agree() {
    for (query, parameters, expected) in [
        (
            "SELECT name FROM users WHERE id = $p",
            vec![number(2.0)],
            vec![vec![text("Grace")]],
        ),
        (
            "SELECT name FROM users WHERE age >= 20 AND active ORDER BY age DESC LIMIT 1 OFFSET 1",
            vec![],
            vec![vec![text("Ada")]],
        ),
        (
            "SELECT name FROM users ORDER BY age ASC LIMIT 3 OFFSET 1",
            vec![],
            vec![vec![text("Ada")], vec![text("Grace")], vec![text("Zed")]],
        ),
        (
            "SELECT name FROM users WHERE id = 1 OR id = 3 ORDER BY id",
            vec![],
            vec![vec![text("Ada")], vec![text("Lin")]],
        ),
        (
            "SELECT name FROM users WHERE age = $p",
            vec![Value::Null],
            vec![],
        ),
        (
            "SELECT name FROM users WHERE age IS NULL",
            vec![],
            vec![vec![text("Zed")]],
        ),
        ("SELECT name FROM users LIMIT 0", vec![], vec![]),
        (
            "SELECT name FROM users ORDER BY id OFFSET 3",
            vec![],
            vec![vec![text("Zed")]],
        ),
    ] {
        for optimized in [false, true] {
            assert_eq!(
                evaluate(query, optimized, &parameters),
                expected,
                "{query}, optimized={optimized}"
            );
        }
    }
}

#[test]
fn sqlite_numbers_keep_float_division_remainder_and_nulls() {
    for optimized in [false, true] {
        assert_eq!(
            evaluate(
                "SELECT 5 / 2, 5.5 % 2, age / 2 FROM users WHERE id = 4",
                optimized,
                &[]
            ),
            vec![vec![number(2.5), number(1.5), Value::Null]]
        );
        assert_eq!(
            evaluate(
                "SELECT id / 2, id % 0.6 FROM users WHERE id = 1",
                optimized,
                &[]
            )[0][0],
            number(0.5)
        );
    }
}

#[test]
fn sqlite_crud_executes_before_and_after_optimization() {
    for optimized in [false, true] {
        let p = source(
            "INSERT INTO users (id, name, active, age) VALUES (5, 'Eve', true, 50), (6, 'Bob', false, 60); UPDATE users SET age = age + 5, name = 'Changed' WHERE id = 5; DELETE FROM users WHERE id = 6; SELECT name, age FROM users WHERE id = 5",
            optimized,
        );
        let db = db();
        db.execute_batch("BEGIN").unwrap();
        let mut rows = vec![];
        for q in compile_program(&p).unwrap() {
            rows = run(&db, &q, &[]);
        }
        db.execute_batch("COMMIT").unwrap();
        assert_eq!(rows, vec![vec![text("Changed"), number(55.0)]]);
        assert_eq!(
            db.query_row("SELECT count(*) FROM users", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            5
        );
    }
}

#[test]
fn sqlite_bindings_are_reused_and_never_interpolated() {
    let p = source(
        "SELECT $p + $p, 'x''); DROP TABLE users; --' FROM users WHERE id = 1",
        false,
    );
    let q = compile_query(&p, QueryId(0)).unwrap();
    assert_eq!(
        q.statements[0]
            .bindings
            .iter()
            .filter(|b| matches!(b, Binding::Parameter(_)))
            .count(),
        1
    );
    assert!(!q.statements[0].sql.contains("DROP TABLE"));
    assert!(q.statements[0].bind_values(&[]).is_err());
    assert!(q.statements[0].bind_values(&[number(f64::NAN)]).is_err());
    assert_eq!(
        run(&db(), &q, &[number(3.0)]),
        vec![vec![number(6.0), text("x'); DROP TABLE users; --")]]
    );
}

#[test]
fn sqlite_quotes_catalog_names_and_avoids_cte_shadowing() {
    for name in ["select", "mx_p0", "odd\"name"] {
        let mut p = source("SELECT name FROM users WHERE id = 1", false);
        p.strings[0] = name.into();
        let db = db();
        db.execute_batch(&format!(
            "ALTER TABLE users RENAME TO \"{}\"",
            name.replace('"', "\"\"")
        ))
        .unwrap();
        assert_eq!(
            run(&db, &compile_query(&p, QueryId(0)).unwrap(), &[]),
            vec![vec![text("Ada")]]
        );
    }
}

#[test]
fn sqlite_projection_keeps_order_columns_hidden() {
    let mut p = source("SELECT name FROM users ORDER BY age DESC LIMIT 2", true);
    let sql = compile_query(&p, QueryId(0)).unwrap();
    assert_eq!(sql.columns.len(), 1);
    assert_eq!(
        run(&db(), &sql, &[]),
        vec![vec![text("Grace")], vec![text("Ada")]]
    );
    let q = &mut p.queries[0];
    q.root = q.add_plan(PlanKind::Materialize {
        input: q.root,
        mode: Materialization::Buffered,
    });
    let sql = compile_query(&p, QueryId(0)).unwrap();
    assert!(sql.statements[0].sql.contains("AS MATERIALIZED"));
    assert_eq!(
        run(&db(), &sql, &[]),
        vec![vec![text("Grace")], vec![text("Ada")]]
    );
}

#[test]
fn sqlite_update_snapshots_old_values_and_respects_limited_input() {
    let mut p = source(
        "UPDATE users SET id = id + 10, age = id WHERE active",
        false,
    );
    let q = &mut p.queries[0];
    let PlanKind::Update { input, .. } = q.plan(q.root).unwrap().kind else {
        panic!()
    };
    let input = q.add_plan(PlanKind::Limit {
        input,
        skip: 0,
        limit: Some(1),
    });
    if let PlanKind::Update { input: slot, .. } = &mut q.plan_mut(q.root).unwrap().kind {
        *slot = input;
    }
    for optimized in [false, true] {
        let mut p = p.clone();
        if optimized {
            optimize(&mut p);
        }
        let db = db();
        run(&db, &compile_query(&p, QueryId(0)).unwrap(), &[]);
        assert_eq!(
            db.query_row("SELECT age FROM users WHERE id = 11", [], |r| r
                .get::<_, f64>(0))
                .unwrap(),
            1.0
        );
        assert_eq!(
            db.query_row("SELECT age FROM users WHERE id = 2", [], |r| r
                .get::<_, f64>(0))
                .unwrap(),
            40.0
        );
    }
}

#[test]
fn sqlite_zero_width_values_empty_and_exists_keep_cardinality() {
    let mut p = catalog();
    let mut q = empty_query();
    q.root = q.add_plan(PlanKind::Values {
        columns: vec![],
        rows: vec![vec![], vec![]],
    });
    p.queries.push(q);
    let sql = compile_query(&p, QueryId(0)).unwrap();
    assert!(sql.discard_dummy_column);
    assert_eq!(run(&db(), &sql, &[]), vec![vec![], vec![]]);
    let q = &mut p.queries[0];
    q.root = q.add_plan(PlanKind::Empty { columns: vec![] });
    let output = q.add_column(None, ScalarType::Bool, false, ColumnOrigin::Derived);
    q.root = q.add_plan(PlanKind::Exists {
        input: q.root,
        output,
    });
    assert_eq!(
        run(&db(), &compile_query(&p, QueryId(0)).unwrap(), &[]),
        vec![vec![Value::Integer(0)]]
    );
}

#[test]
fn sqlite_aggregate_distinct_union_and_empty_aggregate() {
    let mut p = source("SELECT * FROM users", false);
    let q = &mut p.queries[0];
    let age = col_expr(q, ColumnId(3));
    let active = col_expr(q, ColumnId(2));
    let group = q.add_column(None, ScalarType::Bool, false, ColumnOrigin::Derived);
    let sum = q.add_column(None, ScalarType::Number, true, ColumnOrigin::Derived);
    let count = q.add_column(None, ScalarType::Number, false, ColumnOrigin::Derived);
    q.root = q.add_plan(PlanKind::Aggregate {
        input: q.root,
        group_by: vec![ProjectionExpr {
            column: group,
            expr: active,
        }],
        aggregates: vec![
            AggregateExpr {
                column: sum,
                function: AggregateFn::Sum,
                argument: Some(age),
                distinct: false,
            },
            AggregateExpr {
                column: count,
                function: AggregateFn::Count,
                argument: Some(age),
                distinct: true,
            },
        ],
    });
    assert_eq!(
        run(&db(), &compile_query(&p, QueryId(0)).unwrap(), &[]),
        vec![
            vec![Value::Integer(0), number(20.0), Value::Integer(1)],
            vec![Value::Integer(1), number(70.0), Value::Integer(2)]
        ]
    );
    let mut p = source("SELECT id FROM users WHERE id = 1", false);
    let q = &mut p.queries[0];
    let column = q.add_column(None, ScalarType::Number, false, ColumnOrigin::Derived);
    let union = q.add_plan(PlanKind::UnionAll {
        inputs: vec![q.root, q.root],
        columns: vec![column],
    });
    q.root = q.add_plan(PlanKind::Distinct {
        input: union,
        columns: vec![column],
    });
    assert_eq!(
        run(&db(), &compile_query(&p, QueryId(0)).unwrap(), &[]),
        vec![vec![number(1.0)]]
    );
}

#[test]
fn sqlite_all_join_kinds_and_physical_algorithms() {
    for kind in [
        JoinKind::Inner,
        JoinKind::Left,
        JoinKind::Semi,
        JoinKind::Anti,
    ] {
        for algorithm in [
            None,
            Some(JoinAlgorithm::Hash),
            Some(JoinAlgorithm::Merge),
            Some(JoinAlgorithm::IndexNestedLoop(IndexId(0))),
        ] {
            let mut p = catalog();
            let mut q = empty_query();
            let a = q.add_column(None, ScalarType::Number, false, ColumnOrigin::Derived);
            let b = q.add_column(None, ScalarType::Number, true, ColumnOrigin::Derived);
            let one = constant(&mut q, 1.0);
            let two = constant(&mut q, 2.0);
            let left = q.add_plan(PlanKind::Values {
                columns: vec![a],
                rows: vec![vec![one], vec![two]],
            });
            let right = q.add_plan(PlanKind::Values {
                columns: vec![b],
                rows: vec![vec![two]],
            });
            let ae = col_expr(&mut q, a);
            let be = col_expr(&mut q, b);
            let on = q.add_expr(
                ExprKind::Binary {
                    op: BinaryOp::Eq,
                    left: ae,
                    right: be,
                },
                ScalarType::Bool,
                true,
            );
            q.root = q.add_plan(if let Some(algorithm) = algorithm {
                PlanKind::PhysicalJoin {
                    left,
                    right,
                    kind,
                    on,
                    algorithm,
                }
            } else {
                PlanKind::Join {
                    left,
                    right,
                    kind,
                    on,
                }
            });
            p.queries.push(q);
            let expected = match kind {
                JoinKind::Inner => vec![vec![number(2.0), number(2.0)]],
                JoinKind::Left => vec![
                    vec![number(1.0), Value::Null],
                    vec![number(2.0), number(2.0)],
                ],
                JoinKind::Semi => vec![vec![number(2.0)]],
                JoinKind::Anti => vec![vec![number(1.0)]],
            };
            assert_eq!(
                run(&db(), &compile_query(&p, QueryId(0)).unwrap(), &[]),
                expected
            );
        }
    }
}

#[test]
fn sqlite_explicit_range_count_and_late_fetch() {
    let mut p = source("SELECT * FROM users", false);
    let q = &mut p.queries[0];
    let low = constant(q, 20.0);
    let range = KeyRange {
        lower: BoundValue::Inclusive(low),
        ..KeyRange::default()
    };
    let scan = q.add_plan(PlanKind::IndexRangeScan {
        table: TableId(0),
        index: IndexId(0),
        range: range.clone(),
        columns: vec![],
        residual: None,
        direction: ScanDirection::Reverse,
        skip: 1,
        limit: Some(1),
        delivery: ScanDelivery::BorrowedKeys,
        covering: true,
    });
    q.root = q.add_plan(PlanKind::FetchRows {
        input: scan,
        table: TableId(0),
        columns: vec![ColumnId(1)],
    });
    assert_eq!(
        run(&db(), &compile_query(&p, QueryId(0)).unwrap(), &[]),
        vec![vec![text("Ada")]]
    );
    let q = &mut p.queries[0];
    let output = q.add_column(None, ScalarType::Number, false, ColumnOrigin::Derived);
    q.root = q.add_plan(PlanKind::RangeCount {
        output,
        table: TableId(0),
        index: Some(IndexId(0)),
        range,
        skip: 2,
        limit: Some(1),
    });
    assert_eq!(
        run(&db(), &compile_query(&p, QueryId(0)).unwrap(), &[]),
        vec![vec![Value::Integer(1)]]
    );
}

#[test]
fn sqlite_batch_set_update_delete_and_rollback_on_duplicate_insert() {
    for mode in [
        MutationMode::Set,
        MutationMode::Update,
        MutationMode::Delete,
    ] {
        let mut p = source(
            "INSERT INTO users (id, name, active, age) VALUES (1, 'New', true, 90)",
            false,
        );
        let q = &mut p.queries[0];
        let PlanKind::Insert { table, mut rows } = q.plan(q.root).unwrap().kind.clone() else {
            panic!()
        };
        if mode == MutationMode::Delete {
            rows[0].retain(|(f, _)| *f == FieldId(0));
        }
        q.plan_mut(q.root).unwrap().kind = PlanKind::WriteBatch { table, mode, rows };
        let db = db();
        run(&db, &compile_query(&p, QueryId(0)).unwrap(), &[]);
        let name = db.query_row("SELECT name FROM users WHERE id = 1", [], |r| {
            r.get::<_, String>(0)
        });
        if mode == MutationMode::Delete {
            assert!(name.is_err());
        } else {
            assert_eq!(name.unwrap(), "New");
        }
    }
    let p = source(
        "INSERT INTO users (id, active) VALUES (8, true), (1, true)",
        true,
    );
    let sql = compile_query(&p, QueryId(0)).unwrap();
    let db = db();
    db.execute_batch("BEGIN").unwrap();
    let mut failed = false;
    for statement in sql.statements {
        if db
            .execute(
                &statement.sql,
                params_from_iter(statement.bind_values(&[]).unwrap()),
            )
            .is_err()
        {
            failed = true;
            break;
        }
    }
    assert!(failed);
    db.execute_batch("ROLLBACK").unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM users WHERE id = 8", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn sqlite_rejects_invalid_ir_limits_and_constants() {
    let mut p = source("SELECT id FROM users LIMIT 18446744073709551615", false);
    assert!(
        compile_query(&p, QueryId(0))
            .unwrap_err()
            .0
            .contains("64-bit")
    );
    p.queries[0].root = PlanId(99999);
    assert!(compile_query(&p, QueryId(0)).is_err());
    let mut p = source("SELECT id FROM users", false);
    let q = &mut p.queries[0];
    let n = constant(q, f64::NAN);
    let column = q.add_column(None, ScalarType::Number, false, ColumnOrigin::Derived);
    q.root = q.add_plan(PlanKind::Values {
        columns: vec![column],
        rows: vec![vec![n]],
    });
    assert!(compile_query(&p, QueryId(0)).unwrap_err().0.contains("NaN"));
}

#[test]
fn sqlite_scalar_functions_blobs_between_and_in_preserve_values() {
    let mut p = catalog();
    p.strings.push("ÁBC\0D".into());
    p.bytes.push(vec![0, 255, 1]);
    let mut q = empty_query();
    let n = constant(&mut q, -2.5);
    let s = q.add_expr(
        ExprKind::Const(ScalarValue::String(StringId(6))),
        ScalarType::String,
        false,
    );
    let b = q.add_expr(
        ExprKind::Const(ScalarValue::Bytes(BytesId(0))),
        ScalarType::Bytes,
        false,
    );
    let null = q.add_expr(ExprKind::Const(ScalarValue::Null), ScalarType::Null, true);
    let mut values = vec![];
    let mut columns = vec![];
    for (f, a, ty) in [
        (BuiltinFn::Abs, n, ScalarType::Number),
        (BuiltinFn::Floor, n, ScalarType::Number),
        (BuiltinFn::Ceil, n, ScalarType::Number),
        (BuiltinFn::Round, n, ScalarType::Number),
        (BuiltinFn::LowerAscii, s, ScalarType::String),
        (BuiltinFn::UpperAscii, s, ScalarType::String),
        (BuiltinFn::StringLength, s, ScalarType::Number),
        (BuiltinFn::BytesLength, b, ScalarType::Number),
    ] {
        values.push(q.add_expr(
            ExprKind::Call {
                function: f,
                args: vec![a],
            },
            ty,
            false,
        ));
        columns.push(q.add_column(None, ty, false, ColumnOrigin::Derived));
    }
    let low = constant(&mut q, -3.0);
    let high = constant(&mut q, -2.0);
    for e in [
        ExprKind::Between {
            value: n,
            low,
            high,
            negated: false,
        },
        ExprKind::InList {
            value: n,
            items: vec![n, null],
            negated: false,
        },
        ExprKind::InList {
            value: n,
            items: vec![low, null],
            negated: false,
        },
    ] {
        values.push(q.add_expr(e, ScalarType::Bool, true));
        columns.push(q.add_column(None, ScalarType::Bool, true, ColumnOrigin::Derived));
    }
    q.root = q.add_plan(PlanKind::Values {
        columns,
        rows: vec![values],
    });
    p.queries.push(q);
    let expected = vec![vec![
        number(2.5),
        number(-3.0),
        number(-2.0),
        number(-3.0),
        text("Ábc\0d"),
        text("ÁBC\0D"),
        number(5.0),
        number(3.0),
        Value::Integer(1),
        Value::Integer(1),
        Value::Null,
    ]];
    for optimized in [false, true] {
        let mut p = p.clone();
        if optimized {
            optimize(&mut p);
        }
        assert_eq!(
            run(&db(), &compile_query(&p, QueryId(0)).unwrap(), &[]),
            expected
        );
    }
}

#[test]
fn sqlite_index_union_intersection_and_fetch_keep_row_identity() {
    for source_text in [
        "SELECT name FROM users WHERE age >= 30 OR active = false ORDER BY id",
        "SELECT name FROM users WHERE age >= 20 AND active = true ORDER BY id",
    ] {
        let mut p = source(source_text, false);
        p.strings.push("by_active".into());
        p.catalog.tables[0].indexes.push(IndexId(1));
        p.catalog.indexes.push(IndexDef {
            id: IndexId(1),
            table: TableId(0),
            name: StringId(6),
            key_prefix: vec![3],
            columns: vec![IndexColumn {
                field: FieldId(2),
                encoding: KeyEncoding::Bool,
                descending: false,
                nulls: NullOrder::First,
            }],
            included_fields: vec![],
            unique: false,
        });
        let before = run(&db(), &compile_query(&p, QueryId(0)).unwrap(), &[]);
        optimize(&mut p);
        assert!(
            p.queries[0]
                .plans
                .iter()
                .any(|n| matches!(n.kind, PlanKind::FetchRows { .. }))
        );
        assert_eq!(
            run(&db(), &compile_query(&p, QueryId(0)).unwrap(), &[]),
            before,
            "{source_text}"
        );
    }
}

#[test]
fn sqlite_composite_primary_key_bounds_and_delete() {
    let mut p = source("SELECT * FROM users", false);
    p.catalog.tables[0].primary_key = vec![FieldId(2), FieldId(0)];
    let q = &mut p.queries[0];
    let yes = q.add_expr(
        ExprKind::Const(ScalarValue::Bool(true)),
        ScalarType::Bool,
        false,
    );
    let one = constant(q, 1.0);
    let two = constant(q, 2.0);
    q.root = q.add_plan(PlanKind::PrimaryRangeScan {
        table: TableId(0),
        range: KeyRange {
            prefix: vec![yes],
            lower: BoundValue::Exclusive(one),
            upper: BoundValue::Inclusive(two),
        },
        columns: vec![ColumnId(1)],
        residual: None,
        direction: ScanDirection::Reverse,
        skip: 0,
        limit: None,
        delivery: ScanDelivery::BorrowedPairs,
    });
    assert_eq!(
        run(&db(), &compile_query(&p, QueryId(0)).unwrap(), &[]),
        vec![vec![text("Grace")]]
    );
    let q = &mut p.queries[0];
    q.transaction = TransactionMode::ReadWrite;
    q.root = q.add_plan(PlanKind::Delete {
        input: q.root,
        table: TableId(0),
    });
    let db = db();
    run(&db, &compile_query(&p, QueryId(0)).unwrap(), &[]);
    assert_eq!(
        db.query_row("SELECT count(*) FROM users WHERE id = 2", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}
