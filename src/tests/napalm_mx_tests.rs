use crate::napalm_mx_front::parse;
use crate::napalm_mx_ir::*;
use crate::napalm_mx_lowerer::{Parameter, lower};

fn catalog() -> Program {
    let mut p = Program::default();
    p.strings = ["users", "id", "name", "active", "age"]
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
        indexes: vec![],
        stats: TableStats::default(),
    });
    p
}
fn parameters() -> Vec<Parameter> {
    vec![Parameter {
        name: "id".into(),
        ty: ScalarType::Number,
        nullable: false,
    }]
}
fn compile(source: &str) -> Program {
    let mut program = catalog();
    let block = parse(source, 0).unwrap();
    lower(&block, &mut program, &parameters()).unwrap();
    verify_program(&program).unwrap();
    program
}

#[test]
fn mx_select_produces_typed_plan_and_deduplicates_parameters() {
    let p = compile(
        "SELECT name AS label, age + 1 AS next_age FROM users WHERE id = $id OR age = $id ORDER BY age DESC LIMIT 5 OFFSET 2",
    );
    let q = &p.queries[0];
    assert_eq!(q.params.len(), 1);
    assert_eq!(q.params[0].ty, ScalarType::Number);
    assert_eq!(q.transaction, TransactionMode::ReadOnlySnapshot);
    let PlanKind::Project { input, expressions } = &q.plan(q.root).unwrap().kind else {
        panic!()
    };
    assert_eq!(
        p.string(q.column(expressions[0].column).unwrap().name.unwrap()),
        Some("label")
    );
    assert!(q.column(expressions[1].column).unwrap().nullable);
    let PlanKind::Limit {
        input,
        skip: 2,
        limit: Some(5),
    } = &q.plan(*input).unwrap().kind
    else {
        panic!()
    };
    let PlanKind::Sort { keys, .. } = &q.plan(*input).unwrap().kind else {
        panic!()
    };
    assert_eq!(keys[0].direction, SortDirection::Desc);
}

#[test]
fn mx_crud_and_multiple_statements_have_stable_ids() {
    let p = compile(
        "INSERT INTO users (id, name, active) VALUES (1, 'Ada', true), (2, NULL, false); UPDATE users SET age = age + 1 WHERE id = $id; DELETE FROM users WHERE name IS NULL; SELECT * FROM users",
    );
    assert_eq!(p.queries.len(), 4);
    for (i, q) in p.queries.iter().enumerate() {
        assert_eq!(q.id, QueryId(i as u32));
    }
    let q = &p.queries[0];
    let PlanKind::Insert { rows, .. } = &q.plan(q.root).unwrap().kind else {
        panic!()
    };
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].len(), 4);
    assert!(matches!(
        q.expr(rows[0][3].1).unwrap().kind,
        ExprKind::Const(ScalarValue::Null)
    ));
    assert!(matches!(
        p.queries[1].plan(p.queries[1].root).unwrap().kind,
        PlanKind::Update { .. }
    ));
    assert!(matches!(
        p.queries[2].plan(p.queries[2].root).unwrap().kind,
        PlanKind::Delete { .. }
    ));
    assert!(
        p.queries[..3]
            .iter()
            .all(|q| q.transaction == TransactionMode::ReadWrite)
    );
}

#[test]
fn mx_rejects_schema_type_and_parameter_errors_atomically() {
    for source in [
        "SELECT * FROM missing",
        "SELECT missing FROM users",
        "SELECT $missing FROM users",
        "SELECT id + name FROM users",
        "SELECT * FROM users WHERE name",
        "UPDATE users SET active = 1",
        "UPDATE users SET id = NULL",
        "UPDATE users SET id = age",
        "UPDATE users SET id = 1, id = 2",
        "INSERT INTO users (id, id) VALUES (1, 2)",
        "INSERT INTO users (id, active) VALUES (1)",
        "INSERT INTO users (name) VALUES ('x')",
        "INSERT INTO users (id, active) VALUES (id, true)",
        "SELECT 'new string' FROM users; DELETE FROM missing",
    ] {
        let mut p = catalog();
        let strings = p.strings.clone();
        let err = lower(&parse(source, 0).unwrap(), &mut p, &parameters()).unwrap_err();
        assert_eq!(err.code, "MX-S001", "{source}");
        assert_eq!(p.strings, strings, "{source}");
        assert!(p.queries.is_empty(), "{source}");
    }
}

#[test]
fn mx_nullable_boolean_semantics_survive_lowering() {
    let p = compile(
        "SELECT age = NULL AS unknown, name IS NOT NULL AS present FROM users WHERE active AND age > 10",
    );
    let q = &p.queries[0];
    let PlanKind::Project { expressions, .. } = &q.plan(q.root).unwrap().kind else {
        panic!()
    };
    assert!(q.expr(expressions[0].expr).unwrap().nullable);
    assert!(!q.expr(expressions[1].expr).unwrap().nullable);
    assert!(
        expressions
            .iter()
            .all(|p| q.expr(p.expr).unwrap().ty == ScalarType::Bool)
    );
}

#[test]
fn mx_lowered_queries_can_run_through_existing_optimizer() {
    let mut p = compile(
        "SELECT name FROM users WHERE id = $id; UPDATE users SET name = 'Ada' WHERE id = $id",
    );
    crate::napalm_mx_ir_optimizing::optimization_passes()
        .optimize_program(&mut p)
        .unwrap();
    verify_program(&p).unwrap();
    assert!(
        p.queries[0]
            .plans
            .iter()
            .any(|p| matches!(p.kind, PlanKind::PointGet { .. }))
    );
}

#[test]
fn mx_host_ast_can_be_lowered_directly() {
    let source = "const q = [[[ SELECT name FROM users WHERE id = $id ]]];";
    let parsed = crate::napalm_front_main::parse("test", source);
    assert!(!parsed.has_errors());
    let crate::napalm_front::StmtKind::Variable {
        initializer: Some(expr),
        ..
    } = &parsed.module.statements[0].kind
    else {
        panic!()
    };
    let crate::napalm_front::ExprKind::Mx(block) = &expr.kind else {
        panic!()
    };
    let mut p = catalog();
    lower(block, &mut p, &parameters()).unwrap();
    verify_program(&p).unwrap();
}

#[test]
fn mx_host_execution_reports_pending_backend() {
    let source = "const q = [[[SELECT * FROM users]]];";
    let parsed = crate::napalm_front_main::parse("test", source);
    let output = crate::napalm_lowerer::lower_new_program("test", source, "test", &parsed.module);
    assert!(output.has_errors());
    assert!(output.diagnostics.iter().any(|d| d.code == "MX-B001"));
}
