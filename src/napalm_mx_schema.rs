//! Typed schema AST -> SQLite DDL. Schema operations do not enter relational IR.
use crate::napalm_mx_front::{Column, ExprKind, Name, SchemaStatement, TableChange};
use crate::napalm_mx_ir::{ScalarType, UnaryOp};

pub const METADATA: &str = "__napalm_metadata";
pub const MAX_VERSION: u64 = (1 << 53) - 1;

pub fn user_name(name: &str) -> Result<(), String> {
    if name.to_ascii_lowercase().starts_with("__napalm_") {
        Err("the __napalm_ namespace is reserved for database metadata".into())
    } else {
        Ok(())
    }
}

fn quote(name: &Name) -> String {
    format!("\"{}\"", name.value.replace('"', "\"\""))
}
fn object(name: &Name) -> Result<String, String> {
    user_name(&name.value)?;
    Ok(quote(name))
}

pub fn scalar_type(declared: &str) -> Result<ScalarType, String> {
    use ScalarType::*;
    match declared.trim().to_ascii_uppercase().as_str() {
        "NUMBER" | "NUM" | "REAL" | "DOUBLE" | "FLOAT" | "INTEGER" | "INT" | "BIGINT"
        | "NUMERIC" => Ok(Number),
        "STRING" | "TEXT" => Ok(String),
        "BOOL" | "BOOLEAN" => Ok(Bool),
        "BYTES" | "BLOB" => Ok(Bytes),
        _ => Err(format!(
            "unsupported column type {declared:?}; use Number, String, Bool or Bytes"
        )),
    }
}

fn column_sql(column: &Column, primary: bool) -> Result<String, String> {
    let ty = scalar_type(&column.ty.value)?;
    let sql_type = match ty {
        ScalarType::Number => "REAL",
        ScalarType::String => "TEXT",
        ScalarType::Bool => "BOOLEAN",
        ScalarType::Bytes => "BLOB",
        _ => unreachable!(),
    };
    let name = quote(&column.name);
    let mut sql = format!(
        "{name} {sql_type}{}",
        if primary || !column.nullable {
            " NOT NULL"
        } else {
            ""
        }
    );
    if let Some(default) = &column.default {
        let (actual, value) = match &default.value {
            ExprKind::Null => (ScalarType::Null, "NULL".into()),
            ExprKind::Bool(v) => (ScalarType::Bool, if *v { "1" } else { "0" }.into()),
            ExprKind::Number(v) => (ScalarType::Number, v.to_string()),
            ExprKind::String(v) if !v.contains('\0') => {
                (ScalarType::String, format!("'{}'", v.replace('\'', "''")))
            }
            ExprKind::Unary(UnaryOp::Neg, e) => match e.value {
                ExprKind::Number(v) => (ScalarType::Number, (-v).to_string()),
                _ => return Err("DEFAULT requires a scalar literal".into()),
            },
            _ => return Err("unsupported DEFAULT literal".into()),
        };
        if actual == ScalarType::Null {
            if primary || !column.nullable {
                return Err("NOT NULL column cannot default to NULL".into());
            }
        } else if actual != ty {
            return Err(format!("DEFAULT type does not match {}", column.ty.value));
        }
        sql.push_str(&format!(" DEFAULT {value}"));
    }
    if ty == ScalarType::Bool {
        sql.push_str(&format!(" CHECK ({name} IN (0, 1))"));
    }
    Ok(sql)
}

pub fn compile(statement: &SchemaStatement) -> Result<String, String> {
    Ok(match statement {
        SchemaStatement::CreateTable {
            table,
            columns,
            primary,
            if_not_exists,
        } => {
            let table = object(table)?;
            if columns.is_empty() {
                return Err("CREATE TABLE requires columns".into());
            }
            let inline = columns
                .iter()
                .filter(|c| c.primary)
                .map(|c| c.name.clone())
                .collect::<Vec<_>>();
            if inline.len() > 1 || (!inline.is_empty() && !primary.is_empty()) {
                return Err("multiple PRIMARY KEY declarations".into());
            }
            let primary = if primary.is_empty() { &inline } else { primary };
            if primary.is_empty() {
                return Err("CREATE TABLE requires an explicit PRIMARY KEY".into());
            }
            for (i, key) in primary.iter().enumerate() {
                if primary[..i]
                    .iter()
                    .any(|p| p.value.eq_ignore_ascii_case(&key.value))
                    || !columns.iter().any(|c| c.name.value == key.value)
                {
                    return Err(format!(
                        "unknown or duplicate primary key field {}",
                        key.value
                    ));
                }
            }
            let mut fields = vec![];
            for (i, c) in columns.iter().enumerate() {
                if columns[..i]
                    .iter()
                    .any(|p| p.name.value.eq_ignore_ascii_case(&c.name.value))
                {
                    return Err("duplicate column name".into());
                }
                fields.push(column_sql(
                    c,
                    primary.iter().any(|p| p.value == c.name.value),
                )?);
            }
            fields.push(format!(
                "PRIMARY KEY ({})",
                primary.iter().map(quote).collect::<Vec<_>>().join(", ")
            ));
            format!(
                "CREATE TABLE {}{table} ({})",
                if *if_not_exists { "IF NOT EXISTS " } else { "" },
                fields.join(", ")
            )
        }
        SchemaStatement::CreateIndex {
            name,
            table,
            columns,
            unique,
            if_not_exists,
        } => {
            let fields = columns
                .iter()
                .map(|(n, desc)| format!("{} {}", quote(n), if *desc { "DESC" } else { "ASC" }))
                .collect::<Vec<_>>();
            format!(
                "CREATE {}INDEX {}{} ON {} ({})",
                if *unique { "UNIQUE " } else { "" },
                if *if_not_exists { "IF NOT EXISTS " } else { "" },
                object(name)?,
                object(table)?,
                fields.join(", ")
            )
        }
        SchemaStatement::UpdateTable { table, change } => {
            let table = object(table)?;
            let action = match change {
                TableChange::Add(c) => {
                    if c.primary {
                        return Err("ADD COLUMN cannot add a primary key".into());
                    }
                    format!("ADD COLUMN {}", column_sql(c, false)?)
                }
                TableChange::Drop(c) => format!("DROP COLUMN {}", quote(c)),
                TableChange::RenameColumn(old, new) => {
                    format!("RENAME COLUMN {} TO {}", quote(old), quote(new))
                }
                TableChange::Rename(new) => format!("RENAME TO {}", object(new)?),
            };
            format!("ALTER TABLE {table} {action}")
        }
        SchemaStatement::Drop {
            name,
            index,
            if_exists,
        } => format!(
            "DROP {} {}{}",
            if *index { "INDEX" } else { "TABLE" },
            if *if_exists { "IF EXISTS " } else { "" },
            object(name)?
        ),
        SchemaStatement::SetVersion(version) => {
            if *version > MAX_VERSION {
                return Err("version exceeds the exact integer range of Number".into());
            }
            format!("UPDATE \"{METADATA}\" SET version = {version} WHERE id = 1")
        }
    })
}
