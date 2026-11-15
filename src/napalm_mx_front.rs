//! Independent frontend for the database language inside `[[[ ... ]]]`.
//! Keywords are case-insensitive; names are case-sensitive. No host expressions
//! are evaluated here: `$name` denotes a parameter supplied to the MX lowerer.
use crate::napalm_front::{Diagnostic, Span, Spanned};
use crate::napalm_mx_ir::{BinaryOp, UnaryOp};
use logos::Logos;

pub type Name = Spanned<String>;
pub type Expr = Spanned<ExprKind>;
#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Field(Name),
    Param(Name),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    IsNull(Box<Expr>, bool),
}
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub statements: Vec<Spanned<Statement>>,
    pub span: Span,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    pub expr: Expr,
    pub alias: Option<Name>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Select {
        table: Name,
        projection: Option<Vec<Projection>>,
        predicate: Option<Expr>,
        order: Vec<(Expr, bool)>,
        limit: Option<u64>,
        offset: u64,
    },
    Insert {
        table: Name,
        fields: Vec<Name>,
        rows: Vec<Vec<Expr>>,
    },
    Update {
        table: Name,
        assignments: Vec<(Name, Expr)>,
        predicate: Option<Expr>,
    },
    Delete {
        table: Name,
        predicate: Option<Expr>,
    },
}

#[derive(Logos, Clone, Copy, Debug, PartialEq)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip(r"--[^\r\n]*|//[^\r\n]*", allow_greedy = true))]
enum Kind {
    #[regex(r"[A-Za-z_][A-Za-z_0-9]*")]
    Word,
    #[regex(r"\$[A-Za-z_][A-Za-z_0-9]*")]
    Param,
    #[regex(r"[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?")]
    Number,
    #[token("'", quoted)]
    #[token("\"", quoted)]
    String,
    #[token("/*", comment)]
    Comment,
    #[regex(r"!=|<>|<=|>=|==|[(),;.*+/%=<>!-]")]
    Symbol,
    Eof,
}
fn quoted(lex: &mut logos::Lexer<'_, Kind>) -> bool {
    let quote = lex.slice().as_bytes()[0];
    if let Some(end) = quote_end(lex.remainder().as_bytes(), 0, quote) {
        lex.bump(end);
        true
    } else {
        lex.bump(lex.remainder().len());
        false
    }
}
// Returns the position just after the closing quote. SQL doubled quotes escape it.
fn quote_end(bytes: &[u8], mut i: usize, quote: u8) -> Option<usize> {
    while i < bytes.len() {
        if bytes[i] == quote {
            if bytes.get(i + 1) == Some(&quote) {
                i += 2;
            } else {
                return Some(i + 1);
            }
        } else {
            i += 1;
        }
    }
    None
}
fn comment_end(bytes: &[u8], mut i: usize) -> Option<usize> {
    let mut depth = 1usize;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"/*") {
            depth += 1;
            i += 2;
        } else if bytes[i..].starts_with(b"*/") {
            depth -= 1;
            i += 2;
            if depth == 0 {
                return Some(i);
            }
        } else {
            i += 1;
        }
    }
    None
}
fn comment(lex: &mut logos::Lexer<'_, Kind>) -> bool {
    if let Some(end) = comment_end(lex.remainder().as_bytes(), 0) {
        lex.bump(end);
        true
    } else {
        lex.bump(lex.remainder().len());
        false
    }
}
/// Scan after an opening `[[[`. The returned length includes `]]]`. Host lexers
/// must consume the whole island, including comments and quoted delimiters.
pub fn block_end(body: &str) -> Option<usize> {
    let b = body.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(b"]]]") {
            return Some(i + 3);
        }
        if b[i] == b'\'' || b[i] == b'"' {
            i = quote_end(b, i + 1, b[i])?;
        } else if b[i..].starts_with(b"/*") {
            i = comment_end(b, i + 2)?;
        } else if b[i..].starts_with(b"--") || b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    None
}

type Result<T> = std::result::Result<T, Diagnostic>;
#[derive(Clone, Copy)]
struct Token<'a> {
    kind: Kind,
    text: &'a str,
    span: Span,
}
struct Parser<'a> {
    tokens: Vec<Token<'a>>,
    pos: usize,
    depth: usize,
}

/// Parse a block body (without delimiters), preserving absolute UTF-8 byte spans.
pub fn parse(body: &str, offset: usize) -> Result<Block> {
    let mut lexer = Kind::lexer(body);
    let mut tokens = Vec::new();
    while let Some(kind) = lexer.next() {
        let span = Span::new(offset + lexer.span().start, offset + lexer.span().end);
        let kind = kind.map_err(|_| {
            Diagnostic::error(
                "MX-L001",
                "invalid MX token or unterminated string/comment",
                span,
            )
        })?;
        if kind != Kind::Comment {
            tokens.push(Token {
                kind,
                text: lexer.slice(),
                span,
            });
        }
    }
    tokens.push(Token {
        kind: Kind::Eof,
        text: "",
        span: Span::empty(offset + body.len()),
    });
    let mut p = Parser {
        tokens,
        pos: 0,
        depth: 0,
    };
    let mut statements = Vec::new();
    while p.peek().kind != Kind::Eof {
        let start = p.peek().span;
        let statement = p.statement()?;
        let span = start.join(p.tokens[p.pos - 1].span);
        statements.push(Spanned::new(statement, span));
        if !p.eat(";") && p.peek().kind != Kind::Eof {
            return p.err("expected `;` between MX statements");
        }
    }
    if statements.is_empty() {
        return p.err("expected an MX statement");
    }
    Ok(Block {
        statements,
        span: Span::new(offset, offset + body.len()),
    })
}
impl<'a> Parser<'a> {
    fn peek(&self) -> Token<'a> {
        self.tokens[self.pos]
    }
    fn err<T>(&self, message: &str) -> Result<T> {
        Err(Diagnostic::error("MX-P001", message, self.peek().span))
    }
    fn eat(&mut self, text: &str) -> bool {
        if self.peek().kind != Kind::Eof && self.peek().text.eq_ignore_ascii_case(text) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, text: &str) -> Result<()> {
        if self.eat(text) {
            Ok(())
        } else {
            self.err(&format!("expected `{text}`"))
        }
    }
    fn name(&mut self) -> Result<Name> {
        let t = self.peek();
        if t.kind != Kind::Word {
            return self.err("expected a name");
        }
        self.pos += 1;
        Ok(Spanned::new(t.text.into(), t.span))
    }
    fn list<T>(&mut self, mut item: impl FnMut(&mut Self) -> Result<T>) -> Result<Vec<T>> {
        let mut out = vec![item(self)?];
        while self.eat(",") {
            out.push(item(self)?);
        }
        Ok(out)
    }
    fn predicate(&mut self) -> Result<Option<Expr>> {
        if self.eat("where") {
            Ok(Some(self.expr(0)?))
        } else {
            Ok(None)
        }
    }
    fn integer(&mut self) -> Result<u64> {
        let t = self.peek();
        if t.kind != Kind::Number {
            return self.err("expected an unsigned integer");
        }
        let value = t.text.parse().map_err(|_| {
            Diagnostic::error("MX-P002", "expected an unsigned 64-bit integer", t.span)
        })?;
        self.pos += 1;
        Ok(value)
    }
    fn statement(&mut self) -> Result<Statement> {
        if self.eat("select") {
            let projection = if self.eat("*") {
                None
            } else {
                Some(self.list(|p| {
                    let expr = p.expr(0)?;
                    let alias = if p.eat("as") { Some(p.name()?) } else { None };
                    Ok(Projection { expr, alias })
                })?)
            };
            self.expect("from")?;
            let table = self.name()?;
            let predicate = self.predicate()?;
            let order = if self.eat("order") {
                self.expect("by")?;
                self.list(|p| {
                    let expr = p.expr(0)?;
                    let desc = p.eat("desc");
                    if !desc {
                        p.eat("asc");
                    }
                    Ok((expr, desc))
                })?
            } else {
                Vec::new()
            };
            let limit = if self.eat("limit") {
                Some(self.integer()?)
            } else {
                None
            };
            let offset = if self.eat("offset") {
                self.integer()?
            } else {
                0
            };
            Ok(Statement::Select {
                table,
                projection,
                predicate,
                order,
                limit,
                offset,
            })
        } else if self.eat("insert") {
            self.expect("into")?;
            let table = self.name()?;
            self.expect("(")?;
            let fields = self.list(Self::name)?;
            self.expect(")")?;
            self.expect("values")?;
            let rows = self.list(|p| {
                p.expect("(")?;
                let row = p.list(|p| p.expr(0))?;
                p.expect(")")?;
                Ok(row)
            })?;
            Ok(Statement::Insert {
                table,
                fields,
                rows,
            })
        } else if self.eat("update") {
            let table = self.name()?;
            self.expect("set")?;
            let assignments = self.list(|p| {
                let field = p.name()?;
                p.expect("=")?;
                Ok((field, p.expr(0)?))
            })?;
            Ok(Statement::Update {
                table,
                assignments,
                predicate: self.predicate()?,
            })
        } else if self.eat("delete") {
            self.expect("from")?;
            let table = self.name()?;
            Ok(Statement::Delete {
                table,
                predicate: self.predicate()?,
            })
        } else {
            self.err("expected SELECT, INSERT, UPDATE or DELETE")
        }
    }
    fn expr(&mut self, min: u8) -> Result<Expr> {
        if self.depth >= 128 {
            return self.err("MX expression nesting limit exceeded");
        }
        self.depth += 1;
        let result = self.expr_inner(min);
        self.depth -= 1;
        result
    }
    fn expr_inner(&mut self, min: u8) -> Result<Expr> {
        let t = self.peek();
        let mut left = if self.eat("(") {
            let mut e = self.expr(0)?;
            self.expect(")")?;
            e.span = t.span.join(self.tokens[self.pos - 1].span);
            e
        } else if self.eat("-") || self.eat("not") || self.eat("!") {
            let op = if t.text == "-" {
                UnaryOp::Neg
            } else {
                UnaryOp::Not
            };
            let rhs = self.expr(if op == UnaryOp::Not { 3 } else { 6 })?;
            let span = t.span.join(rhs.span);
            Spanned::new(ExprKind::Unary(op, Box::new(rhs)), span)
        } else {
            let value = match t.kind {
                Kind::Number => {
                    let n: f64 = t
                        .text
                        .parse()
                        .map_err(|_| Diagnostic::error("MX-P003", "invalid number", t.span))?;
                    if !n.is_finite() {
                        return self.err("number is out of range");
                    }
                    ExprKind::Number(n)
                }
                Kind::String => {
                    let q = &t.text[..1];
                    ExprKind::String(t.text[1..t.text.len() - 1].replace(&q.repeat(2), q))
                }
                Kind::Param => ExprKind::Param(Spanned::new(t.text[1..].into(), t.span)),
                Kind::Word if t.text.eq_ignore_ascii_case("null") => ExprKind::Null,
                Kind::Word
                    if t.text.eq_ignore_ascii_case("true")
                        || t.text.eq_ignore_ascii_case("false") =>
                {
                    ExprKind::Bool(t.text.eq_ignore_ascii_case("true"))
                }
                Kind::Word => ExprKind::Field(Spanned::new(t.text.into(), t.span)),
                _ => return self.err("expected an MX expression"),
            };
            self.pos += 1;
            Spanned::new(value, t.span)
        };
        loop {
            if min <= 3 && self.eat("is") {
                let negated = self.eat("not");
                self.expect("null")?;
                let span = left.span.join(self.tokens[self.pos - 1].span);
                left = Spanned::new(ExprKind::IsNull(Box::new(left), negated), span);
                continue;
            }
            let (bp, op) = match self.peek().text.to_ascii_lowercase().as_str() {
                "or" => (1, BinaryOp::Or),
                "and" => (2, BinaryOp::And),
                "=" | "==" => (3, BinaryOp::Eq),
                "!=" | "<>" => (3, BinaryOp::Ne),
                "<" => (3, BinaryOp::Lt),
                "<=" => (3, BinaryOp::Le),
                ">" => (3, BinaryOp::Gt),
                ">=" => (3, BinaryOp::Ge),
                "+" => (4, BinaryOp::Add),
                "-" => (4, BinaryOp::Sub),
                "*" => (5, BinaryOp::Mul),
                "/" => (5, BinaryOp::Div),
                "%" => (5, BinaryOp::Rem),
                _ => break,
            };
            if bp < min {
                break;
            }
            self.pos += 1;
            let right = self.expr(bp + 1)?;
            let span = left.span.join(right.span);
            left = Spanned::new(ExprKind::Binary(op, Box::new(left), Box::new(right)), span);
        }
        Ok(left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_front::{self, ExprKind as HostExpr, StmtKind};

    #[test]
    fn mx_island_keeps_strings_comments_and_host_tokens_separate() {
        let source = "const q = [[[ SELECT 'á]]]''x' AS text FROM users /* ]]] /* nested */ */ -- ]]]\n WHERE id = $id; ]]];\nconst after = 7;";
        let out = crate::napalm_front_main::parse("test", source);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.module.statements.len(), 2);
        let StmtKind::Variable {
            initializer: Some(expr),
            ..
        } = &out.module.statements[0].kind
        else {
            panic!()
        };
        let HostExpr::Mx(block) = &expr.kind else {
            panic!()
        };
        assert_eq!(
            &source[expr.span.as_range()],
            "[[[ SELECT 'á]]]''x' AS text FROM users /* ]]] /* nested */ */ -- ]]]\n WHERE id = $id; ]]]"
        );
        let Statement::Select {
            projection: Some(projection),
            predicate: Some(predicate),
            ..
        } = &block.statements[0].value
        else {
            panic!()
        };
        assert_eq!(projection[0].expr.value, ExprKind::String("á]]]'x".into()));
        assert_eq!(&source[predicate.span.as_range()], "id = $id");
    }

    #[test]
    fn mx_legacy_front_and_template_interpolation_delegate_to_mx() {
        let old = napalm_front::parse("test", "var q = [[[SELECT * FROM users]]]\n");
        assert!(!old.has_errors(), "{:?}", old.diagnostics);
        let source = "const text = `result ${[[[ SELECT id FROM users -- }\n ]]]}`;";
        let out = crate::napalm_front_main::parse("test", source);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
    }

    #[test]
    fn mx_preserves_double_bracket_arrays_and_host_strings() {
        let out = crate::napalm_front_main::parse(
            "test",
            "const s = '[[[not a query]]]'; const a = [[1], [2]]; const b = [ [[1]]];",
        );
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let out = crate::napalm_front_main::parse("test", "const a = [[[1], [2]]];");
        assert!(out.has_errors());
        let old = napalm_front::parse("test", "var a = [[1], [2]]\n");
        assert!(!old.has_errors(), "{:?}", old.diagnostics);
    }

    #[test]
    fn mx_requires_three_closing_brackets() {
        assert_eq!(block_end(" SELECT * FROM users ]]"), None);
        let body = " SELECT ']]]' FROM users /* ]]] */ ]]] tail";
        let end = block_end(body).unwrap();
        assert_eq!(&body[end..], " tail");
        let out = crate::napalm_front_main::parse("test", "const q = [[[SELECT * FROM users]];");
        assert!(out.diagnostics.iter().any(|d| d.code == "MX-L002"));
    }

    #[test]
    fn mx_precedence_and_not_follow_sql_rules() {
        let block = parse(
            "SELECT 1 + 2 * 3 FROM users WHERE NOT id = 1 OR active AND false",
            10,
        )
        .unwrap();
        let Statement::Select {
            projection: Some(p),
            predicate: Some(e),
            ..
        } = &block.statements[0].value
        else {
            panic!()
        };
        let ExprKind::Binary(BinaryOp::Add, _, right) = &p[0].expr.value else {
            panic!()
        };
        assert!(matches!(right.value, ExprKind::Binary(BinaryOp::Mul, ..)));
        let ExprKind::Binary(BinaryOp::Or, left, right) = &e.value else {
            panic!()
        };
        assert!(matches!(right.value, ExprKind::Binary(BinaryOp::And, ..)));
        let ExprKind::Unary(UnaryOp::Not, inner) = &left.value else {
            panic!()
        };
        assert!(matches!(inner.value, ExprKind::Binary(BinaryOp::Eq, ..)));
    }

    #[test]
    fn mx_rejects_malformed_input_with_absolute_spans() {
        for source in [
            "",
            "SELECT * users",
            "UPDATE users SET id =",
            "SELECT * FROM users garbage",
            "SELECT * FROM users LIMIT 1.5",
            "SELECT 1e999 FROM users",
            "SELECT 'oops FROM users",
            "SELECT * FROM users /*",
        ] {
            let e = parse(source, 20).unwrap_err();
            assert!(e.labels[0].span.start >= 20, "{source}");
            assert!(e.labels[0].span.end <= 20 + source.len(), "{source}");
        }
        for source in [
            "const q = [[[SELECT * FROM users",
            "const q = [[[SELECT ']]]' FROM users",
            "const q = [[[SELECT * FROM users /* ]]] */",
        ] {
            assert!(crate::napalm_front_main::parse("test", source).has_errors());
        }
    }

    #[test]
    fn mx_bounds_recursive_expressions() {
        let source = format!("SELECT {}1{} FROM users", "(".repeat(200), ")".repeat(200));
        assert!(
            parse(&source, 0)
                .unwrap_err()
                .message
                .contains("nesting limit")
        );
    }
}
