#[test]
fn napalm_mx_vm_crud_types_and_snapshot_captures() {
    run_napalm(
        r#"
        function expect(ok) { if (!ok) throw "database assertion failed"; }
        const db: Database = Database.open(":memory:");
        expect(typeof db === "Database" && db instanceof Database && !db.closed);
        db.execute([[[ CREATE TABLE users (id Number NOT NULL, name String NOT NULL, active Bool NOT NULL, data Bytes NOT NULL, note String, PRIMARY KEY (id)) ]]]);
        let name = "O'Reilly; DROP TABLE users; --";
        const data = new ByteBuffer();
        data.append("abc");
        const query: DatabaseQuery = [[[ INSERT INTO users (id, name, active, data, note) VALUES (1, $name, true, $data, null) ]]];
        expect(typeof query === "DatabaseQuery" && query instanceof DatabaseQuery);
        expect(db === db && query === query && db !== query && query.toString() === "<DatabaseQuery>");
        name = "changed";
        data.append("def");
        System.gc();
        expect(db.execute(query) === 1);
        const rows = db.execute([[[ SELECT * FROM users ]]]);
        expect(rows.length === 1 && rows[0].id === 1 && rows[0].active === true);
        expect(rows[0].name === "O'Reilly; DROP TABLE users; --" && rows[0].data.toUtf8() === "abc" && rows[0].note === null);
        expect(db.execute([[[ UPDATE users SET name = $name WHERE id = 1 ]]]) === 1);
        expect(db.execute([[[ SELECT name FROM users WHERE id = 1 ]]])[0].name === "changed");
        expect(db.execute([[[ DELETE FROM users WHERE id = 1 ]]]) === 1);
        expect(db.execute([[[ SELECT * FROM users ]]]).length === 0);
        expect(db.close() === null && db.closed && db.close() === null);
        const result = "ok";
    "#,
        "ok",
    );
}

#[test]
fn napalm_mx_vm_closures_cache_gc_and_indexes() {
    run_napalm_checked(
        r#"
        const db = Database.open(":memory:");
        db.execute([[[ CREATE TABLE items (id Number NOT NULL, value Number NOT NULL, PRIMARY KEY (id)) ]]]);
        function makeQuery(id) { return () => [[[ SELECT value AS answer FROM items WHERE id = $id ]]]; }
        const captured = makeQuery(42);
        for (let id = 0; id < 100; id++) {
            const value = id * 2;
            db.execute([[[ INSERT INTO items (id, value) VALUES ($id, $value) ]]]);
            if (id % 10 === 0) System.gc();
        }
        let sum = 0;
        for (let id = 0; id < 100; id++) {
            sum += db.execute([[[ SELECT value FROM items WHERE id = $id ]]])[0].value;
            if (id % 10 === 0) System.gc();
        }
        const before = db.execute(captured())[0].answer;
        db.execute([[[ CREATE INDEX item_value ON items (value) ]]]);
        const after = db.execute(captured())[0].answer;
        const sorted = db.execute([[[ SELECT value FROM items WHERE value >= 194 ORDER BY value DESC LIMIT 2 ]]]);
        db.close();
        const result = [sum, before, after, sorted[0].value, sorted[1].value];
    "#,
        "[9900, 84, 84, 198, 196]",
        true,
    );
}

#[test]
fn napalm_mx_vm_atomic_blocks_and_errors() {
    run_napalm(
        r#"
        let errors = 0;
        const db = Database.open(":memory:");
        db.execute([[[ CREATE TABLE items (id Number NOT NULL, name String, PRIMARY KEY (id)) ]]]);
        try { db.execute([[[ INSERT INTO items (id, name) VALUES (1, 'first'); INSERT INTO items (id, name) VALUES (1, 'duplicate') ]]]); }
        catch (e) { errors++; }
        const empty = db.execute([[[ SELECT * FROM items ]]]).length;
        const rows = db.execute([[[ INSERT INTO items (id, name) VALUES (2, 'ok'); SELECT name FROM items ]]]);
        let name = null;
        db.execute([[[ UPDATE items SET name = $name WHERE id = 2 ]]]);
        const nullRows = db.execute([[[ SELECT * FROM items WHERE name IS NULL ]]]).length;
        try { db.execute([[[ SELECT missing FROM items ]]]); } catch (e) { errors++; }
        try { const unsupported = []; const q = [[[ SELECT * FROM items WHERE id = $unsupported ]]]; } catch (e) { errors++; }
        try { db.execute([[[ SELECT id AS same, name AS same FROM items ]]]); } catch (e) { errors++; }
        db.close();
        try { db.execute([[[ SELECT * FROM items ]]]); } catch (e) { errors++; }
        const result = [errors, empty, rows[0].name, nullRows];
    "#,
        "[5, 0, ok, 1]",
    );
}

#[test]
fn napalm_mx_vm_file_reopen() {
    let directory = FsTestDirectory::new();
    run_napalm(
        &directory.source(
            r#"
        const path = Path.join(__ROOT__, "data.sqlite");
        const db = Database.open(path);
        db.execute([[[ CREATE TABLE items (id Number NOT NULL, name String NOT NULL, PRIMARY KEY (id)) ]]]);
        db.execute([[[ INSERT INTO items (id, name) VALUES (1, 'persistent') ]]]);
        db.close();
        const reopened = Database.open(path);
        const name = reopened.execute([[[ SELECT name FROM items ]]])[0].name;
        reopened.close();
        Fs.removeFile(path);
        const result = name;
    "#,
        ),
        "persistent",
    );
}

#[test]
fn napalm_mx_vm_burning_roundtrip() {
    let (program, entry, builtins) = compile_typescript_module(
        r#"
        const db = Database.open(":memory:");
        db.execute([[[ CREATE TABLE items (id Number NOT NULL, name String NOT NULL, PRIMARY KEY (id)) ]]]);
        const name = "burning";
        const rows = db.execute([[[ INSERT INTO items (id, name) VALUES (1, $name); SELECT name FROM items ]]]);
        db.close();
        const result = rows[0].name;
    "#,
    );
    let bootstrap = crate::napalm_burning::Bootstrap {
        entry,
        module_name: "mx".into(),
        builtins,
    };
    let encoded = crate::napalm_burning::encode(&program, &bootstrap).unwrap();
    let artifact = crate::napalm_burning::decode(&encoded).unwrap();
    let mut vm = source_vm(&artifact.program, &artifact.bootstrap.builtins);
    vm.run_module(artifact.bootstrap.entry).unwrap();
    let value = last_source_variable(&vm);
    assert_eq!(
        HostContext { heap: &mut vm.heap }.format_value(value),
        "burning"
    );
}

#[test]
fn napalm_mx_vm_template_nesting_and_method_captures() {
    let expression = (0..50).fold("$id".to_owned(), |e, _| format!("({e} + 1)"));
    let source = r#"
        const db = Database.open(":memory:");
        db.execute([[[ CREATE TABLE items (id Number NOT NULL, PRIMARY KEY (id)) ]]]);
        db.execute([[[ INSERT INTO items (id) VALUES (1) ]]]);
        const id = 2;
        class Queries {
            query() { return [[[ SELECT __EXPR__ AS answer FROM items WHERE id = 1 ]]]; }
        }
        const queries = new Queries();
        const result = db.execute(queries.query())[0].answer;
    "#
    .replace("__EXPR__", &expression);
    run_napalm(&source, "52");
}

#[test]
fn napalm_mx_vm_schema_migrations_and_version_are_atomic() {
    run_napalm(
        r#"
        function expect(ok) { if (!ok) throw "migration assertion failed"; }
        const db = Database.open(":memory:");
        expect(db.getVersion() === 0);
        db.execute([[[
            CREATE TABLE users (id Number PRIMARY KEY, name String NOT NULL);
            INSERT INTO users (id, name) VALUES (1, 'Ada');
            CREATE UNIQUE INDEX users_name ON users (name DESC);
            SET VERSION 1;
        ]]]);
        const all = [[[ SELECT * FROM users ]]];
        expect(db.execute(all)[0].name === 'Ada');
        for (let i = 0; i < 2; i++) {
            if (db.getVersion() < 2) {
                db.execute([[[
                    UPDATE TABLE users ADD COLUMN active Bool NOT NULL DEFAULT true;
                    UPDATE TABLE users ADD COLUMN note String DEFAULT 'it''s new';
                    UPDATE TABLE users RENAME COLUMN note TO comment;
                    SET VERSION 2;
                ]]]);
            }
        }
        const row = db.execute(all)[0];
        expect(row.active === true && row.comment === "it's new" && db.getVersion() === 2);
        let failed = false;
        try {
            db.execute([[[
                UPDATE TABLE users ADD COLUMN score Number DEFAULT -1;
                SET VERSION 3;
                UPDATE users SET name = 'lost' WHERE id = 1;
                SELECT missing FROM users;
            ]]]);
        } catch (e) { failed = true; }
        expect(failed && db.getVersion() === 2 && db.execute(all)[0].name === 'Ada');
        // The failed migration's column must also have rolled back.
        db.execute([[[
            ALTER TABLE users ADD COLUMN score Number DEFAULT -2;
            UPDATE TABLE users DROP COLUMN comment;
            UPDATE TABLE users RENAME TO people;
            DROP INDEX users_name;
            CREATE INDEX IF NOT EXISTS people_score ON people (score);
            CREATE INDEX IF NOT EXISTS people_score ON people (score);
            SET VERSION 3;
        ]]]);
        expect(db.execute([[[ SELECT score FROM people ]]])[0].score === -2);
        db.execute([[[ CREATE TABLE IF NOT EXISTS people (id Number PRIMARY KEY); DROP TABLE people; DROP TABLE IF EXISTS people; ]]]);
        expect(db.getVersion() === 3);
        db.close();
        const result = "ok";
    "#,
        "ok",
    );
}

#[test]
fn napalm_mx_vm_schema_errors_leave_no_partial_schema_or_metadata() {
    run_napalm(
        r#"
        const db = Database.open(":memory:");
        let errors = 0;
        try { db.execute([[[ CREATE TABLE items (id Number PRIMARY KEY); SET VERSION 1; INSERT INTO items (id) VALUES (1), (1); ]]]); } catch (e) { errors++; }
        // CREATE succeeds again because the earlier block was rolled back.
        db.execute([[[ CREATE TABLE items (id Number PRIMARY KEY) ]]]);
        try { db.execute([[[ CREATE INDEX bad ON items (missing) ]]]); } catch (e) { errors++; }
        try { db.execute([[[ CREATE TABLE invalid (name String) ]]]); } catch (e) { errors++; }
        try { db.execute([[[ UPDATE TABLE items ADD COLUMN flag Bool DEFAULT 2 ]]]); } catch (e) { errors++; }
        try { db.execute([[[ CREATE TABLE __napalm_user (id Number PRIMARY KEY) ]]]); } catch (e) { errors++; }
        try { db.execute([[[ DROP TABLE __NAPALM_METADATA ]]]); } catch (e) { errors++; }
        try { db.execute([[[ UPDATE __napalm_metadata SET version = 99 WHERE id = 1 ]]]); } catch (e) { errors++; }
        try { db.execute([[[ UPDATE TABLE items RENAME TO __napalm_metadata ]]]); } catch (e) { errors++; }
        try { db.setVersion(-1); } catch (e) { errors++; }
        try { db.setVersion(1.5); } catch (e) { errors++; }
        try { db.setVersion("2"); } catch (e) { errors++; }
        try { db.execute([[[ SET VERSION 9007199254740992 ]]]); } catch (e) { errors++; }
        const initial = db.getVersion();
        db.setVersion(4);
        const version = db.getVersion();
        db.close();
        try { db.getVersion(); } catch (e) { errors++; }
        const result = [errors, initial, version];
    "#,
        "[13, 0, 4]",
    );
}

#[test]
fn napalm_mx_vm_version_persists_across_connections() {
    let directory = FsTestDirectory::new();
    run_napalm(&directory.source(r#"
        function expect(ok) { if (!ok) throw "version assertion failed"; }
        const path = Path.join(__ROOT__, "version.sqlite");
        const first = Database.open(path);
        const second = Database.open(path);
        expect(first.getVersion() === 0 && second.getVersion() === 0);
        first.execute([[[ CREATE TABLE pair (a Number, b String, PRIMARY KEY (a, b)); SET VERSION 2; ]]]);
        expect(second.getVersion() === 2);
        second.setVersion(7);
        first.close(); second.close();
        const reopened = Database.open(path);
        expect(reopened.getVersion() === 7);
        reopened.execute([[[ INSERT INTO pair (a, b) VALUES (1, 'x') ]]]);
        reopened.close();
        Fs.removeFile(path);
        const result = "ok";
    "#), "ok");
}
