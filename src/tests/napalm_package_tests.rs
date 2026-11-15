// `.orange` packages: Package.package builder, Package.unpackage, and the
// tar + Zstandard on-disk format. Every program runs twice (normal and
// `--build` pipelines), so these scripts must be re-runnable.

const PACKAGE_TEST_PRELUDE: &str = r#"
    let assertion = 0;
    function expect(ok) {
        assertion++;
        if (!ok) { System.print(`package assertion ${assertion} failed`); throw "package assertion failed"; }
    }
    function rejects(action) {
        let failed = false;
        try { action(); } catch (error) { failed = true; }
        expect(failed);
    }
    function contains(list, value) {
        for (const item of list) { if (item === value) return true; }
        return false;
    }
"#;

fn package_test_source(directory: &FsTestDirectory, body: &str) -> String {
    directory.source(&format!("{PACKAGE_TEST_PRELUDE}\n{body}"))
}

#[test]
fn napalm_package_roundtrip_preserves_layout() {
    let directory = FsTestDirectory::new();
    run_napalm_checked(&package_test_source(&directory, r#"
        const root = __ROOT__;
        const src = Path.join(root, "src");
        Fs.mkdirAll(Path.join(src, "nested", "deep"));
        Fs.mkdirAll(Path.join(src, "empty"));
        Fs.writeText(Path.join(src, "main.napalm"), "System.print(1);");
        Fs.writeText(Path.join(src, "nested", "deep", "leaf.txt"), "leaf");
        Fs.writeText(Path.join(root, "README.md"), "readme");
        Fs.writeText(Path.join(root, "LICENSE"), "license");
        Fs.writeText(Path.join(root, "notes.txt"), "notes");

        const dist = Path.join(root, "dist");
        const pkg: Package = Package.package("demo.orange");
        expect(typeof pkg === "Package" && pkg instanceof Package && !(pkg instanceof File));
        expect(pkg.addFile(Path.join(root, "README.md")) === pkg);
        const chained = pkg
            .addFiles([Path.join(root, "LICENSE"), Path.join(root, "notes.txt")])
            .addFolder(src)
            .setPackageDest(dist);
        expect(chained === pkg);
        const created = pkg.finish();
        expect(created === Path.join(dist, "demo.orange"));
        const bytes = Fs.readBytes(created);
        // Zstandard frame magic number.
        expect(bytes[0] === 0x28 && bytes[1] === 0xB5 && bytes[2] === 0x2F && bytes[3] === 0xFD);

        // The `.orange` extension is optional when unpacking.
        const target = Path.join(root, "extracted");
        const files = Package.unpackage(Path.join(dist, "demo"), target);
        expect(files.length === 5);
        expect(contains(files, Path.join(target, "README.md")));
        expect(Fs.readText(Path.join(target, "README.md")) === "readme");
        expect(Fs.readText(Path.join(target, "LICENSE")) === "license");
        expect(Fs.readText(Path.join(target, "notes.txt")) === "notes");

        // Folders keep their path relative to the working directory, without
        // leading `..` components (or just their name when on another root).
        const relative = Path.relative(Process.cwd(), src);
        let folder = target;
        if (Path.isAbsolute(relative)) {
            folder = Path.join(target, "src");
        } else {
            for (const part of relative.replaceAll("\\", "/").split("/")) {
                if (part !== "..") folder = Path.join(folder, part);
            }
        }
        expect(Fs.readText(Path.join(folder, "main.napalm")) === "System.print(1);");
        expect(Fs.readText(Path.join(folder, "nested", "deep", "leaf.txt")) === "leaf");
        expect(contains(files, Path.join(folder, "nested", "deep", "leaf.txt")));
        expect(Fs.stat(Path.join(folder, "empty")).isDirectory);

        // finish() can run again and replaces the previous package; the same
        // package unpacks with an explicit extension too.
        expect(pkg.finish() === created);
        expect(Package.unpackage(created, target).length === 5);
        const result = "ok";
    "#), "ok", false);
}

#[test]
fn napalm_package_excludes_itself_from_packaged_folders() {
    let directory = FsTestDirectory::new();
    run_napalm_checked(&package_test_source(&directory, r#"
        const root = __ROOT__;
        const app = Path.join(root, "app");
        Fs.mkdirAll(app);
        Fs.writeText(Path.join(app, "index.napalm"), "index");
        // The package is written inside the folder being packaged, twice: the
        // second build must not contain the first one.
        const pkg = Package.package("self").addFolder(app).setPackageDest(app);
        pkg.finish();
        const created = pkg.finish();
        const files = Package.unpackage(created, Path.join(root, "out"));
        expect(files.length === 1);
        expect(!files[0].endsWith(".orange"));
        // Explicitly adding the package being written is an error.
        rejects(() => Package.package("self").addFile(created).setPackageDest(app).finish());
        const result = "ok";
    "#), "ok", false);
}

#[test]
fn napalm_package_rejects_invalid_input() {
    let directory = FsTestDirectory::new();
    run_napalm(&package_test_source(&directory, r#"
        const root = __ROOT__;
        const file = Path.join(root, "a.txt");
        const other = Path.join(root, "nested", "a.txt");
        Fs.mkdirAll(Path.join(root, "nested"));
        Fs.writeText(file, "a");
        Fs.writeText(other, "other a");
        Fs.writeText(Path.join(root, "b.txt"), "b");

        rejects(() => Package.package(""));
        rejects(() => Package.package(".orange"));
        rejects(() => Package.package("deps/name"));
        rejects(() => Package.package("deps\\name"));
        rejects(() => Package.package(42));

        const pkg = Package.package("checks").setPackageDest(Path.join(root, "dist"));
        rejects(() => pkg.finish()); // empty package
        rejects(() => pkg.addFile(root)); // a folder
        rejects(() => pkg.addFolder(file)); // a file
        rejects(() => pkg.addFile(Path.join(root, "missing.txt")));
        rejects(() => pkg.addFolder(Path.join(root, "missing")));
        rejects(() => pkg.addFiles("a.txt"));
        rejects(() => pkg.addFiles([file, 7]));
        rejects(() => pkg.setPackageDest(""));
        pkg.addFile(file);
        // Two files cannot share a name in the archive root.
        rejects(() => pkg.addFile(file));
        rejects(() => pkg.addFile(other));
        // addFiles is atomic: nothing is added when one path is invalid.
        rejects(() => pkg.addFiles([Path.join(root, "b.txt"), Path.join(root, "missing.txt")]));
        pkg.addFiles([Path.join(root, "b.txt")]);
        const created = pkg.finish();
        expect(Package.unpackage(created, Path.join(root, "out")).length === 2);

        // The `.orange` extension is mandatory on disk.
        Fs.writeBytes(Path.join(root, "plain.tar.zst"), Fs.readBytes(created));
        rejects(() => Package.unpackage(Path.join(root, "plain.tar.zst")));
        rejects(() => Package.unpackage(Path.join(root, "missing")));
        Fs.writeText(Path.join(root, "corrupt.orange"), "not a package");
        rejects(() => Package.unpackage(Path.join(root, "corrupt"), Path.join(root, "out")));
        const result = "ok";
    "#), "ok");
}

#[test]
fn napalm_package_unpackage_rejects_links_and_escaping_entries() {
    fn write_orange(path: &std::path::Path, build: impl FnOnce(&mut tar::Builder<Vec<u8>>)) {
        let mut builder = tar::Builder::new(Vec::new());
        build(&mut builder);
        let tar = builder.into_inner().unwrap();
        std::fs::write(path, zstd::bulk::compress(&tar, 3).unwrap()).unwrap();
    }
    fn file_header(size: usize) -> tar::Header {
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_size(size as u64);
        header.set_mode(0o644);
        header
    }

    let directory = FsTestDirectory::new();
    write_orange(&directory.0.join("symlink.orange"), |builder| {
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Symlink);
        header.set_size(0);
        header.set_mode(0o777);
        builder.append_link(&mut header, "link", "../../outside").unwrap();
    });
    write_orange(&directory.0.join("hardlink.orange"), |builder| {
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Link);
        header.set_size(0);
        builder.append_link(&mut header, "link", "/etc/passwd").unwrap();
    });
    write_orange(&directory.0.join("escape.orange"), |builder| {
        // `Builder::append_data` refuses `..`, so write the raw name bytes.
        let data = b"escaped";
        let mut header = file_header(data.len());
        header.as_old_mut().name[..13].copy_from_slice(b"../escape.txt");
        header.set_cksum();
        builder.append(&header, &data[..]).unwrap();
    });
    write_orange(&directory.0.join("valid.orange"), |builder| {
        let data = b"fine";
        let mut header = file_header(data.len());
        builder.append_data(&mut header, "inner/ok.txt", &data[..]).unwrap();
    });

    run_napalm(&package_test_source(&directory, r#"
        const root = __ROOT__;
        const out = Path.join(root, "out");
        rejects(() => Package.unpackage(Path.join(root, "symlink"), out));
        rejects(() => Package.unpackage(Path.join(root, "hardlink"), out));
        rejects(() => Package.unpackage(Path.join(root, "escape"), out));
        expect(!Fs.exists(Path.join(root, "escape.txt")));
        expect(!Fs.exists(Path.join(out, "link")));
        // Archives produced by other tar writers unpack fine.
        const files = Package.unpackage(Path.join(root, "valid.orange"), out);
        expect(files.length === 1 && files[0] === Path.join(out, "inner", "ok.txt"));
        expect(Fs.readText(files[0]) === "fine");
        const result = "ok";
    "#), "ok");
}

#[test]
fn napalm_package_files_are_reproducible_tar_zstd_level_18() {
    let directory = FsTestDirectory::new();
    std::fs::create_dir_all(directory.0.join("lib/sub")).unwrap();
    std::fs::write(directory.0.join("lib/a.napalm"), "a".repeat(10_000)).unwrap();
    std::fs::write(directory.0.join("lib/sub/b.napalm"), "b").unwrap();
    run_napalm(&package_test_source(&directory, r#"
        const root = __ROOT__;
        const lib = Path.join(root, "lib");
        Package.package("first").addFolder(lib).setPackageDest(root).finish();
        Package.package("second").addFolder(lib).setPackageDest(root).finish();
        const result = "ok";
    "#), "ok");

    let first = std::fs::read(directory.0.join("first.orange")).unwrap();
    let second = std::fs::read(directory.0.join("second.orange")).unwrap();
    assert_eq!(first, second, "packages of identical inputs must be byte-identical");
    let tar = zstd::stream::decode_all(&first[..]).unwrap();
    let mut archive = tar::Archive::new(&tar[..]);
    let entries = archive.entries().unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            assert_eq!(entry.header().uid().unwrap(), 0);
            (entry.path().unwrap().to_string_lossy().replace('\\', "/"), entry.header().entry_type())
        })
        .collect::<Vec<_>>();
    let names = entries.iter().map(|(name, _)| name.trim_end_matches('/').rsplit("lib").next().unwrap().to_owned()).collect::<Vec<_>>();
    assert_eq!(names, ["", "/a.napalm", "/sub", "/sub/b.napalm"], "entries: {entries:?}");
    assert!(entries[0].1.is_dir() && entries[1].1.is_file() && entries[2].1.is_dir());
}
