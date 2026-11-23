//! Native `.orange` packages for the `Package` global.
//!
//! `.orange` is the official Napalm package format (dependencies are
//! distributed with it). It is always a tar archive compressed as a single
//! Zstandard frame at level 18: the highest ratio before decompression speed
//! starts to degrade noticeably. The extension is mandatory on disk but optional
//! in script paths: `Package.unpackage("deps/http")` reads `deps/http.orange`.
//!
//! A `Package` heap object is a builder. It only records validated source paths
//! and their archive locations; no files are read until `finish()`, which
//! snapshots the folders, writes `<dest>/.<name>.orange.partial` and atomically
//! renames it over `<dest>/<name>.orange`. Packages contain no managed
//! references. Like `Fs`, this is a blocking API.
//!
//! Archive layout:
//! - `addFile` / `addFiles` store each file in the archive root under its name.
//! - `addFolder` keeps the folder path relative to the working directory, with
//!   leading `..` removed: `../../package/zeta` is stored as `package/zeta`.
//!   When nothing remains (`./`, `../`, an ancestor of the working directory) or
//!   the folder is on another root, the folder's own name is used instead.
//!
//! Packages are reproducible: entries are sorted, and tar headers use
//! deterministic metadata (no owners or timestamps). Symlinks, hard links and
//! special files are rejected both when packaging and when unpackaging, and
//! extraction never writes outside the destination folder.

use super::*;
use super::path_vm::{current_directory, parse_path, path_join, path_resolve_from};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};

pub(super) const PACKAGE_EXTENSION: &str = ".orange";
/// Fixed by the format: every `.orange` file is compressed at this level.
pub(super) const PACKAGE_ZSTD_LEVEL: i32 = 18;
const DEFAULT_DESTINATION: &str = "./";

#[derive(Debug)]
struct PackageSource {
    /// Path exactly as the script supplied it, for diagnostics.
    written: String,
    /// Absolute path resolved when added, so a later `Process.chdir` does not
    /// change what `finish()` reads.
    resolved: PathBuf,
    /// `/`-separated location inside the archive.
    archive: String,
    folder: bool,
}

#[derive(Debug)]
pub(super) struct NapalmPackage {
    pub(super) name: String,
    destination: String,
    sources: Vec<PackageSource>,
}

impl NapalmPackage {
    pub(super) fn external_bytes(&self) -> usize {
        self.name.capacity()
            + self.destination.capacity()
            + self.sources.capacity() * std::mem::size_of::<PackageSource>()
            + self.sources.iter()
                .map(|source| source.written.capacity() + source.resolved.capacity() + source.archive.capacity())
                .sum::<usize>()
    }

    pub(super) fn file_name(&self) -> String {
        format!("{}{PACKAGE_EXTENSION}", self.name)
    }

    /// Append sources atomically: either all of them are added or none.
    fn add_sources(&mut self, added: Vec<PackageSource>) -> Result<(), String> {
        for (index, source) in added.iter().enumerate() {
            if self.sources.iter().chain(&added[..index]).any(|existing| existing.archive == source.archive) {
                return Err(format!(
                    "{:?} would be stored as {:?}, which is already in the package",
                    source.written, source.archive,
                ));
            }
        }
        self.sources.extend(added);
        Ok(())
    }
}

/// `name` and `name.orange` are equivalent. A name is a single file name; the
/// output folder is chosen with `setPackageDest`.
fn package_name(text: &str) -> Result<String, String> {
    let name = text.strip_suffix(PACKAGE_EXTENSION).unwrap_or(text);
    if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\', '\0']) {
        return Err("the package name must be a file name without path separators; use setPackageDest to choose its folder".into());
    }
    Ok(name.to_owned())
}

fn with_package_extension(path: &str) -> String {
    if path.ends_with(PACKAGE_EXTENSION) { path.to_owned() } else { format!("{path}{PACKAGE_EXTENSION}") }
}

fn resolve_from(cwd: &str, path: &str) -> Result<PathBuf, String> {
    if path.is_empty() { return Err("path must not be empty".into()); }
    path_resolve_from(cwd.to_owned(), &[path.to_owned()]).map(PathBuf::from)
}

/// Archive name of a file added with `addFile`: its final path component.
fn file_archive_name(path: &str) -> Result<String, String> {
    match parse_path(path)?.basename() {
        "" | "." | ".." => Err("cannot determine the file name".into()),
        name => Ok(name.to_owned()),
    }
}

/// Archive path of a folder added with `addFolder`: its path relative to `cwd`
/// without leading `..` components, or the folder's own name when that leaves
/// nothing (`./`, `../`, ancestors of `cwd`) or the roots differ.
fn folder_archive_path(cwd: &str, folder: &str) -> Result<String, String> {
    let base = parse_path(cwd)?;
    let resolved = parse_path(&path_resolve_from(cwd.to_owned(), &[folder.to_owned()])?)?;
    let mut components: &[String] = &[];
    if base.comparable_root(&resolved) {
        components = &resolved.components[base.common_prefix_len(&resolved)..];
    }
    if components.is_empty() {
        components = resolved.components.last().map(std::slice::from_ref).unwrap_or_default();
    }
    if components.is_empty() {
        return Err("cannot determine the folder name of a filesystem root".into());
    }
    Ok(components.join("/"))
}

struct PlanItem {
    source: PathBuf,
    archive: String,
    directory: bool,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn with_path(path: &Path, error: io::Error) -> io::Error {
    io::Error::new(error.kind(), format!("{}: {error}", path.display()))
}

/// Recursively add `dir` and its contents in name order. `skip` filters out
/// the package being written when it lives inside a packaged folder.
fn plan_folder(
    dir: &Path,
    archive: &str,
    plan: &mut Vec<PlanItem>,
    skip: &dyn Fn(&Path) -> bool,
) -> io::Result<()> {
    plan.push(PlanItem { source: dir.to_path_buf(), archive: archive.to_owned(), directory: true });
    let mut children = Vec::new();
    for entry in fs::read_dir(dir).map_err(|error| with_path(dir, error))? {
        let entry = entry.map_err(|error| with_path(dir, error))?;
        let name = entry.file_name().into_string()
            .map_err(|_| invalid(format!("{}: an entry name is not valid UTF-8", dir.display())))?;
        let kind = entry.file_type().map_err(|error| with_path(&entry.path(), error))?;
        children.push((name, entry.path(), kind));
    }
    children.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    for (name, path, kind) in children {
        let archive = format!("{archive}/{name}");
        if kind.is_symlink() {
            return Err(invalid(format!("{}: symlinks cannot be packaged", path.display())));
        } else if kind.is_dir() {
            plan_folder(&path, &archive, plan, skip)?;
        } else if kind.is_file() {
            if !skip(&path) {
                plan.push(PlanItem { source: path, archive, directory: false });
            }
        } else {
            return Err(invalid(format!("{}: only regular files and folders can be packaged", path.display())));
        }
    }
    Ok(())
}

/// Build the package and return the path of the created `.orange` file.
fn finish_package(package: &NapalmPackage, cwd: &str) -> io::Result<String> {
    if package.sources.is_empty() {
        return Err(invalid("the package is empty; add files or folders before calling finish()"));
    }
    let destination = resolve_from(cwd, &package.destination).map_err(invalid)?;
    fs::create_dir_all(&destination).map_err(|error| with_path(&destination, error))?;
    let file_name = package.file_name();
    let partial_name = format!(".{file_name}.partial");
    let output = destination.join(&file_name);
    let partial = destination.join(&partial_name);

    // A previous build of this package (or a stale partial file) inside a
    // packaged folder must not end up inside the new package.
    let canonical_destination = fs::canonicalize(&destination).map_err(|error| with_path(&destination, error))?;
    let canonical_output = canonical_destination.join(&file_name);
    let canonical_partial = canonical_destination.join(&partial_name);
    let is_package_output = |path: &Path| {
        path.file_name().is_some_and(|name| name == file_name.as_str() || name == partial_name.as_str())
            && fs::canonicalize(path).is_ok_and(|path| path == canonical_output || path == canonical_partial)
    };

    let mut plan = Vec::new();
    for source in &package.sources {
        if source.folder {
            plan_folder(&source.resolved, &source.archive, &mut plan, &is_package_output)?;
        } else {
            if is_package_output(&source.resolved) {
                return Err(invalid(format!("{:?} is the package being created", source.written)));
            }
            let metadata = fs::metadata(&source.resolved).map_err(|error| with_path(&source.resolved, error))?;
            if !metadata.is_file() {
                return Err(invalid(format!("{:?} is no longer a regular file", source.written)));
            }
            plan.push(PlanItem { source: source.resolved.clone(), archive: source.archive.clone(), directory: false });
        }
    }
    let mut seen = HashSet::with_capacity(plan.len());
    for item in &plan {
        if !seen.insert(item.archive.as_str()) {
            return Err(invalid(format!(
                "{:?} would be stored twice in the package; overlapping folders or files share that path",
                item.archive,
            )));
        }
    }

    let written = write_archive(&plan, &partial).and_then(|()| fs::rename(&partial, &output));
    if let Err(error) = written {
        let _ = fs::remove_file(&partial);
        return Err(error);
    }
    path_join(&[package.destination.clone(), file_name]).map_err(invalid)
}

fn write_archive(plan: &[PlanItem], partial: &Path) -> io::Result<()> {
    let file = File::create(partial).map_err(|error| with_path(partial, error))?;
    let mut encoder = zstd::stream::Encoder::new(BufWriter::new(file), PACKAGE_ZSTD_LEVEL)?;
    encoder.include_checksum(true)?;
    let workers = std::thread::available_parallelism().map_or(1, |count| count.get());
    if workers > 1 {
        encoder.multithread(workers.min(u32::MAX as usize) as u32)?;
    }
    let mut builder = tar::Builder::new(encoder);
    builder.mode(tar::HeaderMode::Deterministic);
    for item in plan {
        let appended = if item.directory {
            builder.append_dir(&item.archive, &item.source)
        } else {
            builder.append_path_with_name(&item.source, &item.archive)
        };
        appended.map_err(|error| with_path(&item.source, error))?;
    }
    let writer = builder.into_inner()?.finish()?;
    let file = writer.into_inner().map_err(io::IntoInnerError::into_error)?;
    file.sync_all()
}

/// Extract `package` into `destination` and return the extracted file paths
/// inside the archive, in archive order.
fn unpack_package(package: &Path, destination: &Path) -> io::Result<Vec<String>> {
    let metadata = fs::metadata(package)?;
    if !metadata.is_file() {
        return Err(invalid("not a regular file"));
    }
    let decoder = zstd::stream::Decoder::new(File::open(package)?)?;
    let mut archive = tar::Archive::new(decoder);
    archive.set_preserve_mtime(false);
    archive.set_preserve_permissions(false);
    archive.set_unpack_xattrs(false);
    archive.set_overwrite(true);
    fs::create_dir_all(destination).map_err(|error| with_path(destination, error))?;
    let mut files = Vec::new();
    for entry in archive.entries()? {
        let mut entry = entry?;
        let kind = entry.header().entry_type();
        if kind.is_pax_global_extensions() { continue; }
        let path = entry.path()?.to_str()
            .ok_or_else(|| invalid("an entry path is not valid UTF-8"))?
            .to_owned();
        if !kind.is_file() && !kind.is_dir() {
            return Err(invalid(format!(
                "entry {path:?} is not a regular file or folder; links and special files are not allowed",
            )));
        }
        if !entry.unpack_in(destination)? {
            return Err(invalid(format!("entry {path:?} would be extracted outside the destination")));
        }
        if kind.is_file() { files.push(path); }
    }
    Ok(files)
}

impl Interpreter<'_> {
    fn package_text(&self, value: Value, what: &str) -> Result<String, RuntimeError> {
        let Some(HeapObject::String(text)) = self.heap.get(value) else {
            return Err(RuntimeError::new(format!("TypeError: Package {what} must be a String")));
        };
        std::str::from_utf8(text.as_ref()).map(str::to_owned)
            .map_err(|_| RuntimeError::new(format!("TypeError: Package {what} must contain valid UTF-8")))
    }

    fn package_string(&mut self, text: String) -> Value {
        self.heap.alloc(HeapObject::String(NapalmString::new(StrBytes::from(text.into_bytes()))))
    }

    fn package_string_list(&mut self, items: Vec<String>) -> Value {
        let list = self.heap.alloc(HeapObject::List(Vec::with_capacity(items.len())));
        let root = self.add_root(list);
        for item in items {
            let value = self.package_string(item);
            self.heap.remember_if_reference(list, value);
            if let Some(HeapObject::List(values)) = self.heap.get_mut(list) {
                values.push(value);
            } else {
                unreachable!("rooted Package.unpackage list must remain live");
            }
        }
        self.release_root(root);
        let external = self.heap.get(list).map_or(0, external_bytes_for);
        self.heap.set_external_bytes(list, external);
        self.heap.refresh_jit_view(list);
        list
    }

    fn package_receiver(&mut self, receiver: Value) -> Result<&mut NapalmPackage, RuntimeError> {
        match self.heap.get_mut(receiver) {
            Some(HeapObject::Package(package)) => Ok(&mut **package),
            _ => Err(RuntimeError::new("receiver must be a Package")),
        }
    }

    /// Add sources to the receiver and return it, so builder calls chain.
    fn package_add(&mut self, receiver: Value, operation: &str, sources: Vec<PackageSource>) -> Result<Value, RuntimeError> {
        let package = self.package_receiver(receiver)?;
        package.add_sources(sources).map_err(|error| {
            RuntimeError::new(format!("Package.{operation}: {error}"))
        })?;
        let external = package.external_bytes();
        self.heap.set_external_bytes(receiver, external);
        Ok(receiver)
    }

    fn package_operation(&mut self, native: PackageNative, receiver: Value, args: &[RegisterWord]) -> Result<Value, RuntimeError> {
        let arg = |index: usize| args.get(index).copied().map(Value::from_word).unwrap_or(Value::NULL);
        let cwd = || current_directory().map_err(RuntimeError::new);
        match native {
            PackageNative::Package => {
                let text = self.package_text(arg(0), "name")?;
                let name = package_name(&text)
                    .map_err(|error| RuntimeError::new(format!("Package.package({text:?}): {error}")))?;
                let package = NapalmPackage { name, destination: DEFAULT_DESTINATION.to_owned(), sources: Vec::new() };
                let external = package.external_bytes();
                let value = self.heap.alloc(HeapObject::Package(Box::new(package)));
                self.heap.set_external_bytes(value, external);
                Ok(value)
            }
            PackageNative::Unpackage => {
                let path = with_package_extension(&self.package_text(arg(0), "path")?);
                let destination = match arg(1) {
                    value if value == Value::NULL => DEFAULT_DESTINATION.to_owned(),
                    value => self.package_text(value, "destination")?,
                };
                let failure = |error: &dyn std::fmt::Display| {
                    RuntimeError::new(format!("Package.unpackage({path}): {error}"))
                };
                let cwd = cwd()?;
                let package = resolve_from(&cwd, &path).map_err(|error| failure(&error))?;
                let target = resolve_from(&cwd, &destination).map_err(|error| failure(&error))?;
                let files = unpack_package(&package, &target).map_err(|error| failure(&error))?;
                let mut paths = Vec::with_capacity(files.len());
                for file in files {
                    paths.push(path_join(&[destination.clone(), file]).map_err(|error| failure(&error))?);
                }
                Ok(self.package_string_list(paths))
            }
            PackageNative::AddFile => {
                let path = self.package_text(arg(0), "file path")?;
                let source = file_source(&cwd()?, path)
                    .map_err(|error| RuntimeError::new(format!("Package.addFile: {error}")))?;
                self.package_add(receiver, "addFile", vec![source])
            }
            PackageNative::AddFiles => {
                let items = match self.heap.get(arg(0)) {
                    Some(HeapObject::List(items)) => items.clone(),
                    _ => return Err(RuntimeError::new("TypeError: Package.addFiles expects an Array of file paths")),
                };
                let cwd = cwd()?;
                let mut sources = Vec::with_capacity(items.len());
                for item in items {
                    let path = self.package_text(item, "file path")?;
                    sources.push(file_source(&cwd, path)
                        .map_err(|error| RuntimeError::new(format!("Package.addFiles: {error}")))?);
                }
                self.package_add(receiver, "addFiles", sources)
            }
            PackageNative::AddFolder => {
                let path = self.package_text(arg(0), "folder path")?;
                let source = folder_source(&cwd()?, path)
                    .map_err(|error| RuntimeError::new(format!("Package.addFolder: {error}")))?;
                self.package_add(receiver, "addFolder", vec![source])
            }
            PackageNative::SetPackageDest => {
                let destination = self.package_text(arg(0), "destination")?;
                if destination.is_empty() {
                    return Err(RuntimeError::new("Package.setPackageDest: the destination folder must not be empty"));
                }
                let package = self.package_receiver(receiver)?;
                package.destination = destination;
                let external = package.external_bytes();
                self.heap.set_external_bytes(receiver, external);
                Ok(receiver)
            }
            PackageNative::Finish => {
                let cwd = cwd()?;
                let package = self.package_receiver(receiver)?;
                let path = finish_package(package, &cwd).map_err(|error| {
                    RuntimeError::new(format!("Package.finish({}): {error}", package.file_name()))
                })?;
                Ok(self.package_string(path))
            }
        }
    }

    pub(super) fn execute_package_native(
        &mut self,
        op: &PreparedOp,
        native: PackageNative,
        receiver: RegisterWord,
        args: &[RegisterWord],
    ) {
        self.mark_gc_point(op);
        self.maybe_collect();
        match self.package_operation(native, Value::from_word(receiver), args) {
            Ok(value) => unsafe {
                self.set_reg(op.a, value.word());
                self.advance(op);
            },
            Err(error) => self.fail_at(op, error.message),
        }
    }
}

fn file_source(cwd: &str, written: String) -> Result<PackageSource, String> {
    let resolved = resolve_from(cwd, &written).map_err(|error| format!("{written:?}: {error}"))?;
    let metadata = fs::metadata(&resolved).map_err(|error| format!("{written:?}: {error}"))?;
    if !metadata.is_file() {
        return Err(format!("{written:?} is not a regular file; use addFolder for folders"));
    }
    let archive = file_archive_name(&written).map_err(|error| format!("{written:?}: {error}"))?;
    Ok(PackageSource { written, resolved, archive, folder: false })
}

fn folder_source(cwd: &str, written: String) -> Result<PackageSource, String> {
    let resolved = resolve_from(cwd, &written).map_err(|error| format!("{written:?}: {error}"))?;
    let metadata = fs::metadata(&resolved).map_err(|error| format!("{written:?}: {error}"))?;
    if !metadata.is_dir() {
        return Err(format!("{written:?} is not a folder; use addFile for files"));
    }
    let archive = folder_archive_path(cwd, &written).map_err(|error| format!("{written:?}: {error}"))?;
    Ok(PackageSource { written, resolved, archive, folder: true })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_names_accept_optional_extension() {
        assert_eq!(package_name("http").unwrap(), "http");
        assert_eq!(package_name("http.orange").unwrap(), "http");
        assert_eq!(package_name("http-2.1.0").unwrap(), "http-2.1.0");
        for invalid in ["", ".orange", ".", "..", "deps/http", "deps\\http"] {
            assert!(package_name(invalid).is_err(), "{invalid:?} must be rejected");
        }
        assert_eq!(with_package_extension("deps/http"), "deps/http.orange");
        assert_eq!(with_package_extension("deps/http.orange"), "deps/http.orange");
        assert_eq!(with_package_extension("deps/http.tar.zst"), "deps/http.tar.zst.orange");
    }

    #[test]
    fn files_are_stored_in_the_archive_root() {
        assert_eq!(file_archive_name("src/lib/main.napalm").unwrap(), "main.napalm");
        assert_eq!(file_archive_name("C:\\work\\README.md").unwrap(), "README.md");
        assert_eq!(file_archive_name("../notes.txt").unwrap(), "notes.txt");
        assert!(file_archive_name("a/..").is_err());
        assert!(file_archive_name("/").is_err());
    }

    #[test]
    fn folders_keep_their_path_relative_to_the_working_directory() {
        let cwd = "/work/app";
        assert_eq!(folder_archive_path(cwd, "../../package/zeta").unwrap(), "package/zeta");
        assert_eq!(folder_archive_path(cwd, "src/lib").unwrap(), "src/lib");
        assert_eq!(folder_archive_path(cwd, "./src/../assets/").unwrap(), "assets");
        assert_eq!(folder_archive_path(cwd, "/work/app/src").unwrap(), "src");
        assert_eq!(folder_archive_path(cwd, "/srv/shared/pkg").unwrap(), "srv/shared/pkg");
        // Nothing left after removing `..`: fall back to the folder's name.
        assert_eq!(folder_archive_path(cwd, "./").unwrap(), "app");
        assert_eq!(folder_archive_path(cwd, ".").unwrap(), "app");
        assert_eq!(folder_archive_path(cwd, "../").unwrap(), "work");
        assert_eq!(folder_archive_path(cwd, "../..").unwrap_err(), "cannot determine the folder name of a filesystem root");
    }

    #[test]
    fn windows_folders_compare_case_insensitively_and_fall_back_across_drives() {
        let cwd = "C:\\Work\\App";
        assert_eq!(folder_archive_path(cwd, "c:\\work\\app\\src\\lib").unwrap(), "src/lib");
        assert_eq!(folder_archive_path(cwd, "..\\..\\package\\zeta").unwrap(), "package/zeta");
        assert_eq!(folder_archive_path(cwd, ".\\").unwrap(), "App");
        assert_eq!(folder_archive_path(cwd, "D:\\deps\\zeta").unwrap(), "zeta");
    }
}
