#!/usr/bin/env python3
"""Build and validate a native static LLVM distribution for Napalm CI."""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = os.name == "nt"
EXE = ".exe" if WINDOWS else ""


def run(*args, **kwargs):
    command = [str(arg) for arg in args]
    print("+", subprocess.list2cmdline(command), flush=True)
    try:
        return subprocess.run(command, check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, flush=True)
        if error.stderr:
            print(error.stderr, flush=True)
        raise


def replace_once(text, old, new, path):
    if text.count(old) != 1:
        raise RuntimeError(f"Unexpected source layout in {path}: {old!r}")
    return text.replace(old, new, 1)


def prepare():
    # Cargo features are additive. Replace the dynamic linkage features only in
    # the disposable CI checkout, rather than combining incompatible features.
    edits = {
        "Cargo.toml": [
            ('features = ["llvm23-1-no-llvm-linking"]',
             'features = ["llvm23-1-force-static"]'),
            ('inkwell = { version = "0.11.0", features = ["llvm23-1-prefer-dynamic"], optional = true }',
             'inkwell = { version = "0.11.0", default-features = false, features = ["llvm23-1-force-static"], optional = true }'),
            ('llvm-jit = ["dep:inkwell", "dep:llvm-sys"]',
             'llvm-jit = ["dep:inkwell", "dep:llvm-sys"]\nllvm-static = ["llvm-jit"]'),
        ],
        "build.rs": [
            ('    match env::var("CARGO_CFG_TARGET_OS").as_deref() {',
             '    // llvm-sys supplies all native link directives for static builds.\n'
             '    if env::var_os("CARGO_FEATURE_LLVM_STATIC").is_some() {\n'
             '        return;\n    }\n'
             '    match env::var("CARGO_CFG_TARGET_OS").as_deref() {'),
        ],
        "src/napalm_tracing_jit/llvm.rs": [
            ('#[cfg(windows)]\nfn load_llvm_library()',
             '#[cfg(all(windows, not(feature = "llvm-static")))]\nfn load_llvm_library()'),
            ('#[cfg(not(windows))]\nfn load_llvm_library()',
             '#[cfg(any(not(windows), feature = "llvm-static"))]\nfn load_llvm_library()'),
        ],
        "src/napalm_standalone.rs": [
            ('#[cfg(all(windows, feature = "llvm-jit"))]\nfn llvm_library_to_embed()',
             '#[cfg(all(windows, feature = "llvm-jit", not(feature = "llvm-static")))]\nfn llvm_library_to_embed()'),
            ('#[cfg(not(all(windows, feature = "llvm-jit")))]\nfn llvm_library_to_embed()',
             '#[cfg(not(all(windows, feature = "llvm-jit", not(feature = "llvm-static"))))]\nfn llvm_library_to_embed()'),
        ],
        "tests/compile.rs": [
            ('#[cfg(all(windows, feature = "llvm-jit"))]\n#[test]\nfn embedded_llvm_library_is_extracted_and_loaded()',
             '#[cfg(all(windows, feature = "llvm-jit", not(feature = "llvm-static")))]\n#[test]\nfn embedded_llvm_library_is_extracted_and_loaded()'),
        ],
    }
    # Validate every expected source anchor before touching any file.
    prepared = {}
    for filename, substitutions in edits.items():
        path = ROOT / filename
        original = path.read_text(encoding="utf-8")
        modified = original
        for old, new in substitutions:
            modified = replace_once(modified, old, new, path)
        prepared[path] = modified
    for path, content in prepared.items():
        path.write_text(content, encoding="utf-8", newline="\n")
    run("git", "diff", "--check", cwd=ROOT)
    run("git", "diff", "--stat", cwd=ROOT)


def build_llvm(args):
    prefix = args.prefix.resolve()
    if prefix.exists():
        shutil.rmtree(prefix)
    prefix.mkdir(parents=True)
    # Use the same CRT as the statically linked Rust executable on MSVC.
    # No rpmalloc interposition or optional LLVM system-library dependencies.
    # Napalm builds its own Zstandard dependency through Cargo.
    with tempfile.TemporaryDirectory(prefix="napalm-llvm-", dir=os.environ.get("RUNNER_TEMP")) as tmp:
        work = Path(tmp)
        source = work / "src"
        build = work / "build"
        run("git", "clone", "--depth", "1", "--branch", f"llvmorg-{args.version}",
            "https://github.com/llvm/llvm-project.git", source)
        opts = {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_INSTALL_PREFIX": prefix.as_posix(),
            "BUILD_SHARED_LIBS": "OFF",
            "LLVM_BUILD_LLVM_DYLIB": "OFF",
            "LLVM_LINK_LLVM_DYLIB": "OFF",
            "LLVM_BUILD_LLVM_C_DYLIB": "OFF",
            "LLVM_ENABLE_RPMALLOC": "OFF",
            "LLVM_ENABLE_PROJECTS": "",
            "LLVM_ENABLE_RUNTIMES": "",
            "LLVM_TARGETS_TO_BUILD": args.backend,
            "LLVM_ENABLE_ASSERTIONS": "OFF",
            # These options affect LLVM Support only, not Cargo dependencies.
            # Keep them explicit to avoid linking a second native Zstandard
            # implementation or introducing platform-specific shared libraries.
            "LLVM_ENABLE_ZLIB": "OFF",
            "LLVM_ENABLE_ZSTD": "OFF",
            "LLVM_ENABLE_LIBXML2": "OFF",
            "LLVM_ENABLE_LIBEDIT": "OFF",
            "LLVM_ENABLE_LIBPFM": "OFF",
            "LLVM_ENABLE_TERMINFO": "OFF",
            "LLVM_ENABLE_CURL": "OFF",
            "LLVM_ENABLE_HTTPLIB": "OFF",
            "LLVM_ENABLE_LZMA": "OFF",
            "LLVM_ENABLE_FFI": "OFF",
            "LLVM_ENABLE_Z3_SOLVER": "OFF",
            "LLVM_INCLUDE_TESTS": "OFF",
            "LLVM_INCLUDE_BENCHMARKS": "OFF",
            "LLVM_INCLUDE_EXAMPLES": "OFF",
            "LLVM_INCLUDE_DOCS": "OFF",
            "LLVM_INCLUDE_UTILS": "OFF",
            "LLVM_BUILD_TOOLS": "ON",
            "LLVM_PARALLEL_LINK_JOBS": "1",
            "LLVM_PARALLEL_TABLEGEN_JOBS": "2",
            "LLVM_DISTRIBUTION_COMPONENTS": "llvm-headers;llvm-libraries;llvm-config;cmake-exports",
        }
        if WINDOWS:
            opts.update(CMAKE_C_COMPILER="cl", CMAKE_CXX_COMPILER="cl",
                        CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded",
                        LLVM_USE_CRT_RELEASE="MT", LLVM_ENABLE_DIA_SDK="OFF",
                        LLVM_ENABLE_PDB="OFF")
        elif sys_platform() == "darwin":
            opts.update(CMAKE_OSX_DEPLOYMENT_TARGET="12.0",
                        CMAKE_OSX_ARCHITECTURES="arm64" if args.backend == "AArch64" else "x86_64")
        run("cmake", "-S", source / "llvm", "-B", build, "-G", "Ninja",
            *(f"-D{k}={v}" for k, v in opts.items()))
        run("cmake", "--build", build, "--target", "install-distribution", "--parallel", str(args.jobs))
        shutil.copy2(source / "llvm" / "LICENSE.TXT", prefix / "LLVM-LICENSE.txt")
    verify_llvm(args)


def sys_platform():
    import sys
    return sys.platform


def verify_llvm(args):
    config = args.prefix / "bin" / f"llvm-config{EXE}"
    version = run(config, "--version", capture_output=True, text=True).stdout.strip()
    if version != args.version:
        raise RuntimeError(f"Expected LLVM {args.version}, found {version}")
    backends = run(config, "--targets-built", capture_output=True, text=True).stdout.split()
    if backends != [args.backend]:
        raise RuntimeError(f"Unexpected LLVM targets: {backends}")
    run(config, "--link-static", "--libs", "all")
    run(config, "--link-static", "--system-libs", "all")
    libdir = args.prefix / "lib"
    if not list(libdir.glob("LLVM*.lib" if WINDOWS else "libLLVM*.a")):
        raise RuntimeError(f"No static LLVM libraries under {libdir}")
    if not (args.prefix / "LLVM-LICENSE.txt").is_file():
        raise RuntimeError("LLVM license is missing")


def check_imports(binary):
    if WINDOWS:
        result = run("dumpbin", "/IMPORTS", binary, capture_output=True, text=True, errors="replace")
        names = re.findall(r"(?im)^\s*([^\s]+\.dll)\s*$", result.stdout)
        if not names:
            raise RuntimeError("Could not parse PE imports")
        dependencies = "\n".join(names)
        # Check normal and delay-loaded imports, not arbitrary exported symbols.
        forbidden = r"(?im)\b(?:llvm[^\s]*|vcruntime[^\s]*|msvcp[^\s]*)\.dll\b"
    elif sys_platform() == "darwin":
        dependencies = run("otool", "-L", binary, capture_output=True, text=True).stdout
        forbidden = r"(?i)libLLVM[^\s]*\.dylib"
    else:
        dependencies = run("readelf", "-d", binary, capture_output=True, text=True).stdout
        forbidden = r"(?i)libLLVM[^\s\]]*\.so(?:\.[^\s\]]*)?"
    print(dependencies, flush=True)
    if re.search(forbidden, dependencies):
        raise RuntimeError("Unexpected dynamic LLVM or MSVC runtime dependency")


def smoke(args):
    binary = args.binary.resolve()
    check_imports(binary)
    # Run outside the build tree with no LLVM DLL or LLVM path in the working dir.
    with tempfile.TemporaryDirectory(prefix="napalm-smoke-") as tmp:
        work = Path(tmp)
        isolated = work / f"napalm{EXE}"
        shutil.copy2(binary, isolated)
        source = work / "hot.napalm"
        source.write_text("let total = 0; for (let i = 0; i < 2000000; i += 1) { total += i % 7; } System.print(total);\n", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith("NAPALM_") and not k.startswith("LLVM_SYS_") and k not in {"LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"}}
        env.update(NAPALM_LLVM_THRESHOLD="1", NAPALM_LLVM_SYNC="1", NAPALM_JIT_EVENTS="1",
                   NAPALM_CACHE_DIR=str(work / "cache"))
        result = run(isolated, source, cwd=work, env=env, capture_output=True, text=True, timeout=180)
        print(result.stdout, end="")
        print(result.stderr, end="")
        if result.stdout.strip() != "5999995":
            raise RuntimeError("Incorrect Napalm smoke-test result")
        if not re.search(r"\[napalm-jit\] LLVM (?:tier|region) installed", result.stderr):
            raise RuntimeError("LLVM JIT did not install native code")
        if (work / "cache" / "llvm").exists():
            raise RuntimeError("Static build unexpectedly extracted an LLVM library")

        # Exercise both of Napalm's Zstandard-backed artifact formats using
        # the relocated binary, independently of the LLVM compression options.
        artifact_source = work / "compression.napalm"
        artifact_source.write_text("System.print(6 * 7);\n", encoding="utf-8")
        run(isolated, "--build", artifact_source, cwd=work, env=env, timeout=180)
        artifact = artifact_source.with_suffix(".burning")
        with artifact.open("rb") as file:
            if file.read(8) != b"NAPBURN\x00":
                raise RuntimeError("Invalid .burning artifact header")
        artifact_source.unlink()
        result = run(isolated, artifact, cwd=work, env=env, capture_output=True,
                     text=True, timeout=180)
        if result.stdout.strip() != "42":
            raise RuntimeError("Incorrect .burning round-trip result")

        # Run the repository's existing end-to-end package example in a clean
        # working directory, so its fixtures never pollute the checkout.
        scripts = work / "scripts"
        scripts.mkdir()
        package_example = scripts / "package.napalm"
        shutil.copy2(ROOT / "scripts" / "package.napalm", package_example)
        result = run(isolated, package_example, cwd=work, env=env,
                     capture_output=True, text=True, timeout=180)
        print(result.stdout, end="")
        print(result.stderr, end="")
        if "Package example passed" not in result.stdout.splitlines():
            raise RuntimeError("Napalm .orange package round-trip failed")
        dist = scripts / "__napalm_package_example__" / "dist"
        for name in ("demo.orange", "whole.orange"):
            with (dist / name).open("rb") as file:
                if file.read(4) != b"\x28\xb5\x2f\xfd":
                    raise RuntimeError(f"{name} is not a Zstandard frame")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    for name in ("build-llvm", "verify-llvm"):
        sub = commands.add_parser(name)
        sub.add_argument("--prefix", type=Path, required=True)
        sub.add_argument("--version", required=True)
        sub.add_argument("--backend", choices=["X86", "AArch64"], required=True)
        if name == "build-llvm":
            sub.add_argument("--jobs", type=int, default=4)
    sub = commands.add_parser("smoke")
    sub.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "build-llvm":
        build_llvm(args)
    elif args.command == "verify-llvm":
        verify_llvm(args)
    elif args.command == "smoke":
        smoke(args)


if __name__ == "__main__":
    main()
