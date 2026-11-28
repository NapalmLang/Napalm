// Profile-guided optimization (PGO) build of Napalm.
//
// 1. Builds an instrumented Napalm (`-Cprofile-generate`).
// 2. Trains it: every script in ./scripts and ./lsp/scripts (with the JIT and
//    interpreter-only), the LSP test suite (live LSP sessions over stdio) and
//    the Rust integration tests (which drive the napalm executable).
// 3. Merges the profiles with the toolchain's `llvm-profdata`.
// 4. Rebuilds Napalm with `-Cprofile-use`.
//
// Usage (Node >= 22.18 runs TypeScript natively):
//   node tools/pgo.ts [options]
//
// Options:
//   --quick               Train on the scripts only (no LSP or Rust tests).
//   --no-lsp              Skip the LSP test suite.
//   --no-rust-tests       Skip the Rust integration tests.
//   --unit-tests          Also run the Rust unit tests. Off by default: they
//                         exercise rare paths (errors, edge cases) and would
//                         skew the profile away from real programs.
//   --extra <dir>         Also train on the .napalm files in <dir> (repeatable).
//   --jobs <n>            Parallel training processes (default: half the cores).
//   --timeout <seconds>   Per-script timeout (default 300; instrumented code is
//                         2-3x slower).
//   --compare             Afterwards, time the PGO build against target/release
//                         on the benchmarks (interleaved runs, medians).
//   --rounds <n>          Rounds for --compare (default 5).
//   --copy-to <dir>       Copy the PGO napalm executable and LLVM-C.dll there.
//
// Outputs: target/pgo/<host>/release/napalm[.exe]. The training profile is
// regenerated on every run (target/pgo-data); it is never committed, because
// it goes stale as soon as the code changes.

import { spawn, spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from "node:fs";
import { availableParallelism } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

interface Options {
    lsp: boolean;
    rustTests: boolean;
    unitTests: boolean;
    extra: string[];
    jobs: number;
    timeoutSeconds: number;
    compare: boolean;
    rounds: number;
    copyTo?: string;
}

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const exeName = process.platform === "win32" ? "napalm.exe" : "napalm";
const dataDir = join(root, "target", "pgo-data");
const genTargetDir = join(root, "target", "pgo-gen");
const useTargetDir = join(root, "target", "pgo");

/// Scripts that cannot be training runs: imported-only modules, network
/// access, and servers that never exit (a killed process writes no profile).
const SKIP_SCRIPTS = new Set(["math", "other", "module_exports", "fetch", "server", "server_simple"]);
/// LSP tests that need tools beyond Node (bun, vsce).
const SKIP_LSP_TESTS = new Set(["build-vscode.test.mjs"]);
/// Command-line arguments for scripts that need them (relative to the root).
const SCRIPT_ARGS: Record<string, string[]> = {
    parse_file: ["lsp/src/parser.napalm"],
};
/// Benchmarks timed by --compare (scripts/ and lsp/scripts/ names).
const COMPARE_SCRIPTS = [
    "scripts/fib.napalm", "scripts/trees.napalm", "scripts/lev.napalm", "scripts/jit.napalm",
    "scripts/gc.napalm", "scripts/reactive_bench.napalm", "scripts/jit_entries.napalm",
    "lsp/scripts/parser_bench.napalm", "lsp/scripts/parser_selfhost.napalm", "lsp/scripts/intelligence_smoke.napalm",
];

function parseArgs(argv: string[]): Options {
    const options: Options = {
        lsp: true,
        rustTests: true,
        unitTests: false,
        extra: [],
        jobs: Math.max(1, Math.floor(availableParallelism() / 2)),
        timeoutSeconds: 300,
        compare: false,
        rounds: 5,
    };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        const value = (): string => {
            const next = argv[++i];
            if (next === undefined) throw new Error(`missing value for ${arg}`);
            return next;
        };
        switch (arg) {
            case "--quick": options.lsp = false; options.rustTests = false; break;
            case "--no-lsp": options.lsp = false; break;
            case "--no-rust-tests": options.rustTests = false; break;
            case "--unit-tests": options.unitTests = true; break;
            case "--extra": options.extra.push(resolve(value())); break;
            case "--jobs": options.jobs = Math.max(1, Number(value())); break;
            case "--timeout": options.timeoutSeconds = Number(value()); break;
            case "--compare": options.compare = true; break;
            case "--rounds": options.rounds = Math.max(1, Number(value())); break;
            case "--copy-to": options.copyTo = resolve(value()); break;
            case "-h":
            case "--help":
                console.log("usage: node tools/pgo.ts [--quick] [--no-lsp] [--no-rust-tests] [--unit-tests] [--extra DIR]... [--jobs N] [--timeout S] [--compare] [--rounds N] [--copy-to DIR]");
                process.exit(0);
            default:
                throw new Error(`unknown option ${arg}`);
        }
    }
    return options;
}

function run(command: string, args: string[], env: NodeJS.ProcessEnv = {}, cwd = root): void {
    console.log(`> ${command} ${args.join(" ")}`);
    const result = spawnSync(command, args, { cwd, stdio: "inherit", env: { ...process.env, ...env } });
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`${command} failed with exit code ${result.status}`);
}

function capture(command: string, args: string[]): string {
    const result = spawnSync(command, args, { cwd: root, encoding: "utf8" });
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed: ${result.stderr}`);
    return result.stdout.trim();
}

/// The host triple and the toolchain's llvm-profdata, which must match the
/// LLVM version of the rustc that instruments the build.
function toolchain(): { host: string; profdata: string } {
    const host = /^host: (.+)$/m.exec(capture("rustc", ["-vV"]))?.[1];
    if (!host) throw new Error("cannot determine the host triple from `rustc -vV`");
    const sysroot = capture("rustc", ["--print", "sysroot"]);
    const profdata = join(sysroot, "lib", "rustlib", host, "bin", process.platform === "win32" ? "llvm-profdata.exe" : "llvm-profdata");
    if (!existsSync(profdata)) {
        throw new Error(`llvm-profdata not found at ${profdata}; install it with: rustup component add llvm-tools`);
    }
    return { host, profdata };
}

/// Builds Napalm for `host` with extra rustflags. `--config` arrays are
/// appended to the ones in .cargo/config.toml (crt-static, aes), which a
/// RUSTFLAGS variable would replace. `--target` keeps the flags off build
/// scripts and proc-macros, which run on the host during the build.
function cargo(subcommand: string[], host: string, targetDir: string, rustflags: string[]): void {
    const flags = rustflags.map((flag) => `'${flag}'`).join(", ");
    run("cargo", [...subcommand, "--release", "--target", host, "--config", `target.${host}.rustflags=[${flags}]`], {
        CARGO_TARGET_DIR: targetDir,
    });
}

interface Job {
    label: string;
    command: string;
    args: string[];
    cwd: string;
    env: NodeJS.ProcessEnv;
}

interface JobResult {
    job: Job;
    code: number | null;
    timedOut: boolean;
    millis: number;
}

function runJob(job: Job, timeoutMs: number): Promise<JobResult> {
    const started = performance.now();
    return new Promise((done) => {
        const child = spawn(job.command, job.args, {
            cwd: job.cwd,
            env: { ...process.env, ...job.env },
            stdio: ["ignore", "ignore", "ignore"],
            windowsHide: true,
        });
        let timedOut = false;
        const timer = setTimeout(() => {
            timedOut = true;
            child.kill();
        }, timeoutMs);
        const finish = (code: number | null): void => {
            clearTimeout(timer);
            done({ job, code, timedOut, millis: performance.now() - started });
        };
        child.on("error", () => finish(null));
        child.on("close", (code) => finish(code));
    });
}

async function runPool(jobs: Job[], parallel: number, timeoutMs: number): Promise<JobResult[]> {
    const results: JobResult[] = [];
    let next = 0;
    const worker = async (): Promise<void> => {
        while (next < jobs.length) {
            const job = jobs[next++];
            const result = await runJob(job, timeoutMs);
            results.push(result);
            const status = result.timedOut ? "TIMEOUT" : result.code === 0 ? "ok" : `exit ${result.code}`;
            console.log(`  ${status.padEnd(8)} ${(result.millis / 1000).toFixed(1).padStart(6)}s  ${job.label}`);
        }
    };
    await Promise.all(Array.from({ length: Math.min(parallel, jobs.length) }, worker));
    return results;
}

function napalmFiles(dir: string): string[] {
    if (!existsSync(dir)) return [];
    return readdirSync(dir)
        .filter((file) => file.endsWith(".napalm") && !SKIP_SCRIPTS.has(basename(file, ".napalm")))
        .map((file) => join(dir, file))
        .sort();
}

function profileFiles(): string[] {
    return existsSync(dataDir) ? readdirSync(dataDir).filter((file) => file.endsWith(".profraw")) : [];
}

/// Online merging into a small pool of files (`%4m`) is safe for concurrent
/// processes and keeps the profile directory small.
const profileEnv = (): NodeJS.ProcessEnv => ({ LLVM_PROFILE_FILE: join(dataDir, "napalm-%4m.profraw") });

/// Every script with the JIT first, then interpreter-only: the two runs of one
/// script never overlap (some scripts write fixed scratch files).
function scriptJobs(exe: string, scripts: string[]): Job[] {
    const modes: [string, NodeJS.ProcessEnv][] = [
        ["jit", profileEnv()],
        ["interpreter", { ...profileEnv(), NAPALM_NO_JIT: "1" }],
    ];
    return modes.flatMap(([mode, env]) =>
        scripts.map((file) => ({
            label: `${file.slice(root.length + 1)} (${mode})`,
            command: exe,
            args: [file, ...(SCRIPT_ARGS[basename(file, ".napalm")] ?? [])],
            // Repository scripts resolve their paths from the repository root.
            cwd: file.startsWith(root) ? root : dirname(file),
            env,
        })),
    );
}

/// The LSP test suite, which runs live LSP sessions against `NAPALM_BIN`.
function lspJobs(exe: string): Job[] {
    const testsDir = join(root, "lsp", "tests");
    return readdirSync(testsDir)
        .filter((file) => file.endsWith(".test.mjs") && !SKIP_LSP_TESTS.has(file))
        .sort()
        .map((file) => ({
            label: `lsp/tests/${file}`,
            command: process.execPath,
            args: ["--test", join(testsDir, file)],
            cwd: join(root, "lsp"),
            env: { ...profileEnv(), NAPALM_BIN: exe },
        }));
}

/// Same flags and target directory as the instrumented build, so the napalm
/// executable the integration tests spawn is the instrumented one.
function rustTestTraining(host: string, unitTests: boolean): void {
    console.log("\nTraining: Rust tests");
    const testArgs = unitTests ? ["test"] : ["test", "--test", "*"];
    process.env.LLVM_PROFILE_FILE = profileEnv().LLVM_PROFILE_FILE;
    try {
        cargo(testArgs, host, genTargetDir, [`-Cprofile-generate=${dataDir}`]);
    } catch (error) {
        console.log(`  Rust tests reported failures (${error instanceof Error ? error.message : error}); continuing.`);
    } finally {
        delete process.env.LLVM_PROFILE_FILE;
    }
}

async function train(exe: string, host: string, options: Options): Promise<void> {
    const scripts = [
        ...napalmFiles(join(root, "scripts")),
        ...napalmFiles(join(root, "lsp", "scripts")),
        ...options.extra.flatMap(napalmFiles),
    ];
    const jobs = [...scriptJobs(exe, scripts), ...(options.lsp ? lspJobs(exe) : [])];
    console.log(`\nTraining: ${jobs.length} runs, ${options.jobs} at a time`);
    const results = await runPool(jobs, options.jobs, options.timeoutSeconds * 1000);
    const failed = results.filter((r) => r.timedOut || r.code !== 0).length;
    if (failed > 0) {
        // Failing runs still profile everything they executed before exiting;
        // only killed ones (timeouts) are lost.
        console.log(`  ${failed} run(s) did not pass; their profile counts as long as they exited normally.`);
    }
    if (options.rustTests || options.unitTests) rustTestTraining(host, options.unitTests);
}

function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

async function compare(pgoExe: string, rounds: number): Promise<void> {
    const baseExe = join(root, "target", "release", exeName);
    if (!existsSync(baseExe)) run("cargo", ["build", "--release"]);
    console.log(`\nComparing ${rounds} interleaved rounds: target/release vs PGO`);
    const times = new Map<string, { base: number[]; pgo: number[] }>();
    for (let round = 0; round < rounds; round++) {
        for (const script of COMPARE_SCRIPTS) {
            const file = join(root, script);
            if (!existsSync(file)) continue;
            const entry = times.get(script) ?? { base: [], pgo: [] };
            for (const [kind, exe] of [["base", baseExe], ["pgo", pgoExe]] as const) {
                const job: Job = { label: script, command: exe, args: [file], cwd: root, env: {} };
                entry[kind].push((await runJob(job, 300_000)).millis);
            }
            times.set(script, entry);
        }
    }
    console.log(`\n${"benchmark".padEnd(40)} ${"release".padStart(10)} ${"pgo".padStart(10)}  change`);
    for (const [script, { base, pgo }] of times) {
        const b = median(base);
        const p = median(pgo);
        const change = ((p - b) / b) * 100;
        console.log(`${script.padEnd(40)} ${b.toFixed(0).padStart(8)}ms ${p.toFixed(0).padStart(8)}ms  ${change >= 0 ? "+" : ""}${change.toFixed(1)}%`);
    }
}

async function main(): Promise<void> {
    const options = parseArgs(process.argv.slice(2));
    const { host, profdata } = toolchain();
    const started = performance.now();

    console.log("\n== 1/4 Instrumented build");
    cargo(["build"], host, genTargetDir, [`-Cprofile-generate=${dataDir}`]);
    const genExe = join(genTargetDir, host, "release", exeName);

    console.log("\n== 2/4 Training");
    rmSync(dataDir, { recursive: true, force: true });
    mkdirSync(dataDir, { recursive: true });
    await train(genExe, host, options);
    const raw = profileFiles();
    if (raw.length === 0) throw new Error(`no .profraw files were written to ${dataDir}`);

    console.log("\n== 3/4 Merging profiles");
    const merged = join(dataDir, "napalm.profdata");
    run(profdata, ["merge", "-o", merged, ...raw.map((file) => join(dataDir, file))]);
    console.log(`  ${raw.length} raw profile(s) -> ${merged} (${(statSync(merged).size / 1024).toFixed(0)} KiB)`);

    console.log("\n== 4/4 Optimized build");
    // Functions the training never reached simply get no profile; that is
    // expected, not a mismatch worth a warning per function.
    cargo(["build"], host, useTargetDir, [`-Cprofile-use=${merged}`, "-Cllvm-args=-no-pgo-warn-mismatch"]);
    const pgoExe = join(useTargetDir, host, "release", exeName);
    console.log(`\nPGO build: ${pgoExe} (${((performance.now() - started) / 60000).toFixed(1)} min)`);

    if (options.copyTo) {
        mkdirSync(options.copyTo, { recursive: true });
        for (const file of [exeName, "LLVM-C.dll"]) {
            const source = join(useTargetDir, host, "release", file);
            if (existsSync(source)) copyFileSync(source, join(options.copyTo, file));
        }
        console.log(`Copied to ${options.copyTo}`);
    }
    if (options.compare) await compare(pgoExe, options.rounds);
}

main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
});
