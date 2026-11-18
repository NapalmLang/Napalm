#!/usr/bin/env node
/**
 * Build and optionally install/publish a self-contained Napalm VS Code extension.
 *
 * This build script intentionally has no runtime dependency on a TypeScript
 * transpiler: Bun or Node.js 22+ with --experimental-strip-types can execute it.
 * The language server itself is compiled from Napalm, not JavaScript.
 */
import {spawnSync} from 'node:child_process';
import {
    chmodSync, closeSync, copyFileSync, existsSync, mkdirSync, openSync,
    readFileSync, readSync, rmSync, statSync, writeFileSync,
} from 'node:fs';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {detectTarget, getVsixEntry, validateExecutable} from './editors/vscode/scripts/bundling.mjs';

export const REPOSITORY_ROOT = path.resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
export const EXTENSION_ROOT = path.join(REPOSITORY_ROOT, 'lsp', 'editors', 'vscode');
export const BRAND_SIZES = [16, 32, 48, 64, 256] as const;

export type Options = {
    target: string;
    napalm: string;
    binary: string | null;
    out: string;
    skipCompile: boolean;
    skipNpm: boolean;
    skipTests: boolean;
    install: boolean;
    publish: boolean;
    help: boolean;
};

function optionValue(argv: string[], index: number, option: string): string {
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) {
        throw new Error(`Missing value for ${option}`);
    }
    return value;
}

export function parseOptions(argv: string[], host = detectTarget()): Options {
    const options: Options = {
        target: host,
        napalm: path.join('target', 'release', process.platform === 'win32' ? 'napalm.exe' : 'napalm'),
        binary: null,
        out: path.join('dist', 'vscode'),
        skipCompile: false,
        skipNpm: false,
        skipTests: false,
        install: false,
        publish: false,
        help: false,
    };
    const seen = new Set<string>();
    for (let i = 0; i < argv.length; i++) {
        const name = argv[i];
        if (name === '--help' || name === '-h') {
            options.help = true;
            continue;
        }
        if (seen.has(name)) throw new Error(`Repeated option: ${name}`);
        seen.add(name);
        switch (name) {
            case '--target':
                options.target = optionValue(argv, i++, name);
                break;
            case '--napalm':
                options.napalm = optionValue(argv, i++, name);
                break;
            case '--binary':
                options.binary = optionValue(argv, i++, name);
                break;
            case '--out':
                options.out = optionValue(argv, i++, name);
                break;
            case '--skip-compile':
                options.skipCompile = true;
                break;
            case '--skip-npm':
                options.skipNpm = true;
                break;
            case '--skip-tests':
                options.skipTests = true;
                break;
            case '--install':
                options.install = true;
                break;
            case '--publish':
                options.publish = true;
                break;
            default:
                throw new Error(`Unknown option: ${name}. Use --help for usage.`);
        }
    }
    if (options.help) return options;
    const supported = ['win32-x64', 'win32-arm64', 'linux-x64', 'linux-arm64', 'darwin-x64', 'darwin-arm64'];
    if (!supported.includes(options.target)) throw new Error(`Unsupported target: ${options.target}`);
    if (!options.skipCompile && options.binary !== null) {
        throw new Error('--binary requires --skip-compile (the compiled output is otherwise lsp/main.exe or lsp/main)');
    }
    if (!options.skipCompile && options.target !== host) {
        throw new Error(`Cannot natively compile ${options.target} on ${host}; use --skip-compile --binary PATH with a cross-built executable.`);
    }
    return options;
}

function asAbsolute(file: string): string {
    return path.resolve(REPOSITORY_ROOT, file);
}

function run(command: string, args: string[], cwd: string, shell = false): void {
    console.log(`\n[napalm-vscode] ${path.basename(command)} ${args.join(' ')}`);
    const result = spawnSync(command, args, {cwd, shell, stdio: 'inherit', windowsHide: true, maxBuffer: 8 * 1024 * 1024});
    if (result.error) throw new Error(`Cannot run ${command}: ${result.error.message}`);
    if (result.signal) throw new Error(`${command} terminated by signal ${result.signal}`);
    if (result.status !== 0) throw new Error(`${command} exited with code ${result.status ?? 'unknown'}`);
}

/** Use a JS npm entry point where possible, avoiding cmd.exe quoting on Windows. */
function runNpm(args: string[]): void {
    const fromEnvironment = process.env.npm_execpath;
    const nearNode = path.join(path.dirname(process.execPath), 'node_modules', 'npm', 'bin', 'npm-cli.js');
    const cli = [fromEnvironment, nearNode].find((candidate) =>
        !!candidate && /\.(?:c?js|mjs)$/i.test(candidate) && existsSync(candidate)
    );
    if (cli) {
        run(process.execPath, [cli, ...args], EXTENSION_ROOT);
    } else if (process.platform === 'win32') {
        // These arguments are controlled constants, never user-provided paths.
        run('npm.cmd', args, EXTENSION_ROOT, true);
    } else {
        run('npm', args, EXTENSION_ROOT);
    }
}

function readManifest(): {name: string; version: string; publisher?: string; license?: string; contributes?: {
    languages?: Array<{extensions?: string[]}>;
}} {
    const manifestPath = path.join(EXTENSION_ROOT, 'package.json');
    return JSON.parse(readFileSync(manifestPath, 'utf8'));
}

export function pngSize(data: Buffer): {width: number; height: number} {
    if (data.length < 33 || !data.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) {
        throw new Error('Not a valid PNG image');
    }
    if (data.readUInt32BE(8) !== 13 || data.toString('ascii', 12, 16) !== 'IHDR') {
        throw new Error('PNG image has no valid IHDR header');
    }
    const width = data.readUInt32BE(16);
    const height = data.readUInt32BE(20);
    if (!width || !height || width > 8192 || height > 8192) {
        throw new Error(`Unexpected PNG dimensions: ${width}x${height}`);
    }
    return {width, height};
}

/** A Windows ICO supports storing a complete PNG in each icon directory entry. */
export function icoFromPngs(images: Array<{size: number; data: Buffer}>): Buffer {
    if (images.length === 0 || images.length > 65535) throw new Error('Invalid number of icon images');
    const header = Buffer.alloc(6 + 16 * images.length);
    header.writeUInt16LE(1, 2);
    header.writeUInt16LE(images.length, 4);
    let offset = header.length;
    images.forEach(({size, data}, i) => {
        if (size < 1 || size > 256) throw new Error(`Unsupported ICO image size: ${size}`);
        const actual = pngSize(data);
        if (actual.width !== size || actual.height !== size) throw new Error(`Invalid ${size}x${size} source image`);
        const entry = 6 + i * 16;
        header.writeUInt8(size === 256 ? 0 : size, entry);
        header.writeUInt8(size === 256 ? 0 : size, entry + 1);
        header.writeUInt16LE(1, entry + 4);
        header.writeUInt16LE(32, entry + 6);
        header.writeUInt32LE(data.length, entry + 8);
        header.writeUInt32LE(offset, entry + 12);
        offset += data.length;
    });
    return Buffer.concat([header, ...images.map((item) => item.data)]);
}

function writeIfDifferent(destination: string, data: Buffer): void {
    if (existsSync(destination) && readFileSync(destination).equals(data)) return;
    mkdirSync(path.dirname(destination), {recursive: true});
    writeFileSync(destination, data);
}

/** Copies the user's flame artwork and extension logo, and regenerates the ICO. */
export function prepareBranding(extensionRoot = EXTENSION_ROOT): void {
    const source = path.join(extensionRoot, 'branding');
    const dest = path.join(extensionRoot, 'icons');
    const images: Array<{size: number; data: Buffer}> = [];
    for (const size of BRAND_SIZES) {
        const input = path.join(source, `flame-${size}.png`);
        const data = readFileSync(input);
        const dimensions = pngSize(data);
        if (dimensions.width !== size || dimensions.height !== size) {
            throw new Error(`Expected ${size}x${size} image at ${input}, got ${dimensions.width}x${dimensions.height}`);
        }
        writeIfDifferent(path.join(dest, `flame-${size}.png`), data);
        images.push({size, data});
    }
    const logo = readFileSync(path.join(source, 'extension-logo-256.png'));
    const logoSize = pngSize(logo);
    if (logoSize.width !== 256 || logoSize.height !== 256) throw new Error('Extension Marketplace logo must be 256x256 PNG');
    writeIfDifferent(path.join(dest, 'extension-logo-256.png'), logo);
    writeIfDifferent(path.join(dest, 'napalm-file.ico'), icoFromPngs(images));

    for (const name of ['generic-file.svg', 'generic-folder.svg', 'generic-folder-open.svg']) {
        const file = path.join(extensionRoot, 'themes', name);
        const size = pngSize(readFileSync(file));
        if (size.width !== 48 || size.height !== 48) throw new Error(`Expected 48x48 theme asset: ${file}`);
    }
    console.log('[napalm-vscode] Branding prepared: 5 flame sizes, extension logo, ICO and theme PNGs.');
}

function validateManifest(): ReturnType<typeof readManifest> {
    const manifest = readManifest();
    const registered = new Set((manifest.contributes?.languages ?? []).flatMap((lang) => lang.extensions ?? []));
    for (const name of ['.napalm', '.burning', '.fire', '.orange']) {
        if (!registered.has(name)) throw new Error(`VS Code manifest does not register ${name}`);
    }
    if (!manifest.name || !manifest.version) throw new Error('Extension name/version are missing');
    return manifest;
}

function compileNapalm(options: Options): string {
    const entry = path.join(REPOSITORY_ROOT, 'lsp', 'main.napalm');
    const nativeName = process.platform === 'win32' ? 'main.exe' : 'main';
    const output = options.binary ? asAbsolute(options.binary) : path.join(REPOSITORY_ROOT, 'lsp', nativeName);
    if (!options.skipCompile) {
        if (!existsSync(entry)) throw new Error(`Napalm LSP source not found: ${entry}`);
        const compiler = asAbsolute(options.napalm);
        if (!existsSync(compiler)) throw new Error(`Napalm compiler missing: ${compiler}\nBuild the engine or specify --napalm PATH`);
        // Never accidentally ship an old binary when a compile command fails.
        rmSync(output, {force: true});
        run(compiler, ['--compile', './lsp/main.napalm'], REPOSITORY_ROOT);
    }
    if (!existsSync(output)) throw new Error(`Compiled LSP not found: ${output}`);
    const bytes = validateExecutable(output, options.target);
    console.log(`[napalm-vscode] Native server: ${output} (${bytes.toLocaleString()} bytes, ${options.target})`);
    return output;
}

function hashFile(file: string): string {
    const hash = createHash('sha256');
    const fd = openSync(file, 'r');
    const buffer = Buffer.allocUnsafe(1024 * 1024);
    try {
        let size = 0;
        while ((size = readSync(fd, buffer, 0, buffer.length, null)) > 0) {
            hash.update(buffer.subarray(0, size));
        }
    } finally {
        closeSync(fd);
    }
    return hash.digest('hex');
}

function stageServer(source: string, target: string): {name: string; size: number; sha256: string} {
    const name = target.startsWith('win32-') ? 'napalm-lsp.exe' : 'napalm-lsp';
    const serverDir = path.join(EXTENSION_ROOT, 'server');
    const sourceSize = validateExecutable(source, target);
    mkdirSync(serverDir, {recursive: true});
    for (const previous of ['napalm-lsp.exe', 'napalm-lsp']) {
        const file = path.join(serverDir, previous);
        if (path.resolve(file) !== path.resolve(source)) rmSync(file, {force: true});
    }
    const staged = path.join(serverDir, name);
    if (path.resolve(source) !== path.resolve(staged)) copyFileSync(source, staged);
    if (!target.startsWith('win32-')) chmodSync(staged, 0o755);
    if (statSync(staged).size !== sourceSize) throw new Error('The native server was copied incompletely');
    const original = hashFile(source);
    const copied = hashFile(staged);
    if (original !== copied) throw new Error('The staged native server has a different SHA-256');
    return {name, size: sourceSize, sha256: original};
}

function installedVsce(): string {
    const folder = path.join(EXTENSION_ROOT, 'node_modules', '@vscode', 'vsce');
    const manifestFile = path.join(folder, 'package.json');
    if (!existsSync(manifestFile)) throw new Error('Missing @vscode/vsce; remove --skip-npm and try again');
    const pkg: {bin?: string | Record<string, string>} = JSON.parse(readFileSync(manifestFile, 'utf8'));
    const bin = typeof pkg.bin === 'string' ? pkg.bin : pkg.bin?.vsce;
    if (!bin) throw new Error('Cannot resolve @vscode/vsce CLI from its package.json');
    const cli = path.resolve(folder, bin);
    if (!cli.startsWith(folder + path.sep) || !existsSync(cli)) throw new Error('Invalid VSCE CLI path');
    return cli;
}

function installVsix(file: string): void {
    if (process.platform === 'win32') {
        // Windows extension CLI is normally code.cmd. Quote the path so spaces
        // and shell metacharacters within the VSIX pathname are not evaluated.
        const safe = file.replaceAll('%', '%%').replaceAll('"', '');
        const command = `code --install-extension "${safe}" --force`;
        run(process.env.ComSpec || 'cmd.exe', ['/d', '/s', '/c', command], REPOSITORY_ROOT);
    } else {
        run('code', ['--install-extension', file, '--force'], REPOSITORY_ROOT);
    }
}

function verifyPackage(vsix: string, binary: {name: string; size: number}): void {
    const required = [
        ['extension/package.json', null],
        ['extension/extension.js', null],
        [`extension/server/${binary.name}`, binary.size],
        ['extension/icons/extension-logo-256.png', null],
        ['extension/icons/flame-32.png', null],
        ['extension/themes/napalm-file-icon-theme.json', null],
        ['extension/themes/generic-file.png', null],
        ['extension/syntaxes/napalm.tmLanguage.json', null],
    ] as const;
    for (const [name, expectedBytes] of required) {
        const entry = getVsixEntry(vsix, name);
        if (!entry || entry.uncompressedSize <= 0) throw new Error(`VSIX is missing required resource ${name}`);
        if (expectedBytes !== null && entry.uncompressedSize !== expectedBytes) {
            throw new Error(`VSIX contains a truncated/outdated executable: ${name}`);
        }
    }
}

export function printHelp(): void {
    console.log(`Napalm VS Code extension builder\n\n` +
        `Usage (from repository root):\n` +
        `  bun run lsp/build-vscode.ts [options]\n` +
        `  node --experimental-strip-types lsp/build-vscode.ts [options]\n\n` +
        `Default: compile the LSP with target/release/napalm.exe (or napalm),\n` +
        `prepare artwork, install extension NPM dependencies, run tests,\n` +
        `package a platform-specific VSIX and verify its contents.\n\n` +
        `Options:\n` +
        `  --target TARGET   win32-x64 | win32-arm64 | linux-x64 | linux-arm64 | darwin-x64 | darwin-arm64\n` +
        `  --napalm PATH     Override the Napalm compiler executable\n` +
        `  --out DIR         VSIX output directory (default: dist/vscode)\n` +
        `  --skip-compile    Use an existing compiled LSP binary\n` +
        `  --binary PATH     With --skip-compile, use a specific native LSP binary\n` +
        `  --skip-npm        Skip NPM install/ci (dependencies must exist)\n` +
        `  --skip-tests      Skip extension tests\n` +
        `  --install         Install the resulting VSIX into local VS Code\n` +
        `  --publish         Publish to Marketplace (requires real publisher and VSCE_PAT)\n` +
        `  --help            Print this help\n`);
}

export function build(options: Options): string {
    console.log(`[napalm-vscode] Building for ${options.target} from ${REPOSITORY_ROOT}`);
    const manifest = validateManifest();
    if (options.publish) {
        if (!manifest.publisher || manifest.publisher === 'napalm-local') {
            throw new Error('Configure your real Marketplace publisher ID in package.json before using --publish');
        }
        if (!process.env.VSCE_PAT) {
            throw new Error('Set VSCE_PAT in the environment; do not put Marketplace credentials in source control');
        }
    }
    const compiled = compileNapalm(options);
    prepareBranding();
    if (!options.skipNpm) {
        const lockfile = existsSync(path.join(EXTENSION_ROOT, 'package-lock.json'));
        runNpm([lockfile ? 'ci' : 'install', '--no-audit', '--no-fund']);
    }
    if (!options.skipTests) {
        runNpm(['run', 'typecheck']);
        runNpm(['test']);
        // The build script lives outside the VS Code extension and has separate tests.
        run(process.execPath, ['--experimental-strip-types', '--test', path.join('lsp', 'tests', 'build-vscode.test.mjs')], REPOSITORY_ROOT);
    }
    const binary = stageServer(compiled, options.target);
    const outputFolder = asAbsolute(options.out);
    mkdirSync(outputFolder, {recursive: true});
    const vsix = path.join(outputFolder, `${manifest.name}-${manifest.version}@${options.target}.vsix`);
    rmSync(vsix, {force: true});
    const vsce = installedVsce();
    run(process.execPath, [vsce, 'package', '--target', options.target, '--out', vsix, '--allow-missing-repository'], EXTENSION_ROOT);
    verifyPackage(vsix, binary);
    console.log(`\n[napalm-vscode] SUCCESS: ${vsix}`);
    console.log(`[napalm-vscode] Verified embedded server: ${binary.name}, ${binary.size} bytes, SHA-256 ${binary.sha256}`);
    if (options.install) installVsix(vsix);
    if (options.publish) {
        run(process.execPath, [vsce, 'publish', '--packagePath', vsix], EXTENSION_ROOT);
    }
    return vsix;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
    try {
        const options = parseOptions(process.argv.slice(2));
        if (options.help) printHelp();
        else build(options);
    } catch (error) {
        console.error(`\n[napalm-vscode] ERROR: ${error instanceof Error ? error.message : String(error)}`);
        process.exitCode = 1;
    }
}
