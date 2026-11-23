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
    chmodSync, closeSync, copyFileSync, existsSync, mkdirSync, openSync, readdirSync,
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
    checkAssets: boolean;
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
        checkAssets: false,
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
            case '--check-assets':
                options.checkAssets = true;
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
    if (options.help || options.checkAssets) return options;
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

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] as const;

function firstBytesHex(data: Uint8Array, count = 16): string {
    return Buffer.from(data.subarray(0, count)).toString('hex') || '(empty file)';
}

function detectedFormat(data: Uint8Array): string {
    if (data.length === 0) return 'empty file';
    if (data.length >= 4 && data[0] === 0x00 && data[1] === 0x00 && data[2] === 0x01 && data[3] === 0x00) return 'Windows ICO';
    if (data.length >= 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) return 'JPEG';
    if (data.length >= 6 && String.fromCharCode(...data.subarray(0, 6)).startsWith('GIF8')) return 'GIF';
    if (data.length >= 12 && String.fromCharCode(...data.subarray(0, 4)) === 'RIFF' &&
        String.fromCharCode(...data.subarray(8, 12)) === 'WEBP') return 'WebP';
    const prefix = Buffer.from(data.subarray(0, 128)).toString('utf8').replace(/^\uFEFF/, '').trimStart();
    if (prefix.startsWith('<svg') || prefix.startsWith('<?xml')) return 'XML/SVG';
    if (prefix.startsWith('<!DOCTYPE html') || prefix.startsWith('<html')) return 'HTML (possibly a download error page)';
    return 'unknown format';
}

/** Read the dimensions of a PNG, including its source path in failures. */
export function pngSize(data: Uint8Array, source = 'PNG data'): {width: number; height: number} {
    const validSignature = data.length >= PNG_SIGNATURE.length &&
        PNG_SIGNATURE.every((byte, index) => data[index] === byte);
    if (!validSignature) {
        throw new Error(`Not a valid PNG image: ${source} (${data.length} bytes; detected ${detectedFormat(data)}; ` +
            `first bytes: ${firstBytesHex(data)}). Expected PNG signature 89504e470d0a1a0a.`);
    }
    if (data.length < 33 || data[8] !== 0x00 || data[9] !== 0x00 || data[10] !== 0x00 || data[11] !== 0x0d ||
        data[12] !== 0x49 || data[13] !== 0x48 || data[14] !== 0x44 || data[15] !== 0x52) {
        throw new Error(`Invalid PNG IHDR chunk: ${source} (${data.length} bytes)`);
    }
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const width = view.getUint32(16, false);
    const height = view.getUint32(20, false);
    if (!width || !height || width > 8192 || height > 8192) {
        throw new Error(`Unexpected PNG dimensions at ${source}: ${width}x${height}`);
    }
    return {width, height};
}

function readPng(file: string): Buffer {
    let data: Buffer;
    try {
        data = readFileSync(file);
    } catch (error) {
        throw new Error(`Cannot read image ${file}: ${error instanceof Error ? error.message : String(error)}`);
    }
    pngSize(data, file);
    return data;
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

/** Validate all assets before copying, so a corrupt source cannot leave partial output. */
export function prepareBranding(extensionRoot = EXTENSION_ROOT): void {
    const source = path.join(extensionRoot, 'branding');
    const dest = path.join(extensionRoot, 'icons');
    const images: Array<{size: number; data: Buffer}> = [];
    for (const size of BRAND_SIZES) {
        const input = path.join(source, `flame-${size}.png`);
        const data = readPng(input);
        const dimensions = pngSize(data, input);
        if (dimensions.width !== size || dimensions.height !== size) {
            throw new Error(`Expected ${size}x${size} PNG at ${input}, got ${dimensions.width}x${dimensions.height}`);
        }
        images.push({size, data});
    }
    const logoPath = path.join(source, 'extension-logo-256.png');
    const logo = readPng(logoPath);
    const logoSize = pngSize(logo, logoPath);
    if (logoSize.width !== 256 || logoSize.height !== 256) {
        throw new Error(`Marketplace logo must be 256x256 PNG: ${logoPath}; got ${logoSize.width}x${logoSize.height}`);
    }

    for (const name of ['generic-file.png', 'generic-folder.png', 'generic-folder-open.png']) {
        const file = path.join(extensionRoot, 'themes', name);
        const size = pngSize(readPng(file), file);
        if (size.width !== 48 || size.height !== 48) {
            throw new Error(`Expected 48x48 theme PNG: ${file}; got ${size.width}x${size.height}`);
        }
    }

    // All inputs have passed validation. Staging is now safe to begin.
    for (const {size, data} of images) {
        writeIfDifferent(path.join(dest, `flame-${size}.png`), data);
    }
    writeIfDifferent(path.join(dest, 'extension-logo-256.png'), logo);
    writeIfDifferent(path.join(dest, 'napalm-file.ico'), icoFromPngs(images));

    console.log('[napalm-vscode] Branding validated: five flame PNGs, Marketplace logo and three theme PNGs.');
    console.log('[napalm-vscode] Branding staged: flame icons, extension logo and Windows ICO.');
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

/**
 * Build a VS Code CLI invocation without nested cmd.exe quoting on Windows.
 *
 * code.cmd is typically a shell wrapper. Passing an absolute, quoted VSIX
 * path through Bun -> cmd.exe -> code.cmd can preserve the quote characters
 * literally. Running from the artifact directory with a validated basename
 * eliminates both the quoting ambiguity and shell metacharacter injection.
 */
export function vsixInstallInvocation(file: string, platform = process.platform): {
    command: string;
    args: string[];
    cwd: string;
    shell: boolean;
} {
    const absolute = path.resolve(file);
    const directory = path.dirname(absolute);
    if (platform === 'win32') {
        const filename = path.basename(absolute);
        if (!/^[A-Za-z0-9][A-Za-z0-9._@-]*\.vsix$/i.test(filename)) {
            throw new Error(`Unsafe VSIX filename for Windows cmd.exe: ${filename}`);
        }
        return {
            command: process.env.ComSpec || 'cmd.exe',
            args: ['/d', '/s', '/c', `code --install-extension ${filename} --force`],
            cwd: directory,
            shell: false,
        };
    }
    return {
        command: 'code',
        args: ['--install-extension', absolute, '--force'],
        cwd: directory,
        shell: false,
    };
}

function installVsix(file: string): void {
    if (!existsSync(file)) {
        throw new Error(`Cannot install missing VSIX: ${file}`);
    }
    const invocation = vsixInstallInvocation(file);
    run(invocation.command, invocation.args, invocation.cwd, invocation.shell);
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
        `  --check-assets    Validate icons and logo without compiling or packaging\n` +
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
    if (options.checkAssets) {
        prepareBranding();
        console.log('[napalm-vscode] PASS: all branding assets are valid PNG images.');
        return "";
    }
    if (options.publish) {
        if (!manifest.publisher || manifest.publisher === 'napalm-local') {
            throw new Error('Configure your real Marketplace publisher ID in package.json before using --publish');
        }
        if (!process.env.VSCE_PAT) {
            throw new Error('Set VSCE_PAT in the environment; do not put Marketplace credentials in source control');
        }
    }
    // Check artwork before compiling the potentially large Napalm executable.
    prepareBranding();
    const compiled = compileNapalm(options);
    if (!options.skipNpm) {
        const lockfile = existsSync(path.join(EXTENSION_ROOT, 'package-lock.json'));
        runNpm([lockfile ? 'ci' : 'install', '--include=dev', '--no-audit', '--no-fund']);
    }
    if (!options.skipTests) {
        // Use the installed tools directly. An older local package.json may not
        // declare "typecheck" or "test" scripts, even when the tools are installed.
        // Node.js is required for npm and @vscode/vsce, independently of Bun.
        const node = process.platform === 'win32' ? 'node.exe' : 'node';
        const tsc = path.join(EXTENSION_ROOT, 'node_modules', 'typescript', 'bin', 'tsc');
        if (!existsSync(tsc)) {
            throw new Error('TypeScript compiler not installed: ' + tsc +
                '. Run the builder without --skip-npm, or npm install in lsp/editors/vscode.');
        }
        const nodeTypes = path.join(EXTENSION_ROOT, 'node_modules', '@types', 'node', 'package.json');
        if (!existsSync(nodeTypes)) {
            throw new Error('Node.js type declarations are missing: ' + nodeTypes +
                '. Run npm install --include=dev in lsp/editors/vscode, or run the builder without --skip-npm.');
        }
        run(node, [tsc, '--project', path.join('lsp', 'tsconfig.build-vscode.json')], REPOSITORY_ROOT);

        const testDirectory = path.join(EXTENSION_ROOT, 'tests');
        const extensionTests = readdirSync(testDirectory)
            .filter((name) => name.endsWith('.test.mjs'))
            .sort()
            .map((name) => path.join(testDirectory, name));
        if (extensionTests.length === 0) {
            throw new Error('No VS Code extension tests found in ' + testDirectory);
        }
        run(node, ['--test', ...extensionTests], EXTENSION_ROOT);

        // The build script lives outside the VS Code extension and has its own tests.
        run(node, ['--test', path.join(REPOSITORY_ROOT, 'lsp', 'tests', 'build-vscode.test.mjs')], REPOSITORY_ROOT);
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
