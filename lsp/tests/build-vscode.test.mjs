import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFileSync, mkdtempSync, mkdirSync, rmSync, writeFileSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {
    BRAND_SIZES, EXTENSION_ROOT, icoFromPngs, parseOptions, pngSize, prepareBranding,
} from '../build-vscode.ts';

function withTemp(callback) {
    const root = mkdtempSync(path.join(os.tmpdir(), 'napalm-vscode-build-'));
    try { callback(root); } finally { rmSync(root, {recursive: true, force: true}); }
}

function flame(size) {
    return readFileSync(path.join(EXTENSION_ROOT, 'branding', `flame-${size}.png`));
}

test('build defaults to compile + package for the native host', () => {
    const opts = parseOptions([], 'win32-x64');
    assert.equal(opts.target, 'win32-x64');
    assert.equal(opts.skipCompile, false);
    assert.equal(opts.install, false);
    assert.equal(opts.publish, false);
});

test('cross-platform packaging requires an explicitly precompiled server', () => {
    assert.throws(() => parseOptions(['--target', 'linux-arm64'], 'win32-x64'), /Cannot natively compile/);
    const opts = parseOptions(['--target', 'linux-arm64', '--skip-compile', '--binary', 'release/napalm-lsp'], 'win32-x64');
    assert.equal(opts.binary, 'release/napalm-lsp');
    assert.equal(opts.target, 'linux-arm64');
    assert.throws(() => parseOptions(['--binary', 'server.exe'], 'win32-x64'), /requires --skip-compile/);
});

test('rejects missing, repeated and unknown flags', () => {
    assert.throws(() => parseOptions(['--target'], 'win32-x64'), /Missing value/);
    assert.throws(() => parseOptions(['--target', 'win32-x64', '--target', 'win32-x64'], 'win32-x64'), /Repeated option/);
    assert.throws(() => parseOptions(['--release'], 'win32-x64'), /Unknown option/);
    assert.throws(() => parseOptions(['--target', 'android-arm64'], 'win32-x64'), /Unsupported target/);
});

test('all artwork is a correctly sized PNG, including the Marketplace logo', () => {
    for (const size of BRAND_SIZES) assert.deepEqual(pngSize(flame(size)), {width: size, height: size});
    const logo = readFileSync(path.join(EXTENSION_ROOT, 'branding', 'extension-logo-256.png'));
    assert.deepEqual(pngSize(logo), {width: 256, height: 256});
    assert.throws(() => pngSize(Buffer.alloc(32)), /Not a valid PNG/);
});

test('ICO generation includes each resolution with correct data offsets', () => {
    const images = BRAND_SIZES.map((size) => ({size, data: flame(size)}));
    const output = icoFromPngs(images);
    assert.equal(output.readUInt16LE(4), 5);
    images.forEach(({size, data}, index) => {
        const entry = 6 + index * 16;
        assert.equal(output[entry], size === 256 ? 0 : size);
        const bytes = output.readUInt32LE(entry + 8);
        const position = output.readUInt32LE(entry + 12);
        assert.equal(bytes, data.length);
        assert.deepEqual(output.subarray(position, position + bytes), data);
    });
});

test('branding staging is deterministic and preserves the user artwork', () => withTemp((root) => {
    const source = path.join(root, 'branding');
    const theme = path.join(root, 'themes');
    mkdirSync(source, {recursive: true});
    mkdirSync(theme, {recursive: true});
    for (const size of BRAND_SIZES) writeFileSync(path.join(source, `flame-${size}.png`), flame(size));
    writeFileSync(path.join(source, 'extension-logo-256.png'),
        readFileSync(path.join(EXTENSION_ROOT, 'branding', 'extension-logo-256.png')));
    for (const file of ['generic-file.png', 'generic-folder.png', 'generic-folder-open.png']) {
        writeFileSync(path.join(theme, file), readFileSync(path.join(EXTENSION_ROOT, 'themes', file)));
    }
    prepareBranding(root);
    prepareBranding(root);
    for (const size of BRAND_SIZES) {
        assert.deepEqual(readFileSync(path.join(root, 'icons', `flame-${size}.png`)), flame(size));
    }
    assert.equal(readFileSync(path.join(root, 'icons', 'napalm-file.ico')).readUInt16LE(4), 5);
}));

test('help is available without a compiler, npm installation or Napalm runtime', () => {
    const buildFile = fileURLToPath(new URL('../build-vscode.ts', import.meta.url));
    const output = execFileSync(process.execPath, ['--experimental-strip-types', buildFile, '--help'], {
        encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'],
    });
    assert.match(output, /--skip-compile/);
    assert.match(output, /--publish/);
});
