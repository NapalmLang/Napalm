import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {getVsixEntry, identifyExecutable, validateExecutable} from '../scripts/bundling.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const require = createRequire(import.meta.url);

function withTemporaryDirectory(run) {
    const dir = mkdtempSync(path.join(tmpdir(), 'napalm-bundling-'));
    try { return run(dir); } finally { rmSync(dir, {recursive: true, force: true}); }
}

function pe(machine) {
    const data = Buffer.alloc(1024);
    data.writeUInt16LE(0x5a4d, 0);
    data.writeUInt32LE(0x80, 0x3c);
    data.writeUInt32LE(0x4550, 0x80);
    data.writeUInt16LE(machine, 0x84);
    return data;
}

function elf(machine) {
    const data = Buffer.alloc(1024);
    Buffer.from([0x7f, 0x45, 0x4c, 0x46, 2, 1]).copy(data);
    data.writeUInt16LE(machine, 18);
    return data;
}

function macho(cpu) {
    const data = Buffer.alloc(1024);
    Buffer.from([0xcf, 0xfa, 0xed, 0xfe]).copy(data);
    data.writeUInt32LE(cpu, 4);
    return data;
}

// Minimal ZIP central directory fixture. No compression is necessary because
// the verifier reads only metadata, not entry payloads.
function centralDirectoryZip(name, size) {
    const filename = Buffer.from(name, 'utf8');
    const local = Buffer.alloc(30 + filename.length);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(filename.length, 26);
    filename.copy(local, 30);
    const central = Buffer.alloc(46 + filename.length);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt32LE(size, 20);
    central.writeUInt32LE(size, 24);
    central.writeUInt16LE(filename.length, 28);
    filename.copy(central, 46);
    const end = Buffer.alloc(22);
    end.writeUInt32LE(0x06054b50, 0);
    end.writeUInt16LE(1, 8);
    end.writeUInt16LE(1, 10);
    end.writeUInt32LE(central.length, 12);
    end.writeUInt32LE(local.length, 16);
    return Buffer.concat([local, central, end]);
}

test('binary detection matches Windows, Linux and macOS architectures', () => withTemporaryDirectory((dir) => {
    for (const [name, data, target] of [
        ['win-x64.exe', pe(0x8664), 'win32-x64'],
        ['win-arm64.exe', pe(0xaa64), 'win32-arm64'],
        ['linux-x64', elf(0x3e), 'linux-x64'],
        ['linux-arm64', elf(0xb7), 'linux-arm64'],
        ['mac-x64', macho(0x01000007), 'darwin-x64'],
        ['mac-arm64', macho(0x0100000c), 'darwin-arm64'],
    ]) {
        const file = path.join(dir, name);
        writeFileSync(file, data);
        assert.equal(identifyExecutable(file), target);
        assert.equal(validateExecutable(file, target), 1024);
    }
}));

test('rejects an executable with the wrong target or an invalid header', () => withTemporaryDirectory((dir) => {
    const file = path.join(dir, 'lsp.exe');
    writeFileSync(file, pe(0x8664));
    assert.throws(() => validateExecutable(file, 'win32-arm64'), /Binary is for win32-x64/);
    writeFileSync(file, Buffer.alloc(1024));
    assert.throws(() => validateExecutable(file, 'win32-x64'), /not a supported executable/);
}));

test('VSIX verifier rejects missing entries and accepts the correctly sized server', () => withTemporaryDirectory((dir) => {
    const file = path.join(dir, 'sample.vsix');
    writeFileSync(file, centralDirectoryZip('extension/server/napalm-lsp.exe', 123456));
    assert.deepEqual(getVsixEntry(file, 'extension/server/napalm-lsp.exe'), {
        name: 'extension/server/napalm-lsp.exe', uncompressedSize: 123456,
    });
    assert.equal(getVsixEntry(file, 'extension/server/other.exe'), null);
}));

test('VS Code client prefers a packaged executable unless overridden', () => {
    const client = readFileSync(path.join(root, 'extension.js'), 'utf8');
    assert.match(client, /context\.extensionPath/);
    assert.match(client, /path\.join\(extensionPath, 'server', bundledName\)/);
    assert.match(client, /languageServerPath/);
    assert.ok(client.indexOf("if (specified !== '')") < client.indexOf('const bundled ='));
});

test('extension manifest includes the native server directory for packaging', () => {
    const manifest = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'));
    assert.ok(manifest.files.includes('server'));
    assert.ok(manifest.scripts['package:bundled']);
    assert.ok(existsSync(path.join(root, 'scripts', 'package-bundled.mjs')));
});

test('VSCE includes the language client and its runtime dependencies, excluding build tools', {
    skip: !existsSync(path.join(root, 'node_modules/@vscode/vsce/package.json')) && 'Install extension dependencies first',
}, async () => {
    const {listFiles} = require('@vscode/vsce');
    const files = new Set(await listFiles({cwd: root}));
    for (const entry of [
        'vscode-languageclient/package.json',
        'vscode-languageclient/lib/node/main.js',
        'vscode-languageserver-protocol/lib/node/main.js',
        'vscode-jsonrpc/lib/node/main.js',
        'vscode-languageserver-types/lib/umd/main.js',
        'vscode-languageserver-textdocument/lib/umd/main.js',
        'semver/functions/satisfies.js',
        'minimatch/package.json',
    ]) assert.ok(files.has(`node_modules/${entry}`), `Missing runtime dependency: ${entry}`);
    assert.ok(!files.has('node_modules/typescript/package.json'));
    assert.ok(!files.has('node_modules/@vscode/vsce/package.json'));
});
