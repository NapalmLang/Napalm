#!/usr/bin/env node
import {copyFileSync, existsSync, mkdirSync, chmodSync, rmSync, statSync, readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {detectTarget, getVsixEntry, validateExecutable} from './bundling.mjs';

const extensionRoot = fileURLToPath(new URL('../', import.meta.url));
const manifest = JSON.parse(readFileSync(path.join(extensionRoot, 'package.json'), 'utf8'));

function parseArgs(argv) {
    const options = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!['--target', '--server', '--out'].includes(arg)) {
            throw new Error(`Unknown argument: ${arg}`);
        }
        if (options[arg] !== undefined || ++i >= argv.length || argv[i].startsWith('--')) {
            throw new Error(`Missing or duplicate value for ${arg}`);
        }
        options[arg] = argv[i];
    }
    return options;
}

try {
    const args = parseArgs(process.argv.slice(2));
    const target = args['--target'] || detectTarget();
    const executable = target.startsWith('win32-') ? 'napalm-lsp.exe' : 'napalm-lsp';
    const requested = args['--server'] || path.join('..', '..', target.startsWith('win32-') ? 'main.exe' : 'main');
    const source = path.resolve(extensionRoot, requested);
    if (!existsSync(source)) {
        throw new Error(`Compiled LSP binary not found: ${source}\nBuild it first with: napalm --compile ./lsp/main.napalm`);
    }
    const sourceSize = validateExecutable(source, target);
    const stageDir = path.join(extensionRoot, 'server');
    const staged = path.join(stageDir, executable);
    mkdirSync(stageDir, {recursive: true});
    for (const old of ['napalm-lsp.exe', 'napalm-lsp']) {
        const oldPath = path.join(stageDir, old);
        if (oldPath !== source) rmSync(oldPath, {force: true});
    }
    if (source !== staged) copyFileSync(source, staged);
    if (!target.startsWith('win32-')) chmodSync(staged, 0o755);
    if (statSync(staged).size !== sourceSize) throw new Error('Copy verification failed');

    const output = path.resolve(extensionRoot, args['--out'] || `${manifest.name}-${manifest.version}@${target}.vsix`);
    const vsceRoot = path.join(extensionRoot, 'node_modules', '@vscode', 'vsce');
    const vsceManifest = path.join(vsceRoot, 'package.json');
    if (!existsSync(vsceManifest)) throw new Error('Missing local @vscode/vsce. Run npm install in lsp/editors/vscode.');
    const vscePackage = JSON.parse(readFileSync(vsceManifest, 'utf8'));
    const binRelative = typeof vscePackage.bin === 'string' ? vscePackage.bin : vscePackage.bin?.vsce;
    if (!binRelative) throw new Error('Cannot locate the installed @vscode/vsce CLI');
    const vsceBin = path.resolve(vsceRoot, binRelative);
    if (!vsceBin.startsWith(vsceRoot + path.sep) || !existsSync(vsceBin)) {
        throw new Error('Invalid @vscode/vsce CLI location');
    }
    const run = spawnSync(process.execPath, [vsceBin, 'package', '--target', target, '--out', output, '--allow-missing-repository'], {
        cwd: extensionRoot, stdio: 'inherit', shell: false,
    });
    if (run.error) throw run.error;
    if (run.status !== 0) throw new Error(`vsce exited with code ${run.status}`);
    const found = getVsixEntry(output, `extension/server/${executable}`);
    if (!found || found.uncompressedSize !== sourceSize) {
        throw new Error(`VSIX verification failed: missing or truncated extension/server/${executable}`);
    }
    console.log(`Verified ${output}: bundled ${executable} (${sourceSize} bytes) for ${target}`);
} catch (error) {
    console.error(`[napalm-vsix] ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
}
