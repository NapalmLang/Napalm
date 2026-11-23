#!/usr/bin/env node
// Exercise the real standalone Napalm LSP through binary stdio without VS Code.
import {spawn} from 'node:child_process';
import {existsSync, mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve, join, dirname} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDirectory, '../..');
const defaultExecutable = resolve(projectRoot, process.platform === 'win32' ? 'lsp/main.exe' : 'lsp/main');
const executable = resolve(process.argv[2] ?? defaultExecutable);
const timeoutMs = 20_000;
const maxOutputBytes = 8 * 1024 * 1024;

if (!existsSync(executable)) {
    console.error(`Executable not found: ${executable}`);
    console.error('Compile the language server first: napalm --compile lsp/main.napalm');
    process.exitCode = 2;
} else {
    process.exitCode = await probe(executable);
}

function encodeFrame(message) {
    const payload = Buffer.from(JSON.stringify(message), 'utf8');
    return Buffer.concat([
        Buffer.from(`Content-Length: ${payload.length}\r\n\r\n`, 'ascii'),
        payload,
    ]);
}

function decodeFrames(buffer) {
    const messages = [];
    let position = 0;
    while (position < buffer.length) {
        const boundary = buffer.indexOf('\r\n\r\n', position);
        if (boundary < 0) throw new Error(`Unterminated response headers at byte ${position}`);
        const header = buffer.toString('ascii', position, boundary);
        const lengths = [...header.matchAll(/^Content-Length:\s*(\d+)\s*$/gim)];
        if (lengths.length !== 1) throw new Error(`Expected exactly one Content-Length header at byte ${position}; got: ${JSON.stringify(header)}`);
        const length = Number(lengths[0][1]);
        if (!Number.isSafeInteger(length) || length < 0 || length > maxOutputBytes) {
            throw new Error(`Invalid response length: ${lengths[0][1]}`);
        }
        const bodyStart = boundary + 4;
        const bodyEnd = bodyStart + length;
        if (bodyEnd > buffer.length) throw new Error(`Truncated response body: expected ${length} bytes`);
        try {
            messages.push(JSON.parse(buffer.toString('utf8', bodyStart, bodyEnd)));
        } catch (error) {
            throw new Error(`Invalid response JSON: ${error.message}`, {cause: error});
        }
        position = bodyEnd;
    }
    return messages;
}

async function probe(command) {
    const fixtureDirectory = mkdtempSync(join(tmpdir(), 'napalm-lsp-probe-'));
    const sourcePath = join(fixtureDirectory, 'probe.napalm');
    const source = 'function add(a: number, b: number): number { return a + b; }\nconst result = add(1, 2);\nconst broken = ;\n';
    writeFileSync(sourcePath, source, 'utf8');
    const uri = pathToFileURL(sourcePath).href;
    const rootUri = pathToFileURL(fixtureDirectory).href;
    const input = Buffer.concat([
        encodeFrame({jsonrpc: '2.0', id: 1, method: 'initialize', params: {processId: null, rootUri, workspaceFolders: null, capabilities: {}}}),
        encodeFrame({jsonrpc: '2.0', method: 'initialized', params: {}}),
        encodeFrame({jsonrpc: '2.0', method: 'textDocument/didOpen', params: {textDocument: {uri, languageId: 'napalm', version: 1, text: source}}}),
        encodeFrame({jsonrpc: '2.0', id: 2, method: 'textDocument/hover', params: {textDocument: {uri}, position: {line: 1, character: 16}}}),
        encodeFrame({jsonrpc: '2.0', id: 3, method: 'textDocument/definition', params: {textDocument: {uri}, position: {line: 1, character: 16}}}),
        encodeFrame({jsonrpc: '2.0', id: 4, method: 'shutdown', params: null}),
        encodeFrame({jsonrpc: '2.0', method: 'exit', params: null}),
    ]);
    console.log(`Launching: ${command}`);
    console.log(`Sending initialize, didOpen, hover, definition, shutdown, exit (${input.length} bytes)`);
    let child;
    let timedOut = false;
    const stdoutChunks = [];
    const stderrChunks = [];
    let stdoutBytes = 0;
    let stderrBytes = 0;
    let pipeError = null;
    try {
        child = spawn(command, [], {cwd: projectRoot, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true,
            env: {...process.env, NAPALM_LSP_TRACE: '1'},
        });
        child.stdout.on('data', chunk => {
            stdoutBytes += chunk.length;
            if (stdoutBytes > maxOutputBytes) child.kill();
            else stdoutChunks.push(chunk);
        });
        child.stderr.on('data', chunk => {
            stderrBytes += chunk.length;
            if (stderrBytes <= maxOutputBytes) stderrChunks.push(chunk);
        });
        child.stdin.on('error', error => { pipeError = error; });
        const result = await new Promise(done => {
            const timeout = setTimeout(() => { timedOut = true; child.kill(); }, timeoutMs);
            child.once('error', error => { clearTimeout(timeout); done({error}); });
            child.once('close', (code, signal) => { clearTimeout(timeout); done({code, signal}); });
            child.stdin.end(input);
        });
        console.log(`Process: ${timedOut ? 'timed out' : `exit=${result.code ?? '?'} signal=${result.signal ?? 'none'}`}`);
        if (result.error) console.error(`Spawn error: ${result.error.message}`);
        if (pipeError) console.error(`Input pipe error: ${pipeError.message}`);
        const stdout = Buffer.concat(stdoutChunks);
        const stderr = Buffer.concat(stderrChunks).toString('utf8');
        if (stderr) console.log(`--- Server stderr ---\n${stderr.trimEnd()}\n--- End stderr ---`);
        let frames;
        try {
            frames = decodeFrames(stdout);
        } catch (error) {
            console.error(`INVALID LSP STDOUT: ${error.message}`);
            console.error(`First stdout bytes (UTF-8): ${JSON.stringify(stdout.subarray(0, 300).toString('utf8'))}`);
            return 1;
        }
        const byId = id => frames.find(frame => frame.id === id);
        const init = byId(1);
        const hover = byId(2);
        const definition = byId(3);
        const diagnostics = frames.filter(frame => frame.method === 'textDocument/publishDiagnostics');
        console.log(`Received ${frames.length} valid LSP frame(s), ${diagnostics.length} diagnostic notification(s).`);
        console.log(`initialize: ${JSON.stringify(init ?? null)}`);
        console.log(`hover: ${JSON.stringify(hover ?? null)}`);
        console.log(`definition: ${JSON.stringify(definition ?? null)}`);
        if (diagnostics.length) console.log(`diagnostics: ${JSON.stringify(diagnostics[0].params?.diagnostics ?? null)}`);
        if (timedOut || result.error || !init || !init.result?.capabilities) {
            console.error('FAILED: no valid initialize response. This is a server/transport problem, independent of VS Code.');
            return 1;
        }
        if (!hover || !definition || diagnostics.length === 0) {
            console.error('PARTIAL: initialize works, but at least one document request or diagnostics notification is missing.');
            return 1;
        }
        if (result.code !== 0) {
            console.error('PARTIAL: expected clean shutdown and exit code 0.');
            return 1;
        }
        console.log('PASS: LSP transport, initialization and basic document handling work without VS Code.');
        return 0;
    } finally {
        if (child && child.exitCode === null) child.kill();
        rmSync(fixtureDirectory, {recursive: true, force: true});
    }
}
