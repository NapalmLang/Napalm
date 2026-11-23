import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname, resolve, isAbsolute} from 'node:path';
import {encodeFrame, FrameDecoder} from './protocol.mjs';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const executable = process.env.NAPALM_BIN ?? resolve(
  projectRoot,
  process.platform === 'win32' ? 'target/release/napalm.exe' : 'target/release/napalm',
);
const canExecute = isAbsolute(executable) ? existsSync(executable) : true;

async function runServer(frames, {expectedExit = 0, timeoutMs = 30_000} = {}) {
  const child = spawn(executable, ['lsp/main.napalm'], {
    cwd: projectRoot,
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
  });
  const decoder = new FrameDecoder();
  const responses = [];
  const stderr = [];
  let protocolError = null;
  child.stdout.on('data', (chunk) => {
    if (protocolError !== null) return;
    try {
      responses.push(...decoder.push(chunk));
    } catch (error) {
      protocolError = error;
      child.kill();
    }
  });
  child.stderr.on('data', (chunk) => stderr.push(chunk));
  const done = new Promise((resolveDone, rejectDone) => {
    child.on('error', rejectDone);
    child.on('close', (code, signal) => resolveDone({code, signal}));
  });
  const timeout = setTimeout(() => child.kill(), timeoutMs);
  try {
    for (const frame of frames) child.stdin.write(frame);
    child.stdin.end();
    const result = await done;
    if (protocolError !== null) throw protocolError;
    decoder.finish();
    assert.equal(result.signal, null, 'Server was killed or timed out');
    assert.equal(result.code, expectedExit, Buffer.concat(stderr).toString('utf8'));
    return {responses, stderr: Buffer.concat(stderr).toString('utf8')};
  } finally {
    clearTimeout(timeout);
    if (child.exitCode === null) child.kill();
  }
}

const onlyIfPresent = {skip: !canExecute && 'Build Napalm or set NAPALM_BIN to run live LSP tests'};

test('Initialize, incremental sync, unknown request, shutdown, exit', onlyIfPresent, async () => {
  const uri = 'file:///project/source.napalm';
  const packets = [
    {jsonrpc: '2.0', id: 1, method: 'initialize', params: {capabilities: {}}},
    {jsonrpc: '2.0', method: 'initialized', params: {}},
    {jsonrpc: '2.0', method: 'textDocument/didOpen', params: {
      textDocument: {uri, languageId: 'napalm', version: 1, text: 'A😀é\r\nvalue'},
    }},
    {jsonrpc: '2.0', method: 'textDocument/didChange', params: {
      textDocument: {uri, version: 2},
      contentChanges: [{range: {
        start: {line: 0, character: 1}, end: {line: 0, character: 3},
      }, rangeLength: 2, text: 'Q'}],
    }},
    {jsonrpc: '2.0', method: 'textDocument/didClose', params: {textDocument: {uri}}},
    {jsonrpc: '2.0', id: 'no-method', method: 'textDocument/hover', params: {}},
    {jsonrpc: '2.0', id: 3, method: 'shutdown', params: null},
    {jsonrpc: '2.0', method: 'exit'},
  ];
  const {responses} = await runServer(packets.map((item) => encodeFrame(item)));
  assert.deepEqual(responses.filter((item) => Object.hasOwn(item, 'id')).map((item) => item.id), [1, 'no-method', 3]);
  const publications = responses.filter((item) => item.method === 'textDocument/publishDiagnostics');
  assert.equal(publications.length, 3);
  assert.equal(publications[0].params.uri, uri);
  assert.equal(publications[0].params.version, 1);
  assert.equal(publications[1].params.version, 2);
  assert.deepEqual(publications[2].params.diagnostics, []);
  const requests = responses.filter((item) => Object.hasOwn(item, 'id'));
  assert.equal(requests[0].result.capabilities.positionEncoding, 'utf-16');
  assert.equal(requests[0].result.capabilities.textDocumentSync.change, 2);
  assert.equal(requests[1].error.code, -32601);
  assert.equal(requests[2].result, null);
});

test('Malformed JSON yields parse error, then the server recovers', onlyIfPresent, async () => {
  const invalidJson = Buffer.from('{bad json!', 'utf8');
  const invalidFrame = Buffer.concat([
    Buffer.from(`Content-Length: ${invalidJson.length}\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n`),
    invalidJson,
  ]);
  const packets = [
    invalidFrame,
    encodeFrame({jsonrpc: '2.0', id: 42, method: 'initialize', params: {capabilities: {}}}),
    encodeFrame({jsonrpc: '2.0', id: 43, method: 'shutdown'}),
    encodeFrame({jsonrpc: '2.0', method: 'exit'}),
  ];
  const {responses} = await runServer(packets);
  assert.equal(responses[0].error.code, -32700);
  assert.equal(responses[1].id, 42);
  assert.equal(responses[2].id, 43);
});

test('Exit without shutdown uses the required nonzero exit code', onlyIfPresent, async () => {
  const packets = [encodeFrame({jsonrpc: '2.0', method: 'exit'})];
  const {responses} = await runServer(packets, {expectedExit: 1});
  assert.deepEqual(responses, []);
});

test('Lexer publishes UTF-16 diagnostics for unsaved documents and clears them', onlyIfPresent, async () => {
  const uri = 'file:///project/lexical.napalm';
  const messages = [
    {jsonrpc: '2.0', id: 1, method: 'initialize', params: {capabilities: {}}},
    {jsonrpc: '2.0', method: 'textDocument/didOpen', params: {
      textDocument: {uri, languageId: 'napalm', version: 1, text: 'const a = "A😀é\n'},
    }},
    {jsonrpc: '2.0', method: 'textDocument/didChange', params: {
      textDocument: {uri, version: 2}, contentChanges: [{text: 'const a = "A😀é";\n'}],
    }},
    {jsonrpc: '2.0', method: 'textDocument/didClose', params: {textDocument: {uri}}},
    {jsonrpc: '2.0', id: 2, method: 'shutdown'},
    {jsonrpc: '2.0', method: 'exit'},
  ];
  const {responses} = await runServer(messages.map(encodeFrame));
  const diagnostics = responses.filter((item) => item.method === 'textDocument/publishDiagnostics');
  assert.equal(diagnostics.length, 3);
  assert.equal(diagnostics[0].params.version, 1);
  assert.equal(diagnostics[0].params.diagnostics[0].code, 'NAPALM-LEX-001');
  assert.equal(diagnostics[0].params.diagnostics[0].range.end.character, 15);
  assert.equal(diagnostics[1].params.version, 2);
  assert.deepEqual(diagnostics[1].params.diagnostics, []);
  assert.deepEqual(diagnostics[2].params.diagnostics, []);
});

test('Parser diagnostics recover after a valid incremental edit', onlyIfPresent, async () => {
  const uri = 'file:///project/syntax.napalm';
  const messages = [
    {jsonrpc: '2.0', id: 1, method: 'initialize', params: {capabilities: {}}},
    {jsonrpc: '2.0', method: 'textDocument/didOpen', params: {
      textDocument: {uri, languageId: 'napalm', version: 1, text: 'const answer = ;\nconst valid = 42;'},
    }},
    {jsonrpc: '2.0', method: 'textDocument/didChange', params: {
      textDocument: {uri, version: 2}, contentChanges: [{text: 'const answer = 7;\nconst valid = 42;'}],
    }},
    {jsonrpc: '2.0', id: 2, method: 'shutdown'},
    {jsonrpc: '2.0', method: 'exit'},
  ];
  const {responses} = await runServer(messages.map(encodeFrame));
  const updates = responses.filter((response) => response.method === 'textDocument/publishDiagnostics');
  assert.equal(updates.length, 2);
  assert.equal(updates[0].params.version, 1);
  assert.ok(updates[0].params.diagnostics.some((item) => String(item.code).startsWith('NAPALM-PARSE-')));
  assert.equal(updates[1].params.version, 2);
  assert.deepEqual(updates[1].params.diagnostics, []);
});


test('Semantic navigation, document symbols, references and fresh diagnostics', onlyIfPresent, async () => {
  const uri = 'file:///project/semantic.napalm';
  const messages = [
    {jsonrpc: '2.0', id: 1, method: 'initialize', params: {capabilities: {}}},
    {jsonrpc: '2.0', method: 'textDocument/didOpen', params: {
      textDocument: {uri, languageId: 'napalm', version: 1,
        text: 'const value = 1;\nconst doubled = value + value;\n'},
    }},
    {jsonrpc: '2.0', id: 2, method: 'textDocument/definition', params: {
      textDocument: {uri}, position: {line: 1, character: 17},
    }},
    {jsonrpc: '2.0', id: 3, method: 'textDocument/references', params: {
      textDocument: {uri}, position: {line: 0, character: 7},
      context: {includeDeclaration: true},
    }},
    {jsonrpc: '2.0', id: 4, method: 'textDocument/documentSymbol', params: {textDocument: {uri}}},
    {jsonrpc: '2.0', method: 'textDocument/didChange', params: {
      textDocument: {uri, version: 2},
      contentChanges: [{text: 'const value = 1;\nconst doubled = missing;\n'}],
    }},
    {jsonrpc: '2.0', method: 'textDocument/didClose', params: {textDocument: {uri}}},
    {jsonrpc: '2.0', id: 5, method: 'shutdown'},
    {jsonrpc: '2.0', method: 'exit'},
  ];
  const {responses} = await runServer(messages.map(encodeFrame));
  const requests = responses.filter(item => Object.hasOwn(item, 'id'));
  const [init, definition, references, symbols, shutdown] = requests;
  assert.equal(init.result.capabilities.definitionProvider, true);
  assert.equal(init.result.capabilities.referencesProvider, true);
  assert.equal(init.result.capabilities.documentSymbolProvider, true);
  assert.deepEqual(definition.result, {uri, range: {
    start: {line: 0, character: 6}, end: {line: 0, character: 11},
  }});
  assert.equal(references.result.length, 3);
  assert.deepEqual(Array.from(symbols.result, item => item.name), ['value', 'doubled']);
  assert.equal(shutdown.result, null);
  const publications = responses.filter(item => item.method === 'textDocument/publishDiagnostics');
  assert.equal(publications.length, 3);
  assert.deepEqual(publications[0].params.diagnostics, []);
  assert.ok(publications[1].params.diagnostics.some(item => item.code === 'NAPALM-SEM-001'));
  assert.deepEqual(publications[2].params.diagnostics, []);
});

test('Cross-module definition, references, and re-analysis after unsaved edit', onlyIfPresent, async () => {
  const {readFileSync} = await import('node:fs');
  const {pathToFileURL} = await import('node:url');
  const root = resolve(projectRoot, 'lsp/tests/fixtures/intermodule');
  const entry = pathToFileURL(resolve(root, 'entry.napalm')).href;
  const base = pathToFileURL(resolve(root, 'base.napalm')).href;
  const entryText = readFileSync(resolve(root, 'entry.napalm'), 'utf8');
  const baseText = readFileSync(resolve(root, 'base.napalm'), 'utf8');
  const messages = [
    {jsonrpc: '2.0', id: 1, method: 'initialize', params: {rootUri: pathToFileURL(root).href, capabilities: {}}},
    {jsonrpc: '2.0', method: 'textDocument/didOpen', params: {textDocument: {
      uri: entry, languageId: 'napalm', version: 1, text: entryText}}},
    {jsonrpc: '2.0', id: 2, method: 'textDocument/definition', params: {
      textDocument: {uri: entry}, position: {line: 2, character: 17}}},
    {jsonrpc: '2.0', id: 3, method: 'textDocument/references', params: {
      textDocument: {uri: entry}, position: {line: 2, character: 26},
      context: {includeDeclaration: true}}},
    {jsonrpc: '2.0', method: 'textDocument/didOpen', params: {textDocument: {
      uri: base, languageId: 'napalm', version: 1, text: baseText}}},
    {jsonrpc: '2.0', method: 'textDocument/didChange', params: {
      textDocument: {uri: base, version: 2}, contentChanges: [{text: 'export const count = 7;\n'}]}},
    {jsonrpc: '2.0', id: 4, method: 'textDocument/definition', params: {
      textDocument: {uri: entry}, position: {line: 2, character: 17}}},
    {jsonrpc: '2.0', id: 5, method: 'shutdown'},
    {jsonrpc: '2.0', method: 'exit'},
  ];
  const {responses} = await runServer(messages.map(encodeFrame));
  const request = (id) => responses.find((item) => item.id === id);
  assert.equal(request(2).result.uri, base);
  assert.ok(request(3).result.some((item) => item.uri === base));
  assert.equal(request(4).result, null);
  const publications = responses.filter((item) => item.method === 'textDocument/publishDiagnostics' &&
    item.params.uri === entry);
  assert.ok(publications.at(-1).params.diagnostics.some((item) => item.code === 'NAPALM-MOD-002'));
});
