import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
function load() {
  const files = ['lexer.napalm', 'parser.napalm', 'semantic.napalm'];
  const sources = files.map(file => readFileSync(join(root, 'src', file), 'utf8')
    .replace(/^import \{[^\n]+;\s*$/gm, '')
    .replaceAll('export function ', 'function ')
    .replaceAll('typeof source !== "String"', 'typeof source !== "string"')
    .replace('for (const entry of properties) value[entry.key] = entry.value;',
      'for (const key of Object.keys(properties)) value[key] = properties[key];')
    .replace('if (node[field] === null) return null;', 'if (node[field] == null) return null;'));
  const context = vm.createContext({Buffer});
  vm.runInContext(`
    const HostMap = globalThis.Map;
    class NapalmMap extends HostMap {
      get(key) { return this.has(key) ? super.get(key) : null; }
    }
    const Map = NapalmMap;
    class ByteArray extends Uint8Array { toUtf8() { return Buffer.from(this).toString('utf8'); } }
    const Bytes = {fromUtf8(value) { return ByteArray.from(Buffer.from(value, 'utf8')); }};
    ${sources.join('\n')}
    globalThis.api = {tokenize, parseLexed, analyze, definitionAt, referencesAt};
  `, context, {timeout: 5_000});
  return context.api;
}
const api = load();
function analyzeText(source) {
  const lexical = api.tokenize(source);
  assert.deepEqual(Array.from(lexical.diagnostics), []);
  const parsed = api.parseLexed(source, lexical);
  assert.deepEqual(Array.from(parsed.diagnostics), []);
  return api.analyze(parsed.ast);
}

test('semantic analysis operates on the real Napalm lexer and parser AST', () => {
  const text = 'const answer = 42;\nfunction twice(value: Number) { return value + answer; }\nconst output = twice(answer);\n';
  const result = analyzeText(text);
  assert.deepEqual(Array.from(result.diagnostics), []);
  assert.deepEqual(Array.from(result.documentSymbols, symbol => symbol.name), ['answer', 'twice', 'output']);
  const definition = api.definitionAt(result, 'file:///example.napalm', {line: 2, character: 16});
  assert.deepEqual({line: definition.range.start.line, character: definition.range.start.character}, {line: 1, character: 9});
  const ref = api.referencesAt(result, 'file:///example.napalm', {line: 0, character: 7}, true);
  assert.equal(ref.length, 3);
});

test('builtins, class members, imports, type scopes, and UTF-16 boundaries', () => {
  const text = 'import {feature as localFeature} from "./dep";\n' +
    'class Sample { value = 1; read() { return this.value + localFeature; } }\n' +
    'const result = Bytes.fromUtf8("ok");\n';
  const result = analyzeText(text);
  assert.deepEqual(Array.from(result.diagnostics), []);
  assert.equal(result.documentSymbols.length, 2);
  assert.equal(result.documentSymbols[0].name, 'Sample');
  assert.equal(result.documentSymbols[0].children.length, 2);
  assert.equal(result.documentSymbols[1].name, 'result');
});

test('unknown names and block-local shadowing with the real parser', () => {
  const text = 'let x = 1;\n{ const x = 2; const local = x; }\nconst y = x;\nconst fail = missingName;\n';
  const result = analyzeText(text);
  assert.deepEqual(Array.from(result.diagnostics, d => d.code), ['NAPALM-SEM-001']);
  assert.equal(api.definitionAt(result, 'file:///x', {line: 1, character: 29}).range.start.line, 1);
  assert.equal(api.definitionAt(result, 'file:///x', {line: 2, character: 10}).range.start.line, 0);
});

test('duplicate declaration recovery and functions with default parameter values', () => {
  const text = 'let x = 1;\nconst x = 2;\nfunction greet(name = "Napalm") { return name; }\n';
  const result = analyzeText(text);
  assert.deepEqual(Array.from(result.diagnostics, d => d.code), ['NAPALM-SEM-002']);
});

test('semantic pass handles its own complete source without crashing', () => {
  for (const filename of ['semantic.napalm', 'parser.napalm', 'server.napalm', 'lexer.napalm']) {
    const text = readFileSync(join(root, 'src', filename), 'utf8');
    const lexical = api.tokenize(text);
    const parsed = api.parseLexed(text, lexical);
    assert.equal(lexical.diagnostics.length, 0, filename + ' lexical');
    assert.equal(parsed.diagnostics.length, 0, filename + ' parse: ' + JSON.stringify(parsed.diagnostics[0]));
    const analyzed = api.analyze(parsed.ast);
    assert.deepEqual(Array.from(analyzed.diagnostics), [], filename + ' semantic');
    assert.ok(analyzed.documentSymbols.length > 0, filename);
  }
});


test('enums and getter/setter pairs do not produce spurious duplicate diagnostics', () => {
  const text = 'enum Flags { Read = 1, Write = 2, Both = Read | Write }\n' +
    'class Sample { get value() { return 1; } set value(x) { } }\n';
  const result = analyzeText(text);
  assert.deepEqual(Array.from(result.diagnostics), []);
});

test('definitions preserve UTF-16 columns after non-BMP Unicode text', () => {
  const text = 'const emoji = "😀"; const value = 7; const result = value;';
  const result = analyzeText(text);
  assert.deepEqual(Array.from(result.diagnostics), []);
  const declared = text.indexOf('value');
  const used = text.lastIndexOf('value');
  // Both offsets are UTF-16 code units; the astral character occupies two.
  const definition = api.definitionAt(result, 'file:///emoji.napalm', {line: 0, character: used});
  assert.equal(definition.range.start.character, declared);
  assert.equal(definition.range.end.character, declared + 5);
});
