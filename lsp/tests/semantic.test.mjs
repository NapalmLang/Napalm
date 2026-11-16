import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';

const projectPath = join(dirname(fileURLToPath(import.meta.url)), '..');
const source = readFileSync(join(projectPath, 'src/semantic.napalm'), 'utf8');
const parserSource = readFileSync(join(projectPath, 'src/parser.napalm'), 'utf8');
const serverSource = readFileSync(join(projectPath, 'src/server.napalm'), 'utf8');
const moduleIndexSource = readFileSync(join(projectPath, 'src/module_index.napalm'), 'utf8');

// Execute the same source, with only the differences in missing-key semantics
// between JavaScript Map and Napalm Map adapted for the test environment.
function loadSemantic() {
  const translated = source
    .replaceAll('export function ', 'function ')
    .replace('if (node[field] === null) return null;', 'if (node[field] == null) return null;');
  const context = vm.createContext({});
  vm.runInContext(`const HostMap = globalThis.Map;
    class NapalmMap extends HostMap { get(key) { return this.has(key) ? super.get(key) : null; } }
    const Map = NapalmMap;
    ${translated}
    globalThis.api = {analyze, definitionAt, referencesAt, symbolAt};`, context);
  return context.api;
}

const api = loadSemantic();
function pos(line, character) { return {line, character}; }
function node(kind, start, end, fields = {}) {
  return {kind, startByte: start, endByte: end, range: {start: pos(0, start), end: pos(0, end)}, ...fields};
}
function id(name, start) { return node('Identifier', start, start + name.length, {name}); }
function root(...body) { return node('Module', 0, 200, {body}); }
function variable(name, pos, init, kind = 'const') {
  return node('VariableDeclaration', pos, pos + 8, {declarationKind: kind,
    declarations: [node('VariableDeclarator', pos, pos + 8,
      {id: id(name, pos), init, annotation: null})]});
}
function expr(value) { return node('ExpressionStatement', value.startByte, value.endByte, {expression: value}); }

// Test the actual algorithm and its public API, not just string patterns.
test('block shadowing, forward resolution, definition and reference locations', () => {
  const x = variable('value', 1, node('NumericLiteral', 15, 16));
  const use = id('value', 25);
  const nested = node('BlockStatement', 20, 90, {body: [variable('value', 35, null), expr(use)]});
  const useOuter = id('value', 100);
  const result = api.analyze(root(x, nested, expr(useOuter)));
  assert.equal(result.diagnostics.length, 0);
  assert.equal(api.definitionAt(result, 'file:///x', pos(0, 25)).range.start.character, 35);
  assert.equal(api.definitionAt(result, 'file:///x', pos(0, 100)).range.start.character, 1);
  assert.equal(api.referencesAt(result, 'file:///x', pos(0, 1), false).length, 1);
  assert.equal(api.referencesAt(result, 'file:///x', pos(0, 1), true).length, 2);
});

test('unresolved names, duplicate declarations, built-ins and member names', () => {
  const missing = id('notDeclared', 15);
  const member = node('MemberExpression', 40, 60, {object: id('Process', 40),
    property: id('cwd', 48), computed: false});
  const result = api.analyze(root(variable('x', 1, null), variable('x', 10, null), expr(missing), expr(member)));
  assert.deepEqual(Array.from(result.diagnostics, d => d.code).sort(), ['NAPALM-SEM-001', 'NAPALM-SEM-002']);
  assert.equal(api.definitionAt(result, 'file:///x', pos(0, 48)), null);
});

test('function scopes resolve parameters and return outer captures', () => {
  const parameter = node('Parameter', 30, 32, {name: id('n', 30), annotation: null, defaultValue: null});
  const expression = node('BinaryExpression', 65, 80, {left: id('n', 65), right: id('capture', 69)});
  const fn = node('FunctionDeclaration', 20, 120, {id: id('fn', 20), params: [parameter],
    typeParameters: [], returnType: null,
    body: node('BlockStatement', 50, 110, {body: [node('ReturnStatement', 60, 90, {argument: expression})]})});
  const result = api.analyze(root(variable('capture', 1, null), fn));
  assert.equal(result.diagnostics.length, 0);
  assert.equal(api.definitionAt(result, 'file:///x', pos(0, 65)).range.start.character, 30);
  assert.equal(api.definitionAt(result, 'file:///x', pos(0, 69)).range.start.character, 1);
  assert.equal(result.documentSymbols[1].name, 'fn');
});

test('server registers semantic providers and caches per-document analyses', () => {
  for (const key of ['definitionProvider: true', 'referencesProvider: true', 'documentSymbolProvider: true',
    'state["analyses"].set(uri', 'state["analyses"].delete(uri)']) {
    assert.ok(serverSource.includes(key), key);
  }
  assert.match(moduleIndexSource, /source: "napalm-semantic"/);
  assert.match(parserSource, /let index = endIndex \+ 1;/);
});
