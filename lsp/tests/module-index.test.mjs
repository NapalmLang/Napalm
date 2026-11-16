import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync, existsSync, realpathSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {posix as path} from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {dirname, join} from 'node:path';
import {fileURLToPath as herePath} from 'node:url';
import vm from 'node:vm';

const project = join(dirname(herePath(import.meta.url)), '..');
const sources = ['lexer.napalm', 'parser.napalm', 'semantic.napalm', 'module_index.napalm']
  .map((file) => readFileSync(join(project, 'src', file), 'utf8')
    .replace(/^import \{[^\n]+;\s*$/gm, '')
    .replaceAll('export function ', 'function ')
    .replaceAll('typeof source !== "String"', 'typeof source !== "string"')
    .replaceAll('typeof uri !== "String"', 'typeof uri !== "string"')
    .replaceAll('typeof raw !== "String"', 'typeof raw !== "string"')
    .replaceAll('typeof home === "String"', 'typeof home === "string"')
    .replaceAll('typeof item === "String"', 'typeof item === "string"')
    .replace('for (const entry of properties) value[entry.key] = entry.value;',
      'for (const key of Object.keys(properties)) value[key] = properties[key];')
    .replace('if (node[field] === null) return null;', 'if (node[field] == null) return null;'));

function setup() {
  const temp = mkdtempSync(join(tmpdir(), 'napalm-lsp-m5-'));
  const workspace = temp.replaceAll('\\', '/');
  const context = vm.createContext({Buffer, process, fsImpl: {
    exists: existsSync, read: readFileSync, real: realpathSync,
  }, pathImpl: path, pathToFileURL, fileURLToPath});
  vm.runInContext(`
    const HostMap = globalThis.Map;
    class NapalmMap extends HostMap { get(key) { return this.has(key) ? super.get(key) : null; } }
    const Map = NapalmMap;
    Object.defineProperty(String.prototype, 'utf8Length', {get() { return Buffer.byteLength(this.toString(), 'utf8'); }});
    Object.defineProperty(String.prototype, 'utf16Length', {get() { return this.length; }});
    class ByteArray extends Uint8Array { toUtf8() { return Buffer.from(this).toString('utf8'); } }
    const Bytes = {fromUtf8(value) { return ByteArray.from(Buffer.from(value, 'utf8')); }};
    const Fs = {exists(path) { return fsImpl.exists(path); },
      readText(path) { return fsImpl.read(path, 'utf8'); },
      realPath(path) { return fsImpl.real(path); }};
    const Path = pathImpl;
    const Uri = {toFilePath(value) { return fileURLToPath(value); },
      fromFilePath(value) { return pathToFileURL(value).href; }};
    const Process = {getenv(name) { return null; }};
    ${sources.join('\n')}
    globalThis.NapalmMap = NapalmMap;
    globalThis.api = {buildModuleIndex, moduleDefinitionAt, moduleReferencesAt,
      documentDiagnostics, recordForUri, tokenize, parseLexed, analyze};
  `, context, {timeout: 8_000});
  const api = context.api;
  function file(name, contents) {
    const target = path.join(workspace, name);
    mkdirSync(path.dirname(target), {recursive: true});
    writeFileSync(target, contents, 'utf8');
    return target;
  }
  function open(...items) {
    const documents = new context.NapalmMap();
    for (const [relative, text] of items) {
      const uri = pathToFileURL(path.join(workspace, relative)).href;
      documents.set(uri, {uri, text, version: 1});
    }
    return api.buildModuleIndex(documents, [workspace]);
  }
  return {api, context, file, open, workspace,
    uri(relative) { return pathToFileURL(path.join(workspace, relative)).href; },
    close() { rmSync(temp, {recursive: true, force: true}); }};
}

test('named alias, barrel re-export and cross-file references', () => {
  const t = setup();
  try {
    t.file('dep.napalm', 'export const answer = 42;\n');
    t.file('barrel.napalm', 'export {answer as renamed} from "./dep";\n');
    const text = 'import {renamed as alias} from "./barrel";\nconst value = alias;\n';
    const index = t.open(['main.napalm', text]);
    const at = text.split('\n')[1].lastIndexOf('alias');
    const def = t.api.moduleDefinitionAt(index, t.uri('main.napalm'), {line: 1, character: at});
    assert.equal(def.uri, t.uri('dep.napalm'));
    assert.equal(def.range.start.character, 13);
    const all = t.api.moduleReferencesAt(index, t.uri('main.napalm'), {line: 1, character: at}, true);
    const visited = new Set(Array.from(all, location => location.uri));
    assert.deepEqual(visited, new Set([t.uri('main.napalm'), t.uri('dep.napalm'), t.uri('barrel.napalm')]));
    assert.deepEqual(Array.from(t.api.documentDiagnostics(index, t.uri('main.napalm'))), []);
  } finally { t.close(); }
});

test('default exports, type-only imports, directory index and module precedence', () => {
  const t = setup();
  try {
    t.file('library/index.napalm', 'export default function greet() { return 1; }\nexport interface Shape { size: number; }\n');
    const source = 'import greet, {Shape} from "./library";\nconst result = greet();\nconst data: Shape = null;\n';
    const index = t.open(['main.napalm', source]);
    const def = t.api.moduleDefinitionAt(index, t.uri('main.napalm'), {line: 1, character: 16});
    assert.equal(def.uri, t.uri('library/index.napalm'));
    assert.equal(def.range.start.line, 0);
    const ty = t.api.moduleDefinitionAt(index, t.uri('main.napalm'), {line: 2, character: 13});
    assert.equal(ty.uri, t.uri('library/index.napalm'));
    assert.equal(ty.range.start.line, 1);
    assert.deepEqual(Array.from(t.api.documentDiagnostics(index, t.uri('main.napalm'))), []);
  } finally { t.close(); }
});

test('missing exports, missing paths, and cycles produce bounded diagnostics', () => {
  const t = setup();
  try {
    t.file('a.napalm', 'export {thing} from "./b";\n');
    t.file('b.napalm', 'export {thing} from "./a";\n');
    const source = 'import {thing} from "./a";\nimport {nope} from "./missing";\nconst x = thing;\n';
    const index = t.open(['main.napalm', source]);
    const errors = Array.from(t.api.documentDiagnostics(index, t.uri('main.napalm')), d => d.code);
    assert.ok(errors.includes('NAPALM-MOD-001'), 'unresolved module');
    assert.ok(errors.includes('NAPALM-MOD-002'), 'unresolved export');
    assert.equal(index.order.length, 3);
  } finally { t.close(); }
});

test('unsaved changes override disk, and @lib package resolution', () => {
  const t = setup();
  try {
    t.file('dep.napalm', 'export const oldName = 1;\n');
    t.file('.napalm/packages/pkg.napalm', 'export function pkgFunction() {}\n');
    const src = 'import {newName} from "./dep";\nimport {pkgFunction} from "@lib/pkg";\nconst x = newName;\n';
    const withDisk = t.open(['main.napalm', src]);
    assert.ok(Array.from(t.api.documentDiagnostics(withDisk, t.uri('main.napalm')), d => d.code)
      .includes('NAPALM-MOD-002'));
    const updated = t.open(['main.napalm', src], ['dep.napalm', 'export const newName = 7;\n']);
    assert.deepEqual(Array.from(t.api.documentDiagnostics(updated, t.uri('main.napalm'))), []);
    const def = t.api.moduleDefinitionAt(updated, t.uri('main.napalm'), {line: 2, character: 13});
    assert.equal(def.uri, t.uri('dep.napalm'));
    const library = t.api.moduleDefinitionAt(updated, t.uri('main.napalm'), {line: 1, character: 11});
    assert.equal(library.uri, t.uri('.napalm/packages/pkg.napalm'));
  } finally { t.close(); }
});

test('the lexer/parser accept the new module index and server sources', () => {
  const t = setup();
  try {
    for (const file of ['module_index.napalm', 'server.napalm']) {
      const source = readFileSync(join(project, 'src', file), 'utf8');
      const lexical = t.api.tokenize(source);
      assert.equal(lexical.diagnostics.length, 0, file + ' lexer: ' + JSON.stringify(lexical.diagnostics[0]));
      const parsed = t.api.parseLexed(source, lexical);
      assert.equal(parsed.diagnostics.length, 0, file + ' parser: ' + JSON.stringify(parsed.diagnostics[0]));
      const semantic = t.api.analyze(parsed.ast);
      assert.deepEqual(Array.from(semantic.diagnostics), [], file + ' semantic');
    }
  } finally { t.close(); }
});
