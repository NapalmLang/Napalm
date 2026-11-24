import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const parser = readFileSync(join(root, 'src/parser.napalm'), 'utf8');
const server = readFileSync(join(root, 'src/server.napalm'), 'utf8');
const moduleIndex = readFileSync(join(root, 'src/module_index.napalm'), 'utf8');
const smoke = readFileSync(join(root, 'scripts/parser_smoke.napalm'), 'utf8');

test('parser is a separate Napalm frontend with no native Rust bridge', () => {
  assert.match(parser, /export function parse\(source\)/);
  assert.match(parser, /export function parseLexed\(source, lexical\)/);
  assert.match(parser, /class Parser/);
  assert.match(parser, /expression\(minimum, allowIn\)/);
  assert.doesNotMatch(parser, /new Set\(\s*\[/);
  assert.doesNotMatch(parser, /count\+\+,\s*index\+\+/);
  assert.doesNotMatch(parser, /FFI\.|Dll|ForeignFunction|napalm_front\.rs|@std\/parser/);
});

test('parser provides UTF-8 and UTF-16 ranges, error recovery and complexity guards', () => {
  assert.match(parser, /startByte: startByte/);
  assert.match(parser, /range: \{start: startPosition, end: endPosition\}/);
  assert.match(parser, /recoverStatement\(\)/);
  assert.match(parser, /MAX_PARSE_DEPTH/);
  assert.match(parser, /MAX_PARSE_DIAGNOSTICS/);
  assert.match(parser, /MAX_AST_NODES/);
});

test('server publishes both lexical and syntax diagnostics from the unsaved document', () => {
  assert.match(moduleIndex, /import \{parseLexed\} from "\.\/parser"/);
  assert.match(moduleIndex, /parseLexed\(source, lexical\)/);
  assert.match(moduleIndex, /source: "napalm-parser"/);
  assert.match(server, /method: "textDocument\/publishDiagnostics"/);
});

test('smoke suite exercises precedence, declarations, templates, generics and recovery', () => {
  for (const keyword of ['operator precedence', 'ClassDeclaration', 'TypeAliasDeclaration',
    'TemplateLiteral', 'ForOfStatement', 'ImportDeclaration', 'recovery preserves']) {
    assert.ok(smoke.includes(keyword), `Missing parser test for ${keyword}`);
  }
});
