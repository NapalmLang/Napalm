import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const parser = readFileSync(resolve(root, 'src/parser.napalm'), 'utf8');
const server = readFileSync(resolve(root, 'src/server.napalm'), 'utf8');
const moduleIndex = readFileSync(resolve(root, 'src/module_index.napalm'), 'utf8');
const expectedArity = new Map([
  ['peek', 1], ['text', 1], ['expect', 2], ['node', 4],
  ['error', 4], ['identifier', 1], ['statements', 1],
  ['variableDeclaration', 1], ['commaExpression', 1],
  ['arrowBody', 4], ['expression', 2],
]);

// Count call arguments without mistaking nested parentheses, lists, objects,
// comments or string literals for top-level commas.
function callArity(source, opening) {
  let index = opening + 1;
  let depth = 0;
  let commas = 0;
  let quote = null;
  while (index < source.length) {
    const character = source[index];
    if (quote !== null) {
      if (character === '\\') { index += 2; continue; }
      if (character === quote) quote = null;
      index++;
      continue;
    }
    if (source.startsWith('//', index)) {
      const newline = source.indexOf('\n', index + 2);
      index = newline < 0 ? source.length : newline;
      continue;
    }
    if (source.startsWith('/*', index)) {
      const close = source.indexOf('*/', index + 2);
      index = close < 0 ? source.length : close + 2;
      continue;
    }
    if (character === '"' || character === "'" || character === '`') {
      quote = character;
      index++;
      continue;
    }
    if (character === '(' || character === '[' || character === '{') depth++;
    else if (character === ')' || character === ']' || character === '}') {
      if (depth === 0) {
        assert.equal(character, ')', 'Unexpected delimiter at end of call');
        return commas + (source.slice(opening + 1, index).trim().length > 0 ? 1 : 0);
      }
      depth--;
    } else if (character === ',' && depth === 0) commas++;
    index++;
  }
  assert.fail('Unterminated call');
}

test('Napalm parser contains no unsupported default parameters', () => {
  for (const [name, arity] of expectedArity) {
    assert.match(parser, new RegExp(`\\b${name}\\([^()]*\\) \\{`));
    assert.doesNotMatch(parser, new RegExp(`\\b${name}\\([^()]*\\s=\\s`),
      `A default argument was reintroduced in ${name}`);
  }
  assert.match(parser, /export function parse\(source\)/);
  assert.match(parser, /export function parseLexed\(source, lexical\)/);
});

test('All affected parser method calls provide explicit arguments', () => {
  const matches = parser.matchAll(/\bthis\.([A-Za-z]+)\s*\(/g);
  let checked = 0;
  for (const match of matches) {
    const arity = expectedArity.get(match[1]);
    if (arity === undefined) continue;
    const opening = match.index + match[0].length - 1;
    assert.equal(callArity(parser, opening), arity,
      `Wrong arity for ${match[1]}() at offset ${match.index}`);
    checked++;
  }
  assert.ok(checked >= 150, `Expected to verify many parser calls, got ${checked}`);
});

test('No unsupported Set(iterable) or multiple increment expressions in for', () => {
  assert.doesNotMatch(parser, /new Set\(\s*\[/);
  assert.match(parser, /function createStringSet\(values\)/);
  assert.doesNotMatch(parser, /count\+\+,\s*index\+\+/);
});

test('The editor passes its existing lexical result to parseLexed', () => {
  assert.match(moduleIndex, /import \{parseLexed\} from "\.\/parser"/);
  assert.match(moduleIndex, /parseLexed\(source, lexical\)/);
  assert.match(server, /updateModuleIndex\(state\["moduleIndex"\], state\["documents"\]/);
});
