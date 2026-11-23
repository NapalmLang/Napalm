import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {existsSync} from 'node:fs';
import {dirname, isAbsolute, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const executable = process.env.NAPALM_BIN ?? resolve(
  projectRoot,
  process.platform === 'win32' ? 'target/release/napalm.exe' : 'target/release/napalm',
);
const available = !isAbsolute(executable) || existsSync(executable);

async function runNapalmScript(script, expected) {
  const child = spawn(executable, [script], {
    cwd: projectRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  const output = [];
  const errors = [];
  child.stdout.on('data', (chunk) => output.push(chunk));
  child.stderr.on('data', (chunk) => errors.push(chunk));
  const exited = await new Promise((resolveDone, rejectDone) => {
    child.once('error', rejectDone);
    child.once('close', (code, signal) => resolveDone({code, signal}));
  });
  const stderr = Buffer.concat(errors).toString('utf8');
  assert.equal(exited.signal, null, stderr);
  assert.equal(exited.code, 0, stderr);
  assert.match(Buffer.concat(output).toString('utf8'), expected);
}

const requiresEngine = {skip: !available && 'Napalm executable is unavailable'};

test('Napalm parser AST smoke suite', requiresEngine, async () => {
  await runNapalmScript('lsp/scripts/parser_smoke.napalm', /Napalm parser smoke tests passed/);
});

test('Napalm parser parses its own source code', requiresEngine, async () => {
  await runNapalmScript('lsp/scripts/parser_selfhost.napalm', /Napalm parser self-hosting smoke tests passed/);
});
