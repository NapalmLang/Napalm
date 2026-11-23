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

test('Napalm lexer smoke suite', {skip: !available && 'Napalm executable is unavailable'}, async () => {
  const child = spawn(executable, ['lsp/scripts/lexer_smoke.napalm'], {
    cwd: projectRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  const output = [];
  const errors = [];
  child.stdout.on('data', (chunk) => output.push(chunk));
  child.stderr.on('data', (chunk) => errors.push(chunk));
  const result = await new Promise((resolveDone, rejectDone) => {
    child.on('error', rejectDone);
    child.on('close', (code, signal) => resolveDone({code, signal}));
  });
  const stderr = Buffer.concat(errors).toString('utf8');
  assert.equal(result.signal, null, stderr);
  assert.equal(result.code, 0, stderr);
  assert.match(Buffer.concat(output).toString('utf8'), /Napalm lexer smoke tests passed/);
});
