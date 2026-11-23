import assert from 'node:assert/strict';
import {readFileSync, existsSync} from 'node:fs';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';

const root = fileURLToPath(new URL('../', import.meta.url));
const load = (path) => JSON.parse(readFileSync(join(root, path), 'utf8'));
const manifest = load('package.json');
const theme = load('themes/napalm-file-icon-theme.json');

const languageExtensions = ['napalm', 'burning', 'fire', 'orange'];

test('extension logo is a real bundled PNG at least 128px square', () => {
  assert.ok(manifest.icon?.startsWith('./icons/'));
  const png = readFileSync(join(root, manifest.icon));
  assert.equal(png.subarray(1, 4).toString(), 'PNG');
  assert.ok(png.readUInt32BE(16) >= 128);
  assert.equal(png.readUInt32BE(16), png.readUInt32BE(20));
});

test('all file types provide two existing fallback language icons', () => {
  for (const ext of languageExtensions) {
    const language = manifest.contributes.languages.find((entry) => entry.extensions.includes('.' + ext));
    assert.ok(language, ext);
    for (const target of [language.icon?.light, language.icon?.dark]) {
      assert.ok(target, `Missing ${ext} icon`);
      assert.ok(existsSync(join(root, target)), `Missing ${target}`);
    }
  }
});

test('optional icon theme maps the four extensions to a bundled flame image', () => {
  const contribution = manifest.contributes.iconThemes.find((entry) => entry.id === 'napalm-file-icons');
  assert.ok(contribution);
  for (const ext of languageExtensions) {
    const id = theme.fileExtensions[ext];
    assert.ok(id, ext);
    const definition = theme.iconDefinitions[id];
    assert.ok(definition);
    assert.ok(existsSync(join(root, 'themes', definition.iconPath)));
  }
  for (const definition of Object.values(theme.iconDefinitions)) {
    assert.ok(existsSync(join(root, 'themes', definition.iconPath)));
  }
});

test('bundled Windows icon contains the original five native resolutions', () => {
  const ico = readFileSync(join(root, 'icons/napalm-file.ico'));
  assert.equal(ico.readUInt16LE(0), 0);
  assert.equal(ico.readUInt16LE(2), 1);
  assert.equal(ico.readUInt16LE(4), 5);
  const sizes = [];
  for (let i = 0; i < 5; i++) {
    const offset = 6 + i * 16;
    const size = ico[offset] === 0 ? 256 : ico[offset];
    const length = ico.readUInt32LE(offset + 8);
    const fileOffset = ico.readUInt32LE(offset + 12);
    assert.equal(ico.subarray(fileOffset + 1, fileOffset + 4).toString(), 'PNG');
    assert.ok(fileOffset + length <= ico.length);
    sizes.push(size);
  }
  assert.deepEqual(sizes, [16, 32, 48, 64, 256]);
});
