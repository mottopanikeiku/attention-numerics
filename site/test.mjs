import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { attention, decodeFP8, encodeFP8, floatFromBits, rotate, toy } from './numerics.mjs';
const fixture = JSON.parse(await readFile(new URL('./fp8-fixture.json', import.meta.url)));
test('E4M3 raw and saturating bytes match the Python fixture', () => {
  for (const [bits, raw, saturated] of fixture.cases) {
    const value = floatFromBits(bits);
    assert.equal(encodeFP8(value, false), raw, `raw float32 bits ${bits.toString(16)}`);
    assert.equal(encodeFP8(value), saturated, `saturating float32 bits ${bits.toString(16)}`);
  }
});
test('All E4M3 storage bytes round-trip, including negative zero and signed NaNs', () => {
  for (let byte = 0; byte < 256; byte++) {
    assert.equal(encodeFP8(decodeFP8(byte)), byte);
  }
});
test('Common key component and exact rotation leave softmax unchanged', () => {
  const low = toy(0), high = toy(64);
  high.exact.forEach((value, i) => assert.ok(Math.abs(value - low.exact[i]) < 1e-14));
  const query = [1, 2, 3, 4, 5, 6, 7, 8];
  const exact = attention(query, high.keys);
  const rotated = attention(rotate(query), high.keys.map(rotate));
  rotated.forEach((value, i) => assert.ok(Math.abs(value - exact[i]) < 1e-14));
});
test('The default illustration shows harm and smoothing removes shared-coordinate dependence', () => {
  assert.ok(toy().rotatedTV > toy().plainTV);
  const low = toy(0, true), high = toy(64, true);
  assert.deepEqual(low.plain, high.plain);
  assert.deepEqual(low.rotated, high.rotated);
});
test('Export contains every physical head and published harm counts', async () => {
  const data = JSON.parse(await readFile(new URL('./heads.json', import.meta.url)));
  assert.equal(data.points.length, 2880);
  assert.equal(new Set(data.points.map(p => p.slice(0, 3).join('/'))).size, 2880);
  assert.equal(data.points.filter(p => p[4] > 0).length, 250);
  const unseen = data.points.filter(p => ['smol17', 'tiny11', 'olmo1'].includes(p[0]));
  assert.equal(unseen.length, 1728);
  assert.equal(unseen.filter(p => p[4] > 0).length, 132);
  assert.ok(data.points.every(p => p.slice(1).every(Number.isFinite)));
});
