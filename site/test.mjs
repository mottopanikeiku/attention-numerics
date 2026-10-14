import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { attention, decodeFP8, encodeFP8, floatFromBits, rotate, toy } from './numerics.mjs';
import { accuracySummary } from './accuracy.mjs';
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

test('Real FA3 headline uses the committed answer-accuracy comparisons', async () => {
  const summary = JSON.parse(await readFile(new URL('../results/accuracy/summary.json', import.meta.url)));
  assert.equal(summary.baseline, 'bf16');
  assert.equal(accuracySummary(summary), 'I measure Qwen2.5-7B HellaSwag accuracy: 80.9% BF16 → 63.7% plain FP8 → 45.5% rotated FP8. With centered keys, with or without rotation, 0 of 8 checkpoints lose ≥2 points vs BF16 (worst −1.5 points, Qwen2.5-1.5B MMLU).');
  // A point drop is not a significance flag; include both centered variants.
  const changed = structuredClone(summary);
  for (const variant of ['smooth_k', 'rotate_smooth_k']) {
    const row = changed.rows.find(row => row.model === 'qwen7' && row.task === 'hellaswag' && row.variant === variant);
    row.delta = -0.02;
    row.harm_flag = false;
  }
  assert.match(accuracySummary(changed), /1 of 8 checkpoints lose ≥2 points vs BF16 \(worst −2.0 points, Qwen2.5-7B HellaSwag\)/);
  changed.rows.find(row => row.model === 'qwen7' && row.task === 'hellaswag' && row.variant === 'tile').accuracy = 0.65;
  assert.match(accuracySummary(changed), /80.9% BF16 → 65.0% plain FP8 → 45.5% rotated FP8/);
});

test('The intro links real-kernel accuracy separately from unchanged emulation stats', async () => {
  const html = await readFile(new URL('./index.html', import.meta.url), 'utf8');
  assert.match(html, /Real FlashAttention-3 \(FA3\)/);
  assert.match(html, /id="accuracy-result"/);
  assert.match(html, /href="https:\/\/github.com\/mottopanikeiku\/attention-numerics#answer-accuracy"/);
  assert.ok(html.indexOf('id="accuracy-result"') < html.indexOf('Earlier CPU FP8 emulation'));
  assert.ok(html.indexOf('Earlier CPU FP8 emulation') < html.indexOf('class="stats"'));
  for (const text of ['250 / 2,880', '0.978 AUC', '1.91×', 'CPU FP8 emulation results']) assert.ok(html.includes(text));
});
