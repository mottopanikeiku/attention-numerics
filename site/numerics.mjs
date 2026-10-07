// E4M3FN has 3 fraction bits, exponent bias 7, finite maximum 448.
// Convert through float32, exactly like attention.raw_storage_cast.
const buffer = new ArrayBuffer(4);
const view = new DataView(buffer);
export function floatFromBits(bits) {
  view.setUint32(0, bits, true);
  return view.getFloat32(0, true);
}
export function decodeFP8(byte) {
  const sign = byte & 128 ? -1 : 1;
  const magnitude = byte & 127;
  const exponent = magnitude >> 3;
  const fraction = magnitude & 7;
  if (magnitude === 127) return floatFromBits(byte & 128 ? 0xffc00000 : 0x7fc00000);
  return sign * (exponent === 0 ? fraction * 2 ** -9 : (1 + fraction / 8) * 2 ** (exponent - 7));
}
const levels = Array.from({ length: 127 }, (_, code) => decodeFP8(code));
export function encodeFP8(value, saturate = true) {
  view.setFloat32(0, value, true);
  const sign = (view.getUint32(0, true) >>> 24) & 128;
  let magnitude = Math.abs(view.getFloat32(0, true));
  if (Number.isNaN(magnitude)) return sign | 127;
  if (saturate) magnitude = Math.min(magnitude, 448);
  if (magnitude > 464) return sign | 127;
  if (magnitude >= 448) return sign | 126;
  let lo = 0, hi = 126;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (levels[mid] <= magnitude) lo = mid;
    else hi = mid;
  }
  const below = magnitude - levels[lo], above = levels[hi] - magnitude;
  const code = below < above ? lo : above < below ? hi : (lo % 2 === 0 ? lo : hi);
  return sign | code;
}
export const roundFP8 = value => decodeFP8(encodeFP8(value));
export function quantize(rows) {
  const f = Math.fround;
  const input = rows.map(row => row.map(f));
  const maximum = Math.max(...input.flat().map(Math.abs));
  const scale = maximum ? f(maximum / 448) : 1;
  return input.map(row => row.map(value => f(roundFP8(f(value / scale)) * scale)));
}
// Fixed random-sign Hadamard; an orthogonal transform applied to both Q and K.
const signs = [1, -1, 1, 1, -1, 1, -1, -1];
export function rotate(row) {
  const out = row.map((value, i) => value * signs[i]);
  for (let width = 1; width < out.length; width *= 2) {
    for (let start = 0; start < out.length; start += width * 2) {
      for (let j = 0; j < width; j++) {
        const a = out[start + j], b = out[start + j + width];
        out[start + j] = a + b;
        out[start + j + width] = a - b;
      }
    }
  }
  return out.map(value => value / Math.sqrt(out.length));
}
export function attention(query, keys) {
  const scores = keys.map(key => key.reduce((sum, x, i) => sum + x * query[i], 0) / Math.sqrt(query.length));
  const maximum = Math.max(...scores);
  const weights = scores.map(score => Math.exp(score - maximum));
  const total = weights.reduce((a, b) => a + b, 0);
  return weights.map(weight => weight / total);
}
export function toy(shared = 32, smooth = false) {
  const query = [0.6, 1.1, -0.7, 0.4, 1.3, -0.2, 0.8, -0.5];
  const keys = [
    [shared, 0.3, -1.2, 0.8, 0.6, -0.5, 0.2, 1.0],
    [shared, -0.8, 0.4, 1.1, -0.2, 0.7, -0.6, 0.3],
    [shared, 1.2, 0.1, -0.5, 0.9, -0.3, 0.8, -0.7],
    [shared, 0.5, 0.9, -0.4, -0.7, 0.2, 1.1, 0.6],
  ];
  const mean = query.map((_, i) => keys.reduce((sum, key) => sum + key[i], 0) / keys.length);
  const centered = smooth ? keys.map(key => key.map((x, i) => x - mean[i])) : keys;
  const plainKeys = quantize(centered), rotatedKeys = quantize(centered.map(rotate));
  const exact = attention(query, keys);
  const plain = attention(quantize([query])[0], plainKeys);
  const rotated = attention(quantize([rotate(query)])[0], rotatedKeys);
  const tv = weights => weights.reduce((sum, w, i) => sum + Math.abs(w - exact[i]), 0) / 2;
  return { keys, plainKeys, rotatedKeys, exact, plain, rotated, plainTV: tv(plain), rotatedTV: tv(rotated) };
}
