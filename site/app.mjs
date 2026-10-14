import { toy } from './numerics.mjs';
const $ = selector => document.querySelector(selector);
const names = { qwen05: 'Qwen2.5-0.5B', qwen15: 'Qwen2.5-1.5B', smol036: 'SmolLM2-360M', smol17: 'SmolLM2-1.7B', tiny11: 'TinyLlama-1.1B', olmo1: 'OLMo-2-1B' };
const unseen = new Set(['smol17', 'tiny11', 'olmo1']);
const percent = value => `${(value * 100).toFixed(2)}%`;
const score = value => value.toFixed(4);
$('#theme').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
  $('#theme').textContent = dark ? 'Use dark theme' : 'Use light theme';
});
function updateToy() {
  const shared = Number($('#shared').value), smooth = $('#smooth').checked;
  $('#shared-value').value = shared;
  const data = toy(shared, smooth);
  $('#weights').innerHTML = data.exact.map((_, i) => `<div class="weight-row"><span>Key ${i + 1}</span><div class="bars">${['exact', 'plain', 'rotated'].map(field => `<div class="bar ${field}"><span style="width:${data[field][i] * 80}%"></span><span>${percent(data[field][i])}</span></div>`).join('')}</div></div>`).join('');
  $('#weights').setAttribute('aria-label', data.exact.map((_, i) => `Key ${i + 1}: exact ${percent(data.exact[i])}, FP8 without rotation ${percent(data.plain[i])}, with rotation ${percent(data.rotated[i])}`).join('; '));
  $('#plain-error').textContent = percent(data.plainTV);
  $('#rotated-error').textContent = percent(data.rotatedTV);
  $('#toy-description').textContent = smooth ? 'Mean subtraction removes the shared first coordinate. Moving it no longer changes either FP8 result.' : `At this setting, rotation ${data.rotatedTV > data.plainTV ? 'increases' : 'decreases'} weight error. Rounding makes the response jump rather than change smoothly.`;
  $('#vectors tbody').innerHTML = data.keys.map((key, i) => `<tr><th>${i + 1}</th>${[key, data.plainKeys[i], data.rotatedKeys[i]].map(row => `<td>[${row.map(x => x.toFixed(2)).join(', ')}]</td>`).join('')}</tr>`).join('');
}
$('#shared').addEventListener('input', updateToy);
$('#smooth').addEventListener('change', updateToy);
updateToy();

let points = [], filtered = [], page = 0, selected = null;
const svgNS = 'http://www.w3.org/2000/svg';
function svg(tag, attributes, text) {
  const element = document.createElementNS(svgNS, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}
function inspect(point) {
  selected = point;
  const [model, layer, head, predicted, observed, energy, tile, rotated] = point;
  $('#inspection').innerHTML = `<h3>${names[model]}</h3><p>Layer ${layer} · query head ${head}<br>${unseen.has(model) ? 'Unseen evaluation model' : model === 'qwen15' ? 'Additional model, not in the unseen group' : 'Predictor development model'}</p><p class="verdict">Observed: rotation ${observed > 0 ? 'hurts' : observed < 0 ? 'helps' : 'ties'}.<br>Predicted: rotation ${predicted > 0 ? 'hurts' : 'does not hurt'}.</p><dl><div><dt>Predicted harm score</dt><dd>${score(predicted)}</dd></div><div><dt>Observed harm score</dt><dd>${score(observed)}</dd></div><div><dt>Unrotated output error (log-mean)</dt><dd>${percent(tile)}</dd></div><div><dt>Rotated output error (log-mean)</dt><dd>${percent(rotated)}</dd></div><div><dt>Mean-key energy fraction (text mean)</dt><dd>${percent(energy)}</dd></div></dl><p>These output errors are not the toy's weight errors.</p>`;
  for (const circle of $('#plot').querySelectorAll('circle')) circle.classList.toggle('selected', filtered[Number(circle.dataset.index)] === point);
}
function renderPlot() {
  const root = $('#plot');
  root.querySelectorAll('g').forEach(node => node.remove());
  const group = svg('g', {});
  // Keep the same scale across filters so model/layer comparisons stay meaningful.
  const minimum = Math.min(0, ...points.flatMap(p => [p[3], p[4]]));
  const maximum = Math.max(0, ...points.flatMap(p => [p[3], p[4]]));
  const padding = (maximum - minimum) * 0.05;
  const lo = minimum - padding, hi = maximum + padding;
  const x = value => 75 + (value - lo) / (hi - lo) * 565;
  const y = value => 440 - (value - lo) / (hi - lo) * 400;
  for (let i = 0; i <= 4; i++) {
    const value = lo + (hi - lo) * i / 4;
    group.append(svg('line', { x1: x(value), x2: x(value), y1: 40, y2: 440, class: 'grid' }), svg('line', { x1: 75, x2: 640, y1: y(value), y2: y(value), class: 'grid' }));
    group.append(svg('text', { x: x(value), y: 464, 'text-anchor': 'middle' }, value.toFixed(2)), svg('text', { x: 64, y: y(value) + 4, 'text-anchor': 'end' }, value.toFixed(2)));
  }
  group.append(svg('line', { x1: 75, x2: 640, y1: y(0), y2: y(0), class: 'zero' }), svg('line', { x1: x(0), x2: x(0), y1: 40, y2: 440, class: 'zero' }), svg('line', { x1: 75, x2: 640, y1: 440, y2: 40, class: 'diagonal' }));
  group.append(svg('text', { x: 357, y: 501, 'text-anchor': 'middle' }, 'Predicted harm score →'), svg('text', { x: -240, y: 19, transform: 'rotate(-90)', 'text-anchor': 'middle' }, 'Observed harm score →'));
  filtered.forEach((point, index) => {
    const circle = svg('circle', { cx: x(point[3]), cy: y(point[4]), r: point === selected ? 6 : 3.5, 'data-index': index, class: `${point[4] > 0 ? 'hurt' : ''} ${point === selected ? 'selected' : ''}` });
    circle.append(svg('title', {}, `${names[point[0]]}, layer ${point[1]}, head ${point[2]}; predicted ${score(point[3])}, observed ${score(point[4])}`));
    group.append(circle);
  });
  root.append(group);
}
function renderTable() {
  const start = page * 20;
  $('#head-table tbody').innerHTML = filtered.slice(start, start + 20).map((point, i) => `<tr><td>${names[point[0]]}</td><td>${point[1]} / ${point[2]}</td><td>${score(point[3])}</td><td>${score(point[4])}</td><td><button type="button" data-row="${start + i}" aria-label="Inspect ${names[point[0]]} layer ${point[1]} head ${point[2]}">Inspect</button></td></tr>`).join('');
  $('#page-count').textContent = filtered.length ? `${start + 1}–${Math.min(start + 20, filtered.length)} of ${filtered.length.toLocaleString()}` : 'No matching heads';
  $('#previous').disabled = page === 0;
  $('#next').disabled = start + 20 >= filtered.length;
}
function updateFilters() {
  const model = $('#model').value, layer = $('#layer').value, group = $('#group').value;
  filtered = points.filter(p => (model === 'all' || p[0] === model) && (layer === 'all' || p[1] === Number(layer)) && (group === 'all' || (group === 'unseen' ? unseen.has(p[0]) : p[4] > 0))).sort((a, b) => b[4] - a[4]);
  page = 0;
  $('#head-count').textContent = `${filtered.length.toLocaleString()} heads shown · ${filtered.filter(p => p[4] > 0).length} harmed · ${filtered.filter(p => p[3] > 0).length} predicted harmed`;
  if (!filtered.includes(selected)) {
    selected = null;
    $('#inspection').innerHTML = '<h3>Inspect a head</h3><p>Select a point or table row in this filtered group.</p>';
  }
  renderPlot();
  renderTable();
}
function updateLayers() {
  const model = $('#model').value, current = $('#layer').value;
  const layers = [...new Set(points.filter(p => model === 'all' || p[0] === model).map(p => p[1]))].sort((a, b) => a - b);
  $('#layer').innerHTML = '<option value="all">All layers</option>' + layers.map(layer => `<option value="${layer}">Layer ${layer}</option>`).join('');
  $('#layer').value = layers.includes(Number(current)) && current !== 'all' ? current : 'all';
}
$('#model').addEventListener('change', () => { updateLayers(); updateFilters(); });
$('#layer').addEventListener('change', updateFilters);
$('#group').addEventListener('change', updateFilters);
$('#plot').addEventListener('click', event => {
  const index = event.target.closest('circle')?.dataset.index;
  if (index !== undefined) inspect(filtered[Number(index)]);
});
$('#head-table').addEventListener('click', event => {
  const index = event.target.closest('button')?.dataset.row;
  if (index !== undefined) inspect(filtered[Number(index)]);
});
$('#previous').addEventListener('click', () => { page--; renderTable(); });
$('#next').addEventListener('click', () => { page++; renderTable(); });
try {
  const response = await fetch(new URL('./heads.json', import.meta.url));
  if (!response.ok) throw new Error(`Data request failed: ${response.status}`);
  points = (await response.json()).points;
  for (const [id, name] of Object.entries(names)) $('#model').append(new Option(name, id));
  updateLayers();
  updateFilters();
} catch {
  $('#head-count').textContent = 'I could not load the measurements. Reload the page, or read results/v2/heads.csv in the repository.';
  for (const selector of ['#model', '#layer', '#group', '#previous', '#next']) $(selector).disabled = true;
}
