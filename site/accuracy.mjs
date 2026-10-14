const models = {
  qwen05: 'Qwen2.5-0.5B', qwen15: 'Qwen2.5-1.5B', qwen3: 'Qwen2.5-3B',
  qwen7: 'Qwen2.5-7B', qwen14: 'Qwen2.5-14B', mistral7: 'Mistral-7B',
  olmo7: 'OLMo-2-7B', smol17: 'SmolLM2-1.7B',
};
const tasks = { arc_challenge: 'ARC-Challenge', hellaswag: 'HellaSwag', mmlu: 'MMLU' };
const percent = value => `${(value * 100).toFixed(1)}%`;

export function accuracySummary(summary) {
  const example = summary.rows.filter(row => row.model === 'qwen7' && row.task === 'hellaswag');
  const accuracy = variant => percent(example.find(row => row.variant === variant).accuracy);
  const centered = summary.rows.filter(row => ['smooth_k', 'rotate_smooth_k'].includes(row.variant));
  const checkpoints = new Set(centered.map(row => row.model)).size;
  const threshold = summary.harm_rule.minimum_drop;
  const losing = new Set(centered.filter(row => row.delta <= -threshold).map(row => row.model)).size;
  const worst = centered.reduce((a, b) => a.delta < b.delta ? a : b);
  const change = (worst.delta * 100).toFixed(1).replace('-', '−');
  return `I measure Qwen2.5-7B HellaSwag accuracy: ${accuracy(summary.baseline)} BF16 → ${accuracy('tile')} plain FP8 → ${accuracy('rotate')} rotated FP8. With centered keys, with or without rotation, ${losing} of ${checkpoints} checkpoints lose ≥${threshold * 100} points vs BF16 (worst ${change} points, ${models[worst.model]} ${tasks[worst.task]}).`;
}
