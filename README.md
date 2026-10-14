# attention-numerics

I measure how FP8 attention changes multiple-choice answers.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/hero-dark.svg"><img alt="Qwen2.5-7B HellaSwag accuracy with FlashAttention-3 on one H100: 80.9% BF16, 63.7% FP8, 45.5% FP8 with rotated queries and keys, 81.0% when the keys are also centered." src="docs/hero-light.svg" width="600"></picture>

**Question:** Does rotating queries and keys before rounding hurt answer accuracy, and does centering keys first help?

**Result:** On Qwen2.5-7B HellaSwag, I measure **80.90% with BF16** → **63.70% with plain FP8 (`tile`, unrotated FA3)** → **45.50% with rotated FP8**. With keys centered, with or without rotation, no checkpoint loses **2 percentage points or more** versus BF16; the worst change is **−1.50pp** on Qwen2.5-1.5B MMLU. [All measurements](results/accuracy/summary.json).

I used one H100, full-sequence per-head scales, and batch scoring rather than streaming generation.

Recompute the table and figure from committed item records on CPU:

```sh
uv sync --locked --python 3.13
uv run python -m study.accuracy.report --additional-plan data/accuracy/qwen14-plan.json --additional-plan data/accuracy/smol17-plan.json
```

[Interactive mechanism and earlier emulated-head study](https://mottopanikeiku.github.io/attention-numerics/).

## Answer accuracy

Eight checkpoints, four families, five attention variants, one H100: full **1,172-item ARC-Challenge test**, seeded **2,000-item HellaSwag validation** and subject-stratified **2,000-item MMLU test**, all zero-shot without chat templates. ARC/HellaSwag use character-normalized continuation likelihood; MMLU scores answer letters without normalization. [Item IDs, prompts and source revisions](data/accuracy/selection.json).

The table gives BF16 accuracy percentages and the count of flagged FP8 comparisons. A flag requires a drop **≥2pp** and a paired bootstrap interval whose upper endpoint is below zero. There are 12 comparisons per checkpoint: three tasks × four nonbaseline variants. [Point estimates, all 95% intervals and raw-file hashes](results/accuracy/summary.json).

| Checkpoint | ARC | HellaSwag | MMLU | Harm flags / 12 |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 33.87 | 53.75 | 44.75 | 3 |
| Qwen2.5-1.5B-Instruct | 46.84 | 69.55 | 61.95 | 6 |
| Qwen2.5-3B-Instruct | 48.38 | 76.80 | 65.00 | 1 |
| Qwen2.5-7B-Instruct | 55.20 | 80.90 | 71.60 | 6 |
| Qwen2.5-14B-Instruct | 62.29 | 85.40 | 78.75 | 0 |
| Mistral-7B-v0.3 | 54.35 | 81.50 | 58.40 | 0 |
| OLMo-2-1124-7B | 57.17 | 81.25 | 59.50 | 0 |
| SmolLM2-1.7B | 47.44 | 72.95 | 48.60 | 0 |

The other three families are **base** checkpoints, so this is not a controlled ranking of family quality. Damage is not monotone with size: 7B Qwen is much worse than 14B here.

![All 96 paired accuracy changes and 95% intervals; red cells meet the fixed harm rule](results/accuracy/paired-change.svg)

Using published [FlashAttention-3](https://arxiv.org/abs/2407.08608) and pinned [lm-evaluation-harness prompts](docs/PRIOR_WORK.md#fixed-item-multiple-choice-evaluation), my [data preparation](study/accuracy/data.py) fixes the items, [scorer](study/accuracy/worker.py) changes attention in every layer, and [reporter](study/accuracy/report.py) pairs answers with native GPU BF16. Projections, RoPE, norms and MLP remain native.

`tile` names unrotated FA3 with **full-sequence per-head scales**, not the earlier block-scaled emulator. `rotate` applies a shared signed Hadamard to Q/K; `smooth_k` subtracts the key mean; `rotate_smooth_k` does both. V is never rotated.

## Why centering can help

A shared key vector adds the same offset to every visible score in a query row; exact softmax cancels it. Rotation can spread that vector across channels and coarsen the token-specific residual during rounding. Removing it preserves exact attention, not necessarily quantized attention. The results support a contributor, not one universal cause. [Derivation and earlier controls](docs/V2_PREDICTION.md).

The earlier real-kernel study measured a **12.56× batch exp-CE ratio** for rotated Qwen1.5B and **1.004×** after centering. It also showed that high predictor AUC can coexist with poor Sage threshold precision. These are separate numerical/loss experiments, not task-accuracy forecasts. [Earlier results](results/hardware/summary.json).

## Reproduce

Repeating model scoring requires **H100 80GB** for the native FA3 FP8 API. CPU-only cloud stages download/hash weights and tokenize before GPU allocation. The complete new-study compute bound is **$4.77 including the failed preparation and pilot**, not an invoice; downloaded checkpoint copies were removed afterward. [Cost accounting](results/accuracy/cost.json); [full stages, licenses and methods](docs/REPRODUCE.md#fixed-item-multiple-choice-accuracy).

## Limits and prior work

- Fixed samples and standard zero-shot prompts, not chat or task-general ability.
- Itemwise intervals, no multiple-comparison correction; absence of a flag is not equivalence.
- Full-sequence scales/means can see later tokens: batch scoring, not streaming generation.
- One GPU/kernel configuration; no latency or memory-saving benchmark, no new-model risk forecast.
- Qwen3B is noncommercial research/evaluation only; the other model checkpoints are Apache-2.0. No weights are redistributed.

Built on [FlashAttention-3](https://arxiv.org/abs/2407.08608), earlier [SageAttention](https://arxiv.org/abs/2410.02367) comparisons and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9); upstream work is attributed, not presented as my kernel.

Written with AI coding assistance.
