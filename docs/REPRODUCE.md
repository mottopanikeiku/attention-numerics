# Reproduction

Use Python 3.13 and `uv`. `uv.lock` pins library builds. Synthetic sweeps use a single head, independent float32 Gaussian Q/K/V, seeds 3/17/29, dimensions 64/128, and causal/non-causal settings. No GPU or paid service is used. The experiments collect errors, not speed; do not interpret run wall-clock logs as benchmarks.

## Checks, exactly as CI

```
uv sync --locked --python 3.13
uv run ruff check .
uv run ruff format --check .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python sweep.py --study smoke --seeds 3 --output /tmp/attention-smoke
```

Tests include exact uniform attention/prefix means and a two-key logistic answer; independent dense float64 and slow scalar fp32/reduced14 references; FP8 representability/ties/saturation; rotation invariance; masked reverse traversal; sampled/full equality; promotion; and a denominator-only compensation example. They also check that key centering preserves the float64 reference under both masks and sampled/full rows, leaves Q uncentered, and that diagnostic logits reconstruct FP32 attention.

## Synthetic studies

On the shared workstation use the installed wrapper. Each invocation below is independent and bounded in memory; do not combine them into a single long allocation of a heavy slot. Elsewhere use `nice -n 19` instead of `pp-run heavy` and set equivalent thread limits.

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
pp-run heavy uv run python sweep.py --study length
pp-run heavy uv run python sweep.py --study fixes
pp-run heavy uv run python sweep.py --study tiles
pp-run heavy uv run python sweep.py --study softmax
pp-run heavy uv run python sweep.py --study full
pp-run heavy uv run python sweep.py --study full64 --seeds 3
pp-run heavy uv run python sweep.py --study dots
pp-run heavy uv run python sweep.py --study denominator
uv run python figures.py
```

The wrapper on the workstation is `/home/alp/Projects/profile-program/bin/pp-run`, with a default 1400 MB memory cap. `full` evaluates every query at N=4096, both dimensions/masks and three seeds. `full64` evaluates every query at N=65536 in one unit-Gaussian, d=64, non-causal case with seed 3 (FP32, BF16, E4M3/tensor). Other studies use all queries at N<=1024 and four 32-row query tiles at longer lengths, attending to the full allowed key context. The reference never allocates a large N×N array. CSVs retain query counts, seeds, worst original row, tile size, rounding choices and all error metrics. JSONs record software/commands. `summary.json` reports seed medians/ranges, not confidence bounds.

`dots.csv` compares long matrix reductions on already-quantized E4M3 **storage** values to float64 products of those same values. Thus it isolates arithmetic from input quantization; its absolute errors are in storage units, not dequantized attention units. K is a GEMM reduction dimension, not a statement that the tiled attention kernel reduces all N keys at once.

`denominator.csv` is a known-answer construction: one score is 0, the others are −16, the first V is 1 and all others are 0. The exact answer is `1 / (1 + (N−1)*exp(−16))`. Its numerator is exactly 1, so compensation can be evaluated without cancellation between numerator and denominator errors. This is an intentionally favorable construction for denominator compensation, not evidence that it improves typical attention.

## Real Q/K/V

`data/alice.txt` is a public-domain excerpt of Lewis Carroll's *Alice's Adventures in Wonderland*, chapter I ([source](https://www.gutenberg.org/files/11/11-0.txt)); [data/NOTICE](../data/NOTICE) records attribution. No text is generated or repeated.

The optional capture uses `Qwen/Qwen2.5-0.5B-Instruct`, pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`, under Apache-2.0. Weights stay in the shared Hugging Face cache and are not committed. `data/model-hashes.json` pins downloaded checkpoint/tokenizer/config SHA-256 hashes; `capture.py` checks them on every subsequent run. The capture is after rotary embedding and before GQA repetition, layers 0/12, query heads 0/7 paired with KV heads 0/1, all zero-based. Arrays are actual BF16 operands expanded exactly to float32. The model's preceding layers run in BF16 on CPU using Transformers eager attention, not the emulator. This is not a GPU run.

To **evaluate the committed small operand file**, only the second command is needed:

```
export HF_HOME=/home/alp/Projects/profile-program/cache/hf OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
pp-run heavy uv run --extra capture python real.py
```

To regenerate the capture, run `pp-run heavy uv run --extra capture python capture.py` before evaluation. The optional extra installs CPU-only PyTorch and Transformers; an uncached model download is about 1 GB, so perform it through `pp-run heavy`. Do not load multiple models simultaneously. The capture metadata includes checkpoint, text and operand-file hashes. Prefix evaluations at 128/512/1024 tokens reuse the captured forward pass; they are not independent samples. The CPU PyTorch SDPA comparison uses these same arrays and mask at BF16/FP32, never GPU timing or hardware inference.

## Rotation and key-mean diagnosis

The committed operand file suffices; this command needs no Torch, checkpoint download or model load:

```
pp-run heavy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python diagnosis.py --input data/qwen-qkv.npz --output results --seeds 0 1 2
uv run python figures.py
```

The numerical run writes `diagnosis.csv` (20 captured head/variant cases), `means.csv` (eight original Q/K energy records), `bias-controls.csv` (72 matched synthetic cases), and `diagnosis.json` (including per-query TV/error arrays). Every case uses all 1024 causal rows, d=64. Five captured conditions compare tensor scaling, tile scaling, rotation and key-only smoothing; synthetic conditions compare exactly constant, additive-bias and multiplicative-outlier channels at levels 8/32 over seeds 0/1/2. Other Q/K channels and V stay matched. The mean is over all K tokens before quantization; original float64 references never change.

The completed recorded invocation used `.venv/bin/python` instead of `uv run python`, with the same installed lockfile environment and thread settings, and additionally passed:

```
--checkpoint-header /home/alp/Projects/profile-program/cache/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors
```

That optional flag only inspects the cached safetensors header for bias tensor names/shapes; it does not load tensor contents or rehash the full checkpoint. Omit it if weights are not cached. It changes only the header-corroboration metadata, not the accuracy results. [MODEL.md](MODEL.md#rotation-diagnosis) defines normalized pre-probability-rounding TV and the score-only output comparison.

## Reading the results

The main length figure isolates context/input effects. The fix figure changes one rounding choice from E4M3 tensor scaling. The tile figure changes PV's per-tile reduction size. The dot figure tests genuinely long reduced-precision reductions and 128-term promotion. These are distinct questions; a large long-dot error does not imply the same error in a tiled attention kernel. Consult [MODEL.md](MODEL.md) before comparing this model to a specific GPU implementation.

`figures.py` refreshes `results/machine.json` with the **figure-rendering environment**. Recorded numerical-run provenance belongs to each study's JSON, particularly `diagnosis.json` and `real.json`; a later plotting environment must not be substituted for it.
