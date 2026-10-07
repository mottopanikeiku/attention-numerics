# Reproduction

Use Python 3.13 and `uv`. `uv.lock` pins library builds. Synthetic sweeps use a single head, independent float32 Gaussian Q/K/V, seeds 3/17/29, dimensions 64/128, and causal/non-causal settings. No GPU or paid service is used. The experiments collect errors, not speed; do not interpret run wall-clock logs as benchmarks.

## Checks, exactly as CI

```
uv sync --locked --extra capture --python 3.13
uv run ruff check .
uv run ruff format --check .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python sweep.py --study smoke --seeds 3 --output /tmp/attention-smoke
```

Tests include exact uniform attention/prefix means and a two-key logistic answer; independent dense float64 and slow scalar fp32/reduced14 references; FP8 representability/ties/saturation; rotation invariance; masked reverse traversal; sampled/full equality; promotion; and a denominator-only compensation example. They also check that key centering preserves the float64 reference under both masks and sampled/full rows, leaves Q uncentered, and that diagnostic logits reconstruct FP32 attention.

## Synthetic studies

Run each study separately with single-thread BLAS. Long-context studies are compute-intensive despite bounded working memory.

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 19 uv run python sweep.py --study length
nice -n 19 uv run python sweep.py --study fixes
nice -n 19 uv run python sweep.py --study tiles
nice -n 19 uv run python sweep.py --study softmax
nice -n 19 uv run python sweep.py --study full
nice -n 19 uv run python sweep.py --study full64 --seeds 3
nice -n 19 uv run python sweep.py --study dots
nice -n 19 uv run python sweep.py --study denominator
uv run python figures.py
```

`full` evaluates every query at N=4096, both dimensions/masks and three seeds. `full64` evaluates every query at N=65536 in one unit-Gaussian, d=64, non-causal case with seed 3 (FP32, BF16, E4M3/tensor). Other studies use all queries at N<=1024 and four 32-row query tiles at longer lengths, attending to the full allowed key context. The reference never allocates a large N×N array. CSVs retain query counts, seeds, worst original row, tile size, rounding choices and all error metrics. JSONs record software/commands. `summary.json` reports seed medians/ranges, not confidence bounds.

`dots.csv` compares long matrix reductions on already-quantized E4M3 **storage** values to float64 products of those same values. Thus it isolates arithmetic from input quantization; its absolute errors are in storage units, not dequantized attention units. K is a GEMM reduction dimension, not a statement that the tiled attention kernel reduces all N keys at once.

`denominator.csv` is a known-answer construction: one score is 0, the others are −16, the first V is 1 and all others are 0. The exact answer is `1 / (1 + (N−1)*exp(−16))`. Its numerator is exactly 1, so compensation can be evaluated without cancellation between numerator and denominator errors. This is an intentionally favorable construction for denominator compensation, not evidence that it improves typical attention.

## Real Q/K/V

`data/alice.txt` is a public-domain excerpt of Lewis Carroll's *Alice's Adventures in Wonderland*, chapter I ([source](https://www.gutenberg.org/files/11/11-0.txt)); [data/NOTICE](../data/NOTICE) records attribution. No text is generated or repeated.

The optional capture uses `Qwen/Qwen2.5-0.5B-Instruct`, pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`, under Apache-2.0. Weights stay in your configured Hugging Face cache and are not committed. `data/model-hashes.json` pins downloaded checkpoint/tokenizer/config SHA-256 hashes; `capture.py` checks them on every subsequent run. The capture is after rotary embedding and before GQA repetition, layers 0/12, query heads 0/7 paired with KV heads 0/1, all zero-based. Arrays are actual BF16 operands expanded exactly to float32. The model's preceding layers run in BF16 on CPU using Transformers eager attention, not the emulator. This is not a GPU run.

To **evaluate the committed small operand file**, only the second command is needed:

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 19 uv run --extra capture python real.py
```

To regenerate the capture, run `nice -n 19 uv run --extra capture python capture.py` before evaluation. The optional extra installs CPU-only PyTorch and Transformers; an uncached model download is about 1 GB. Hugging Face chooses its standard user cache, honoring `HF_HOME` or `XDG_CACHE_HOME` if configured; the script does not change those variables. Do not load multiple models simultaneously. The capture metadata includes checkpoint, text and operand-file hashes. Prefix evaluations at 128/512/1024 tokens reuse the captured forward pass; they are not independent samples. The CPU PyTorch SDPA comparison uses these same arrays and mask at BF16/FP32, never GPU timing or hardware inference.

## Rotation and key-mean diagnosis

The committed operand file suffices; this command needs no Torch, checkpoint download or model load:

```
nice -n 19 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python diagnosis.py --input data/qwen-qkv.npz --output results --seeds 0 1 2
uv run python figures.py
```

The numerical run writes `diagnosis.csv` (20 captured head/variant cases), `means.csv` (eight original Q/K energy records), `bias-controls.csv` (72 matched synthetic cases), and `diagnosis.json` (including per-query TV/error arrays). Every case uses all 1024 causal rows, d=64. Five captured conditions compare tensor scaling, tile scaling, rotation and key-only smoothing; synthetic conditions compare exactly constant, additive-bias and multiplicative-outlier channels at levels 8/32 over seeds 0/1/2. Other Q/K channels and V stay matched. The mean is over all K tokens before quantization; original float64 references never change.

The recorded invocation used `.venv/bin/python` instead of `uv run python`, with the same installed lockfile environment and thread settings. For optional header corroboration, point at your cached snapshot:

```
HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
nice -n 19 uv run python diagnosis.py --checkpoint-header "$HF_HOME/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"
```

That optional flag only inspects the cached safetensors header for bias tensor names/shapes; it does not load tensor contents or rehash the full checkpoint. Omit it if weights are not cached. It changes only the header-corroboration metadata, not the accuracy results. [MODEL.md](MODEL.md#rotation-diagnosis) defines normalized pre-probability-rounding TV and the score-only output comparison.

Recorded checkpoint paths in `diagnosis.json` use the portable `$HF_HOME/hub/...` form. This path-label normalization leaves every numerical result, snapshot revision and file hash unchanged.

## All-layer multi-model study

[models.json](../data/v2/models.json) pins the six open checkpoints, their upstream
file hashes, and architecture configurations. [texts.json](../data/v2/texts.json)
attributes the three public-domain excerpts and pins their hashes. No checkpoint
weights or large captures are committed.

```sh
uv sync --locked --extra capture --python 3.13
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 19 uv run --extra capture python -m study.pipeline --model all --seconds 500
```

Repeat the final command until its JSON reports `"complete": true`. Each invocation
resumes completed layers from the working cache. Downloads use the normal Hugging
Face cache; set `HF_HOME` if needed. Intermediate BF16 operands and hidden states
use `~/.cache/attention-numerics`, or `ATTENTION_NUMERICS_CACHE` / `--work-dir`.
Choose a fresh working directory to reproduce a changed checkpoint or input.
The model loader keeps only one decoder layer in memory, including for the larger
models; it does not construct a complete larger model.
The seconds limit is checked between complete layers. A download, whole layer, or
final vocabulary-scoring/plotting step can run past it; use `timeout` for a hard
deadline, then resume. Capture manifests stay beside their hidden-state
checkpoints and are republished on resume, including already-complete captures.

The first 1,024 raw-text tokens supply native BF16 post-RoPE Q/K/V captures at every
layer and query head, before GQA repetition. The next 1,025 tokens supply a disjoint
1,024-label next-token batch. Both use no chat template and reset context. The
baseline here is native Transformers **SDPA** BF16 attention, distinct from the
earlier eager-attention capture above. Layerwise final hidden states feed native
BF16 vocabulary projections; FP64 reductions cover the full vocabulary for CE and
`KL(BF16 || variant)`. `expCE` is the exponential of token-weighted mean CE.
Full-K centering and block scales may use later batch tokens: these are **batch
teacher-forced measurements**, not streaming-decoder perplexity or generation.

[design.json](../data/v2/design.json) lists development, calibration, and untouched
evaluation models and fixes the zero-threshold rotation-harm rule and prefix-sink
definition. Prediction code, thresholds, and model lists were published in
[f756745 (now tagged prediction-locked)](https://github.com/mottopanikeiku/attention-numerics/tree/prediction-locked) before
any untouched evaluation checkpoint was loaded. The error model and its omissions
are derived in [V2_PREDICTION.md](V2_PREDICTION.md). The parameter-free rule uses
Q/K quantization-error statistics and ideal attention sensitivity, not measured
surrogate output errors; the separately reported affine calibration fits Qwen only.

The pipeline writes raw head errors/features, sink measurements, downstream
metrics, fitted summaries, classifier denominators, and figures under
[`results/v2/`](../results/v2/). It also checks every BF16 storage pattern against
CPU Torch E4M3FN conversion, validates the small streamed Qwen baseline against a
full native model, and compares selected captured heads against the independent
NumPy emulator. Cached checkpoints are verified against the pinned file hashes on
initial download; a private revision marker avoids repeated whole-file hashing
when resuming the same immutable snapshot. Delete that marker to repeat byte checks.

## Reading the results

The main length figure isolates context/input effects. The fix figure changes one rounding choice from E4M3 tensor scaling. The tile figure changes PV's per-tile reduction size. The dot figure tests genuinely long reduced-precision reductions and 128-term promotion. These are distinct questions; a large long-dot error does not imply the same error in a tiled attention kernel. Consult [MODEL.md](MODEL.md) before comparing this model to a specific GPU implementation.

`figures.py` refreshes `results/machine.json` with the **figure-rendering environment**. Recorded numerical-run provenance belongs to each study's JSON, particularly `diagnosis.json` and `real.json`; a later plotting environment must not be substituted for it.

## Real-kernel comparison

I compare actual FA3 E4M3 forward on Hopper and SageAttention INT8-QK/FP8-PV on Ada with the earlier uniform-E4 CPU emulator. These are different arithmetic models, not interchangeable implementations. The wrappers are [FA3](../study/hardware/fa3.py) and [Sage](../study/hardware/sage.py); the shared [operand transform and explicit FP64 reference](../study/hardware/common.py) hold the comparison together.

### Operands and sampling

I selected the hardware cases before GPU numerical outputs. [selection.json](../data/hardware/selection.json) records the unchanged predictor commit, head-table hash, random seed and each physical head. Both Qwens include every head. Each other model contributes the top 32 locked-predictor scores and an independent uniform sample of 32 heads from its full population. Overlaps retain both labels and are measured once. The combined enriched sample is descriptive, not a population estimate; the report separates exhaustive Qwens, uniform samples and top-ranked samples.

I use the exact earlier CPU captures: native BF16 post-RoPE Q/K/V, before GQA expansion, over the same Alice, Moby-Dick and Pride and Prejudice windows. I do not regenerate them on the GPUs or silently compare new operands with old predictor features. [operands.json](../data/hardware/operands.json) lists every bundle member, byte count and SHA256, plus original capture hashes and head mappings. The bundle is retained in my private Modal Volume `attention-numerics-inputs`; it is **not currently a public download**. Publishing it is a separate owner decision. The committed [capture pipeline](../study/pipeline.py) can regenerate captures; my original CPU capture run used $0 paid compute and 16.4 GB of pinned model files ([model byte counts](../data/v2/models.json)). A different CPU/library build may produce different BF16 captures; its new hashes and matching emulator table must be recorded, not compared as if identical.

The public input preparation command is:

```sh
uv run python -m study.hardware.inputs --capture-cache "$ATTENTION_NUMERICS_CACHE" --output "$HOME/.cache/attention-numerics/night2-inputs"
```

It verifies the published source hashes, packages only selected heads, retains exact uint16 BF16 bits, and restores each original KV mapping explicitly. If rebuilding the original capture study, first complete [the all-layer instructions above](#all-layer-multi-model-study), including its matched emulator head table and capture manifests.

### What the real APIs compute

| Setting | Uniform-E4 emulator | FA3 public FP8 API | SageAttention2 Ada API |
|---|---|---|---|
| Q/K | E4M3, Q32/K128 token-block scales | E4M3, independent full-sequence batch/head max scales | INT8, explicit per-warp Q32/K64 scales |
| V | E4M3, 128-token-block scale | E4M3, full-sequence batch/head max scale | E4M3, per batch/head/channel scale over tokens |
| P | E4M3 numerators, scale 1/448 | E4M3 numerators with exponent offset 8 (scale 1/256) | Native E4M3 numerators, upstream exponent offset |
| QK accumulation | software grouped 32-product FP32 | Hopper FP8 GMMA, float accumulator registers | INT32 tensor-core product, FP32 dequantization |
| PV accumulation | grouped 32-product FP32 recurrence | Native FP8 GMMA with FP32 online rescaling | `fp32+fp32`: fresh Ada instruction buffer per 64-key tile, then FP32 running accumulation |
| Denominator | FP32 sum before P rounding | FP32 sum before P rounding | FP32 sum before P rounding |
| Output | BF16 | BF16 | BF16 |

“Tile” remains the legacy unrotated-condition label; it does **not** imply the real FA3 API uses the emulator's Q32/K128 scales. FP32 accumulator registers do not guarantee IEEE FP32 intermediate rounding within tensor-core instructions. Sage's upstream describes its Ada FP8 MMA precision as 22 valid bits. I do not claim access to hidden accumulator traces or probability codes.

All variants use the same shared PCG64(seed1729) random-sign normalized Hadamard on **Q and K only**, with the exact v2 FP32 butterfly routine. V is never rotated. FA3 key smoothing uses the emulator's FP64 token mean/subtraction before FP32 rotation and packing. Sage's public API takes BF16, so rotated FP32 Q/K first round to BF16; its native `smooth_k` computes a BF16 mean after that rotation/narrowing and subtracts it in the fused FP32 quantizer. I record this precision/order difference rather than present Sage as bit-identical preprocessing. I explicitly set `smooth_v=False`, `qk_quant_gran='per_warp'`, `pv_accum_dtype='fp32+fp32'` and full causal masking.

### Reference, classifier and downstream loss

The reference expands the original BF16 inputs exactly to FP64, computes explicit QK, causal softmax and PV in FP64, and chunks queries only to bound memory. Per-head reference energy must agree with the earlier independent NumPy reference to relative 1e-9. Both hardware runs use the same packed hashes. CUDA profiler events record actual launched kernel names. CPU tests validate math and API contracts only; they do not claim GPU execution in CI.

For each physical head I average `log1p(relative Frobenius output error)` across the three texts before subtraction or ranking. Rotation hurts iff the rotated mean exceeds the unrotated mean; the locked predictor uses the corresponding predicted difference with threshold 0. I do not refit it to kernels. ROC AUC uses tied ranks and remains undefined without both classes. The report includes base rates, balanced accuracy, precision/recall, kernel/emulator rank correlations and signed harm disagreements. Tiny effects are counted without changing the threshold.

For downstream evaluation I load each pinned Qwen entirely in GPU BF16 and replace only native attention's per-instance dispatch. Native projections, norms, RoPE, MLP and output projection remain unchanged. Every nonbaseline forward must launch the real kernel exactly once in every layer. The same heldout 1025-token window supplies 1024 inputs/next-token labels; positions reset. Native CUDA BF16 SDPA is the matched baseline, not the earlier CPU baseline. I use native BF16 vocabulary projections and FP64 cross-entropy and KL(BF16||variant). Full-batch means/scales may see later tokens: this is batch teacher forcing, not streaming decoder perplexity or generation.

### Builds and bounded reruns

The FA3 image acquires only the publisher's pinned stable-ABI CUDA 12.8 binary, verifies every publisher SHA256, and loads it directly. An isolated hub 1.10 acquisition environment avoids the latest `kernels` package's incompatible hub requirement beside Transformers 4.57.6. Runtime Torch is 2.9.1+cu128. Sage 2.2.0 has no official wheel in the inspected releases, so its pinned source builds in the CPU image stage, targeting only sm89, with GCC and two compiler workers; no source compilation runs in a GPU container. Runtime Torch is 2.6.0+cu124.

Install the locked optional SDK with `uv sync --locked --extra capture --extra hardware`. Prepare/upload the bundle in a CPU-only run:

```sh
ATTENTION_BACKEND=fa3 ATTENTION_GPU=none ATTENTION_MINUTES=15 ATTENTION_INPUT_BUNDLE="$HOME/.cache/attention-numerics/night2-inputs.tar" uv run modal run study/hardware/modal_app.py --mode upload --output /tmp/attention-upload.json
```

Then run each backend with resource limits matching the intended allocation. `ATTENTION_MINUTES` sets the actual function timeout; `max_containers=1` prevents multiplication. Use a 5-minute pilot before choosing full-run limits. CPU-only image checks use `--mode image`; numerical modes reject a missing GPU, wrong architecture or unavailable API without a BF16/emulation fallback. Results return to local gzip JSON; regenerate summary/figure with `uv run python -m study.hardware.report`.

The SVG rasterizes scatter points at 150 dpi for a smaller download; text and axes stay vector. The physical-head CSV retains the exact numerical values.

Elapsed records size the paid experiments and bound their cost, not benchmark latency. Cost bounds include image-build attempts, uploads, pilots and full runs; they are not a Modal invoice. Quantization granularity, P representation, transform rounding and accumulation change together, so disagreement does not establish which one individually caused an error difference.

### Primary implementation sources

- [Pinned FA3 build metadata and digests](https://huggingface.co/kernels/kernels-community/flash-attn3/raw/7cb368cf8278b583132eb72cbf312d54586df2e2/build/torch-stable-abi29-cu128-x86_64-linux/metadata.json), [public API](https://huggingface.co/kernels/kernels-community/flash-attn3/raw/7cb368cf8278b583132eb72cbf312d54586df2e2/build/torch-stable-abi29-cu128-x86_64-linux/flash_attn_interface.py), [source accumulation](https://github.com/huggingface/kernels-community/tree/8a730d96c37560ccf1d3e09f7bbccdc886818f33/flash-attn3).
- [Pinned Sage2.2.0 source](https://github.com/thu-ml/SageAttention/tree/eb615cf6cf4d221338033340ee2de1c37fbdba4a), [API and native smoothing](https://github.com/thu-ml/SageAttention/blob/eb615cf6cf4d221338033340ee2de1c37fbdba4a/sageattention/core.py), [tensor-core precision](https://github.com/thu-ml/SageAttention/blob/eb615cf6cf4d221338033340ee2de1c37fbdba4a/csrc/mma.cuh).
- [FA3 paper](https://arxiv.org/abs/2407.08608), [SageAttention](https://arxiv.org/abs/2410.02367), [SageAttention2](https://arxiv.org/abs/2411.10958); the earlier [prior-work notes](PRIOR_WORK.md) distinguish their arithmetic from uniform E4.
