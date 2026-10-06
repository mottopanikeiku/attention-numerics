# Numerical model

`attention.py` emulates one attention head on a CPU. It does **not** execute CUDA, measure GPU error, predict a particular GPU bit pattern, or reproduce a complete FlashAttention-3 kernel. The input generator produces float32 arrays; the reference treats these exact arrays as its input. There is no BF16 pre-rounding hidden in the FP8 synthetic conditions.

## Optional key smoothing

`Config(smooth_k=True)` subtracts K's per-channel mean over **all key tokens**, before optional rotation and storage conversion. The mean and subtraction are computed in float64, and centered K is then converted to float32. Q and V are not centered. For a constant vector μ, `q_i · (k_j − μ) = q_i · k_j − q_i · μ`: every allowed key logit for query i receives the same shift, so softmax is unchanged in real arithmetic, including under a causal mask. A test checks the independent float64 reference before/after centering, and FP32 tests bound preprocessing roundoff. Naively centering Q is not invariant and is not part of this option.

This is batch preprocessing using the full K array, including future keys in a causal batch. The exact shift is still softmax-invariant, but approximate quantization can depend on that batch mean; this is **not** a streaming implementation or a measured speed improvement. The smoothing attribution and distinction from SageAttention2's corrected Q smoothing are in [PRIOR_WORK.md](PRIOR_WORK.md).

## Storage, products, and accumulators

1. Optional Q/K rotation applies the same fixed-seed random signs and normalized Walsh–Hadamard butterflies in float32. In real arithmetic `(QR)(KR)^T = QK^T`. V is not rotated. Rotation has its own float32 error, tested separately from quantization.
2. BF16 conversion uses `ml_dtypes.bfloat16`, then expands back to float32 exactly. FP8 uses `float8_e4m3fn` or `float8_e5m2`. Conversions are round-to-nearest-even; FP8 overflow is saturated to ±448 or ±57344 before conversion. Subnormals are retained by these conversions; there is no flush-to-zero switch.
3. FP8 quantization scale is `float32(amax / max_finite)`, with 1 for an all-zero block. Divide in float32, saturate/cast, and retain scale separately. “Tensor” means one scale per whole Q, K, or V **head**, not per multi-head batch. “Tile” means a Q block of 32 rows or K/V block of the selected key-tile size, across all channels. Q tile boundaries do not change when query rows are sampled. There is no delayed calibration, clipping percentile, or power-of-two restriction.
4. Matrix reductions are partitioned into consecutive groups of 32 inner-dimension terms. The fp32 condition calls NumPy float32 matrix multiplication on each group and adds group outputs in float32. Products and group accumulators are fp32. The BLAS reduction/FMA order inside a group is library-dependent; this model does not prescribe individual tensor-core instruction order. The slow scalar test checks this path against left-to-right fp32 arithmetic within a small tolerance, and float64 tests bound its attention error.
5. The reduced14 condition first forms each group's 32 products in float32. For each output element, `frexp(max(abs(products)))` determines exponent `e`. Every product is truncated toward zero to a multiple of `2^(e-14)`; the products are summed in float32. Adding this sum to the running accumulator is followed by truncation toward zero to 14 significant bits **including the leading bit**. This is the explicit shared-exponent surrogate inspired by the [DeepSeek-V3 report](PRIOR_WORK.md), not a fully specified H800 model. In particular, internal hardware reduction order, guard bits, signs/alignment conventions, saturation, and underflow behavior may differ.
6. Promotion every 128 terms resets the reduced partial after adding it into an fp32 total. A partial is also promoted at a GEMM/tile boundary. No promotion means the reduced accumulator persists through that GEMM. QK reduces over d, while PV reduces over **the key tile**, not the full N. For FP8, dequantized PV tile outputs are added to the cross-tile numerator in float32. For BF16/global updates the unscaled numerator can be carried directly into subsequent matrix accumulation. Thus longer N does not automatically mean a single long low-precision accumulator. `dots.csv` separately isolates the case of a genuinely long inner reduction, with reference on already-quantized storage values.
7. QK results are multiplied in float32 by the Q×K dequantization scales and then the softmax scale. Dequantization is after the GEMM, not before product truncation. PV dequantization is also after the GEMM, using the V scale and probability scale.

## Online softmax

All scores, maxima, exponential outputs, running denominators, rescalings, output numerators and final divisions are float32. NumPy's `exp` is used, **not** a GPU approximate `exp2` instruction. Masked scores are −∞. Causal queries at original position i attend keys 0 through i, inclusive. Reverse order handles initially fully masked tiles without NaNs.

The default “global” update stores unnormalized numerator `u`, denominator `l`, and running maximum `m`. At each tile:

```
m_new = max(m, max(scores))
a = exp(m - m_new)
p = exp(scores - m_new)
l = float32(float32(a*l) + sum_float32(p))
u = float32(a*u) + GEMM(round_storage(p), V)
m = m_new
```

The denominator uses the **unrounded** float32 p. The PV product uses p rounded to the selected probability storage: by default BF16 for BF16 inputs, the chosen FP8 for FP8 inputs, and float32 for float32 inputs. FP8 p uses fixed scale `1/max_finite`, because p is in [0,1]; this avoids direct unscaled FP8 underflow. The output is `round_output(u/l)`; BF16 is the default output for both BF16 and FP8 inputs, while the fp32 baseline retains fp32 output. These choices are independently switchable.

The “local” alternative computes p relative to the **tile** maximum, computes a tile PV contribution, then rescales it by `b=exp(tile_max-m_new)` before addition. It is algebraically equivalent before rounding, but changes probability quantization and rounding order. “Reverse” reverses tile traversal; it does not reorder keys inside tiles. Neither is claimed to be a drop-in GPU scheduling improvement.

Compensation applies Kahan summation to **inter-tile denominator additions only**. Its correction is rescaled by a when the maximum changes. It does not compensate within-tile `sum`, exponential error, rescaling multiplications, or numerator accumulation. It can improve the denominator without improving total attention error; the separate constructed test isolates a denominator-only error.

## Reference and measurements

The reference is independent of the online recurrence: a first float64 chunked pass finds the final maximum, then a second pass computes `exp(scores-final_max)`, denominator and numerator in float64. It uses query chunks of 32 and key chunks of 1024; no large N×N matrix is allocated. Full results cost O(N²d) compute despite bounded memory.

Main sweeps evaluate **all query rows at N=1024**. At N=4096, 16384, and 65536 they evaluate four contiguous 32-row blocks starting at 0, N/4, N/2 and N−32. These deterministic blocks are not a random/unbiased sample; “worst” is only over those rows. A separate `full` study evaluates all 4096 rows for both dimensions and masks over three seeds. The `full64` study evaluates all 65,536 rows for one unit-Gaussian case: d=64, non-causal, seed 3, FP32/BF16/E4M3 tensor scaling. Only this supplemental 64k case has global 64k maxima. Every evaluated query uses its full permitted key context.

Metrics: max absolute element error, `||output-reference||_F / ||reference||_F`, original query index with largest row L2 error, and that error. A zero reference norm produces a null relative metric, not a made-up denominator. “Worst row” and “max” refer only to evaluated rows when sampling is used. The summary reports medians and ranges over independent input seeds; this is input variability, not repeated timing variability. No timings are reported.

The sweep is deliberately not a full Cartesian product. `sweep.py::study_cases` gives exact combinations: lengths and input distributions, separate fix ablations, tile sizes, softmax scales, and full-row checks. Results metadata records versions, commands and thread settings. Real model captures are a separate dataset and never represented as synthetic results.

## Rotation diagnosis

`diagnosis.py` evaluates all 1024 causal rows of each committed Qwen head, preserving the original-operand float64 reference across all five E4M3 variants. It also measures the original Q/K energy in the all-token mean: `N * ||mean(X)||_2² / ||X||_F²`. These are descriptive statistics, not an attribution of the means to one architectural parameter.

| Layer / query head | Q mean energy | K mean energy | Reference output Frobenius norm |
|---|---:|---:|---:|
| 0 / 0 | 19.4266% | 97.4710% | 4.507749 |
| 0 / 7 | 60.4716% | 94.8067% | 3.192995 |
| 12 / 0 | 44.6994% | 59.4170% | 165.598905 |
| 12 / 7 | 77.0569% | 50.7188% | 66.973508 |

Source: [means.csv](../results/means.csv). Header-only inspection of the pinned cached checkpoint confirms Q/K/V projection bias tensors in both captured layers, recorded in [diagnosis.json](../results/diagnosis.json). That inspection did not reload the model or recheck the full checkpoint hash, and does not establish that projection bias alone causes the post-RoPE means.

### Attention-mass movement, not just a relative-error denominator

The diagnostic reconstructs the emulator's QK logits using the same packed operands, natural query/key GEMM shapes, accumulation and float32 scale operations. It normalizes both these logits and the original float64 logits with stable **float64 softmax before probability storage conversion**. Total variation is `0.5 * sum(abs(p_emulated - p_reference))`, between normalized distributions. It isolates score-driven mass movement; it is **not** TV of the final rounded, potentially non-normalized P coefficients or a reconstruction of every online-softmax rounding step.

For the original tile-plus-rotation condition:

| Layer / head | Mean TV | Worst output row | TV there | Reference row L2 | Row error L2 | Top key: reference → approximate |
|---|---:|---:|---:|---:|---:|---:|
| 0 / 0 | 0.899519 | 572 | 0.988045 | 0.137486 | 0.381958 | 570 → 551 |
| 0 / 7 | 0.473372 | 213 | 0.981962 | 0.143988 | 0.297225 | 164 → 29 |
| 12 / 0 | 0.022308 | 238 | 0.083865 | 8.514739 | 0.798539 | 238 → 238 |
| 12 / 7 | 0.052176 | 421 | 0.130546 | 3.587868 | 0.597564 | 419 → 420 |

Source: [diagnosis.csv](../results/diagnosis.csv); per-query TV and output-error arrays are retained in its JSON companion. Layer-0 reference norms are smaller than layer-12 norms, but the failures also move almost all probability mass at the worst rows. They are not merely a divide-by-zero artifact.

Using probabilities from the approximate logits with **original V**, in float64, gives layer-0 relative errors of **117.3171% / 73.2477%**, versus full-emulation **117.0663% / 73.2544%**. Thus most of this failure is already present before P/V storage conversion. Full-minus-score-only residual norms are **3.4135% / 2.7540%** of the reference norm; these include P/V/output rounding and online recurrence and are **not an additive decomposition** of the total error.

### What key smoothing repairs

Key smoothing changes neither the original reference nor Q. On layer 0, unrotated tile errors fall from **23.6521% / 21.1675%** to **5.4869% / 11.9612%**. Rotated errors fall from **117.0663% / 73.2544%** to **44.8814% / 27.3203%**; mean TV falls to **0.379685 / 0.159246**, but smoothing does not make rotation preferable. On layer 12, rotation helps before smoothing; smoothing modestly improves the rotated errors further to **3.5844% / 7.1503%**. Smoothing alone slightly worsens head 0 (**4.9085% → 5.4242%**), so it is not an unconditional improvement. See [diagnosis.svg](../results/diagnosis.svg).

### Matched synthetic controls

For each seed 0/1/2, draw standard-normal float32 Q/K/V at N=1024, d=64. Keep all V and the other Q/K channels identical across constructions. In channel 0 of both Q and K:

- **Constant channel:** replace every token's value by B. This contributes only a common logit shift.
- **Additive bias:** add B to the original varying channel. Genuine bias-times-variation logits remain; this is not the same control as an exactly constant channel.
- **Varying outlier:** multiply the original channel by B.

Run B=8/32, causal attention, all query rows, and four tile-based variants. Every construction retains its own original float64 reference. These constructions are not variance-matched equivalents.

At B=32, relative-error medians [observed min–max] across three seeds are:

| Construction | Tile | Tile + rotation | Smooth K + tile | Smooth K + tile + rotation |
|---|---:|---:|---:|---:|
| Constant channel | 4.65 [4.46–4.70]% | 40.77 [39.11–48.85]% | 4.58 [4.47–4.68]% | 14.52 [13.99–14.56]% |
| Additive bias | 90.51 [64.21–91.78]% | 36.76 [19.65–39.35]% | 18.90 [16.37–23.60]% | 9.71 [9.52–11.14]% |
| Varying outlier | 43.01 [15.56–58.07]% | 4.31 [4.30–4.31]% | 30.01 [9.65–37.36]% | 4.59 [4.23–4.62]% |

Source: [bias-controls.csv](../results/bias-controls.csv), with both levels in [bias-controls.svg](../results/bias-controls.svg). Seed ranges are not confidence intervals.

**Qualified conclusion:** The exact-constant control demonstrates that rotation can turn a harmless shared component into harmful key-dependent rounding; the varying-outlier control shows the opposite sign. Together with the real K energy and centering intervention, this supports a shared-component contributor to the captured failure. It does not show that high mean energy is sufficient, that every channel bias behaves identically, or that projection bias alone is the cause. Substantial rotated error remains after key centering. FP8's per-element exponent also makes outlier-spreading less uniformly beneficial than for a scaled integer grid.
