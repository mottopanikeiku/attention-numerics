# Numerical model

`attention.py` emulates one attention head on a CPU. It does **not** execute CUDA, measure GPU error, predict a particular GPU bit pattern, or reproduce a complete FlashAttention-3 kernel. The input generator produces float32 arrays; the reference treats these exact arrays as its input. There is no BF16 pre-rounding hidden in the FP8 synthetic conditions.

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
