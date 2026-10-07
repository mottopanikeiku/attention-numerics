# Cold review

Two independent, read-only reviews on 2026-10-06 used `openai-codex/gpt-6.1-sol:high`. Neither reviewer ran tests, experiments, model loads, builds, linters or formatters. This is review of the source, documentation and recorded data, not independent hardware validation.

## Numerical implementation

The code reviewer found no actionable correctness or methodology blocker and judged the project publishable **as a disclosed CPU surrogate**. The review covered scaling placement, shared-exponent truncation, promotion intervals, online max/sum updates, probability conversion, masked reverse traversal, compensation, output conversion and the independent two-pass float64 reference.

It specifically examined the unexpectedly poor FP8 rotation result on captured first-layer heads. The transform is shared by Q and K; an unrotated tile-scaled comparator exists; the FP32-rotation control is small; removing probability conversion barely changes the large FP8-rotation error. Those controls do not implicate a broken transform or probability rounding as the dominant source. They do **not** establish the detailed operand-level mechanism or any conclusion about GPU hardware.

The reviewer also noted that some integration tests reuse conversion helpers or compare sampled/full execution, and that checksum/BF16-representability tests do not independently establish capture provenance. Independent dense/scalar references and analytic tests cover the arithmetic model; the capture remains reproducible from its pinned source model.

## Claims and prior work

The claims reviewer reproduced the main table's medians/ranges from CSVs, traced all five SVG plots to their data, checked the capture/text hashes and primary-source attribution, and verified the exact DeepSeek-V3 quotations and section numbers. It found no material mismatch in the reported numerical results. The supplemental all-row 64k values were separately checked against `results/full64.csv` and distinguished from three-seed sampled results.

Five low-severity presentation findings were corrected, including one from the final skim:

1. Full/full64 JSON metadata now explicitly says **all query rows**, matching the actual implementation and CSVs.
2. Exact-zero FP32 medians in the long-dot log plot are explicitly annotated and omitted from log coordinates. No positive numerical floor is inserted; zero lower bounds are also not shaded.
3. The tile/dot figure captions identify seed medians and observed minimum–maximum ranges, not confidence intervals.
4. The real-operands README paragraph explicitly states **causal attention** and evaluation of all captured query rows.
5. The fixes paragraph explicitly refers to the **main sampled table**, not the supplemental single-seed global result. The final skim checked the supplemental table's numbers and one-seed/all-row disclosure.

A separate source cleanup gave dot-product, analytic-denominator and real-operand metadata their own correct reference descriptions instead of inheriting a generic attention description. No numerical CSV values were changed by these presentation corrections.

See [MODEL.md](MODEL.md) for the rounding assumptions and [REPRODUCE.md](REPRODUCE.md) for the test and experiment commands. Passing these reviews does not make this a bit-exact H800 model or a FlashAttention-3 reproduction.

## Follow-up: key smoothing and rotation diagnosis

Two fresh independent read-only reviews on the same prescribed model covered the smoothing implementation, diagnostic logits/TV, new raw data, SageAttention citations and revised README. Neither reviewer ran tests, experiments, builds or model loads.

The numerical review identified one P2 precision-boundary defect: an original float64 K was converted to float32 before the documented float64 centering. The implementation now preserves original K until centering, then converts the centered result. A regression uses `K = 2^30 + [1, 3]`, whose variation would otherwise disappear, checking packed centered keys and the end-to-end independent reference. A focused static recheck passed. This correction does not change the recorded float32 captured/synthetic experiment arithmetic.

The claims review checked every captured table entry, mean-energy fraction, reference norm, worst-row TV/key change and synthetic median/range against the raw records. It independently confirmed the pinned SageAttention §4.2 Eq. (6) and SageAttention2 §3.1–3.4 citations. It judged the explanation appropriately qualified: evidence for a shared-component contributor, not projection bias as the sole cause.

Two P3 presentation/provenance findings were corrected: the diagnostic figure's footer was moved clear of its tick labels, and the README's experiment-version link now points to recorded numerical-run metadata rather than the regenerable rendering environment. [REPRODUCE.md](REPRODUCE.md) explicitly distinguishes the two. The reviewed README meets the total/opening/mechanism limits, and the exact-constant, additive-bias and varying-outlier controls remain separate.

## All-layer study: theory and inference

Two independent read-only reviews on `openai-codex/gpt-6.1-sol:high` covered the
locked prediction/statistical design and the streamed inference/FP8/metric code.
Neither ran numerical work or loaded an untouched evaluation model.

The theory review found no actionable defect. It checked the retained Q/K
cross-noise term, removal of row-common score noise, the softmax/output Jacobian,
relative-MSE units, exclusion of measured surrogate outputs from features, one
Qwen-only affine calibration, and explicit exclusion of both development pilots
from untouched-model evaluation. It also checked the zero-threshold classifier,
ties, majority and balanced accuracy, tied-rank AUC, minority precision/recall,
denominators, and the operational three-text prefix-sink definition. Scalar
covariance contraction, causal sensitivity with full-pair noise statistics,
linearization, and omitted P/V/output/transform rounding remain disclosed
approximations, not guarantees of accurate prediction.

The inference review found three interrupted-run or cross-cache provenance
defects, not observed arithmetic corruption. Capture file records now live in a
cache-local manifest written before advancing the hidden-state checkpoint;
resumption republishes that manifest even when all layers are already complete.
Score metadata is now atomically published before the metrics file that marks
scoring complete. Real tiny-model regressions cover switching between completed
and partial caches and interrupted private/public/score metadata publication.
The three existing completed capture caches were migrated from their actual
matching published manifests; no numerical operands or results were changed.
A focused read-only recheck found these fixes addressed all three findings.

No additional actionable arithmetic defect was found in the native architecture
dispatch, selective checkpoint loading, E4M3 recurrence, or full-vocabulary CE/KL
paths. That is source review, not independent validation of every real model.
The actual runtime, checks, and scoped native/fast/storage validation records are
under [`results/v2/`](../results/v2/). The prediction code and thresholds remain
those published in [f756745 (prediction-locked)](https://github.com/mottopanikeiku/attention-numerics/tree/prediction-locked)
before untouched evaluation inputs were loaded.

### Final numerical-claims audit

After all six models finished, a cold reader independently recomputed the
README's quantities from CSV/JSON. It confirmed 8,640 unique matched head/text
rows, all 2,880 physical heads, every pinned layer/head/text identity, and the
designated 1,728-head untouched split. The audit checked all headline risk
statistics, physical-head rotation-gain R² versus pooled transformation-gain
R², Qwen-only calibration, every downstream ΔCE/KL cell, token-weighted expCE
ratios, validation counts, and the undefined AUC/recall for OLMo's zero positives.
This was independent data arithmetic, not a rerun of the model experiments.

One publication defect was corrected: the compressed README had described
actual harm using predicted errors. It now states that **observed** rotated
log-error exceeding tile log-error defines the label, while the classifier uses
predicted counterparts. The distinction matters: the untouched group contains
132 observed-positive heads and 143 predicted-positive heads.

## Real kernels: source and independent data arithmetic

Two independent read-only cold reviews on 2026-10-07 used
`openai-codex/gpt-6.1-sol:high`. Neither ran CI, builds, capture preparation or
GPU jobs. These are source and recorded-data reviews, not independent GPU reruns.

The numerical/API reviewer found no actionable defect. It checked shared
PCG64/FWHT rotation, GQA mapping, unchanged V, original-operand FP64 causal
reference, pinned FA3 binary acquisition/digests and actual FP8 dispatch, Sage
INT8/FP8 settings and native smoothing/narrowing, and per-instance Qwen dispatch
with exact all-layer call counts and FP64 vocabulary CE/KL. The raw records
contain native H100 FP8 and L4 INT8/FP8 CUDA traces, 390 operand files per backend,
and all 30 FA3 downstream rows. Each nonbaseline forward records 24 or 28 native
kernel calls, matching the model's layers.

The claims reviewer independently aggregated raw records with standard-library
arithmetic. It confirmed 918 heads × three texts per backend, every 336+336 Qwen
head, reconstructed top32/uniform32 selections, all 398 operand-manifest members,
1,836 physical-head CSV rows and 12 per-model classifiers. Summary, CSV and
locked-score differences were at most 2.23e-16. It reproduced Qwen AUCs
0.956145/0.965517; FA3 TP79/FP44/FN8 versus Sage TP5/FP118/FN0; unrotated error
rank correlations 0.992052/0.849987; rotation-effect correlations
0.982007/0.262593; 28/146 harm disagreements; all token-weighted CE/KL values;
the 12.560327×/1.004379× Qwen1.5B ratios; and the $0.8699 conservative cost sum.

The comparison is transfer to real kernels, not matched-arithmetic reproduction:
scales, probability representation, accumulation and Sage BF16 preprocessing
differ together. Private operand access, enriched non-Qwen sampling,
future-token-dependent batch preprocessing and FA3-only downstream evaluation
remain explicit limitations. The README does not equate Sage's high AUC with
useful zero-threshold precision or claim a latency benchmark.
