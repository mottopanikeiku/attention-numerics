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
