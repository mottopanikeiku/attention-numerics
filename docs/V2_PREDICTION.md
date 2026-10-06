# Outcome-independent attention-error prediction

`study/prediction.py` predicts per-head Q/K-driven rounding error from the original
operands, the reference attention distribution, and deterministic E4M3 operand
residuals. **It never reads an emulated output or a measured target-output error.**
`summarize_fit` is a separate evaluation function: only that function receives
observed errors. The analytic prediction has no fitted coefficient.

This is a first-principles approximation to **local attention-output error**, not
a prediction of downstream cross-entropy, KL, or error accumulated through a
whole decoder. No empirical success or cross-family performance is asserted in
this document. Those require the model/text measurements under `results/v2/`.

## Input and CSV contract

```python
from study.prediction import predict_head, summarize_fit

features = predict_head(q, k, v, scale=None, sign_seed=1729)
```

Q and K have matching original `N x d` float32 shapes; V has N rows, ordinarily
also d channels. This is one batch/head at a time, before GQA duplication. The
function uses the existing `attention.quantize_qk`, `Config.sign_seed`, key
centering and normalized signed-Hadamard conventions. The four variants are
`tile`, `rotate`, `smooth_k`, and `rotate_smooth_k`, with Q tiles of 32, K tiles
of 128, and expanded E4M3 operands. The default softmax scale is $d^{-1/2}$;
pass the captured native attention scale when it differs. Rotated variants
require the shared Hadamard implementation's power-of-two head dimension.

The result is a flat numeric dictionary suitable for a head CSV:

- Original `q_`/`k_` token-mean L2 norms, total/broadcast-mean energies,
  mean-energy fractions, RMS and maximum absolute entries.
- Reference output mean square, value sensitivity, and mean per-row
  $\sum_j p_{ij}^2$.
- For each variant, Q/K residual mean squares, centered/common K residual mean
  squares, centered K and expanded Q mean squares, score-noise second moment,
  and its query, key, and signed cross terms.
- `{variant}_predicted_relative_mse` and `{variant}_predicted_error`, the latter
  exactly the square root of the former.

**Relative MSE here means squared Frobenius error divided by the reference
squared Frobenius norm**, not per-element absolute MSE. CSV conversion makes
long evaluation rows containing `model`, `family`, `variant`, `observed_mse`,
`predicted_mse`, and identifying `text`, `layer`, `head`, and, when needed,
`batch`. Both `*_mse` evaluation fields must use that same relative definition.
Use the predicted relative MSE field, and square the measured relative
Frobenius error for `observed_mse`. No target-dependent floor is added.

Reference P and/or O can be supplied. Without P, the function computes causal
float64 probabilities from original Q/K; without O, it computes $O=PV$ in
float64. Supplied P/O must describe the same reference. Supplied full-attention
P is also permitted; O alone does not change the default causal mask. The
routine checks supplied P's shape, finiteness, nonnegativity and normalization,
not whether it is mathematically the attention distribution for Q/K. Reference
computation and sensitivity use 32-row blocks, with no new dense $N\times N$
probability or score matrix, and no $N\times N\times d$ distance tensor.
Scratch space is $O(32N+d^2)$ besides operands. This also avoids quadratic
scratch allocation for sequences longer than 1024. Supplied dense P is caller
storage. Covariance evaluation is $O(Nd^2)$; reference attention remains
quadratic arithmetic, not a speed claim.

## Exact operand-noise statistic

Work in each variant's coordinate basis. For smoothed K, the reference operand
is token-centered K; that change shifts every key score by the same constant
per query and therefore preserves exact softmax. Rotation is the shared
$row\;x\;\mathrm{diag}(signs)\;H/\sqrt d$ operation. Let expanded stored
operands be $\widehat Q,\widehat K$, and define

$$
E_Q=\widehat Q-Q,\qquad E_K=\widehat K-K,
\quad K_c=K-\overline K,\quad E_{K,c}=E_K-\overline{E_K}.
$$

The **row-centered** difference between expanded-operand and reference scores
is exactly

$$
D=s\left(E_Q K_c^T+\widehat Q E_{K,c}^T\right).
$$

Using $\widehat Q$ in the second term retains the $E_QE_K^T$ product. Centering
only K and its residual removes the common score shift
$s(E_Q\overline K^T+\widehat Q\overline{E_K}^T)$, which has zero softmax effect.
It is wrong to count a constant K channel or a constant K residual as score
noise. It is also wrong to center Q here: Q's token mean can couple to
key-dependent residuals and change attention.

For any pair of row matrices define the **uncentered** cross second moment
$M_{AB}=A^TB/n$, with n the row count. Expanding the squared Frobenius norm
gives the exact full-pair result

$$
\sigma_D^2=\frac{\|D\|_F^2}{N_QN_K}
=s^2\left[
\langle M_{E_QE_Q},M_{K_cK_c}\rangle_F
+\langle M_{\widehat Q\widehat Q},M_{E_{K,c}E_{K,c}}\rangle_F
+2\langle M_{E_Q\widehat Q},M_{K_cE_{K,c}}\rangle_F
\right].
$$

Each matrix is only $d\times d$. The identity uses actual deterministic
rounding residuals, including their channel cross-correlations and the signed
cross term. **It assumes neither independent operands nor independent
rounding errors.** These matrices are sometimes called covariance products,
but centering Q would turn them into the wrong statistic. The implementation
clamps only a negative final second moment caused by floating-point
cancellation to zero; it leaves the signed cross term visible.

Expansion of storage values and scales is in float64. The reference operands
for this statistic are the actual float32 prepared/rotated operands. Thus the
identity is exact for their mathematical score difference, not a model of
FP32 GEMM rounding, float32 Hadamard's deviation from exact orthogonality, or
the float32 rounding of key centering and scale multiplication. Those effects
are not folded into a measured residual or fitted correction.

## From score noise to relative output error

For reference output $o_i=\sum_jp_{ij}v_j$, the derivative with respect to score
$j$ is

$$
\frac{\partial o_i}{\partial z_{ij}}=p_{ij}(v_j-o_i).
$$

The implementation computes the reference Jacobian/value sensitivity

$$
S=\frac{1}{N d_v}\sum_{i,j}p_{ij}^2\|v_j-o_i\|_2^2,
\qquad R=\frac{1}{N d_v}\|O_{ref}\|_F^2.
$$

Its fixed prediction is

$$
\widehat{\mathrm{relative\ MSE}}=\frac{\sigma_D^2S}{R},
\qquad \widehat{\mathrm{relative\ Frobenius\ error}}
=\sqrt{\widehat{\mathrm{relative\ MSE}}}.
$$

This contraction replaces the actual score-noise covariance seen by the
Jacobian with a scalar second moment. It would be exact at first order for a
suitable isotropic perturbation covariance, but that condition is **not**
asserted for FP8 residuals. Deterministic score correlations, heterogeneous
row/key magnitudes and alignment with values can all break the approximation.
Subtracting a row-common component is harmless because the true Jacobian
annihilates it; that does not justify independence of the remaining entries.

For zero reference energy the relative quantity is undefined in general. The
implementation rejects a zero-energy reference with positive sensitivity.
If both reference energy and sensitivity are zero, it reports zero by explicit
convention, including all-zero V. There is no hidden positive denominator
floor. This convention concerns the analytic score-only predictor, not an
assertion about full emulated error.

### Causal/full-pair approximation

The covariance statistic averages **all** query/key pairs, including future
keys. The default P and sensitivity are causal, so masked entries contribute
zero sensitivity. Combining a full-pair noise average with causal sensitivity
is an approximation: early queries, prefix-specific statistics, and
quantization tiles can have different noise from the all-pair mean. Removing
the global token mean still removes a true common shift on every allowed
prefix; it need not be the prefix's optimal mean. No extra causal factor of
one half is inserted, since sensitivity already excludes masked keys.

### Missing effects and large errors

This is deliberately a score-only predictor. It has **no P/V rounding floor**:
probability storage, V storage, FP32 online recurrence, output conversion,
transform arithmetic, and their interactions are not captured by
$\sigma_D^2S$. A fitted floor would cease to be parameter-free; a separately
justified floor would require specifying those operations and their
correlations. Do not interpret zero predicted error as exact full FP8
attention output.

Softmax linearization can fail when perturbations move substantial attention
mass, change the winning key, or cross saturated regions. An originally
peaked distribution can have small reference sensitivity while a large score
perturbation changes the selected value completely. A scalar second moment
also discards rare/extreme errors. Neither smoothing nor rotation is guaranteed
to improve any given head or family.

## Prospective model evaluation

```python
report = summarize_fit(
    long_rows, train_family="qwen",
    development_models=("qwen05", "smol036"),
    evaluation_models=("smol17", "tiny11", "olmo1"),
)
```

The split is fixed in [the design file](../data/v2/design.json). Qwen0.5 and
Smol360 were inspected during development. Qwen0.5 and Qwen1.5 train the affine
map; Smol360 remains a development model, not an untouched test. Smol1.7,
TinyLlama and OLMo are the prospective evaluation models. Their study operands
and outcomes were not accessed before fixing this split and the diagnostic
thresholds; public configurations and literature were already known.

The fixed error-level target and predictor are respectively
$y=\log(1+\sqrt{\mathrm{observed\ relative\ MSE}})$ and
$x=\log(1+\sqrt{\mathrm{predicted\ relative\ MSE}})$. They remain defined at
exact zero with no epsilon. For every identified head/text, gains compare each
non-tile variant to its tile baseline: $g=y_{tile}-y_{variant}$, and likewise
for predicted x. Positive gain means lower relative Frobenius error.
Duplicate identity/variant rows are rejected; missing tile baselines are
counted and excluded only from paired gain metrics.

The report separates:

1. Parameter-free Spearman (average tied ranks), direct predictive
   $R^2=1-\sum(y-x)^2/\sum(y-\overline y)^2$, MAE, and paired gain-sign
   accuracy. Negative $R^2$ is retained. Undefined correlations/constant-target
   $R^2$ are JSON null. Exact zero is a third sign, not silently counted as an
   improvement.
2. One affine OLS map $a x+b$, pooled over **qwen rows only** by default.
   It minimizes training-family squared log-space error. Other families do not
   affect either coefficient, even when their observed errors/features change.
   Rank-deficient training uses the training target mean; absent training rows
   produce null calibrated results. Negative calibrated log predictions remain
   in log space. Gains are $a(x_{tile}-x_{variant})$: the intercept cancels.
3. Separate training, development and untouched-model metrics; per-model,
   per-family and per-variant views; rotation after key-centering; and the ten
   largest parameter-free discrepancies. Every view uses the same coefficients.
   Untouched-model failures are evaluation, not feedback for changing features.

The pooled and subgroup summaries weight input rows equally. Repeated texts,
heads from the same model/layer, and GQA-sharing heads are dependent; these are
**not thousands of independent trials**. The report provides no p-values,
confidence intervals or effective sample count. Family/model-level replication
and the largest failures matter more than an impressive pooled correlation.

`study/classification.py` also reports **rotation-only** risk for one physical
head, averaging three text-level log errors first. The fixed hurt score is
$\log(1+E_{rotate})-\log(1+E_{tile})$; both the observed label and parameter-free
classifier use threshold **strictly greater than zero**. Exact observed ties
are separately reported and binary non-hurt. Hurt prevalence and majority
accuracy accompany accuracy, balanced accuracy, tie-aware ROC AUC, precision
and recall. These quantities, not pooled three-transform sign accuracy, test
whether the statistic finds rotation-harmed heads.

`study/sinks.py` computes ideal FP64 causal attention mass on the first four
tokens, averaged over queries 128..1023. A physical head's three-text mean mass
of at least 0.5 is the operational **prefix-sink** label. Layer concentration
and sink/non-sink harm rates are descriptive associations, not permanent head
identities or causal explanations of downstream loss.

## Analytic checks

`tests/test_prediction.py` covers the covariance identity against a small
dense score perturbation with nonzero means and a nonzero cross term; exact
constant-K/common-error cancellation; constant-key predictions; supplied/default
causal-reference agreement and relative normalization; nondefault shared sign
seeds; zero values; gain pairing/zero-safe targets; missing training-family
behavior; duplicate identities; and exclusion of held-out errors **and**
features from affine fitting.

Checks to run from the repository root:

```sh
uv run pytest tests/test_prediction.py tests/test_attention.py tests/test_diagnosis.py
uv run ruff check study/prediction.py tests/test_prediction.py
uv run ruff format --check study/prediction.py tests/test_prediction.py
```

The analytic checks test the identity and implementation. Predictor quality
comes only from saved model/text measurements, not the derivation or tests.
