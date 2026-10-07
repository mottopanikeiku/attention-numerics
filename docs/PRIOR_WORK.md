# Prior work

This project is CPU emulation, not a new attention kernel or a hardware accuracy measurement. All implementation code is original; the algorithms below supply the mathematical ideas.

- **Dao, Fu, Ermon, Rudra and Ré (2022), [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).** Tiling and online softmax avoid storing the full score and probability matrices. “Exact” means no sparse/low-rank approximation, not exact floating-point arithmetic.
- **Dao (2023), [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691).** Changes work partitioning and reduces non-matmul operations. This emulator uses its unnormalized output numerator and running denominator, dividing only at the end. It does not reproduce GPU scheduling.
- **Shah, Bikshandi, Zhang, Thakkar, Ramani and Dao (2024), [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/html/2407.08608v1#S3.SS3), §3.3.** FP8 block quantization and incoherent processing apply a shared orthogonal transform to Q and K before quantization. Their hardware result is not a result of this project. We test per-block scaling and a randomized normalized Walsh–Hadamard transform independently; we do not emulate asynchronous execution or claim kernel equivalence.
- **Micikevicius et al. (2022), [FP8 Formats for Deep Learning](https://arxiv.org/html/2209.05433v2#S3), Table 1 and §2.** E4M3 has 4 exponent and 3 trailing significand bits, max finite 448, smallest subnormal 2^-9, and no infinity. E5M2 has 5 exponent and 2 trailing significand bits, max finite 57344, smallest subnormal 2^-16, and IEEE-like infinities. The paper explicitly leaves conversion rounding choices to implementations. We use `ml_dtypes.float8_e4m3fn` and `float8_e5m2`, round-to-nearest-even conversions, saturation for FP8, and current-block absolute-max scaling.
- **DeepSeek-AI (2025), [DeepSeek-V3 Technical Report, arXiv:2412.19437v2](https://arxiv.org/html/2412.19437v2#S3.SS3.SSS2), §3.3.2, “Increasing Accumulation Precision”.** Exact wording: “However, we observe that the accumulation precision of FP8 GEMM on NVIDIA H800 GPUs is limited to retaining around 14 bits, which is significantly lower than FP32 accumulation precision.” It reports promotion to FP32 CUDA-core registers every N_C=128 inner-dimension elements, equivalent to four WGMMAs. [§3.5.2, “Higher FP8 GEMM Accumulation Precision in Tensor Cores”](https://arxiv.org/html/2412.19437v2#S3.SS5.SSS2) states: “After aligning 32 mantissa products by right-shifting based on the maximum exponent, the Tensor Core only uses the highest 14 bits of each mantissa product for addition, and truncates bits exceeding this range. The accumulation of addition results into registers also employs 14-bit precision.” This motivates a **surrogate**, not a bit-exact H800 model: 32-product shared-exponent truncation, 14 significant bits (including the leading bit), truncation toward zero, and optional 128-term promotion. The report does not specify every rounding detail; the chosen reduction order and significand interpretation are explicit assumptions.
- **Ashkboos et al. (2024), [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/html/2404.00456v2), §3–4.** Randomized Hadamard transforms remove activation outliers while preserving unquantized computation; the method includes integer 4-bit weights/activations/KV caches. Here only Q/K rotation is tested with FP8; this is not a reproduction of QuaRot's end-to-end method.

## SageAttention: key smoothing before integer quantization

Zhang et al., **SageAttention: Accurate 8-Bit Attention for Plug-and-Play Inference Acceleration**, ICLR 2025. These details were checked against the [full paper, arXiv:2410.02367v9](https://arxiv.org/pdf/2410.02367v9), dated 1 October 2025; the version is pinned because later revisions need not retain the same numbering.

[§4.2, Eq. (6)](https://arxiv.org/html/2410.02367v9#S4.SS2) defines `gamma(K) = K - mean(K)`, where `mean(K) = sum_t K[t, :] / N` is a channel vector averaged over **all tokens**, not a scalar averaged over channels or a separate mean for each key tile. In the paper's single-head `N × d` notation, this is one vector per head, broadcast to every token. The subtraction precedes quantization; Algorithm 1 explicitly places it in preprocessing. The motivation is the authors' observation that key-channel outliers can be a large common bias across tokens plus a smaller token-dependent signal. This is an observed pattern in their operands, not a guarantee about every model's keys.

Write `K_c = K - 1 mu_K`. For query row `q_r`,

$$
\frac{q_r K_c^\top}{\sqrt d}
= \frac{q_r K^\top}{\sqrt d}
- \frac{q_r\mu_K^\top}{\sqrt d}\mathbf{1}^\top.
$$

The removed term is constant across keys within that query row, so row softmax is unchanged in exact arithmetic. The **logits do change**; the probabilities and attention output do not. This does not promise identical results after mean/subtraction rounding, quantization, or finite-precision matmul.

[§4.3–4.5 and Table 6](https://arxiv.org/html/2410.02367v9#S4.SS3) distinguish SageAttention's variants: Q/K use INT8, with per-token or per-block scales. SAGEAttn-B/T retain the unnormalized online-softmax probabilities and V in FP16 and use an FP16 matmul accumulator. The vB/vT variants instead quantize those operands to INT8. Thus neither “all operands are FP8” nor “every SageAttention variant uses FP16 P/V” describes the paper.

## SageAttention2: query smoothing needs a correction

Zhang et al., **SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization**, [ICML 2025 publication record](https://proceedings.mlr.press/v267/zhang25ae.html). Method details were checked against the [full paper, arXiv:2411.10958v7](https://arxiv.org/pdf/2411.10958v7), dated 1 October 2025.

[§3.1, Eq. (2) and the following decomposition](https://arxiv.org/html/2411.10958v7#S3.SS1) center each **query block** using its token-axis mean `mu_Qi`, while K uses the all-token mean `mu_K`. Both are channel vectors. After quantizing the centered operands and dequantizing their product, the method adds `Delta S_ij = mu_Qi (K_j - mu_K)^T` before softmax. This vector varies across keys and is broadcast across query rows in the block. Only the remaining row-constant term can be discarded. The §3.1 decomposition suppresses attention's `1/sqrt(d)` factor; that factor applies to the corrected logits as in §2.1, Eq. (1).

Naively replacing Q by `Q - mu_Q` subtracts `mu_Q K^T`, which is generally **not** constant across keys. Unlike K centering, Q centering alone is not softmax-invariant. SageAttention2's compensating GEMV is essential, not an optional accuracy adjustment.

[§3.2 and Appendix A.6, Eq. (8)](https://arxiv.org/html/2411.10958v7#A1.SS6) describe hardware-layout-aware **per-thread INT4 Q/K** quantization; the INT4 MMA accumulates into INT32 before dequantization. [§3.3](https://arxiv.org/html/2411.10958v7#S3.SS3) uses **E4M3 FP8** for V and the unnormalized online-softmax quantity `P_tilde = exp(S - m)`, not already-normalized P. P_tilde has a static scale `1/448`, while V is quantized per channel. [§4.1, Table 3](https://arxiv.org/html/2411.10958v7#S4.SS1) also defines SageAttention2-8b: it uses INT8 Q/K and omits Q smoothing, while retaining the other techniques.

[§3.4 and Algorithm 1](https://arxiv.org/html/2411.10958v7#S3.SS4) separate FP8 **operand format** from accumulation. The authors report an effective FP22 internal accumulator for the tested Ada/Hopper `mma(f32f8f8f32)` instruction: one sign bit, eight exponent bits, and thirteen trailing significand bits. They compute a block's P_tilde/V product in that accumulator, then combine block products and online rescaling in a separate FP32 register buffer. This is the paper's hardware observation and two-level strategy, not a universal specification of every FP8 instruction or a measurement by this project. FP32 destination registers alone do not establish full-FP32 internal accumulation.

## Why the FP8 surrogate is not a Sage reproduction

[Micikevicius et al., arXiv:2209.05433v2, §2–3 and Table 1](https://arxiv.org/html/2209.05433v2#S3) specify an exponent and significand in **each FP8 element**, alongside software-managed tensor scaling. Within a fixed scaled quantization group, integer quantization has a uniform absolute step. Normal FP8 spacing instead grows with the element's exponent, with approximately relative precision within its normal range; subnormals and saturation are exceptions. Sharing a block scale does not make FP8 a uniform integer grid. Consequently, removing a large absolute maximum or rotating outliers is not, by itself, proof that FP8 error must improve.

This project's key-centering-only FP8 surrogate borrows SageAttention's softmax-invariant preprocessing while leaving Q uncentered and retaining the emulator's FP8 scaling and rounding choices. It does not reproduce SageAttention's INT8 Q/K kernels or SageAttention2's corrected Q smoothing, per-thread INT4 groups, static P_tilde/per-channel V FP8 quantization, and hardware two-level accumulation. The optional shared Q/K Hadamard transform is a separate ablation, not Sage's mean-subtraction method. It cannot establish Sage kernel accuracy, speed, end-to-end model quality, or a unique cause of rotation failure. The controlled results and remaining uncertainty are in [MODEL.md](MODEL.md#rotation-diagnosis).

## What this adds

A small inspectable rounding model and controlled ablations separate input quantization, probability quantization, accumulation, online rescaling, and output rounding. The study compares the same synthetic arrays to a two-pass chunked float64 reference and reports error as context grows. It adds no claim of a new algorithm, GPU speed, model quality, or H800 prediction.

The locally available accuracy baselines are the NumPy FP32 condition and PyTorch CPU scaled-dot-product attention at BF16/FP32 on the same captured operands ([results](../results/real.csv)). CPU SDPA/BF16 is effectively tied in median error and wins on the first layer's first tested head. FlashAttention GPU implementations cannot run on this GPU-free machine; comparisons to their published numerical tables would mix hardware, inputs and rounding models, so they are not made.

## Cross-family study: known ideas and the narrower question

The preceding surrogate description and results concern the original study. The cross-family extension asks about attention-only FP8 rounding on actual post-RoPE operands; it does not turn those earlier results into downstream model-quality evidence. The table below distinguishes established methods from a potentially useful empirical comparison. It reports no results of the extension.

| Topic | Already established in primary sources | Scope for this study |
| --- | --- | --- |
| Mean-key cancellation | [SageAttention v9, §4.2, Eq. (6)](https://arxiv.org/html/2410.02367v9#S4.SS2) subtracts the token-axis key mean and explicitly derives softmax invariance. | Measure how much common key bias is present and what centering changes under specified FP8 rounding; not a new identity or smoothing algorithm. |
| Corrected query smoothing | [SageAttention2 v7, §3.1, Eq. (2)](https://arxiv.org/html/2411.10958v7#S3.SS1) gives query-block centering and its key-dependent correction. Appendix A.5 also analyzes smoothing under Gaussian assumptions. | Apply the existing correction in an FP8 surrogate, separately from the paper's integer kernel. |
| Rotation and outliers | [FlashAttention-3 v1, §3.3](https://arxiv.org/html/2407.08608v1#S3.SS3), [QuaRot v2, §3–4](https://arxiv.org/html/2404.00456v2#S3) and [SpinQuant v4, §2–3](https://arxiv.org/html/2405.16406v4#S2) already study incoherent processing, outlier reduction and rotation-dependent quantization error. | Characterize when a fixed shared Q/K transform helps or hurts these FP8 operands; not invent rotation or assert universal improvement. |
| Integer versus FP8 | [SageAttention v9, §4.3, Tables 2–3](https://arxiv.org/html/2410.02367v9#S4.SS3) compares integer and floating-point operand formats. [FP8 Formats v2, §3, Table 1](https://arxiv.org/html/2209.05433v2#S3) specifies FP8 encodings. | Keep the distinction between an integer uniform grid and FP8 exponent-dependent spacing; a smaller absolute maximum alone is not an FP8 error guarantee. |
| Layer/head coverage and downstream comparison | [KIVI v2, §3.2, Table 2](https://arxiv.org/html/2402.02750v2#S3.SS2) averages errors over all layers and heads. [SageAttention2 v7, §4.1–4.3, Tables 4–5](https://arxiv.org/html/2411.10958v7#S4.SS1) compares rotation and smoothing and evaluates downstream metrics. | Neither all-head coverage, multiple models, nor a rotation/smoothing comparison alone is new. |
| Potential contribution | The reviewed sources establish these ingredients, but do not establish this particular matched post-RoPE FP8 study. | A cross-family, per-head characterization linked to a prediction tested on held-out families/texts, plus a matched attention-only comparison using held-out next-token CE, exp(CE) and output-distribution KL. This is a research question, not a demonstrated contribution or a claim of priority. |

### Exact transform and correction conventions

[FlashAttention-3 v1, §3.3](https://arxiv.org/html/2407.08608v1#S3.SS3) uses **row-matrix right multiplication**:

$$
Q'=QM,\qquad K'=KM,\qquad MM^\top=I,\qquad Q'K'^\top=QK^\top.
$$

Its incoherent-processing step rotates **Q and K only**, not V. V has block quantization, which is a different operation. The paper describes M as a product of random sign-diagonal matrices and a Hadamard matrix, but does not specify their factor order/count or a random-number generator/seed. A concrete row-vector realization is `x D H`, with sign diagonal D and normalized Walsh–Hadamard H: signs precede mixing. This is an implementation choice consistent with the stated mathematics, not a source-prescribed factorization. In contrast, `x H D` only flips the mixed coordinates' signs. For symmetric magnitude-based quantization, those final sign flips do not provide the randomized cancellation of input contributions. If V were rotated as well, the output basis would change and would require an inverse transform; that is not this FA3 step.

For [SageAttention2 v7, §3.1](https://arxiv.org/html/2411.10958v7#S3.SS1), write `Q_i^c = Q_i - mu_Qi` and `K_j^c = K_j - mu_K`. The means are channel vectors: Q's mean is over each query block's tokens, K's over all key tokens, independently for each batch/head. The corrected logits are

$$
\widetilde S_{ij}
=\alpha\left[
\operatorname{dequant}(\widehat Q_i^c(\widehat K_j^c)^\top)
+\mathbf1\,\mu_{Qi}(K_j^c)^\top
\right].
$$

The correction uses centered K **before quantization**, varies over keys, and is broadcast over query rows before softmax. The discarded term is the row-constant `alpha Q_i mu_K^T`. The paper suppresses alpha in its §3.1 decomposition; its §2.1 attention definition supplies the scaling. Without the correction, Q centering changes attention even in exact arithmetic.

The earlier datatype distinctions remain essential: [SageAttention2 v7, §3.2–3.4 and Appendix A.6](https://arxiv.org/html/2411.10958v7#S3.SS2) uses per-thread INT4 Q/K with INT32 MMA accumulation, E4M3 unnormalized online-softmax probabilities with scale `1/448`, per-channel E4M3 V, and a separate FP32 block-accumulation buffer around the reported effective FP22 hardware accumulator. FP8 Q/K plus FP32 recurrence borrows preprocessing, not this kernel's quantization or accumulation behavior.

In this study's [attention emulator](../study/attention.py), the rotation is shared Q/K `D H`, leaves V unrotated, and uses NumPy PCG64 with default `sign_seed=1729`; these are declared emulator choices, not paper-specified constants. The rounded conditions use blockwise E4M3 Q/K/V and FP32 recurrence. `smooth_kq` restores the mean-query correction using prequantized centered K. These conditions do not reproduce SageAttention2's INT4 Q/K, per-channel V or effective FP22 hardware accumulation.

### Sinks, massive activations and no-op heads are related, not equivalent

- **Xiao, Tian, Chen, Han and Lewis, [Efficient Streaming Language Models with Attention Sinks, arXiv:2309.17453v3](https://arxiv.org/html/2309.17453v3#S3.SS1), 2023, §3.1–3.3.** The authors observe semantically unimportant initial tokens attracting attention and show that retaining their KV states stabilizes windowed inference. This concerns attention allocation and cache eviction. Subtracting a common key vector preserves that allocation in exact arithmetic; it does not remove sink tokens.
- **Mingjie Sun, Xinlei Chen, Kolter and Zhuang Liu, [Massive Activations in Large Language Models, arXiv:2402.17762v2](https://arxiv.org/html/2402.17762v2#S2), 2024, §2–4.** The measured objects are rare scalar outliers in post-residual hidden states, not a mean vector shared by all key tokens. Their interventions support a bias-like role and a connection to attention concentration. The paper explicitly distinguishes massive activations from widespread outlier features (§2.3); neither is automatically Sage's common key bias.
- **Bondarenko, Nagel and Blankevoort, [Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing, arXiv:2306.12929v2](https://arxiv.org/html/2306.12929v2#S3), 2023, §3–4.** Their no-op hypothesis connects concentrated attention on low-value tokens to small residual updates and pressure toward large logit differences. Clipped softmax (§4.1, Eq. (4)) and learned gated attention (§4.2, Eq. (5)) alter the trained architecture; mean-key cancellation does not. Large keys alone do not identify a no-op head: values and the resulting update matter.
- **Shangwen Sun, Canziani, LeCun and Zhu, [The Spike, the Sparse and the Sink, arXiv:2603.05498v1](https://arxiv.org/html/2603.05498v1#S4.SS2.SSS2), 2026 preprint, §3 and §4.2.2.** The authors analyze Llama/Qwen and use normalization ablations to suppress spikes while retaining sinks. This newer evidence further cautions against treating co-occurrence as equivalence or proposing one universal cause from operand statistics alone.

### KV-cache quantization addresses a different numerical workload

- **Mengzhao Chen et al., [PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language Models Quantization, arXiv:2410.05265v2](https://arxiv.org/html/2410.05265v2#S4), 2025 revision, §4.1–4.3.** The revised title differs from v1's *Static Quantization Beats Dynamic through Prefixed Outliers in LLMs*. PrefixQuant isolates token-wise outliers using cached prefix tokens, builds on Hadamard rotations, and adds block-wise fine-tuning. Its prefix intervention changes context; it is not algebraic key centering. Its Q/K analysis also includes unusually small-magnitude tokens, so not every relevant outlier is a large coordinate.
- **Hooper et al., [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization, arXiv:2401.18079v6](https://arxiv.org/html/2401.18079v6#S3), 2025 revision of the 2024 work, §3.1–3.6.** It combines per-channel pre-RoPE keys, per-token values, calibrated non-uniform codebooks and separate high-precision outliers. Its sink-aware step (§3.5) retains the first token in FP16. Pre-RoPE storage with RoPE after dequantization is not post-RoPE FP8 attention arithmetic.
- **Zirui Liu et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache, arXiv:2402.02750v2](https://arxiv.org/html/2402.02750v2#S3), 2024, §3.1–3.3.** It uses affine integer quantization with per-channel keys, per-token values, and a full-precision recent residual cache. Its analysis distinguishes operand reconstruction error from attention-output error; its prefill passes exact KV tensors onward while storing the quantized cache. That is not quantizing both Q/K and online-softmax operands throughout prefill.

These papers motivate checking token structure, channel structure, RoPE location, quantization groups and downstream sensitivity separately. They do not justify inferring cache-memory savings, kernel speed, sink removal or a common causal mechanism from this study's FP8 emulation.
