"""Outcome-independent Q/K rounding predictions and family-held-out evaluation.

Only summarize_fit consumes observed errors. predict_head uses original operands,
reference attention, and deterministic storage residuals, never emulated outputs.
"""

import numpy as np

from attention import Config, center_keys, hadamard, quantize_qk

VARIANTS = ("tile", "rotate", "smooth_k", "rotate_smooth_k")
IDENTITY_FIELDS = ("model", "family", "text", "layer", "head", "batch")


def score_noise_moments(q, k, qhat, khat, scale):
    """Exact full-pair second moment after removing token-common score error.

    Covariances below are uncentered second/cross moments; Q's token mean must
    remain. Only K and its residual are centered. No independence is assumed.
    Operands must be in the same (possibly rotated/key-smoothed) coordinate basis.
    """
    q, k, qhat, khat = (np.asarray(x, dtype=np.float64) for x in (q, k, qhat, khat))
    if (
        q.ndim != 2
        or k.ndim != 2
        or q.shape != qhat.shape
        or k.shape != khat.shape
        or q.shape[1] != k.shape[1]
        or not q.size
        or not k.size
    ):
        raise ValueError("incompatible nonempty Q/K operand shapes")
    if not all(np.all(np.isfinite(x)) for x in (q, k, qhat, khat)):
        raise ValueError("operands must be finite")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    eq = qhat - q
    ek = khat - k
    kc = k - k.mean(axis=0)
    ekc = ek - ek.mean(axis=0)
    nq, nk = len(q), len(k)
    query_term = float(np.sum((eq.T @ eq / nq) * (kc.T @ kc / nk))) * scale**2
    key_term = float(np.sum((qhat.T @ qhat / nq) * (ekc.T @ ekc / nk))) * scale**2
    cross_term = 2 * float(np.sum((eq.T @ qhat / nq) * (kc.T @ ekc / nk))) * scale**2
    return {
        "score_noise_second_moment": max(0.0, query_term + key_term + cross_term),
        "score_noise_query_term": query_term,
        "score_noise_key_term": key_term,
        "score_noise_cross_term": cross_term,
        "q_residual_mean_square": float(np.mean(eq**2)),
        "k_residual_mean_square": float(np.mean(ek**2)),
        "k_centered_residual_mean_square": float(np.mean(ekc**2)),
        "k_common_residual_mean_square": float(np.mean(ek.mean(axis=0) ** 2)),
        "k_centered_mean_square": float(np.mean(kc**2)),
        "qhat_mean_square": float(np.mean(qhat**2)),
    }


def _mean_statistics(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    total = float(np.sum(x**2))
    mean_energy = float(len(x) * np.sum(mean**2))
    return {
        "token_mean_l2": float(np.linalg.norm(mean)),
        "total_energy": total,
        "broadcast_mean_energy": mean_energy,
        "mean_energy_fraction": mean_energy / total if total else 0.0,
        "rms": float(np.sqrt(np.mean(x**2))),
        "max_abs": float(np.max(np.abs(x))),
    }


def _reference_statistics(q, k, v, probabilities, output, scale):
    """Compute Jacobian/value sensitivity in 32-row blocks, not N x N x d."""
    n, dv = v.shape
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if probabilities.shape != (n, n):
            raise ValueError("reference_probabilities must have shape (N,N)")
    if output is not None:
        output = np.asarray(output)
        if output.shape != (n, dv) or not np.all(np.isfinite(output)):
            raise ValueError("reference_output must be finite with shape (N,value_dimension)")
    vnorm = np.sum(v**2, axis=1)
    sensitivity, output_energy, probability_energy = 0.0, 0.0, 0.0
    for start in range(0, n, 32):
        stop = min(start + 32, n)
        if probabilities is None:
            scores = (q[start:stop] @ k.T) * scale
            scores = np.where(
                np.arange(n)[None, :] <= np.arange(start, stop)[:, None], scores, -np.inf
            )
            p = np.exp(scores - scores.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
        else:
            p = np.asarray(probabilities[start:stop], dtype=np.float64)
            if (
                not np.all(np.isfinite(p))
                or np.any(p < 0)
                or not np.allclose(p.sum(axis=1), 1, rtol=1e-6, atol=1e-8)
            ):
                raise ValueError(
                    "reference probabilities must be finite normalized nonnegative rows"
                )
        o = p @ v if output is None else np.asarray(output[start:stop], dtype=np.float64)
        onorm = np.sum(o**2, axis=1)
        distance = np.maximum(0.0, onorm[:, None] + vnorm[None, :] - 2 * (o @ v.T))
        sensitivity += float(np.sum(p**2 * distance))
        probability_energy += float(np.sum(p**2))
        output_energy += float(np.sum(onorm))
    return {
        "value_sensitivity": sensitivity / (n * dv),
        "reference_output_mean_square": output_energy / (n * dv),
        "reference_probability_square_sum_per_row": probability_energy / n,
    }


def predict_head(
    q, k, v, reference_probabilities=None, reference_output=None, scale=None, sign_seed=1729
):
    """Return flat numeric features and four parameter-free relative error predictions.

    Inputs are one original N x d head. The default reference is causal float64
    attention. Supplied reference P/O may instead describe full attention; they
    must describe the same reference, and supplying O alone retains causal P.
    Quantization uses E4M3, Q tiles of 32 and K tiles of 128. Scratch reference
    storage is O(32*N), even when N exceeds 1024; no dense perturbed scores or
    emulated output errors are constructed.
    """
    q, k, v = (np.asarray(x, dtype=np.float32) for x in (q, k, v))
    if any(x.ndim != 2 or not x.size for x in (q, k, v)) or q.shape != k.shape or len(q) != len(v):
        raise ValueError("Q/K must have matching nonempty (N,d) shapes; V must have N rows")
    if not all(np.all(np.isfinite(x)) for x in (q, k, v)):
        raise ValueError("Q/K/V must be finite")
    scale = float(q.shape[1] ** -0.5 if scale is None else scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    result = {"n": len(q), "d": q.shape[1], "value_dimension": v.shape[1], "scale": scale}
    for name, x in (("q", q), ("k", k)):
        result.update({f"{name}_{key}": value for key, value in _mean_statistics(x).items()})
    reference = _reference_statistics(
        q.astype(np.float64),
        k.astype(np.float64),
        v.astype(np.float64),
        reference_probabilities,
        reference_output,
        scale,
    )
    result.update(reference)
    if reference["reference_output_mean_square"] == 0 and reference["value_sensitivity"] > 0:
        raise ValueError("relative error is undefined for zero-energy reference output")
    for variant in VARIANTS:
        cfg = Config(
            storage="e4m3",
            scaling="tile",
            rotate="rotate" in variant,
            smooth_k="smooth_k" in variant,
            sign_seed=sign_seed,
        )
        qs, qscale, ks, kscale = quantize_qk(q, k, cfg)
        work_q = q
        work_k = center_keys(k).astype(np.float32) if cfg.smooth_k else k
        if cfg.rotate:
            signs = np.random.default_rng(sign_seed).choice([-1, 1], size=q.shape[1])
            work_q, work_k = hadamard(work_q, signs), hadamard(work_k, signs)
        # Expand in float64 to isolate storage residuals from GEMM/scale rounding.
        qhat = qs.astype(np.float64) * qscale[:, None].astype(np.float64)
        khat = ks.astype(np.float64) * kscale[:, None].astype(np.float64)
        noise = score_noise_moments(work_q, work_k, qhat, khat, scale)
        result.update({f"{variant}_{key}": value for key, value in noise.items()})
        predicted_mse = noise["score_noise_second_moment"] * reference["value_sensitivity"]
        reference_mse = reference["reference_output_mean_square"]
        relative_mse = predicted_mse / reference_mse if reference_mse else 0.0
        result[f"{variant}_predicted_relative_mse"] = relative_mse
        result[f"{variant}_predicted_error"] = float(np.sqrt(relative_mse))
    return result


def _ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def _metrics(target, prediction):
    y, x = (np.asarray(a, dtype=np.float64) for a in (target, prediction))
    if not len(y):
        return {"n": 0, "spearman": None, "r2": None, "mae": None, "sign_accuracy": None}
    yr, xr = _ranks(y), _ranks(x)
    yr, xr = yr - yr.mean(), xr - xr.mean()
    rank_denominator = float(np.linalg.norm(yr) * np.linalg.norm(xr))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(y),
        "spearman": float(yr @ xr / rank_denominator) if rank_denominator else None,
        "r2": 1 - float(np.sum((y - x) ** 2)) / total if total else None,
        "mae": float(np.mean(np.abs(y - x))),
        "sign_accuracy": float(np.mean(np.sign(y) == np.sign(x))),
    }


def summarize_fit(rows, train_family="qwen", development_models=(), evaluation_models=None):
    """Evaluate fixed features, then calibrate on the named family only.

    Long rows require model, family, variant, predicted_mse and observed_mse.
    Identity fields text/layer/head/batch are optional, but the identity plus
    variant must be unique. Gains are paired against tile within that identity.
    Both mse fields are RELATIVE squared Frobenius errors. Levels and gains use
    log1p(sqrt(relative MSE)), with no hidden epsilon/floor.
    Explicit evaluation_models restricts the prospective model test; development
    models are excluded from that test even when outside the calibration family.
    """
    train_family = str(train_family)
    development_models = {str(model) for model in development_models}
    evaluation_models = (
        None if evaluation_models is None else {str(model) for model in evaluation_models}
    )
    records = []
    seen = set()
    for source in rows:
        identity = {
            key: source[key].item() if isinstance(source[key], np.generic) else source[key]
            for key in IDENTITY_FIELDS
            if key in source
        }
        model, family, variant = (str(source[field]) for field in ("model", "family", "variant"))
        identity.update(model=model, family=family)
        if any(
            not isinstance(value, str | int | float | bool | type(None))
            or (isinstance(value, float) and not np.isfinite(value))
            for value in identity.values()
        ):
            raise ValueError("identity metadata must contain finite JSON scalars")
        if variant not in VARIANTS:
            raise ValueError(f"unknown prediction variant: {variant}")
        predicted, observed = float(source["predicted_mse"]), float(source["observed_mse"])
        if not all(np.isfinite(x) and x >= 0 for x in (predicted, observed)):
            raise ValueError("predicted_mse and observed_mse must be finite and nonnegative")
        key = tuple(identity.get(field) for field in IDENTITY_FIELDS)
        if (key, variant) in seen:
            raise ValueError("duplicate head/text identity and variant")
        seen.add((key, variant))
        records.append(
            {
                **identity,
                "model": model,
                "family": family,
                "variant": variant,
                "predicted_mse": predicted,
                "observed_mse": observed,
                "x": float(np.log1p(np.sqrt(predicted))),
                "y": float(np.log1p(np.sqrt(observed))),
                "key": key,
            }
        )
    training = [row for row in records if row["family"] == train_family]
    nontraining = [row for row in records if row["family"] != train_family]
    if evaluation_models is not None and any(
        row["model"] in evaluation_models
        and (row["family"] == train_family or row["model"] in development_models)
        for row in records
    ):
        raise ValueError("evaluation models overlap calibration or development models")
    heldout = [
        row
        for row in nontraining
        if row["model"] not in development_models
        and (evaluation_models is None or row["model"] in evaluation_models)
    ]
    heldout_keys = {row["key"] for row in heldout}
    development_nontraining = [row for row in nontraining if row["model"] in development_models]
    development_keys = {row["key"] for row in development_nontraining}
    slope, intercept = None, None
    if training:
        x, y = (np.array([row[field] for row in training]) for field in ("x", "y"))
        centered = x - x.mean()
        denominator = float(centered @ centered)
        # Rank-deficient training uses the training target mean, not test data.
        slope = float(centered @ (y - y.mean()) / denominator) if denominator else 0.0
        intercept = float(y.mean() - slope * x.mean())
    baselines = {row["key"]: row for row in records if row["variant"] == "tile"}
    gains = []
    for row in records:
        if row["variant"] == "tile" or row["key"] not in baselines:
            continue
        baseline = baselines[row["key"]]
        gains.append({**row, "x": baseline["x"] - row["x"], "y": baseline["y"] - row["y"]})
    centered_baselines = {row["key"]: row for row in records if row["variant"] == "smooth_k"}
    centered_gains = []
    for row in records:
        if row["variant"] == "rotate_smooth_k" and row["key"] in centered_baselines:
            before = centered_baselines[row["key"]]
            centered_gains.append({**row, "x": before["x"] - row["x"], "y": before["y"] - row["y"]})

    def evaluate(selected, gain=False):
        target = [row["y"] for row in selected]
        raw = [row["x"] for row in selected]
        calibrated = (
            None
            if slope is None
            else _metrics(target, [slope * x + (0 if gain else intercept) for x in raw])
        )
        parameter_free = _metrics(target, raw)
        if not gain:
            parameter_free.pop("sign_accuracy")
            if calibrated is not None:
                calibrated.pop("sign_accuracy")
        return {"parameter_free": parameter_free, "calibrated": calibrated}

    def grouped(field):
        groups = {}
        for value in sorted({row[field] for row in records}):
            groups[value] = {
                "errors": evaluate([row for row in records if row[field] == value]),
                "gains": evaluate([row for row in gains if row[field] == value], gain=True),
            }
        return groups

    def failures(selected, gain=False):
        worst = sorted(selected, key=lambda row: abs(row["y"] - row["x"]), reverse=True)[:10]
        return [
            {
                **{field: row[field] for field in IDENTITY_FIELDS if field in row},
                "variant": row["variant"],
                "target": row["y"],
                "prediction": row["x"],
                "absolute_log_error": abs(row["y"] - row["x"]),
                "calibrated_prediction": (
                    None if slope is None else slope * row["x"] + (0 if gain else intercept)
                ),
            }
            for row in worst
        ]

    return {
        "definitions": {
            "error_target": "log1p(sqrt(observed_mse))",
            "error_prediction": "log1p(sqrt(predicted_mse))",
            "gain": (
                "log1p(sqrt(tile_mse)) - log1p(sqrt(variant_mse)); positive means improvement"
            ),
            "r2": "1 - sum((target-prediction)^2)/sum((target-mean(target))^2)",
            "spearman": "Pearson correlation of average ranks; undefined constants return null",
            "sign_accuracy": "exact agreement of signs including zero; gains only",
            "calibration": (
                "OLS affine map of log1p(relative Frobenius error), pooled training variants only"
            ),
            "dependence": (
                "Repeated texts, layers, and GQA-sharing heads are dependent; no p-values"
            ),
            "units": "Both MSE fields = squared Frobenius error / reference squared Frobenius norm",
        },
        "calibration": {
            "train_family": train_family,
            "train_rows": len(training),
            "heldout_rows": len(heldout),
            "train_models": sorted({row["model"] for row in training}),
            "heldout_families": sorted({row["family"] for row in heldout}),
            "slope": slope,
            "intercept": intercept,
            "status": "fit" if training else "no training-family rows; calibration not fit",
            "negative_calibrated_predictions": "retained in log space, not clipped",
        },
        "design_split": {
            "development_models": sorted(development_models),
            "requested_evaluation_models": (
                None if evaluation_models is None else sorted(evaluation_models)
            ),
            "observed_evaluation_models": sorted({row["model"] for row in heldout}),
        },
        "all": {"errors": evaluate(records), "gains": evaluate(gains, gain=True)},
        "training": {
            "errors": evaluate(training),
            "gains": evaluate([row for row in gains if row["family"] == train_family], gain=True),
        },
        "heldout": {
            "errors": evaluate(heldout),
            "gains": evaluate([row for row in gains if row["key"] in heldout_keys], gain=True),
        },
        "development_nontraining": {
            "errors": evaluate(development_nontraining),
            "gains": evaluate([row for row in gains if row["key"] in development_keys], gain=True),
        },
        "per_family": grouped("family"),
        "per_model": grouped("model"),
        "per_variant": grouped("variant"),
        "heldout_per_variant": {
            variant: {
                "errors": evaluate([row for row in heldout if row["variant"] == variant]),
                "gains": evaluate(
                    [
                        row
                        for row in gains
                        if row["variant"] == variant and row["key"] in heldout_keys
                    ],
                    gain=True,
                ),
            }
            for variant in VARIANTS
        },
        "rotation_after_key_centering": {
            "definition": "log1p(error_smooth_k) - log1p(error_rotate_smooth_k)",
            "all": evaluate(centered_gains, gain=True),
            "heldout": evaluate(
                [row for row in centered_gains if row["key"] in heldout_keys], gain=True
            ),
        },
        "paired_gain_rows": len(gains),
        "unpaired_non_tile_rows": len(records) - len(baselines) - len(gains),
        "largest_failures": failures(records),
        "largest_gain_failures": failures(gains, gain=True),
        "largest_heldout_failures": failures(heldout),
        "largest_heldout_gain_failures": failures(
            [row for row in gains if row["key"] in heldout_keys], gain=True
        ),
    }
