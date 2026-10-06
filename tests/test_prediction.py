import json

import numpy as np
import pytest

from attention import Config, center_keys, hadamard, quantize_qk
from study.prediction import VARIANTS, predict_head, score_noise_moments, summarize_fit


def test_covariance_formula_matches_dense_row_centered_score_noise():
    rng = np.random.default_rng(51)
    q = rng.normal(size=(5, 4)) + 3
    k = rng.normal(size=(7, 4)) + 11
    eq = 0.03 * q + rng.normal(scale=0.07, size=q.shape)
    ek = 0.04 * k + rng.normal(scale=0.09, size=k.shape)
    qhat, khat = q + eq, k + ek
    scale = 0.37
    delta = scale * (qhat @ khat.T - q @ k.T)
    delta -= delta.mean(axis=1, keepdims=True)
    result = score_noise_moments(q, k, qhat, khat, scale)
    np.testing.assert_allclose(result["score_noise_second_moment"], np.mean(delta**2), rtol=2e-13)
    np.testing.assert_allclose(
        result["score_noise_second_moment"],
        result["score_noise_query_term"]
        + result["score_noise_key_term"]
        + result["score_noise_cross_term"],
        rtol=1e-15,
    )
    assert result["score_noise_cross_term"] != 0


def test_exact_constant_key_and_common_residual_cancellation():
    q = np.arange(24, dtype=np.float64).reshape(6, 4) / 8
    k = np.broadcast_to(np.array([2.0, 4.0, -8.0, 1.0]), (9, 4))
    qhat = q + 0.125
    khat = k + np.array([0.25, -0.5, 1.0, 0.125])
    result = score_noise_moments(q, k, qhat, khat, 0.5)
    assert result["score_noise_second_moment"] == 0
    assert result["k_centered_residual_mean_square"] == 0
    assert result["k_common_residual_mean_square"] > 0


def test_common_key_error_cancels_with_nonconstant_keys():
    q = np.arange(24, dtype=np.float64).reshape(6, 4) / 8
    k = np.arange(36, dtype=np.float64).reshape(9, 4) / 8
    khat = k + np.array([0.25, -0.5, 1.0, 0.125])
    assert score_noise_moments(q, k, q, khat, 0.5)["score_noise_second_moment"] == 0


def test_constant_keys_give_zero_score_prediction_for_all_variants():
    rng = np.random.default_rng(17)
    q, v = (rng.normal(size=(137, 8)).astype(np.float32) for _ in range(2))
    k = np.broadcast_to(np.arange(8, dtype=np.float32) + 0.13, q.shape).copy()
    result = predict_head(q, k, v)
    assert result["k_mean_energy_fraction"] == pytest.approx(1)
    for variant in VARIANTS:
        assert result[f"{variant}_score_noise_second_moment"] == 0
        assert result[f"{variant}_predicted_relative_mse"] == 0
        assert result[f"{variant}_predicted_error"] == 0


def test_supplied_causal_reference_matches_default_and_relative_normalization():
    rng = np.random.default_rng(22)
    q, k, v = (rng.normal(size=(13, 8)).astype(np.float32) for _ in range(3))
    scale = 0.21
    scores = scale * (q.astype(np.float64) @ k.astype(np.float64).T)
    scores[np.triu_indices(len(q), 1)] = -np.inf
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    o = p @ v.astype(np.float64)
    expected = predict_head(q, k, v, scale=scale)
    supplied = predict_head(q, k, v, p, o, scale=scale)
    for key in expected:
        assert supplied[key] == pytest.approx(expected[key], rel=2e-13, abs=1e-16)
    distance = np.sum((v.astype(np.float64)[None, :, :] - o[:, None, :]) ** 2, axis=2)
    sensitivity = float(np.sum(p**2 * distance)) / np.sum(o**2)
    for variant in VARIANTS:
        predicted = expected[f"{variant}_score_noise_second_moment"] * sensitivity
        assert expected[f"{variant}_predicted_relative_mse"] == pytest.approx(predicted)
        assert expected[f"{variant}_predicted_error"] == pytest.approx(np.sqrt(predicted))


def test_nondefault_seed_reuses_shared_quantization_conventions():
    rng = np.random.default_rng(59)
    q, k, v = (rng.normal(size=(39, 8)).astype(np.float32) for _ in range(3))
    seed = 71
    result = predict_head(q, k, v, sign_seed=seed)
    signs = np.random.default_rng(seed).choice([-1, 1], size=8)
    for variant in ("rotate", "rotate_smooth_k"):
        smooth = variant == "rotate_smooth_k"
        cfg = Config(storage="e4m3", scaling="tile", rotate=True, smooth_k=smooth, sign_seed=seed)
        qs, qscale, ks, kscale = quantize_qk(q, k, cfg)
        work_q = hadamard(q, signs)
        work_k = hadamard(center_keys(k).astype(np.float32) if smooth else k, signs)
        expected = score_noise_moments(
            work_q,
            work_k,
            qs.astype(np.float64) * qscale[:, None],
            ks.astype(np.float64) * kscale[:, None],
            8**-0.5,
        )
        for name, value in expected.items():
            assert result[f"{variant}_{name}"] == value


def test_zero_values_have_zero_relative_error_without_hidden_floor():
    q = np.ones((5, 4), dtype=np.float32)
    result = predict_head(q, q, np.zeros_like(q))
    assert result["reference_output_mean_square"] == 0
    for variant in VARIANTS:
        assert result[f"{variant}_predicted_relative_mse"] == 0


def _fit_rows():
    rows = []
    for family in ("qwen", "smol", "olmo"):
        for head in range(3):
            for variant, factor in (("tile", 2.0), ("rotate", 1.0), ("smooth_k", 0.5)):
                rows.append(
                    {
                        "model": f"{family}-model",
                        "family": family,
                        "text": "first",
                        "layer": 0,
                        "head": head,
                        "variant": variant,
                        "predicted_mse": (head + 1) * factor,
                        "observed_mse": (head + 1) * factor * 3,
                    }
                )
    return rows


def test_heldout_families_are_excluded_from_affine_fit():
    rows = _fit_rows()
    original = summarize_fit(rows)
    changed = [
        {
            **row,
            "observed_mse": row["observed_mse"] * (1e8 if row["family"] != "qwen" else 1),
            "predicted_mse": row["predicted_mse"] * (1e4 if row["family"] != "qwen" else 1),
        }
        for row in rows
    ]
    second = summarize_fit(changed)
    assert original["calibration"] == second["calibration"]
    assert original["training"] == second["training"]
    assert original["heldout"] != second["heldout"]
    assert original["calibration"]["train_rows"] == 9
    assert original["calibration"]["train_models"] == ["qwen-model"]
    x, y = (
        np.array([np.log1p(np.sqrt(row[field])) for row in rows if row["family"] == "qwen"])
        for field in ("predicted_mse", "observed_mse")
    )
    coefficient = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
    assert original["calibration"]["slope"] == pytest.approx(coefficient[0])
    assert original["calibration"]["intercept"] == pytest.approx(coefficient[1])
    json.dumps(original, allow_nan=False)


def test_gain_metrics_pair_heads_and_use_zero_safe_relative_frobenius_target():
    rows = _fit_rows()
    for row in rows:
        row["observed_mse"] = row["predicted_mse"]
    for row in rows:
        if row["variant"] == "smooth_k":
            row["predicted_mse"] = row["observed_mse"] = 0
    result = summarize_fit(rows)
    assert result["paired_gain_rows"] == 18
    assert result["all"]["gains"]["parameter_free"]["sign_accuracy"] == 1
    assert result["all"]["gains"]["parameter_free"]["r2"] == 1
    assert result["all"]["errors"]["parameter_free"]["spearman"] == pytest.approx(1)
    assert result["all"]["errors"]["parameter_free"]["r2"] == 1
    assert result["per_family"]["smol"]["errors"]["parameter_free"]["n"] == 9
    assert result["per_model"]["olmo-model"]["gains"]["parameter_free"]["n"] == 6


def test_missing_training_family_has_no_calibrated_metrics():
    result = summarize_fit([row for row in _fit_rows() if row["family"] == "smol"])
    assert result["calibration"]["slope"] is None
    assert result["heldout"]["errors"]["calibrated"] is None
    assert result["training"]["errors"]["parameter_free"]["spearman"] is None
    json.dumps(result, allow_nan=False)


def test_duplicate_identity_does_not_silently_pair_different_texts():
    row = _fit_rows()[0]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_fit([row, row])


def test_prospective_model_split_excludes_nontraining_development_data():
    rows = []
    for model, family in (("qwen05", "qwen"), ("smol036", "smol"), ("smol17", "smol")):
        for variant, prediction, observation in (("tile", 0.04, 0.09), ("rotate", 0.09, 0.16)):
            rows.append(
                {
                    "model": model,
                    "family": family,
                    "head": 0,
                    "variant": variant,
                    "predicted_mse": prediction,
                    "observed_mse": observation,
                }
            )
    options = {"development_models": ("qwen05", "smol036"), "evaluation_models": ("smol17",)}
    first = summarize_fit(rows, **options)
    assert first["calibration"]["heldout_rows"] == 2
    assert first["design_split"]["observed_evaluation_models"] == ["smol17"]
    assert first["development_nontraining"]["errors"]["parameter_free"]["n"] == 2
    assert first["heldout_per_variant"]["rotate"]["gains"]["parameter_free"]["n"] == 1
    changed = [
        {**row, "observed_mse": 100 if row["model"] == "smol036" else row["observed_mse"]}
        for row in rows
    ]
    second = summarize_fit(changed, **options)
    assert first["calibration"] == second["calibration"]
    assert first["heldout"] == second["heldout"]
    with pytest.raises(ValueError, match="overlap"):
        summarize_fit(rows, development_models=("smol036",), evaluation_models=("qwen05",))


def test_rotation_after_centering_reuses_the_single_training_slope():
    rows = [
        {
            "model": "qwen05",
            "family": "qwen",
            "variant": variant,
            "predicted_mse": predicted,
            "observed_mse": observed,
        }
        for variant, predicted, observed in (
            ("tile", 0.16, 0.36),
            ("rotate", 0.09, 0.25),
            ("smooth_k", 0.04, 0.09),
            ("rotate_smooth_k", 0.01, 0.04),
        )
    ]
    result = summarize_fit(rows)
    contrast = result["rotation_after_key_centering"]["all"]
    predicted = np.log1p(0.2) - np.log1p(0.1)
    target = np.log1p(0.3) - np.log1p(0.2)
    assert contrast["parameter_free"]["mae"] == pytest.approx(abs(target - predicted))
    assert contrast["calibrated"]["mae"] == pytest.approx(
        abs(target - result["calibration"]["slope"] * predicted)
    )
