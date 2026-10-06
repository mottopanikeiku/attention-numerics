"""Independent full NumPy recurrence versus the packed Torch CPU implementation.

These are synthetic checks, not captured-head parity measurements. Allow one
BF16-scale rounding step plus a small absolute allowance near zero, and also
bound aggregate error. Tolerances are acceptance thresholds, not measurements.
"""

from importlib import import_module

import numpy as np
import pytest

from attention import Config, emulate, quantize_qk, reference

torch = pytest.importorskip("torch")

apply_attention = import_module("study.attention").apply_attention

FP8_VARIANTS = ("tile", "rotate", "smooth_k", "rotate_smooth_k", "smooth_kq")
ATOL = 0.002
RTOL = 0.008
RELATIVE_L2 = 0.003


def _config(variant, key_tile=128, query_tile=32, sign_seed=1729, scale=None):
    return Config(
        storage="e4m3",
        scaling="tile",
        tile=key_tile,
        query_tile=query_tile,
        probability="e4m3",
        output="bf16",
        causal=True,
        scale=scale,
        rotate=variant in ("rotate", "rotate_smooth_k"),
        smooth_k=variant in ("smooth_k", "rotate_smooth_k", "smooth_kq"),
        smooth_q=variant == "smooth_kq",
        sign_seed=sign_seed,
    )


def _full_numpy(q, k, v, cfg):
    q, k, v = (x.float().numpy() for x in (q, k, v))
    expected = np.empty_like(q)
    group = q.shape[1] // k.shape[1]
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            expected[batch, head] = emulate(
                q[batch, head], k[batch, head // group], v[batch, head // group], cfg
            )
    return expected


def _assert_parity(actual, expected):
    actual = actual.float().numpy()
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
    norm = np.linalg.norm(expected.astype(np.float64))
    error = np.linalg.norm(actual.astype(np.float64) - expected)
    assert error <= RELATIVE_L2 * norm + 1e-8


def _inputs(kind, kvheads, n=139, d=64):
    rng = np.random.default_rng(98)
    q = rng.normal(size=(2, 4, n, d)).astype(np.float32)
    k = rng.normal(size=(2, kvheads, n, d)).astype(np.float32)
    v = rng.normal(size=(2, kvheads, n, d)).astype(np.float32)
    if kind == "constant":
        q.fill(1.5)
        k.fill(-0.75)
        # Distinct batch/KV-head values expose incorrect GQA mapping/scales.
        for b in range(2):
            for h in range(kvheads):
                v[b, h].fill((b + 1) * (h + 1) / 4)
    elif kind == "mean_rich":
        q += rng.normal(size=(2, 4, 1, d)).astype(np.float32) * 3
        k += rng.normal(size=(2, kvheads, 1, d)).astype(np.float32) * 8
        # Different query-block means exercise local (not global) Q smoothing.
        q[:, :, 32:64] += 4
        v *= np.linspace(0.25, 2, kvheads, dtype=np.float32)[None, :, None, None]
    return tuple(torch.from_numpy(x).to(torch.bfloat16) for x in (q, k, v))


@pytest.mark.parametrize("variant", FP8_VARIANTS)
@pytest.mark.parametrize("kind", ["constant", "mean_rich", "varying"])
@pytest.mark.parametrize("kvheads", [2, 4])
def test_fast_matches_full_causal_numpy_all_variants(variant, kind, kvheads):
    q, k, v = _inputs(kind, kvheads)
    actual = apply_attention(q, k, v, variant)
    assert actual.shape == q.shape
    assert actual.dtype == torch.bfloat16
    _assert_parity(actual, _full_numpy(q, k, v, _config(variant)))


@pytest.mark.parametrize("variant", FP8_VARIANTS)
def test_custom_tiles_scale_and_seed_match_full_numpy(variant):
    q, k, v = _inputs("varying", 1, n=41, d=16)
    args = dict(key_tile=11, query_tile=7, sign_seed=193, scale=0.3)
    _assert_parity(
        apply_attention(q, k, v, variant, **args),
        _full_numpy(q, k, v, _config(variant, **args)),
    )


@pytest.mark.parametrize("kvheads", [1, 2, 4])
def test_native_bf16_matches_independent_float64_causal_attention(kvheads):
    q, k, v = _inputs("varying", kvheads, n=19, d=16)
    actual = apply_attention(q, k, v, "bf16").float().numpy()
    arrays = [x.float().numpy() for x in (q, k, v)]
    expected = np.empty_like(actual)
    group = 4 // kvheads
    for b in range(2):
        for h in range(4):
            expected[b, h] = reference(
                arrays[0][b, h], arrays[1][b, h // group], arrays[2][b, h // group], causal=True
            )
    np.testing.assert_allclose(actual, expected, atol=0.008, rtol=0.008)


def test_gqa_consecutive_groups_and_batch_independence():
    # Ten heads cross the eight-head chunk boundary inside one GQA group.
    q = torch.zeros((2, 10, 37, 8), dtype=torch.bfloat16)
    k = torch.zeros((2, 2, 37, 8), dtype=torch.bfloat16)
    v = torch.empty_like(k)
    v[0, 0].fill_(1)
    v[0, 1].fill_(2)
    v[1, 0].fill_(4)
    v[1, 1].fill_(8)
    for variant in ("bf16", *FP8_VARIANTS):
        actual = apply_attention(q, k, v, variant)
        expected = torch.tensor([[1] * 5 + [2] * 5, [4] * 5 + [8] * 5])
        expected = expected[:, :, None, None].expand_as(actual).to(torch.bfloat16)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("rotate", [False, True])
def test_query_smoothing_restores_key_dependent_scores_in_full_emulator(rotate):
    rng = np.random.default_rng(79)
    q, k, v = (rng.normal(size=(39, 16)).astype(np.float32) for _ in range(3))
    q += 4
    q[16:32] -= 7
    k += 12
    cfg = Config(
        storage="fp32",
        probability="fp32",
        output="fp32",
        smooth_k=True,
        smooth_q=True,
        rotate=rotate,
        query_tile=16,
        tile=11,
        causal=True,
    )
    np.testing.assert_allclose(
        emulate(q, k, v, cfg), reference(q, k, v, causal=True), atol=1e-5, rtol=1e-4
    )
    centered_q = q.copy()
    for start in range(0, len(q), cfg.query_tile):
        centered_q[start : start + cfg.query_tile] -= centered_q[
            start : start + cfg.query_tile
        ].mean(axis=0)
    # Merely centering Q without correction changes the attention distribution.
    assert (
        np.max(np.abs(reference(centered_q, k, v, causal=True) - reference(q, k, v, causal=True)))
        > 0.1
    )


def test_query_smoothing_keeps_original_float64_keys_until_centering():
    q = np.array([[1.0], [3.0], [-1.0]])
    k = np.array([[2**30 + 1.0], [2**30 + 3.0], [2**30 + 5.0]])
    v = np.array([[0.0], [1.0], [2.0]])
    cfg = Config(
        storage="fp32",
        output="fp32",
        smooth_k=True,
        smooth_q=True,
        causal=True,
        query_tile=2,
        tile=2,
    )
    _, _, packed_keys, _ = quantize_qk(q, k, cfg)
    np.testing.assert_array_equal(packed_keys[:, 0], [-2, 0, 2])
    np.testing.assert_allclose(
        emulate(q, k, v, cfg), reference(q, k, v, causal=True), atol=2e-7, rtol=1e-6
    )


def test_legacy_defaults_and_custom_sign_seed():
    rng = np.random.default_rng(13)
    q, k, v = (rng.normal(size=(17, 16)).astype(np.float32) for _ in range(3))
    old_cfg = Config(storage="e4m3", rotate=True, smooth_k=True, causal=True)
    explicit_cfg = Config(
        storage="e4m3",
        rotate=True,
        smooth_k=True,
        causal=True,
        smooth_q=False,
        sign_seed=1729,
    )
    np.testing.assert_array_equal(emulate(q, k, v, old_cfg), emulate(q, k, v, explicit_cfg))
    alternate = Config(storage="e4m3", rotate=True, smooth_k=True, causal=True, sign_seed=193)
    assert not np.array_equal(quantize_qk(q, k, old_cfg)[0], quantize_qk(q, k, alternate)[0])


def test_fast_causality_without_global_key_centering():
    # K smoothing is intentionally global preprocessing; it can change FP8
    # rounding when future K changes, despite exact softmax shift invariance.
    q, k, v = _inputs("varying", 2, n=139, d=16)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, :, 128:] *= 4
    changed_v[:, :, 128:] *= -3
    for variant in ("tile", "rotate"):
        actual = apply_attention(q, k, v, variant)
        changed = apply_attention(q, changed_k, changed_v, variant)
        assert torch.equal(actual[:, :, :128], changed[:, :, :128])


def test_invalid_gqa_and_rotation_dimensions():
    q = torch.zeros((1, 3, 5, 6), dtype=torch.bfloat16)
    k = torch.zeros((1, 2, 5, 6), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="GQA"):
        apply_attention(q, k, k, "tile")
    with pytest.raises(ValueError, match="power-of-two"):
        apply_attention(q[:, :2], k, k, "rotate")
