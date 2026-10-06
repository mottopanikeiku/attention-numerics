import math
from dataclasses import replace

import numpy as np
import pytest

from attention import (
    Config,
    cast,
    emulate,
    hadamard,
    matmul,
    metrics,
    quantize,
    reference,
    shared_exponent_sum,
    truncate_significand,
)


def dense(q, k, v, causal=False, scale=None):
    scores = np.asarray(q, dtype=np.float64) @ np.asarray(k, dtype=np.float64).T
    scores *= q.shape[1] ** -0.5 if scale is None else scale
    if causal:
        scores = np.where(np.arange(len(k))[None, :] <= np.arange(len(q))[:, None], scores, -np.inf)
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities @ np.asarray(v, dtype=np.float64)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("tile", [1, 7, 32])
def test_chunked_reference_matches_independent_dense(causal, tile):
    rng = np.random.default_rng(11)
    q, k, v = (rng.normal(size=(19, 8)) for _ in range(3))
    actual = reference(q, k, v, causal=causal, tile=tile, query_tile=3)
    np.testing.assert_allclose(actual, dense(q, k, v, causal), rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("storage", ["fp32", "bf16", "e4m3", "e5m2"])
@pytest.mark.parametrize("update", ["global", "local"])
@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_constant_values_known_answer(storage, update, order):
    rng = np.random.default_rng(17)
    q, k = (rng.normal(size=(13, 8)).astype(np.float32) for _ in range(2))
    v = np.ones((13, 4), dtype=np.float32)
    cfg = Config(storage=storage, tile=4, query_tile=3, update=update, order=order, causal=True)
    # Low-precision probabilities need not sum exactly to the unrounded denominator.
    # With uniform logits, all probabilities are exactly 1 before final division.
    np.testing.assert_array_equal(emulate(q * 0, k, v, cfg), v)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("tile", [1, 5, 64])
@pytest.mark.parametrize("update", ["global", "local"])
@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_fp32_online_matches_dense(causal, tile, update, order):
    rng = np.random.default_rng(42)
    q, k, v = (rng.normal(size=(23, 8)).astype(np.float32) for _ in range(3))
    cfg = Config(
        storage="fp32", output="fp32", tile=tile, causal=causal, update=update, order=order
    )
    np.testing.assert_allclose(emulate(q, k, v, cfg), dense(q, k, v, causal), atol=8e-7, rtol=8e-6)


def test_uniform_causal_prefix_mean():
    q = np.zeros((17, 4), dtype=np.float32)
    v = np.arange(17, dtype=np.float32)[:, None] / 16
    cfg = Config(tile=3, causal=True, output="fp32", probability="fp32")
    expected = np.arange(17, dtype=np.float32)[:, None] / 32
    np.testing.assert_array_equal(emulate(q, q, v, cfg), expected)


def test_two_key_logistic_known_answer():
    q = np.ones((1, 1))
    k = np.array([[0], [math.log(3)]])
    v = np.array([[0], [1]])
    cfg = Config(storage="fp32", output="fp32", tile=1)
    np.testing.assert_allclose(reference(q, k, v), [[0.75]], atol=1e-15)
    np.testing.assert_allclose(emulate(q, k, v, cfg), [[0.75]], atol=1e-7)


@pytest.mark.parametrize("fmt,expected", [("bf16", 1.0), ("e4m3", 1.0), ("e5m2", 1.0)])
def test_nearest_even_ties(fmt, expected):
    half_ulp = {"bf16": 2**-8, "e4m3": 2**-4, "e5m2": 2**-3}[fmt]
    np.testing.assert_array_equal(cast([1 + half_ulp], fmt), [expected])


def test_fp8_format_boundaries_and_saturation():
    np.testing.assert_array_equal(cast([448, 1000, 2**-9, 2**-11], "e4m3"), [448, 448, 2**-9, 0])
    np.testing.assert_array_equal(cast([57344, 1e6, 2**-16], "e5m2"), [57344, 57344, 2**-16])


def test_quantization_zero_and_tile_scale():
    zero, scales = quantize(np.zeros((5, 4)), "e4m3", 2)
    np.testing.assert_array_equal(zero, 0)
    np.testing.assert_array_equal(scales, 1)
    x = np.array([[1, -1], [2, -2], [100, -100]], dtype=np.float32)
    packed, scales = quantize(x, "e4m3", 2)
    np.testing.assert_allclose(packed * scales[:, None], x, rtol=1e-7)
    assert scales[0] == scales[1]
    assert scales[0] != scales[2]


@pytest.mark.parametrize("dimension", [64, 128])
def test_rotation_preserves_dot_products_and_norms(dimension):
    rng = np.random.default_rng(12)
    q, k = (rng.normal(size=(7, dimension)).astype(np.float32) for _ in range(2))
    signs = rng.choice([-1, 1], size=dimension)
    qr, kr = hadamard(q, signs), hadamard(k, signs)
    np.testing.assert_allclose(qr @ kr.T, q @ k.T, atol=5e-6, rtol=5e-6)
    np.testing.assert_allclose(np.linalg.norm(qr, axis=1), np.linalg.norm(q, axis=1), rtol=2e-7)


def test_fourteen_bits_include_leading_bit():
    x = np.array([1 + 2**-14, -(1 + 2**-14), 2**-140, 0], dtype=np.float32)
    np.testing.assert_array_equal(truncate_significand(x), [1, -1, 2**-140, 0])


def scalar_reduced(a, b, promote=0):
    # Deliberately separate scalar implementation of the specified surrogate.
    output, partial = np.float32(0), np.float32(0)
    for start in range(0, len(a), 32):
        products = [
            float(np.float32(x * y))
            for x, y in zip(a[start : start + 32], b[start : start + 32], strict=True)
        ]
        maximum = max(abs(x) for x in products)
        exponent = math.frexp(maximum)[1] if maximum else 0
        quantum = math.ldexp(1, exponent - 14)
        total = np.float32(sum(math.trunc(x / quantum) * quantum for x in products))
        value = np.float32((partial if promote else output) + total)
        fraction, exponent = math.frexp(float(value))
        value = np.float32(math.ldexp(math.trunc(fraction * 2**14), exponent - 14))
        if promote:
            partial = value
            if (start + 32) % promote == 0 or start + 32 >= len(a):
                output = np.float32(output + partial)
                partial = np.float32(0)
        else:
            output = value
    return output


@pytest.mark.parametrize("promote", [0, 128])
def test_reduced_matmul_matches_scalar_model(promote):
    rng = np.random.default_rng(7)
    # Exact FP8 representable inputs, not already dequantized values.
    a = cast(rng.normal(size=(2, 131)), "e4m3")
    b = cast(rng.normal(size=(131, 3)), "e4m3")
    expected = np.array([[scalar_reduced(row, column, promote) for column in b.T] for row in a])
    np.testing.assert_array_equal(matmul(a, b, "reduced14", promote), expected)


def test_shared_exponent_discards_small_products():
    np.testing.assert_array_equal(shared_exponent_sum([[1, 2**-15, -(2**-15)]]), [1])


def test_fp32_matmul_matches_slow_scalar_reference():
    rng = np.random.default_rng(4)
    a = cast(rng.normal(size=(3, 67)), "bf16")
    b = cast(rng.normal(size=(67, 2)), "bf16")
    expected = np.zeros((3, 2), dtype=np.float32)
    for i in range(3):
        for j in range(2):
            for index in range(67):
                expected[i, j] = np.float32(expected[i, j] + np.float32(a[i, index] * b[index, j]))
    np.testing.assert_allclose(matmul(a, b), expected, atol=3e-6, rtol=1e-6)


def test_promotion_reduces_long_dot_error():
    rng = np.random.default_rng(3)
    a = cast(rng.normal(size=(1, 4096)), "e4m3")
    b = cast(rng.normal(size=(4096, 1)), "e4m3")
    exact = a.astype(np.float64) @ b.astype(np.float64)
    ordinary = matmul(a, b, "reduced14")
    promoted = matmul(a, b, "reduced14", 128)
    assert abs(promoted - exact).item() < abs(ordinary - exact).item()


def test_compensation_on_many_small_denominator_updates():
    q = np.ones((1, 1), dtype=np.float32)
    k = np.full((4096, 1), -16, dtype=np.float32)
    k[0] = 0
    # Isolate denominator error: the numerator is exactly 1 throughout.
    v = np.zeros_like(k)
    v[0] = 1
    cfg = Config(storage="fp32", output="fp32", tile=1)
    expected = reference(q, k, v)
    plain = emulate(q, k, v, cfg)
    compensated = emulate(q, k, v, replace(cfg, compensated=True))
    assert np.linalg.norm(compensated - expected) < np.linalg.norm(plain - expected) / 10


@pytest.mark.parametrize("storage", ["bf16", "e4m3", "e5m2"])
@pytest.mark.parametrize("update", ["global", "local"])
def test_sampled_rows_equal_full_rows(storage, update):
    rng = np.random.default_rng(19)
    q, k, v = (rng.normal(size=(21, 8)).astype(np.float32) for _ in range(3))
    rows = np.array([20, 0, 4, 4, 8])
    cfg = Config(storage=storage, scaling="tile", tile=5, query_tile=4, causal=True, update=update)
    np.testing.assert_array_equal(emulate(q, k, v, cfg, rows), emulate(q, k, v, cfg)[rows])
    np.testing.assert_allclose(
        reference(q, k, v, rows=rows, causal=True), reference(q, k, v, causal=True)[rows]
    )


def test_independent_probability_and_output_rounding():
    rng = np.random.default_rng(15)
    q, k, v = (rng.normal(size=(9, 8)).astype(np.float32) for _ in range(3))
    cfg = Config(storage="bf16", probability="e4m3", output="fp32", tile=16)
    qs, _ = quantize(q, "bf16")
    ks, _ = quantize(k, "bf16")
    vs, _ = quantize(v, "bf16")
    scores = matmul(qs, ks.T) * np.float32(8**-0.5)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True)).astype(np.float32)
    pscale = np.float32(1 / 448)
    expected = matmul(cast(weights / pscale, "e4m3"), vs) * pscale
    expected /= weights.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(emulate(q, k, v, cfg), expected, atol=1e-7)
    np.testing.assert_array_equal(
        emulate(q, k, v, replace(cfg, output="bf16")), cast(expected, "bf16")
    )


@pytest.mark.parametrize("storage", ["e4m3", "e5m2"])
def test_fp8_single_tile_matches_independent_dense_rounding(storage):
    rng = np.random.default_rng(37)
    q, k, v = (rng.normal(size=(11, 8)).astype(np.float32) for _ in range(3))
    qs, qscale = quantize(q, storage)
    ks, kscale = quantize(k, storage)
    vs, vscale = quantize(v, storage)
    scores = (qs.astype(np.float64) @ ks.astype(np.float64).T).astype(np.float32)
    scores *= qscale[:, None] * kscale[None, :]
    scores *= np.float32(8**-0.5)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True)).astype(np.float32)
    pscale = np.float32(1 / (448 if storage == "e4m3" else 57344))
    packed = cast(weights / pscale, storage)
    expected = (packed.astype(np.float64) @ vs.astype(np.float64)).astype(np.float32)
    expected *= vscale[0] * pscale
    expected /= weights.sum(axis=1, keepdims=True)
    cfg = Config(storage=storage, output="fp32", tile=16)
    np.testing.assert_allclose(emulate(q, k, v, cfg), expected, atol=3e-6, rtol=3e-6)


def test_metrics_identify_original_worst_row():
    result = metrics(np.array([[1, 2], [2, 4]]), np.ones((2, 2)), [7, 18])
    assert result["max_abs"] == 3
    assert result["worst_row"] == 18
    assert metrics(np.ones((1, 1)), np.zeros((1, 1)))["relative_frobenius"] is None


@pytest.mark.parametrize("kwargs", [{"tile": 0}, {"promote": 3}, {"scale": -1}, {"storage": "bad"}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_invalid_inputs():
    q = np.ones((2, 4))
    with pytest.raises(ValueError):
        emulate(q, q, q, rows=[2])
    with pytest.raises(ValueError):
        reference(q, q[:, :3], q)
    with pytest.raises(ValueError):
        emulate(q * np.nan, q, q)
    with pytest.raises(ValueError):
        hadamard(np.ones((2, 3)), np.ones(3))
