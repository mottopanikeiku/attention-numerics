import numpy as np
import pytest

from attention import Config, emulate
from diagnosis import emulated_logits, mean_energy, reference_logits, softmax64


def test_mean_energy_has_known_fraction():
    values = mean_energy(np.array([[1.0, 2.0], [3.0, 2.0]]))
    assert values["total_energy"] == 18
    assert values["broadcast_mean_energy"] == 16
    assert values["mean_energy_fraction"] == 8 / 9


def test_softmax64_known_distribution_and_constant_shift():
    logits = np.array([[1.0, 2.0, -np.inf], [3.0, -np.inf, -np.inf]])
    expected = [[1 / (1 + np.e), np.e / (1 + np.e), 0], [1, 0, 0]]
    np.testing.assert_allclose(softmax64(logits), expected, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(softmax64(logits + 10000), softmax64(logits))


@pytest.mark.parametrize("smooth_k", [False, True])
@pytest.mark.parametrize("rotate", [False, True])
def test_diagnostic_logits_reconstruct_fp32_attention(smooth_k, rotate):
    rng = np.random.default_rng(67)
    q, k, v = (rng.normal(size=(19, 8)).astype(np.float32) for _ in range(3))
    k[:, 0] += 16
    cfg = Config(
        storage="fp32",
        output="fp32",
        smooth_k=smooth_k,
        rotate=rotate,
        causal=True,
        query_tile=5,
        tile=7,
    )
    probabilities = softmax64(emulated_logits(q, k, cfg))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(np.triu(probabilities, k=1), 0)
    np.testing.assert_allclose(
        probabilities @ v.astype(np.float64), emulate(q, k, v, cfg), atol=3e-6, rtol=1e-5
    )
    np.testing.assert_allclose(
        probabilities, softmax64(reference_logits(q, k)), atol=3e-7, rtol=1e-5
    )
