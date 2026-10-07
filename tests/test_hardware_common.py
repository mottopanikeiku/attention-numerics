"""CPU math/API tests, not a claim that real GPU kernels ran in CI."""

import math

import numpy as np
import pytest
import torch

from attention import reference
from study.attention import _center_tokens, _rotate
from study.hardware.common import error_metrics, exact_attention, prepare
from study.hardware.inputs import select_heads
from study.hardware.worker import distribution_metrics, install_attention, remove_attention


def _native_test_dispatch(module, q, k, v, mask, scaling):
    return q.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS = {"sdpa": _native_test_dispatch}


def operands():
    generator = torch.Generator().manual_seed(31)
    q = torch.randn(1, 4, 9, 8, generator=generator).bfloat16()
    k = torch.randn(1, 2, 9, 8, generator=generator).bfloat16()
    v = torch.randn(1, 2, 9, 8, generator=generator).bfloat16()
    return q, k, v


@pytest.mark.parametrize("chunk", [1, 4, 20])
def test_reference_matches_independent_numpy_with_gqa(chunk):
    q, k, v = operands()
    actual = exact_attention(q, k, v, scale=0.3, query_chunk=chunk)
    for head in range(4):
        expected = reference(
            q[0, head].float().numpy(),
            k[0, head // 2].float().numpy(),
            v[0, head // 2].float().numpy(),
            scale=0.3,
            causal=True,
        )
        np.testing.assert_allclose(actual[0, head].numpy(), expected, rtol=2e-14, atol=2e-14)


def test_reference_is_causal_and_uses_original_values():
    q, k, v = operands()
    before = exact_attention(q, k, v)
    k[:, :, 5:] += 100
    v[:, :, 5:] -= 100
    after = exact_attention(q, k, v)
    torch.testing.assert_close(before[:, :, :5], after[:, :, :5], rtol=0, atol=0)
    torch.testing.assert_close(
        before[:, :, 0], operands()[2][:, [0, 0, 1, 1], 0].double(), rtol=0, atol=0
    )


def test_prepare_is_exact_emulator_rotation_and_never_changes_inputs():
    q, k, v = operands()
    original = [value.clone() for value in (q, k, v)]
    signs = torch.from_numpy(np.random.default_rng(1729).choice([-1, 1], size=8).astype(np.float32))
    expected_k, _ = _center_tokens(k[:, [0, 0, 1, 1]].float())
    actual = prepare(q, k, v, "rotate_smooth_k")
    torch.testing.assert_close(actual[0], _rotate(q.float(), signs), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], _rotate(expected_k, signs), rtol=0, atol=0)
    torch.testing.assert_close(actual[2], v[:, [0, 0, 1, 1]].float(), rtol=0, atol=0)
    for value, saved in zip((q, k, v), original, strict=True):
        torch.testing.assert_close(value, saved, rtol=0, atol=0)
    native = prepare(q, k, v, "rotate_smooth_k", native_smoothing=True)
    torch.testing.assert_close(
        native[1], _rotate(k[:, [0, 0, 1, 1]].float(), signs), rtol=0, atol=0
    )


def test_errors_are_per_head_and_reject_nonfinite_or_zero_reference():
    q, k, v = operands()
    expected = exact_attention(q, k, v)
    measured = error_metrics(expected * 1.1, expected)
    np.testing.assert_allclose(measured["relative_fro"], [[0.1] * 4], rtol=1e-14)
    with pytest.raises(ValueError, match="zero reference"):
        error_metrics(torch.zeros_like(expected), torch.zeros_like(expected))
    with pytest.raises(FloatingPointError):
        error_metrics(expected * math.inf, expected)


def synthetic_rows():
    rows = []
    for model in ("qwen05", "smol17", "olmo1"):
        for head in range(70):
            for text in ("alice", "moby", "pride"):
                rows.append(
                    {
                        "model": model,
                        "layer": 0,
                        "head": head,
                        "text": text,
                        "rotate_predicted_error": head / 100,
                        "tile_predicted_error": 0.2,
                        "rotate_relative_fro": 1234.0,
                        "tile_relative_fro": 9876.0,
                    }
                )
    return rows


def test_selection_uses_locked_predictions_not_measured_outcomes():
    rows = synthetic_rows()
    first = select_heads(rows)
    for row in rows:
        row["rotate_relative_fro"], row["tile_relative_fro"] = 0, 1e20
    assert first == select_heads(list(reversed(rows)))
    assert len(first["qwen05"]) == 70
    for model in ("smol17", "olmo1"):
        assert sum("top" in point["selected_by"] for point in first[model]) == 32
        assert sum("random" in point["selected_by"] for point in first[model]) == 32
        assert {point["head"] for point in first[model] if "top" in point["selected_by"]} == set(
            range(38, 70)
        )
    extended = rows + [dict(row, model="unrelated") for row in rows if row["model"] == "smol17"]
    assert select_heads(extended)["smol17"] == first["smol17"]


def test_selection_rejects_missing_text():
    with pytest.raises(ValueError, match="three texts"):
        select_heads(synthetic_rows()[1:])


def test_downstream_distribution_is_fp64_native_projection():
    baseline = torch.tensor([[[0, 2, -1], [1, 0, -2]]], dtype=torch.bfloat16)
    actual = baseline.clone()
    actual[0, 0, 1] += 1
    labels = torch.tensor([[1, 2]])
    measured = distribution_metrics(actual, baseline, labels, row_chunk=1)
    alt_log = torch.log_softmax(actual.double(), -1)
    base_log = torch.log_softmax(baseline.double(), -1)
    assert measured["tokens"] == 2
    assert measured["next_token_ce"] == pytest.approx(-alt_log[0, [0, 1], [1, 2]].mean().item())
    assert measured["kl_from_bf16"] == pytest.approx(
        (base_log.exp() * (base_log - alt_log)).sum(-1).mean().item()
    )
    with pytest.raises(ValueError, match="BF16"):
        distribution_metrics(actual.float(), baseline, labels)


def test_native_attention_interception_is_per_instance_and_restored():
    class Attention:
        layer_idx = 0

        def forward(self, q, k, v):
            return ALL_ATTENTION_FUNCTIONS["sdpa"](self, q, k, v, None, 0.25)

    class Layer:
        self_attn = Attention()

    class Decoder:
        layers = [Layer()]

    class Model:
        model = Decoder()

    class Adapter:
        @staticmethod
        def apply_attention(q, k, v, variant, scale=None):
            assert variant == "rotate" and scale == 0.25
            return q

    q, k, v = operands()
    model, other, calls = Model(), Attention(), []
    native = type(other).forward
    original = install_attention(model, Adapter, "rotate", calls)
    output, weights = model.model.layers[0].self_attn.forward(q, k, v)
    assert output.shape == (1, 9, 4, 8) and weights is None and calls == [0]
    assert "forward" not in other.__dict__ and type(other).forward is native
    remove_attention(original)
    assert "forward" not in model.model.layers[0].self_attn.__dict__
