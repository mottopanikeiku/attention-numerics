"""CPU layout/argument/packing contracts only; no FA3 GPU validation.

Fake APIs never calculate attention. Device checks alone are bypassed in the
contract fixture, and common.prepare is injected to avoid pretending CPU
operands are a valid real-kernel execution. No optional kernels import occurs.
"""

import importlib
import json
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

fa3 = importlib.import_module("study.hardware.fa3")


def _inputs(d=64, n=3):
    q = torch.arange(2 * 4 * n * d).reshape(2, 4, n, d).float() / 100
    k = torch.arange(2 * 2 * n * d).reshape(2, 2, n, d).float() / 50 - 2
    v = -k / 3
    return tuple(x.to(torch.bfloat16) for x in (q, k, v))


@pytest.fixture
def fake_contract(monkeypatch):
    calls = {}

    def prepare(q, k, v, variant, *, native_smoothing, sign_seed):
        calls["prepare"] = (q, k, v, variant, native_smoothing, sign_seed)
        # Deliberately distinct from input expansion: the adapter must pack the
        # helper's FP32 operands, not the original BF16 or a re-created rotation.
        prepared = (
            q.float() * 1.25,
            k.float().repeat_interleave(2, dim=1) - 0.125,
            v.float().repeat_interleave(2, dim=1),
        )
        calls["prepared"] = prepared
        return prepared

    def flash_attn_func(q, k, v, **kwargs):
        calls["api"] = ((q, k, v), kwargs)
        output = torch.arange(q.numel()).reshape(q.shape).to(torch.bfloat16)
        calls["output"] = output
        return output

    monkeypatch.setattr(fa3, "_check_device", lambda *args: None)
    monkeypatch.setattr(fa3, "prepare", prepare)
    monkeypatch.setattr(fa3, "_kernel", SimpleNamespace(flash_attn_func=flash_attn_func))
    return calls


@pytest.mark.parametrize("variant", fa3.VARIANTS)
@pytest.mark.parametrize("scale", [None, 0.3])
def test_exact_helper_operands_layout_descales_and_native_output(fake_contract, variant, scale):
    inputs = _inputs()
    diagnostics = {}
    output = fa3.apply_attention(
        *inputs, variant, scale=scale, sign_seed=91, diagnostics=diagnostics
    )
    call = fake_contract["prepare"]
    assert all(call[i] is inputs[i] for i in range(3))
    assert call[3:] == (variant, False, 91)
    packed, kwargs = fake_contract["api"]
    assert kwargs["softmax_scale"] == (64**-0.5 if scale is None else scale)
    assert kwargs["causal"] is True
    assert kwargs["num_splits"] == 1
    assert kwargs["pack_gqa"] is False
    assert kwargs["return_attn_probs"] is False
    for name, codes, operand in zip(
        ("q", "k", "v"), packed, fake_contract["prepared"], strict=True
    ):
        expected = operand.abs().amax(dim=(-2, -1)) / 448
        expected = torch.where(expected == 0, 1.0, expected)
        descale = kwargs[f"{name}_descale"]
        assert descale.shape == (2, 4)
        assert descale.dtype == torch.float32
        assert descale.is_contiguous()
        torch.testing.assert_close(descale, expected, rtol=0, atol=0)
        expected_codes = (operand / expected[:, :, None, None]).clamp(-448, 448)
        expected_codes = expected_codes.to(torch.float8_e4m3fn).transpose(1, 2).contiguous()
        assert codes.shape == (2, 3, 4, 64)
        assert codes.dtype == torch.float8_e4m3fn
        assert codes.is_contiguous()
        torch.testing.assert_close(codes.float(), expected_codes.float(), rtol=0, atol=0)
        assert diagnostics[name]["finite_fraction"] == 1
    assert output.shape == inputs[0].shape
    assert output.dtype == torch.bfloat16
    assert output.is_contiguous()
    torch.testing.assert_close(output, fake_contract["output"].transpose(1, 2), rtol=0, atol=0)


def test_scales_reduce_full_tokens_and_dimensions_independently():
    x = torch.zeros(2, 3, 5, 16, dtype=torch.float32)
    peaks = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    x[:, :, -1, -1] = peaks
    codes, scales = fa3._quantize(x)
    torch.testing.assert_close(scales, peaks / 448, rtol=0, atol=0)
    assert torch.all(codes.float()[:, -1, :, -1] == 448)
    assert torch.all(codes.float()[:, :-1] == 0)


def test_zero_operands_have_unit_descale_and_zero_codes():
    stats = {}
    codes, scales = fa3._quantize(torch.zeros(2, 3, 5, 16), stats)
    assert torch.all(scales == 1)
    assert torch.all(codes.float() == 0)
    assert stats["max_code_fraction"] == 0
    assert stats["clipped_fraction"] == 0
    assert stats["finite_fraction"] == 1


def test_max_code_occupancy_is_not_clipping():
    # Both round to a representable endpoint, but neither exceeds the range.
    stats = {}
    codes, _ = fa3._quantize(torch.tensor([[[[1.0, 0.97, -1.0, 0.0]]]]), stats)
    assert codes.float().flatten().tolist() == [448.0, 448.0, -448.0, 0.0]
    assert stats["max_code_fraction"] == 0.75
    assert stats["clipped_fraction"] == 0
    assert "not saturation" in stats["max_code_definition"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 1e-40])
def test_quantization_rejects_nonfinite_and_underflowing_scales(bad):
    with pytest.raises(ValueError, match="nonfinite|underflows"):
        fa3._quantize(torch.full((1, 1, 2, 16), bad))


def test_cpu_inputs_rejected_without_loading(monkeypatch):
    def unexpected_load():
        pytest.fail("Invalid CPU input must not download a kernel")

    monkeypatch.setattr(fa3, "_load_kernel", unexpected_load)
    with pytest.raises(ValueError, match="CUDA"):
        fa3.apply_attention(*_inputs(), "tile")


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (10, 0)])
def test_non_hopper_devices_rejected_without_gpu_allocation(monkeypatch, capability):
    operand = SimpleNamespace(device=torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    with pytest.raises(ValueError, match="Hopper"):
        fa3._check_device(operand, operand, operand)


def test_mixed_cuda_devices_rejected_without_gpu_allocation():
    q = SimpleNamespace(device=torch.device("cuda:0"))
    k = SimpleNamespace(device=torch.device("cuda:1"))
    with pytest.raises(ValueError, match="same CUDA device"):
        fa3._check_device(q, k, q)


def test_default_seed_and_no_diagnostics_contract(fake_contract):
    fa3.apply_attention(*_inputs(), "rotate")
    assert fake_contract["prepare"][-1] == 1729


@pytest.mark.parametrize("variant", ["bf16", "smooth_kq", "unknown"])
def test_unsupported_variants_are_not_fallbacks(variant):
    with pytest.raises(ValueError, match="variant"):
        fa3.apply_attention(*_inputs(), variant)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), 1e40, 1e-40])
def test_invalid_softmax_scale_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        fa3.apply_attention(*_inputs(), "tile", scale=scale)


@pytest.mark.parametrize("d", [8, 17, 272])
def test_unsupported_head_dimensions_rejected(d):
    with pytest.raises(ValueError, match="head dimension"):
        fa3.apply_attention(*_inputs(d=d), "tile")


def test_non_power_two_rotation_rejected():
    with pytest.raises(ValueError, match="power-of-two"):
        fa3.apply_attention(*_inputs(d=96), "rotate")


def test_wrong_input_dtype_rejected():
    q, k, v = _inputs()
    with pytest.raises(ValueError, match="BF16"):
        fa3.apply_attention(q.float(), k, v, "tile")


@pytest.mark.parametrize("kind", ["rank", "empty", "gqa", "tokens", "kv"])
def test_incompatible_shapes_rejected(kind):
    q, k, v = _inputs()
    if kind == "rank":
        q = q[0]
    elif kind == "empty":
        q = q[:, :0]
    elif kind == "gqa":
        q = q[:, :3]
    elif kind == "tokens":
        k, v = k[:, :, :-1], v[:, :, :-1]
    else:
        v = v[:, :1]
    with pytest.raises(ValueError, match="shape"):
        fa3.apply_attention(q, k, v, "tile")


@pytest.mark.parametrize("wrong", ["float32", "tuple", "shape"])
def test_wrong_native_output_is_rejected(fake_contract, monkeypatch, wrong):
    def bad_output(q, k, v, **kwargs):
        if wrong == "float32":
            return torch.zeros_like(q, dtype=torch.float32)
        if wrong == "tuple":
            return (torch.zeros_like(q, dtype=torch.bfloat16), None)
        return torch.zeros(1, dtype=torch.bfloat16)

    monkeypatch.setattr(fa3, "_kernel", SimpleNamespace(flash_attn_func=bad_output))
    with pytest.raises(RuntimeError, match="output contract|native BF16"):
        fa3.apply_attention(*_inputs(), "tile")


def test_kernel_errors_propagate_without_bf16_fallback(fake_contract, monkeypatch):
    def unsupported(*args, **kwargs):
        raise RuntimeError("FP8 disabled in this build")

    monkeypatch.setattr(fa3, "_kernel", SimpleNamespace(flash_attn_func=unsupported))
    with pytest.raises(RuntimeError, match="FP8 disabled"):
        fa3.apply_attention(*_inputs(), "tile")


def test_lazy_hub_loader_requests_version_one_once(monkeypatch):
    calls = []
    module = SimpleNamespace(flash_attn_func=lambda *args: None)

    def get_kernel(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return module

    monkeypatch.delenv("ATTENTION_FA3_ROOT", raising=False)
    monkeypatch.setattr(fa3, "_kernel", None)
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=get_kernel))
    assert fa3._load_kernel() is module
    assert fa3._load_kernel() is module
    assert calls == [(fa3.REPO_ID, {"version": 1})]


def test_explicit_prebuilt_import_preserves_relative_imports_and_build_ids(tmp_path, monkeypatch):
    root = tmp_path / fa3.PREBUILT_VARIANT
    root.mkdir()
    name = "_fa3_contract_only_no_gpu"
    metadata = {"name": "flash-attn3", "version": 1, "id": name}
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "__init__.py").write_text("from .api import flash_attn_func\n")
    (root / "api.py").write_text("def flash_attn_func(*args, **kwargs):\n    return None\n")
    monkeypatch.setattr(torch, "__version__", "2.9.1+cu128")
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(fa3.platform, "system", lambda: "Linux")
    monkeypatch.setattr(fa3.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(fa3, "_kernel", None)
    monkeypatch.setenv("ATTENTION_FA3_ROOT", str(root))
    # Preserve cleanup even though importlib populates sys.modules itself.
    monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, name + ".api", None)
    monkeypatch.delitem(sys.modules, name + ".api")
    loaded = fa3._load_kernel()
    assert callable(loaded.flash_attn_func)
    info = fa3.describe()
    assert info["loaded"]["module"] == name
    assert info["loaded"]["metadata"] == metadata
    assert info["loaded"]["api_module"] == name + ".api"
    json.dumps(info, allow_nan=False)


def test_prebuilt_rejects_abi_mismatch_before_import(tmp_path, monkeypatch):
    root = tmp_path / fa3.PREBUILT_VARIANT
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps({"name": "flash-attn3", "version": 1, "id": "unused"})
    )
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    with pytest.raises(RuntimeError, match="CUDA12.8"):
        fa3._import_prebuilt(root)


def test_describe_does_not_load_or_invent_build_observations(monkeypatch):
    monkeypatch.setattr(fa3, "_kernel", None)
    info = fa3.describe()
    assert info["loaded"] is None
    assert info["quantization"]["emulator_q32_kv128"] is False
    assert info["accumulation"]["internal_probability_diagnostics"] == "not exposed"
    assert info["settings"]["causal"] is True
    json.dumps(info, allow_nan=False)
