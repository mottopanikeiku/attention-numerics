"""CPU-only API/layout checks with an injected kernel; NOT GPU validation.

Hardware checks are bypassed only in the fake-kernel fixture. No test computes
SageAttention numerics or claims that a CUDA extension was loaded or executed.
"""

import importlib
import json
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
sage = importlib.import_module("study.hardware.sage")


def _inputs(d=64, n=67, h=4, hkv=2):
    # Distinct batch/head/sequence/channel values expose layout or GQA errors.
    q = torch.arange(2 * h * n * d).reshape(2, h, n, d).remainder(23).to(torch.bfloat16)
    k = torch.arange(2 * hkv * n * d).reshape(2, hkv, n, d).remainder(29)
    v = torch.arange(2 * hkv * n * d).reshape(2, hkv, n, d).remainder(31)
    return q, (k + 10).to(torch.bfloat16), (v - 15).to(torch.bfloat16)


@pytest.fixture
def fake_api(monkeypatch):
    calls = []

    def kernel(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return v.clone()

    monkeypatch.setattr(sage, "_check_device", lambda *args: None)
    monkeypatch.setattr(sage, "_load_kernel", lambda: kernel)
    return calls


@pytest.mark.parametrize("variant", sage.VARIANTS)
@pytest.mark.parametrize("d", [64, 128])
def test_explicit_api_settings_shared_preparation_and_gqa(monkeypatch, fake_api, variant, d):
    q, k, v = _inputs(d=d)
    original = [x.clone() for x in (q, k, v)]
    prepare_calls = []
    prepare = sage.common.prepare

    def spy(q, k, v, variant, *, native_smoothing, sign_seed):
        prepare_calls.append((variant, native_smoothing, sign_seed))
        return prepare(
            q, k, v, variant, native_smoothing=native_smoothing, sign_seed=sign_seed
        )

    monkeypatch.setattr(sage.common, "prepare", spy)
    diagnostics = {"caller_field": 7}
    actual = sage.apply_attention(
        q, k, v, variant, scale=0.375, sign_seed=811, diagnostics=diagnostics
    )
    assert prepare_calls == [(variant, True, 811)]
    q_api, k_api, v_api, options = fake_api[0]
    expected = prepare(q, k, v, variant, native_smoothing=True, sign_seed=811)
    for operand, fp32 in zip((q_api, k_api, v_api), expected, strict=True):
        assert operand.shape == q.shape
        assert operand.dtype == torch.bfloat16
        assert operand.is_contiguous()
        assert torch.equal(operand, fp32.to(torch.bfloat16))
    expanded_v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    assert torch.equal(v_api, expanded_v)
    assert options == {
        "tensor_layout": "HND",
        "is_causal": True,
        "qk_quant_gran": "per_warp",
        "sm_scale": 0.375,
        "pv_accum_dtype": "fp32+fp32",
        "smooth_k": variant in ("smooth_k", "rotate_smooth_k"),
        "smooth_v": False,
        "return_lse": False,
    }
    assert actual.dtype == torch.bfloat16
    assert actual.shape == q.shape
    assert torch.equal(actual, expanded_v)
    assert diagnostics["caller_field"] == 7
    assert diagnostics["quantization_stats"]["available"] is False
    for before, after in zip(original, (q, k, v), strict=True):
        assert torch.equal(before, after)


@pytest.mark.parametrize("variant", ["tile", "smooth_k"])
def test_native_smoothing_is_not_applied_externally(fake_api, variant):
    q, k, v = _inputs(d=96)
    sage.apply_attention(q, k, v, variant)
    _, k_api, _, options = fake_api[0]
    # High-mean K must reach Sage uncentered; native smooth_k owns subtraction.
    assert torch.equal(k_api, k.repeat_interleave(2, dim=1))
    assert options["sm_scale"] == 96**-0.5
    assert options["smooth_k"] is (variant == "smooth_k")


def test_api_narrows_fp32_transforms_to_bf16(monkeypatch, fake_api):
    q, k, v = _inputs()
    prepared = tuple(
        torch.full(q.shape, value, dtype=torch.float32).transpose(1, 2).contiguous()
        .transpose(1, 2)
        for value in (1.003, 2.006, 3.009)
    )
    monkeypatch.setattr(sage.common, "prepare", lambda *args, **kwargs: prepared)
    sage.apply_attention(q, k, v, "rotate")
    for api_operand, fp32 in zip(fake_api[0][:3], prepared, strict=True):
        assert api_operand.is_contiguous()
        assert api_operand.dtype == torch.bfloat16
        assert torch.equal(api_operand, fp32.to(torch.bfloat16))
        assert not torch.equal(api_operand.float(), fp32)


@pytest.mark.parametrize("variant", ["bf16", "smooth_kq", "unknown"])
def test_rejects_unsupported_variants(fake_api, variant):
    with pytest.raises(ValueError, match="variant"):
        sage.apply_attention(*_inputs(), variant)
    assert fake_api == []


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("rank", "shape"),
        ("dtype", "BF16"),
        ("batch", "shapes"),
        ("length", "shapes"),
        ("kv", "shapes"),
        ("gqa", "shapes"),
        ("empty", "shapes"),
        ("head_dim", "head_dim"),
    ],
)
def test_rejects_incompatible_shapes_before_call(fake_api, case, message):
    q, k, v = _inputs()
    if case == "rank":
        q = q[0]
    elif case == "dtype":
        k = k.float()
    elif case == "batch":
        q = q[:1]
    elif case == "length":
        q = q[:, :, :-1]
    elif case == "kv":
        v = v[:, :1]
    elif case == "gqa":
        q = q[:, :3]
    elif case == "empty":
        k = k[:, :0]
        v = v[:, :0]
    elif case == "head_dim":
        q, k, v = _inputs(d=256)
    with pytest.raises(ValueError, match=message):
        sage.apply_attention(q, k, v, "tile")
    assert fake_api == []


@pytest.mark.parametrize("scale", [0.0, -0.1, float("inf"), float("nan")])
def test_rejects_invalid_scale(fake_api, scale):
    with pytest.raises(ValueError, match="scale"):
        sage.apply_attention(*_inputs(), "tile", scale=scale)
    assert fake_api == []


@pytest.mark.parametrize("variant", ["rotate", "rotate_smooth_k"])
def test_rejects_non_power_of_two_rotation(fake_api, variant):
    with pytest.raises(ValueError, match="power-of-two"):
        sage.apply_attention(*_inputs(d=96), variant)
    assert fake_api == []


def test_rejects_cpu_without_importing_optional_sage(monkeypatch):
    def no_import():
        pytest.fail("unsupported devices must be rejected before importing Sage")

    monkeypatch.setattr(sage, "_load_kernel", no_import)
    with pytest.raises(ValueError, match="CUDA.*no CPU fallback"):
        sage.apply_attention(*_inputs(), "tile")


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (9, 0), (12, 0)])
def test_rejects_non_ada_gpu(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    q = SimpleNamespace(device=torch.device("cuda:0"))
    with pytest.raises(ValueError, match="requires sm89"):
        sage._check_device(q, q, q)


def test_accepts_sm89_and_rejects_mixed_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    q = SimpleNamespace(device=torch.device("cuda:0"))
    k = SimpleNamespace(device=torch.device("cuda:1"))
    sage._check_device(q, q, q)
    with pytest.raises(ValueError, match="same CUDA device"):
        sage._check_device(q, k, q)


@pytest.mark.parametrize("case", ["import", "extension", "symbol", "signature"])
def test_missing_or_incompatible_api_has_no_fallback(monkeypatch, case):
    def incompatible_api(q, k, v, **kwargs):
        pytest.fail("incompatible API must never be called")

    def load(name):
        assert name == "sageattention.core"
        if case == "import":
            raise ImportError("package not installed")
        core = SimpleNamespace(SM89_ENABLED=case != "extension")
        if case == "signature":
            core.sageattn_qk_int8_pv_fp8_cuda = incompatible_api
        return core

    monkeypatch.setattr(sage, "import_module", load)
    with pytest.raises(RuntimeError, match="SageAttention"):
        sage._load_kernel()


@pytest.mark.parametrize("bad_result", ["shape", "dtype", "tuple"])
def test_rejects_incompatible_kernel_output(monkeypatch, fake_api, bad_result):
    def kernel(q, k, v, **kwargs):
        if bad_result == "shape":
            return q[:, :, :-1]
        if bad_result == "dtype":
            return q.float()
        return q, torch.empty(0)

    monkeypatch.setattr(sage, "_load_kernel", lambda: kernel)
    with pytest.raises(RuntimeError, match="incompatible BF16"):
        sage.apply_attention(*_inputs(), "tile")


def test_description_is_json_safe_without_importing_sage(monkeypatch):
    def no_import(name):
        pytest.fail("describe must not import SageAttention")

    monkeypatch.setattr(sage, "import_module", no_import)
    info = sage.describe()
    json.dumps(info, allow_nan=False)
    assert info["settings"]["qk_quant_gran"] == "per_warp"
    assert info["settings"]["pv_accum_dtype"] == "fp32+fp32"
    assert info["settings"]["smooth_v"] is False
    assert info["quantization"]["v_scale_max"] == 448.0
    assert "BF16" in info["transforms"]["api_rounding"]
    assert "FP64" in info["transforms"]["native_smoothing"]
    assert info["expected_source_commit"] == sage.SOURCE_COMMIT


def test_loads_only_explicit_fp8_cuda_api(monkeypatch):
    def kernel(
        q, k, v, tensor_layout, is_causal, qk_quant_gran, sm_scale, pv_accum_dtype,
        smooth_k, smooth_v, return_lse,
    ):
        pytest.fail("this test inspects the API only, never executes a kernel")

    def wrong_api(*args, **kwargs):
        pytest.fail("automatic dispatch / FP16-PV alternatives must not be selected")

    core = SimpleNamespace(
        SM89_ENABLED=True,
        sageattn_qk_int8_pv_fp8_cuda=kernel,
        sageattn=wrong_api,
        sageattn_qk_int8_pv_fp16_triton=wrong_api,
    )
    monkeypatch.setattr(sage, "import_module", lambda name: core)
    assert sage._load_kernel() is kernel
