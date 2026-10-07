"""Ada SageAttention INT8-QK / E4M3-PV CUDA adapter, never a CPU surrogate.

The shared FP32 operand transform is rounded back to BF16 for Sage's public
API. Native K smoothing consequently happens after rotation and BF16 rounding,
not before rotation with the emulator's FP64 mean/subtraction. See describe().
"""

import inspect
import math
from importlib import import_module, metadata

import torch

from . import common

VARIANTS = ("tile", "rotate", "smooth_k", "rotate_smooth_k")
SOURCE_COMMIT = "eb615cf6cf4d221338033340ee2de1c37fbdba4a"
_SOURCE = f"https://github.com/thu-ml/SageAttention/blob/{SOURCE_COMMIT}"
_API = "sageattn_qk_int8_pv_fp8_cuda"


def describe():
    """JSON-safe settings; the source pin is intended, not a runtime attestation."""
    try:
        installed_version = metadata.version("sageattention")
    except metadata.PackageNotFoundError:
        installed_version = None
    return {
        "package": "sageattention",
        "expected_version": "2.2.0",
        "installed_version": installed_version,
        "expected_source_commit": SOURCE_COMMIT,
        "api": f"sageattention.core.{_API}",
        "supported_compute_capabilities": [[8, 9]],
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "settings": {
            "tensor_layout": "HND",
            "is_causal": True,
            "qk_quant_gran": "per_warp",
            "pv_accum_dtype": "fp32+fp32",
            "smooth_k": "true only for smooth_k and rotate_smooth_k",
            "smooth_v": False,
            "return_lse": False,
            "sm_scale": "explicit argument or original head_dim**-0.5",
        },
        "quantization": {
            "q": "symmetric INT8, one scale per 32-token warp across head_dim",
            "k": "symmetric INT8, one scale per 64-token block across head_dim",
            "qk_scale": "max(abs(x), 1e-7) / 127; nearest integer codes",
            "qk_dot_accumulator": "INT32 tensor-core dot product, then FP32 dequantization",
            "v": "float8_e4m3fn, one FP32 scale per batch/head/channel over sequence",
            "v_scale_max": 448.0,
            "p": "E4M3 online-softmax numerators; native exponent offset, not uniform E4",
            "pv_accumulation": (
                "Ada FP8 MMA FP32 accumulator (upstream describes 22 valid bits), "
                "fresh instruction buffer per 64-key tile added to FP32 running output"
            ),
            "softmax_denominator": "FP32 CUDA-core sum before E4M3 numerator rounding",
        },
        "transforms": {
            "prepare": "study.hardware.common.prepare(native_smoothing=True)",
            "gqa": "consecutive KV groups expanded to query-head count before kernel",
            "rotation": "shared exact study.attention._rotate FWHT with PCG64 signs",
            "sign_seed_default": 1729,
            "api_rounding": (
                "prepare returns contiguous FP32; Q/K/V converted to BF16 before API; "
                "rotated Q/K incur BF16 transform-rounding absent from the emulator"
            ),
            "native_smoothing": (
                "no external K centering; core computes k.mean over tokens with BF16 "
                "result after rotation and API rounding; fused CUDA quantizer subtracts "
                "that BF16 mean in FP32 before INT8 quantization. Emulator instead "
                "centers exact BF16 expansion in FP64 before FP32 rotation"
            ),
            "v": "no shared transform or native V smoothing; API then quantizes per channel",
            "head_dim": "1..128; upstream pads to 64 or 128; rotation needs power of two",
        },
        "diagnostics": "public API does not expose quantized tensors or max-code counts",
        "sources": [
            f"{_SOURCE}/sageattention/core.py",
            f"{_SOURCE}/sageattention/quant.py",
            f"{_SOURCE}/csrc/fused/fused.cu",
            f"{_SOURCE}/csrc/qattn/attn_utils.cuh",
            f"{_SOURCE}/setup.py",
        ],
    }


def _check_device(q, k, v):
    if any(x.device.type != "cuda" for x in (q, k, v)):
        raise ValueError("SageAttention requires CUDA BF16 Q/K/V; there is no CPU fallback")
    if not q.device == k.device == v.device:
        raise ValueError("Q/K/V must be on the same CUDA device")
    capability = torch.cuda.get_device_capability(q.device)
    if capability != (8, 9):
        raise ValueError(f"Sage FP8 adapter requires sm89 (L4/L40S), got {capability}")


def _load_kernel():
    """Import only when called, and reject missing/incompatible compiled APIs."""
    try:
        core = import_module("sageattention.core")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Install compiled SageAttention 2.2.0 for this Torch/CUDA ABI and sm89; "
            "the PyPI SageAttention 1.x Triton package is not this kernel"
        ) from exc
    if not getattr(core, "SM89_ENABLED", False):
        raise RuntimeError("SageAttention sm89 CUDA extension is unavailable; no fallback")
    kernel = getattr(core, _API, None)
    required = {
        "tensor_layout",
        "is_causal",
        "qk_quant_gran",
        "sm_scale",
        "pv_accum_dtype",
        "smooth_k",
        "smooth_v",
        "return_lse",
    }
    if not callable(kernel) or not required.issubset(inspect.signature(kernel).parameters):
        raise RuntimeError(f"SageAttention {_API} lacks the required explicit API settings")
    return kernel


@torch.no_grad()
def apply_attention(q, k, v, variant, scale=None, sign_seed=1729, diagnostics=None):
    """Causal CUDA BF16 [B,Hq,N,D] attention using the actual sm89 FP8 kernel."""
    if variant not in VARIANTS:
        raise ValueError(f"unsupported Sage attention variant: {variant}")
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Q/K/V must have shape [B,H,N,D]")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("Q/K/V must be BF16 tensors")
    b, h, n, d = q.shape
    hkv = k.shape[1]
    if (
        min(b, h, n, d, hkv) <= 0
        or k.shape != v.shape
        or k.shape[0] != b
        or k.shape[2:] != (n, d)
        or h % hkv
    ):
        raise ValueError("incompatible causal Q/K/V or GQA shapes")
    if d > 128:
        raise ValueError("SageAttention FP8 CUDA supports head_dim at most 128")
    if variant in ("rotate", "rotate_smooth_k") and d & (d - 1):
        raise ValueError("Hadamard rotation requires a power-of-two head dimension")
    scale = d**-0.5 if scale is None else scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax scale must be finite and positive")
    _check_device(q, k, v)
    kernel = _load_kernel()
    transformed = common.prepare(q, k, v, variant, native_smoothing=True, sign_seed=sign_seed)
    q_api, k_api, v_api = (x.to(torch.bfloat16).contiguous() for x in transformed)
    result = kernel(
        q_api,
        k_api,
        v_api,
        tensor_layout="HND",
        is_causal=True,
        qk_quant_gran="per_warp",
        sm_scale=float(scale),
        pv_accum_dtype="fp32+fp32",
        smooth_k=variant in ("smooth_k", "rotate_smooth_k"),
        smooth_v=False,
        return_lse=False,
    )
    if (
        not isinstance(result, torch.Tensor)
        or result.shape != q.shape
        or result.dtype != torch.bfloat16
        or result.device != q.device
    ):
        raise RuntimeError("SageAttention returned an incompatible BF16 [B,Hq,N,D] output")
    if diagnostics is not None:
        diagnostics["quantization_stats"] = {
            "available": False,
            "reason": "Sage public API does not expose its INT8/E4M3 packed operands",
        }
    return result
