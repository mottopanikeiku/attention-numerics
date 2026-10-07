"""Real FA3 E4M3 forward, using the published version-1 Hub API.

The API takes [B,N,H,D] operands and FP32 descales [B,Hkv], including
q_descale (not [B,Hq] for unexpanded GQA). common.prepare expands GQA,
centers K in FP64 before FP32 narrowing, then optionally rotates Q/K.
Each operand here has one independent max scale per expanded batch/head,
over all tokens and dimensions. These are NOT the v2 emulator's Q32/KV128
scales, and the name ``tile`` does not select those emulator tiles.

Hopper QK and PV use E4M3 tensor-core GMMA with float accumulator registers.
Online softmax uses FP32; its unnormalized exponentials are scaled by 2**8
and cast to E4M3 before PV, then normalized with the FP32 (pre-cast) sum.
PV accumulates across native tiles with online FP32 rescaling; it does not
perform the emulator's software 32-product FP32 recurrence. Float registers
do not guarantee IEEE FP32 rounding inside FP8 tensor-core instructions.
V descale is applied at final normalization, and the kernel writes BF16.
The API exposes neither internal probability codes nor accumulator traces.

Sources: https://huggingface.co/kernels/kernels-community/flash-attn3
https://github.com/huggingface/kernels-community/tree/8a730d96c37560ccf1d3e09f7bbccdc886818f33/flash-attn3

ATTENTION_FA3_ROOT selects an explicitly acquired prebuilt module directory.
It avoids installing latest kernels (hub>=1.10) beside Transformers4.57.6
(hub<1). Without it, the optional kernels loader is imported lazily and uses
get_kernel("kernels-community/flash-attn3", version=1). Neither path falls
back to BF16 attention or emulation. This adapter is inference-only.
"""

import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path

import torch

from .common import prepare

REPO_ID = "kernels-community/flash-attn3"
VERSION = 1
PINNED_REVISION = "7cb368cf8278b583132eb72cbf312d54586df2e2"
PREBUILT_VARIANT = "torch-stable-abi29-cu128-x86_64-linux"
VARIANTS = ("tile", "rotate", "smooth_k", "rotate_smooth_k")
_FP8_MAX = 448.0
_kernel = None


def _import_prebuilt(root):
    """Load the Hub package exactly as a package, preserving relative imports."""
    root = Path(root).resolve()
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("name") != "flash-attn3" or metadata.get("version") != VERSION:
        raise RuntimeError("ATTENTION_FA3_ROOT must contain a flash-attn3 version-1 build")
    if root.name != PREBUILT_VARIANT:
        raise RuntimeError(f"ATTENTION_FA3_ROOT must select {PREBUILT_VARIANT}")
    torch_version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if (
        torch_version < (2, 9)
        or torch.version.cuda != "12.8"
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
    ):
        raise RuntimeError("Pinned FA3 artifact requires Linux x86_64, Torch>=2.9, CUDA12.8")
    name = metadata["id"]
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != root / "__init__.py":
            raise RuntimeError("FA3 build identifier is already loaded from a different directory")
        return module
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import FA3 prebuilt module at {root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_kernel():
    global _kernel
    if _kernel is None:
        root = os.environ.get("ATTENTION_FA3_ROOT")
        if root:
            module = _import_prebuilt(root)
        else:
            from kernels import get_kernel

            module = get_kernel(REPO_ID, version=VERSION)
        if not callable(getattr(module, "flash_attn_func", None)):
            raise RuntimeError("Loaded FA3 build does not export flash_attn_func")
        _kernel = module
    return _kernel


def _check_device(q, k, v):
    if any(x.device.type != "cuda" for x in (q, k, v)):
        raise ValueError("FA3 requires CUDA BF16 Q/K/V; CPU emulation is not supported")
    if k.device != q.device or v.device != q.device:
        raise ValueError("Q/K/V must be on the same CUDA device")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("FA3 E4M3 requires Hopper sm90 (H100/H800); no BF16 fallback")


def _validate_inputs(q, k, v, variant, scale):
    if variant not in VARIANTS:
        raise ValueError(f"unsupported FA3 variant: {variant}")
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Q/K/V must have shape [B,H,N,D]")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("Q/K/V must be BF16 tensors")
    b, h, n, d = q.shape
    hk = k.shape[1]
    if (
        min(b, h, n, d, hk) == 0
        or k.shape != v.shape
        or k.shape[0] != b
        or k.shape[2:] != (n, d)
        or h % hk
    ):
        raise ValueError("incompatible Q/K/V or GQA shapes")
    if d > 256 or d % 16:
        raise ValueError("FA3 E4M3 requires head dimension divisible by16 and at most256")
    if variant in ("rotate", "rotate_smooth_k") and d & (d - 1):
        raise ValueError("Hadamard rotation requires a power-of-two head dimension")
    scale = d**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0 or scale > torch.finfo(torch.float32).max:
        raise ValueError("softmax scale must be finite, positive and representable in FP32")
    if scale < torch.finfo(torch.float32).tiny:
        raise ValueError("softmax scale must not underflow CUDA FP32")
    _check_device(q, k, v)
    return scale


def _quantize(x, diagnostics=None):
    """Pack FP32 [B,H,N,D]; return E4M3 [B,N,H,D] and FP32 [B,H]."""
    amax = x.abs().amax(dim=(-2, -1), keepdim=True)
    descale = torch.where(amax == 0, 1.0, amax / _FP8_MAX)
    if not bool((torch.isfinite(descale) & (descale >= torch.finfo(torch.float32).tiny)).all()):
        raise ValueError("FP8 operand is nonfinite or its scale underflows CUDA FP32")
    normalized = x / descale
    # Clamp only roundoff beyond the max-scaled endpoint. Measure this separately
    # from codes at +/-448: max-code occupancy is not evidence of clipping.
    packed = normalized.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    if diagnostics is not None:
        unpacked = packed.float()
        diagnostics.update(
            elements=packed.numel(),
            finite_fraction=float(torch.isfinite(unpacked).float().mean().item()),
            max_code_fraction=float((unpacked.abs() == _FP8_MAX).float().mean().item()),
            clipped_fraction=float((normalized.abs() > _FP8_MAX).float().mean().item()),
            max_code_definition="abs(E4M3 code)==448, not saturation/clipping",
            clipped_definition="abs(FP32 operand/descale)>448 before explicit clamp",
            descale_min=float(descale.min().item()),
            descale_max=float(descale.max().item()),
        )
    return packed.transpose(1, 2).contiguous(), descale[:, :, 0, 0].contiguous()


@torch.no_grad()
def apply_attention(q, k, v, variant, scale=None, sign_seed=1729, diagnostics=None):
    """Causal real FP8 attention: BF16 [B,Hq,N,D] in and out, expanded GQA."""
    scale = _validate_inputs(q, k, v, variant, scale)
    module = _load_kernel()
    operands = prepare(q, k, v, variant, native_smoothing=False, sign_seed=sign_seed)
    packed, descales = [], []
    for name, operand in zip(("q", "k", "v"), operands):
        stats = {} if diagnostics is not None else None
        codes, descale = _quantize(operand, stats)
        packed.append(codes)
        descales.append(descale)
        if diagnostics is not None:
            diagnostics[name] = stats
    out = module.flash_attn_func(
        *packed,
        q_descale=descales[0],
        k_descale=descales[1],
        v_descale=descales[2],
        softmax_scale=scale,
        causal=True,
        num_splits=1,
        pack_gqa=False,
        return_attn_probs=False,
    )
    expected = (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
    if not isinstance(out, torch.Tensor) or out.shape != expected:
        raise RuntimeError("FA3 flash_attn_func returned an unexpected output contract")
    if out.dtype != torch.bfloat16 or out.device != q.device:
        raise RuntimeError("FA3 E4M3 must return native BF16 on the input CUDA device")
    return out.transpose(1, 2).contiguous()


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def describe():
    """JSON-safe settings and observed loaded build IDs; never triggers a load."""
    loaded = None
    if _kernel is not None:
        file = getattr(_kernel, "__file__", None)
        loaded = {
            "module": getattr(_kernel, "__name__", None),
            "module_file": file,
            "api_module": getattr(_kernel.flash_attn_func, "__module__", None),
        }
        if file is not None:
            root = Path(file).parent
            metadata_file = root / "metadata.json"
            if metadata_file.is_file():
                loaded["metadata"] = json.loads(metadata_file.read_text())
            loaded["extension_files"] = sorted(p.name for p in root.glob("*.so"))
            # The parent records actual artifact hashes; metadata digest is the
            # publisher's declared digest, not a new runtime hash measurement.
            loaded["metadata_digest_status"] = "publisher-declared; not rehashed at runtime"
    return {
        "adapter": "fa3",
        "repo_id": REPO_ID,
        "requested_version": VERSION,
        "recipe_revision": PINNED_REVISION,
        "recipe_build_variant": PREBUILT_VARIANT,
        "packages": {name: _package_version(name) for name in ("torch", "kernels", "huggingface-hub")},
        "loaded": loaded,
        "api": "flash_attn_func",
        "input_dtype": "bfloat16",
        "kernel_dtype": "float8_e4m3fn",
        "output_dtype": "bfloat16",
        "input_layout": "B,H,N,D",
        "kernel_layout": "B,N,H,D",
        "quantization": {
            "max_code": _FP8_MAX,
            "scales": "independent Q/K/V amax/448 per batch and expanded head over full N,D",
            "zero_scale": 1.0,
            "descale_shape": "B,Hq after GQA expansion (API requires B,Hkv)",
            "rounding": "torch FP32-to-E4M3 cast after explicit [-448,448] clamp",
            "emulator_q32_kv128": False,
        },
        "accumulation": {
            "qk": "E4M3 GMMA with float accumulator registers",
            "softmax": "FP32 online recurrence, exp2 with max offset8",
            "pv": "E4M3 unnormalized exp*256 and E4M3 V GMMA; float O registers rescaled across native tiles",
            "normalization": "FP32 pre-E4M3-cast probability sum; V descale at finalize",
            "ieee_fp32_tensor_core_rounding_claim": False,
            "internal_probability_diagnostics": "not exposed",
        },
        "settings": {"causal": True, "num_splits": 1, "pack_gqa": False, "return_attn_probs": False},
        "variants": list(VARIANTS),
        "smoothing": "external K centering in FP64 before FP32; before optional shared FWHT; V untouched",
        "hardware": "Hopper sm90 only; D multiple16, D<=256",
        "sources": [
            "https://huggingface.co/kernels/kernels-community/flash-attn3",
            "https://github.com/huggingface/kernels-community/tree/8a730d96c37560ccf1d3e09f7bbccdc886818f33/flash-attn3",
        ],
    }
