"""Shared operand geometry and an explicit causal FP64 hardware reference.

Rotation intentionally calls the v2 FWHT: the same PCG64 Rademacher vector,
Q/K factor order, FP32 butterfly additions and normalization; never rotate V.
The real kernels differ in quantization and accumulation, not in this matrix.
"""

import math

import numpy as np
import torch

from study.attention import _center_tokens, _rotate

VARIANTS = ("tile", "rotate", "smooth_k", "rotate_smooth_k")


def validate_operands(q, k, v):
    """Check geometry without restricting device (CPU contract tests are useful)."""
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Q/K/V must have shape [B,H,N,D]")
    b, h, n, d = q.shape
    if (
        min(b, h, n, d, k.shape[1]) == 0
        or k.shape != v.shape
        or k.shape[0] != b
        or k.shape[2:] != (n, d)
        or h % k.shape[1]
    ):
        raise ValueError("Incompatible Q/K/V or grouped-query geometry")
    if any(x.device != q.device for x in (k, v)):
        raise ValueError("Q/K/V must share a device")
    if any(x.dtype not in (torch.bfloat16, torch.float32) for x in (q, k, v)):
        raise ValueError("Operand storage must be BF16 or FP32")


def expand_gqa(q, k, v):
    validate_operands(q, k, v)
    if q.shape[1] != k.shape[1]:
        groups = q.shape[1] // k.shape[1]
        mapping = torch.arange(q.shape[1], device=q.device) // groups
        k = k.index_select(1, mapping)
        v = v.index_select(1, mapping)
    return q, k, v


@torch.no_grad()
def prepare(q, k, v, variant, *, native_smoothing=False, sign_seed=1729):
    """Prepare contiguous FP32 operands, expanding GQA without changing head mapping.

    Sage's public API takes BF16 and optionally centers K itself. With
    native_smoothing=True I leave that centering to its smooth_k flag; it runs
    after rotation and has its native reduction/narrowing rather than v2's
    FP64-before-FP32 subtraction. The Sage adapter records this difference.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown hardware attention variant: {variant}")
    q, k, v = expand_gqa(q, k, v)
    q, k, v = q.float(), k.float(), v.float()
    if "smooth_k" in variant and not native_smoothing:
        k, _ = _center_tokens(k)
    if variant in ("rotate", "rotate_smooth_k"):
        signs = torch.from_numpy(
            np.random.default_rng(sign_seed).choice([-1, 1], size=q.shape[-1]).astype(np.float32)
        ).to(q.device)
        q, k = _rotate(q, signs), _rotate(k, signs)
    return q.contiguous(), k.contiguous(), v.contiguous()


@torch.no_grad()
def exact_attention(q, k, v, scale=None, query_chunk=128):
    """Explicit FP64 dot/causal-softmax/PV, not an optimized attention backend.

    The query chunk bounds [B,H,chunk,N] storage; every key is included before
    the causal mask. Evaluation inputs are original captured BF16 operands,
    not transformed or quantized operands.
    """
    q, k, v = expand_gqa(q, k, v)
    if query_chunk <= 0:
        raise ValueError("Reference query chunk must be positive")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Softmax scale must be finite and positive")
    q, k, v = q.double(), k.double(), v.double()
    output = torch.empty_like(q)
    keys = torch.arange(k.shape[-2], device=q.device)
    kt = k.transpose(-1, -2)
    for first in range(0, q.shape[-2], query_chunk):
        stop = min(first + query_chunk, q.shape[-2])
        scores = torch.matmul(q[..., first:stop, :], kt) * scale
        rows = torch.arange(first, stop, device=q.device)
        scores.masked_fill_(keys.unsqueeze(0) > rows.unsqueeze(1), -math.inf)
        output[..., first:stop, :] = torch.matmul(torch.softmax(scores, dim=-1), v)
    if not torch.isfinite(output).all():
        raise FloatingPointError("Nonfinite FP64 attention reference")
    return output


@torch.no_grad()
def error_metrics(actual, reference):
    """Per batch/head relative Frobenius and maximum absolute output error."""
    if actual.shape != reference.shape or actual.ndim != 4:
        raise ValueError("Actual/reference outputs must share [B,H,N,D] shape")
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise FloatingPointError("Nonfinite hardware/reference output")
    delta = actual.double() - reference
    numerator = delta.square().sum(dim=(-2, -1)).sqrt()
    denominator = reference.square().sum(dim=(-2, -1)).sqrt()
    if torch.any(denominator == 0):
        raise ValueError("Relative output error is undefined for a zero reference")
    return {
        "relative_fro": (numerator / denominator).cpu().tolist(),
        "max_abs": delta.abs().amax(dim=(-2, -1)).cpu().tolist(),
        "output_nonfinite": 0,
    }
