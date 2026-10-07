"""Bounded-memory CPU FP8 attention surrogate, not a GPU kernel reproduction.

FA3 incoherent processing rotates Q/K only (https://arxiv.org/abs/2407.08608):
row x @ diag(signs) @ H/sqrt(D). Factor order, PCG64 and seed1729 are explicit
emulator choices; the paper does not specify a unique RNG/factorization.
SageAttention2 query smoothing restores mu_Q @ centered_K.T before softmax
(https://arxiv.org/abs/2411.10958). Here Q/K/V are blockwise E4M3, not Sage's
INT4/per-channel-V arithmetic. FP32 recurrence matches the full NumPy model's
32-term products, up to CPU reduction/exp rounding; there is no timing claim.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

VARIANTS = ("bf16", "tile", "rotate", "smooth_k", "rotate_smooth_k", "smooth_kq")
_FP8_MAX = 448.0
_PSCALE = np.float32(1 / _FP8_MAX).item()


def _rotate(x, signs):
    d = x.shape[-1]
    if d & (d - 1):
        raise ValueError("Hadamard rotation requires a power-of-two head dimension")
    y = x * signs
    width = 1
    while width < d:
        blocks = y.reshape(*y.shape[:-1], d // (2 * width), 2, width)
        left = blocks[..., 0, :].clone()
        right = blocks[..., 1, :].clone()
        blocks[..., 0, :] = left + right
        blocks[..., 1, :] = left - right
        width *= 2
    return y / np.float32(np.sqrt(d)).item()


def _pack(x, block):
    """One scale per token block, independently for each batch/head."""
    packed = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0], math.ceil(x.shape[1] / block), 1, 1), dtype=torch.float32)
    for index, start in enumerate(range(0, x.shape[1], block)):
        part = x[:, start : start + block]
        amax = part.abs().amax(dim=(-2, -1), keepdim=True)
        scale = torch.where(amax == 0, 1.0, amax / _FP8_MAX)
        if torch.any(scale == 0):
            raise ValueError("FP8 scale underflows float32")
        packed[:, start : start + block] = (
            (part / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        )
        scales[:, index] = scale
    return packed, scales


def _matmul32(a, b):
    """Grouped FP32 products, avoiding an H*Q*K*D product tensor."""
    out = torch.zeros((a.shape[0], a.shape[1], b.shape[2]), dtype=torch.float32)
    for start in range(0, a.shape[2], 32):
        out.add_(torch.bmm(a[:, :, start : start + 32], b[:, start : start + 32]))
    return out


def _center_tokens(x):
    # BF16 input is expanded exactly. Center before narrowing the mean/subtract
    # to FP32, matching the NumPy reference and retaining small residuals.
    wide = x.double()
    mean = wide.mean(dim=-2, keepdim=True)
    return (wide - mean).float(), mean.float()


@torch.no_grad()
def apply_attention(q, k, v, variant, scale=None, key_tile=128, query_tile=32, sign_seed=1729):
    """Causal CPU attention for Q[B,Hq,N,D], K/V[B,Hkv,N,D], returning BF16.

    Inputs must be actual BF16 tensors; expansion to FP32 is exact. Quantization
    uses Q blocks=query_tile and K/V blocks=key_tile (defaults32/128). Packed
    E4M3 is expanded only for active tiles. Head chunks bound working storage
    independently of total head count; no N*N attention matrix is allocated in
    the FP8 path. GQA maps consecutive groups of query heads to each KV head.

    ``bf16`` calls native Torch SDPA, not the FP8 recurrence. LayerStream must
    retain native Transformers BF16 attention as its experimental baseline.
    ``smooth_kq`` centers Q per query block and K over all tokens, restoring
    the unquantized key-dependent mean-query score correction. All other
    smoothing variants center only K, before any rotation/packing.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown attention variant: {variant}")
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Q/K/V must have shape [B,H,N,D]")
    if any(x.device.type != "cpu" or x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("Q/K/V must be CPU BF16 tensors")
    batch, heads, n, d = q.shape
    kvheads = k.shape[1]
    if (
        min(batch, heads, n, d, kvheads) == 0
        or k.shape != v.shape
        or k.shape[0] != batch
        or k.shape[2:] != (n, d)
        or heads % kvheads
    ):
        raise ValueError("incompatible Q/K/V or GQA shapes")
    if key_tile <= 0 or query_tile <= 0:
        raise ValueError("tiles must be positive")
    scale = d**-0.5 if scale is None else scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax scale must be finite and positive")
    if variant == "bf16":
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale, enable_gqa=heads != kvheads
        )
    scale = np.float32(scale).item()
    rotated = variant in ("rotate", "rotate_smooth_k")
    smooth_k = variant in ("smooth_k", "rotate_smooth_k", "smooth_kq")
    smooth_q = variant == "smooth_kq"
    signs = None
    if rotated:
        signs = torch.from_numpy(
            np.random.default_rng(sign_seed).choice([-1, 1], size=d).astype(np.float32)
        )
    output = torch.empty_like(q)
    # Limit expanded per-head operands to about8MiB; at most8 query heads.
    # One very large single head still requires O(ND) storage, never O(N^2).
    chunk = max(1, min(8, (8 * 1024 * 1024) // (n * d * 4)))
    # Also bound the active H*query_tile*key_tile score/probability tensors.
    chunk = min(chunk, max(1, (8 * 1024 * 1024) // (query_tile * key_tile * 4)))
    group = heads // kvheads
    for b in range(batch):
        for first in range(0, heads, chunk):
            last = min(first + chunk, heads)
            mapping = torch.arange(first, last) // group
            qwork = q[b, first:last].float()
            kwork = k[b].index_select(0, mapping).float()
            vwork = v[b].index_select(0, mapping).float()
            if smooth_k:
                kwork, _ = _center_tokens(kwork)
            means = None
            if smooth_q:
                means = torch.empty(
                    (last - first, math.ceil(n / query_tile), 1, d), dtype=torch.float32
                )
                for qi, start in enumerate(range(0, n, query_tile)):
                    centered, mean = _center_tokens(qwork[:, start : start + query_tile])
                    qwork[:, start : start + query_tile] = centered
                    means[:, qi] = mean
            if rotated:
                qwork, kwork = _rotate(qwork, signs), _rotate(kwork, signs)
            qp, qs = _pack(qwork, query_tile)
            kp, ks = _pack(kwork, key_tile)
            vp, vs = _pack(vwork, key_tile)
            # Only the smoothing correction needs unquantized centered K.
            del qwork, vwork
            if not smooth_q:
                del kwork
            for qi, qstart in enumerate(range(0, n, query_tile)):
                qstop = min(qstart + query_tile, n)
                qr = torch.arange(qstart, qstop)
                query = qp[:, qstart:qstop].float()
                maximum = torch.full(
                    (last - first, qstop - qstart), -torch.inf, dtype=torch.float32
                )
                denominator = torch.zeros_like(maximum)
                numerator = torch.zeros((last - first, qstop - qstart, d), dtype=torch.float32)
                for ki, kstart in enumerate(range(0, qstop, key_tile)):
                    kstop = min(kstart + key_tile, n)
                    keys = kp[:, kstart:kstop].float()
                    scores = _matmul32(query, keys.transpose(1, 2))
                    scores.mul_(qs[:, qi] * ks[:, ki])
                    if smooth_q:
                        correction = _matmul32(means[:, qi], kwork[:, kstart:kstop].transpose(1, 2))
                        scores.add_(correction)
                    scores.mul_(scale)
                    valid = torch.arange(kstart, kstop)[None, :] <= qr[:, None]
                    scores.masked_fill_(~valid, -torch.inf)
                    new_max = torch.maximum(maximum, scores.amax(dim=-1))
                    alpha = torch.exp(maximum - new_max)
                    weights = torch.exp(scores - new_max[:, :, None])
                    denominator.mul_(alpha).add_(weights.sum(dim=-1))
                    numerator.mul_(alpha[:, :, None])
                    probabilities = (
                        (weights / _PSCALE).clamp(0, _FP8_MAX).to(torch.float8_e4m3fn).float()
                    )
                    contribution = _matmul32(probabilities, vp[:, kstart:kstop].float())
                    numerator.add_(contribution * (vs[:, ki] * _PSCALE))
                    maximum = new_max
                output[b, first:last, qstart:qstop] = (numerator / denominator[:, :, None]).to(
                    torch.bfloat16
                )
    return output
