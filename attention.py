"""Inspectable CPU rounding model; not a bit-exact model of any GPU."""

from dataclasses import dataclass
from typing import Literal

import ml_dtypes
import numpy as np

Format = Literal["fp32", "bf16", "e4m3", "e5m2"]
DTYPES = {
    "fp32": np.float32,
    "bf16": ml_dtypes.bfloat16,
    "e4m3": ml_dtypes.float8_e4m3fn,
    "e5m2": ml_dtypes.float8_e5m2,
}
FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}


@dataclass(frozen=True)
class Config:
    storage: Format = "bf16"
    tile: int = 128
    query_tile: int = 32
    scaling: Literal["tensor", "tile"] = "tensor"
    accumulator: Literal["fp32", "reduced14"] = "fp32"
    promote: int = 0
    probability: Format | None = None  # None means same format as storage.
    output: Format = "bf16"
    rotate: bool = False
    smooth_k: bool = False
    compensated: bool = False
    update: Literal["global", "local"] = "global"
    order: Literal["forward", "reverse"] = "forward"
    causal: bool = False
    scale: float | None = None

    def __post_init__(self):
        if self.storage not in DTYPES or self.output not in DTYPES:
            raise ValueError("unknown storage/output format")
        if self.probability is not None and self.probability not in DTYPES:
            raise ValueError("unknown probability format")
        if self.tile <= 0 or self.query_tile <= 0:
            raise ValueError("tiles must be positive")
        if self.promote < 0 or self.promote % 32:
            raise ValueError("promotion interval must be zero or a positive multiple of 32")
        if self.scaling not in ("tensor", "tile"):
            raise ValueError("unknown scaling")
        if self.accumulator not in ("fp32", "reduced14"):
            raise ValueError("unknown accumulator")
        if self.update not in ("global", "local") or self.order not in ("forward", "reverse"):
            raise ValueError("unknown update/order")
        if self.scale is not None and (not np.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("softmax scale must be finite and positive")


def cast(x, fmt: Format):
    """One storage conversion, with saturating FP8 round-to-nearest-even."""
    x = np.asarray(x, dtype=np.float32)
    if fmt in FP8_MAX:
        x = np.clip(x, -FP8_MAX[fmt], FP8_MAX[fmt])
    return x.astype(DTYPES[fmt]).astype(np.float32)


def quantize(x, fmt: Format, block: int | None = None):
    """Return expanded storage values and one dequantization scale per row."""
    x = np.asarray(x, dtype=np.float32)
    if fmt not in FP8_MAX:
        return cast(x, fmt), np.ones(len(x), dtype=np.float32)
    y = np.empty_like(x)
    scales = np.empty(len(x), dtype=np.float32)
    block = len(x) if block is None else block
    for start in range(0, len(x), block):
        part = x[start : start + block]
        amax = np.max(np.abs(part))
        scale = np.float32(amax / FP8_MAX[fmt]) if amax else np.float32(1)
        if scale == 0:
            raise ValueError("FP8 scale underflows float32")
        y[start : start + block] = cast(part / scale, fmt)
        scales[start : start + block] = scale
    return y, scales


def hadamard(x, signs):
    """Row-vector x diag(signs) H/sqrt(d), butterflies rounded in float32."""
    y = np.asarray(x, dtype=np.float32) * np.asarray(signs, dtype=np.float32)
    d = y.shape[-1]
    if d == 0 or d & (d - 1) or np.shape(signs) != (d,):
        raise ValueError("Hadamard requires power-of-two dimension and d signs")
    width = 1
    while width < d:
        blocks = y.reshape(-1, d // (2 * width), 2, width)
        left = blocks[:, :, 0, :].copy()
        right = blocks[:, :, 1, :].copy()
        blocks[:, :, 0, :] = left + right
        blocks[:, :, 1, :] = left - right
        width *= 2
    return y / np.float32(np.sqrt(d))


def truncate_significand(x, bits=14):
    """Truncate magnitude to `bits` significant binary digits, leading bit included."""
    x = np.asarray(x, dtype=np.float32)
    if not 1 <= bits <= 24:
        raise ValueError("significant bits must be in [1,24]")
    fraction, exponent = np.frexp(x)
    return np.ldexp(np.trunc(np.ldexp(fraction, bits)), exponent - bits).astype(np.float32)


def shared_exponent_sum(products, bits=14):
    """Align a final-axis group to its largest exponent; truncate before fp32 sum."""
    products = np.asarray(products, dtype=np.float32)
    _, exponent = np.frexp(np.max(np.abs(products), axis=-1, keepdims=True))
    aligned = np.ldexp(products, bits - exponent)
    clipped = np.ldexp(np.trunc(aligned), exponent - bits)
    return np.sum(clipped, axis=-1, dtype=np.float32)


def matmul(a, b, accumulator="fp32", promote=0, initial=None):
    """32-term fp32 GEMMs, or shared-exponent reduced-significand surrogate."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible matrix shapes")
    out = np.zeros((len(a), b.shape[1]), dtype=np.float32)
    if initial is not None:
        out[:] = initial
    partial = np.zeros_like(out)
    for start in range(0, a.shape[1], 32):
        left, right = a[:, start : start + 32], b[start : start + 32]
        if accumulator == "fp32":
            out = np.add(out, left @ right, dtype=np.float32)
        elif accumulator == "reduced14":
            products = left[:, None, :] * right.T[None, :, :]
            group = shared_exponent_sum(products)
            if promote:
                partial = truncate_significand(partial + group)
                if (start + 32) % promote == 0 or start + 32 >= a.shape[1]:
                    out = np.add(out, partial, dtype=np.float32)
                    partial.fill(0)
            else:
                out = truncate_significand(out + group)
        else:
            raise ValueError("unknown accumulator")
    return out


def _inputs(q, k, v, rows):
    q, k, v = (np.asarray(x) for x in (q, k, v))
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("Q/K/V must be two-dimensional")
    if not len(q) or not len(k) or not q.shape[1] or not v.shape[1]:
        raise ValueError("Q/K/V must be nonempty")
    if q.shape[1] != k.shape[1] or len(k) != len(v):
        raise ValueError("incompatible Q/K/V shapes")
    if not all(np.all(np.isfinite(x)) for x in (q, k, v)):
        raise ValueError("inputs must be finite")
    rows = np.arange(len(q)) if rows is None else np.asarray(rows)
    if rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("rows must be a nonempty integer vector")
    if np.any(rows < 0) or np.any(rows >= len(q)):
        raise ValueError("query rows out of range")
    return q, k, v, rows


def center_keys(k):
    """Subtract the token mean in float64; a constant logit shift per query."""
    k = np.asarray(k, dtype=np.float64)
    return k - np.mean(k, axis=0, dtype=np.float64)


def quantize_qk(q, k, cfg):
    """Shared Q/K preparation for attention and score-distribution diagnostics."""
    q = np.asarray(q, dtype=np.float32)
    k = center_keys(k).astype(np.float32) if cfg.smooth_k else np.asarray(k, dtype=np.float32)
    if cfg.rotate:
        signs = np.random.default_rng(1729).choice([-1, 1], size=q.shape[1])
        q, k = hadamard(q, signs), hadamard(k, signs)
    qblock = cfg.query_tile if cfg.scaling == "tile" else None
    kblock = cfg.tile if cfg.scaling == "tile" else None
    qs, qscale = quantize(q, cfg.storage, qblock)
    ks, kscale = quantize(k, cfg.storage, kblock)
    return qs, qscale, ks, kscale


def _probabilities(p, fmt):
    if fmt in FP8_MAX:
        scale = np.float32(1 / FP8_MAX[fmt])
        return cast(p / scale, fmt), scale
    return cast(p, fmt), np.float32(1)


def emulate(q, k, v, config=None, rows=None):
    """Online attention; memory O(Nd + query_tile*key_tile), including sampled rows."""
    q, k, v, rows = _inputs(q, k, v, rows)
    q, v = (x.astype(np.float32) for x in (q, v))
    cfg = Config() if config is None else config
    qs, qscale, ks, kscale = quantize_qk(q, k, cfg)
    kvblock = cfg.tile if cfg.scaling == "tile" else None
    vs, vscale = quantize(v, cfg.storage, kvblock)
    softmax_scale = np.float32(cfg.scale if cfg.scale is not None else q.shape[1] ** -0.5)
    output = np.empty((len(rows), v.shape[1]), dtype=np.float32)
    probability = cfg.storage if cfg.probability is None else cfg.probability
    # Keep natural query-block boundaries even when only some queries are requested.
    row_blocks = rows // cfg.query_tile
    for row_block in np.unique(row_blocks):
        selected = np.flatnonzero(row_blocks == row_block)
        qr = rows[selected]
        maximum = np.full(len(qr), -np.inf, dtype=np.float32)
        denominator = np.zeros(len(qr), dtype=np.float32)
        correction = np.zeros_like(denominator)
        numerator = np.zeros((len(qr), v.shape[1]), dtype=np.float32)
        limit = min(len(k), int(qr.max()) + 1) if cfg.causal else len(k)
        starts = list(range(0, limit, cfg.tile))
        if cfg.order == "reverse":
            starts.reverse()
        for start in starts:
            stop = min(start + cfg.tile, len(k))
            scores = matmul(qs[qr], ks[start:stop].T, cfg.accumulator, cfg.promote)
            # Q and K scales are constant inside each natural quantization block.
            scores *= qscale[qr, None] * kscale[None, start:stop]
            scores *= softmax_scale
            if cfg.causal:
                scores = np.where(np.arange(start, stop)[None, :] <= qr[:, None], scores, -np.inf)
            tile_max = np.max(scores, axis=1)
            new_max = np.maximum(maximum, tile_max)
            safe_max = np.where(np.isfinite(new_max), new_max, np.float32(0))
            alpha = np.exp(maximum - safe_max).astype(np.float32)
            if cfg.update == "global":
                weights = np.exp(scores - safe_max[:, None]).astype(np.float32)
                beta = np.ones_like(alpha)
            else:
                safe_tile_max = np.where(np.isfinite(tile_max), tile_max, np.float32(0))
                weights = np.exp(scores - safe_tile_max[:, None]).astype(np.float32)
                beta = np.exp(tile_max - safe_max).astype(np.float32)
            tile_sum = np.sum(weights, axis=1, dtype=np.float32) * beta
            denominator *= alpha
            correction *= alpha
            if cfg.compensated:
                delta = tile_sum - correction
                updated = denominator + delta
                correction = (updated - denominator) - delta
                denominator = updated
            else:
                denominator += tile_sum
            numerator *= alpha[:, None]
            packed_weights, pscale = _probabilities(weights, probability)
            # Each V block has one scale; dequantization follows the GEMM.
            value_scale = vscale[start] * pscale
            if cfg.update == "global" and cfg.storage not in FP8_MAX and probability not in FP8_MAX:
                numerator = matmul(
                    packed_weights, vs[start:stop], cfg.accumulator, cfg.promote, numerator
                )
            else:
                contribution = matmul(packed_weights, vs[start:stop], cfg.accumulator, cfg.promote)
                numerator += contribution * (beta * value_scale)[:, None]
            maximum = new_max
        output[selected] = numerator / denominator[:, None]
    return cast(output, cfg.output)


def reference(q, k, v, *, rows=None, causal=False, scale=None, query_tile=32, tile=1024):
    """Two-pass float64 softmax, with no online max-rescaling recurrence."""
    q32, k32, v32, rows = _inputs(q, k, v, rows)
    q, k, v = (x.astype(np.float64) for x in (q32, k32, v32))
    if tile <= 0 or query_tile <= 0:
        raise ValueError("tiles must be positive")
    scale = q.shape[1] ** -0.5 if scale is None else scale
    output = np.empty((len(rows), v.shape[1]), dtype=np.float64)
    for offset in range(0, len(rows), query_tile):
        qr = rows[offset : offset + query_tile]
        limit = min(len(k), int(qr.max()) + 1) if causal else len(k)
        maximum = np.full(len(qr), -np.inf)
        for start in range(0, limit, tile):
            scores = (q[qr] @ k[start : start + tile].T) * scale
            if causal:
                valid = np.arange(start, min(start + tile, len(k)))[None, :] <= qr[:, None]
                scores = np.where(valid, scores, -np.inf)
            maximum = np.maximum(maximum, np.max(scores, axis=1))
        denominator = np.zeros(len(qr))
        numerator = np.zeros((len(qr), v.shape[1]))
        for start in range(0, limit, tile):
            scores = (q[qr] @ k[start : start + tile].T) * scale
            if causal:
                valid = np.arange(start, min(start + tile, len(k)))[None, :] <= qr[:, None]
                scores = np.where(valid, scores, -np.inf)
            weights = np.exp(scores - maximum[:, None])
            denominator += weights.sum(axis=1)
            numerator += weights @ v[start : start + tile]
        output[offset : offset + len(qr)] = numerator / denominator[:, None]
    return output


def metrics(actual, expected, rows=None):
    """Worst row is the original row with the largest absolute row L2 error."""
    difference = np.asarray(actual, dtype=np.float64) - expected
    row_error = np.linalg.norm(difference, axis=1)
    worst = int(np.argmax(row_error))
    norm = float(np.linalg.norm(expected))
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_frobenius": float(np.linalg.norm(difference) / norm) if norm else None,
        "worst_row": int(rows[worst]) if rows is not None else worst,
        "worst_row_l2": float(row_error[worst]),
        "reference_frobenius": norm,
    }
