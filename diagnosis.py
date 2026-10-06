"""Diagnose QK-driven attention-mass movement, separately from P/V rounding."""

import argparse
import csv
import hashlib
import json
import struct
from dataclasses import asdict
from pathlib import Path

import numpy as np

from attention import Config, emulate, matmul, metrics, quantize_qk, reference
from sweep import environment

CONFIGS = {
    "e4_tensor": Config(storage="e4m3", causal=True),
    "e4_tile": Config(storage="e4m3", scaling="tile", causal=True),
    "e4_rotate": Config(storage="e4m3", scaling="tile", rotate=True, causal=True),
    "e4_smooth_tile": Config(storage="e4m3", scaling="tile", smooth_k=True, causal=True),
    "e4_smooth_rotate": Config(
        storage="e4m3", scaling="tile", rotate=True, smooth_k=True, causal=True
    ),
}
CONTROL_VARIANTS = ["e4_tile", "e4_rotate", "e4_smooth_tile", "e4_smooth_rotate"]


def sha256(path):
    """Hash the capture without importing the optional Torch capture module."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mean_energy(x):
    """Energy in the all-token, per-channel mean of the original operands."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=0, dtype=np.float64)
    total = float(np.sum(x * x, dtype=np.float64))
    mean_norm = float(np.linalg.norm(mean))
    broadcast_energy = float(len(x) * np.sum(mean * mean, dtype=np.float64))
    return {
        "token_mean_l2": mean_norm,
        "total_energy": total,
        "broadcast_mean_energy": broadcast_energy,
        "mean_energy_fraction": broadcast_energy / total if total else 0.0,
    }


def emulated_logits(q, k, cfg):
    """Reproduce emulate's natural query/key GEMM shapes and fp32 scale products."""
    qs, qscale, ks, kscale = quantize_qk(q, k, cfg)
    scale = np.float32(cfg.scale if cfg.scale is not None else q.shape[1] ** -0.5)
    logits = np.full((len(q), len(k)), -np.inf, dtype=np.float64)
    for offset in range(0, len(q), cfg.query_tile):
        qr = np.arange(offset, min(offset + cfg.query_tile, len(q)))
        limit = min(len(k), int(qr.max()) + 1) if cfg.causal else len(k)
        for start in range(0, limit, cfg.tile):
            stop = min(start + cfg.tile, len(k))
            scores = matmul(qs[qr], ks[start:stop].T, cfg.accumulator, cfg.promote)
            scores *= qscale[qr, None] * kscale[None, start:stop]
            scores *= scale
            if cfg.causal:
                scores = np.where(np.arange(start, stop)[None, :] <= qr[:, None], scores, -np.inf)
            logits[qr, start:stop] = scores
    return logits


def reference_logits(q, k):
    """Causal, float64 original-operand logits, in natural query blocks."""
    q64, k64 = (np.asarray(x, dtype=np.float64) for x in (q, k))
    logits = np.empty((len(q), len(k)), dtype=np.float64)
    for offset in range(0, len(q), 32):
        qr = np.arange(offset, min(offset + 32, len(q)))
        scores = (q64[qr] @ k64.T) * q.shape[1] ** -0.5
        logits[qr] = np.where(np.arange(len(k))[None, :] <= qr[:, None], scores, -np.inf)
    return logits


def softmax64(logits):
    """Normalized stable float64 softmax, BEFORE probability storage conversion."""
    weights = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return weights / np.sum(weights, axis=1, keepdims=True, dtype=np.float64)


def evaluate(q, k, v, names):
    expected = reference(q, k, v, causal=True)
    probabilities = softmax64(reference_logits(q, k))
    v64 = np.asarray(v, dtype=np.float64)
    energies = {
        f"{operand}_{key}": value
        for operand, x in [("q", q), ("k", k)]
        for key, value in mean_energy(x).items()
    }
    results, row_details = [], {}
    for name in names:
        cfg = CONFIGS[name]
        actual = emulate(q, k, v, cfg)
        approximate_p = softmax64(emulated_logits(q, k, cfg))
        score_only = approximate_p @ v64
        output_metrics = metrics(actual, expected)
        score_metrics = metrics(score_only, expected)
        worst = output_metrics["worst_row"]
        tv = 0.5 * np.sum(np.abs(approximate_p - probabilities), axis=1)
        residual = np.asarray(actual, dtype=np.float64) - score_only
        entry = {
            "n": len(q),
            "d": q.shape[1],
            "causal": True,
            "rows_evaluated": len(q),
            "all_rows": True,
            "variant": name,
            **energies,
            **output_metrics,
            "tv_mean": float(np.mean(tv)),
            "tv_max": float(np.max(tv)),
            "tv_max_row": int(np.argmax(tv)),
            "tv_at_worst_output_row": float(tv[worst]),
            "reference_worst_row_output_l2": float(np.linalg.norm(expected[worst])),
            "reference_worst_row_argmax_key": int(np.argmax(probabilities[worst])),
            "approx_worst_row_argmax_key": int(np.argmax(approximate_p[worst])),
            "reference_worst_row_max_probability": float(np.max(probabilities[worst])),
            "approx_worst_row_max_probability": float(np.max(approximate_p[worst])),
            "argmax_disagreement_fraction": float(
                np.mean(np.argmax(probabilities, axis=1) != np.argmax(approximate_p, axis=1))
            ),
            **{f"score_only_{key}": value for key, value in score_metrics.items()},
            "full_minus_score_only_frobenius": float(np.linalg.norm(residual)),
            "full_minus_score_only_relative_frobenius": float(
                np.linalg.norm(residual) / np.linalg.norm(expected)
            ),
            "full_minus_score_only_max_row_l2": float(np.max(np.linalg.norm(residual, axis=1))),
            **{f"cfg_{key}": value for key, value in asdict(cfg).items()},
        }
        results.append(entry)
        row_details[name] = {
            "tv": tv.tolist(),
            "output_row_l2_error": np.linalg.norm(actual - expected, axis=1).tolist(),
            "score_only_row_l2_error": np.linalg.norm(score_only - expected, axis=1).tolist(),
            "reference_output_row_l2": np.linalg.norm(expected, axis=1).tolist(),
        }
    return results, row_details


def write_csv(path, entries):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


def checkpoint_bias_header(path, layers):
    """Inspect only the safetensors JSON header; never load tensor contents."""
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("truncated safetensors header length")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > 16 * 1024 * 1024:
            raise ValueError("unexpectedly large safetensors header")
        header = json.loads(stream.read(length))
    tensors = {}
    for layer in layers:
        for projection in ["q_proj", "k_proj", "v_proj"]:
            name = f"model.layers.{layer}.self_attn.{projection}.bias"
            tensor = header.get(name)
            tensors[name] = (
                None if tensor is None else {"shape": tensor["shape"], "dtype": tensor["dtype"]}
            )
    return {
        "path": str(path),
        "header_only": True,
        "weight_checksum_checked": False,
        "bias_tensors": tensors,
        "interpretation": (
            "Existence of projection bias does not attribute all measured post-RoPE "
            "token means solely to that bias. No model was loaded."
        ),
    }


def control_summary(entries):
    summaries = []
    for scenario in ["constant_channel", "additive_bias", "multiplicative_outlier"]:
        for level in [8, 32]:
            for name in CONTROL_VARIANTS:
                selected = [
                    row
                    for row in entries
                    if row["scenario"] == scenario
                    and row["level"] == level
                    and row["variant"] == name
                ]
                summary = {"scenario": scenario, "level": level, "variant": name}
                for key in [
                    "relative_frobenius",
                    "score_only_relative_frobenius",
                    "tv_mean",
                    "tv_max",
                    "q_mean_energy_fraction",
                    "k_mean_energy_fraction",
                ]:
                    values = [row[key] for row in selected]
                    summary[key] = {
                        "median": float(np.median(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                summaries.append(summary)
    return summaries


def main(path, destination, seeds, checkpoint):
    capture = json.loads(path.with_suffix(".json").read_text())
    digest = sha256(path)
    if digest != capture["capture_sha256"]:
        raise ValueError("capture checksum mismatch")
    if capture["tokens"] != 1024 or capture["head_dimension"] != 64:
        raise ValueError("this diagnosis requires the 1024-token, d=64 capture")
    destination.mkdir(parents=True, exist_ok=True)
    captured, means, detailed = [], [], {}
    with np.load(path, allow_pickle=False) as data:
        for layer in capture["layers_zero_based"]:
            for head in capture["query_heads_zero_based"]:
                prefix = f"layer{layer}_head{head}"
                q, k, v = (data[f"{prefix}_{name}"] for name in ["q", "k", "v"])
                if any(x.shape != (1024, 64) or x.dtype != np.float32 for x in (q, k, v)):
                    raise ValueError(f"unexpected captured shape/dtype for {prefix}")
                results, rows = evaluate(q, k, v, CONFIGS)
                identity = {"model": capture["model"], "layer": layer, "head": head}
                captured.extend({**identity, **entry} for entry in results)
                detailed[prefix] = rows
                for operand, x in [("q", q), ("k", k)]:
                    means.append(
                        {
                            **identity,
                            "n": len(x),
                            "d": x.shape[1],
                            "operand": operand,
                            **mean_energy(x),
                            "reference_frobenius": results[0]["reference_frobenius"],
                        }
                    )
                print(f"Captured {prefix}: all 1024 causal rows, five variants", flush=True)
    write_csv(destination / "diagnosis.csv", captured)
    write_csv(destination / "means.csv", means)

    controls = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        base_q, base_k, v = (rng.standard_normal((1024, 64), dtype=np.float32) for _ in range(3))
        for scenario in ["constant_channel", "additive_bias", "multiplicative_outlier"]:
            for level in [8, 32]:
                q, k = base_q.copy(), base_k.copy()
                if scenario == "constant_channel":
                    q[:, 0] = np.float32(level)
                    k[:, 0] = np.float32(level)
                elif scenario == "additive_bias":
                    q[:, 0] += np.float32(level)
                    k[:, 0] += np.float32(level)
                else:
                    q[:, 0] *= np.float32(level)
                    k[:, 0] *= np.float32(level)
                results, _ = evaluate(q, k, v, CONTROL_VARIANTS)
                controls.extend(
                    {"scenario": scenario, "level": level, "seed": seed, **entry}
                    for entry in results
                )
                print(f"Control seed={seed} {scenario} level={level}: all rows", flush=True)
    write_csv(destination / "bias-controls.csv", controls)
    summary = environment()
    summary.update(
        {
            "capture": capture,
            "capture_hash": {"expected": capture["capture_sha256"], "observed": digest},
            "query_rows": "all 1024 causal query rows for every captured and synthetic case",
            "reference": "attention.reference: original float32 operands, float64 causal attention",
            "mean_energy_fraction": "N * ||mean_over_tokens(X)||_2^2 / ||X||_F^2; original Q and K",
            "logit_reconstruction": (
                "attention.quantize_qk and matmul in natural query blocks32/key tiles128; "
                "multiply Q/K scale product and softmax scale in float32, exactly as emulate"
            ),
            "probability_diagnostic": (
                "Both original float64 logits and emulated float32 logits are normalized with "
                "stable float64 softmax BEFORE P storage conversion. TV isolates QK-driven "
                "attention-mass movement, not probability-rounding effects; it is NOT TV of "
                "emulate's final rounded, potentially non-normalized P coefficients."
            ),
            "score_only_output": (
                "P_from_emulated_logits @ original V in float64, compared to original reference. "
                "Full-minus-score-only residual includes P/V/output rounding and the online "
                "float32 recurrence; it is not an additive attribution of relative error."
            ),
            "smoothing": (
                "K only: subtract all-token per-channel float64 mean, convert to float32, "
                "then optional shared Hadamard rotation and quantization; reference is unchanged"
            ),
            "synthetic": {
                "n": 1024,
                "d": 64,
                "causal": True,
                "seeds": seeds,
                "levels": [8, 32],
                "construction": (
                    "Matched standard-normal float32 Q/K/V draws for each seed. Channel0 of "
                    "both Q and K is replaced by level (constant_channel), receives +level "
                    "(additive_bias), or *=level (multiplicative_outlier); V and every other "
                    "Q/K channel are unchanged. The exact constant channel contributes only "
                    "a common logit shift; additive_bias retains genuine bias-times-variation "
                    "logits and is a separate scenario. These are distinct constructions, "
                    "not variance-matched equivalents. Each scenario keeps its own original "
                    "reference through every numerical variant."
                ),
            },
            "configs": {name: asdict(cfg) for name, cfg in CONFIGS.items()},
            "captured_per_query": detailed,
            "synthetic_summary": control_summary(controls),
            "checkpoint_header": (
                checkpoint_bias_header(checkpoint, capture["layers_zero_based"])
                if checkpoint is not None
                else None
            ),
        }
    )
    (destination / "diagnosis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Wrote {len(captured)} captured rows, {len(means)} mean rows, "
        f"{len(controls)} synthetic rows; no timings measured",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/qwen-qkv.npz"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seeds", nargs=3, type=int, default=[0, 1, 2])
    parser.add_argument("--checkpoint-header", type=Path, default=None)
    args = parser.parse_args()
    if len(set(args.seeds)) != 3:
        parser.error("provide three distinct fixed input seeds")
    main(args.input, args.output, args.seeds, args.checkpoint_header)
