"""Summarize complete real-kernel runs without fitting the locked predictor.

python -m study.hardware.report --results-dir results/hardware
"""

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from study.classification import _summary
from study.prediction import _ranks

from .common import VARIANTS

ROOT = Path(__file__).resolve().parents[2]
COLORS = {
    "qwen05": "#1565c0",
    "qwen15": "#663399",
    "smol036": "#008577",
    "smol17": "#d97900",
    "tiny11": "#bd334b",
    "olmo1": "#59636d",
}


def load_run(path):
    with gzip.open(path, "rt") as handle:
        run = json.load(handle)
    if run["mode"] != "full" or not run["complete"]:
        raise ValueError("Only complete full hardware runs may enter the result summary")
    return run


def spearman(x, y):
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def physical_points(run, original):
    groups = defaultdict(list)
    for row in run["heads"]["rows"]:
        groups[(row["model"], row["layer"], row["head"])].append(row)
    expected = {
        (model, point["layer"], point["head"])
        for model, points in run["selection"]["models"].items()
        for point in points
    }
    if set(groups) != expected:
        raise ValueError("Hardware physical-head coverage differs from the committed selection")
    points = []
    for identity, rows in sorted(groups.items()):
        if len(rows) != 3 or {row["text"] for row in rows} != {"alice", "moby", "pride"}:
            raise ValueError(f"Incomplete physical-head text coverage: {identity}")
        old = original[identity]
        if any(row["revision"] != old["revision"] for row in rows):
            raise ValueError("Hardware and locked predictor model revisions disagree")
        logs = {}
        for kind in ("emulator", "hardware"):
            logs[kind] = {
                variant: math.fsum(math.log1p(row[kind][variant]["relative_fro"]) for row in rows)
                / 3
                for variant in VARIANTS
            }
        score = (
            math.fsum(
                math.log1p(row["emulator"]["rotate"]["predicted_error"])
                - math.log1p(row["emulator"]["tile"]["predicted_error"])
                for row in rows
            )
            / 3
        )
        if abs(score - old["predicted_hurt_score"]) > 1e-13:
            raise ValueError("Hardware predictor score does not match the published lock")
        observed = logs["hardware"]["rotate"] - logs["hardware"]["tile"]
        emulator_score = logs["emulator"]["rotate"] - logs["emulator"]["tile"]
        point = dict(old)
        point.update(
            {
                "selected_by": rows[0]["selected_by"],
                "text_count": 3,
                "observed_hurt_score": observed,
                "predicted_hurt_score": score,
                "hurts": observed > 0,
                "helps": observed < 0,
                "ties": observed == 0,
                "predict_hurt": score > 0,
                "emulator_hurt_score": emulator_score,
                "emulator_hurts": emulator_score > 0,
                "log_error": logs,
                "native_bf16_log_error": math.fsum(
                    math.log1p(row["hardware"]["bf16"]["relative_fro"]) for row in rows
                )
                / 3,
            }
        )
        points.append(point)
    return points


def compare(points):
    risk = _summary(points)
    # Reuse the existing tie-aware classifier. Sink values are from the same
    # published original operands, but are not a new hardware measurement.
    risk.pop("sink_concentration")
    risk.pop("log_gain")
    discordant = [point for point in points if point["hurts"] != point["emulator_hurts"]]
    risk["emulator_harm_disagreement"] = {
        "count": len(discordant),
        "fraction": len(discordant) / len(points) if points else None,
        "emulator_hurts_kernel_does_not": sum(
            point["emulator_hurts"] and not point["hurts"] for point in points
        ),
        "kernel_hurts_emulator_does_not": sum(
            point["hurts"] and not point["emulator_hurts"] for point in points
        ),
    }
    risk["emulator_rotation_effect_spearman"] = spearman(
        [point["emulator_hurt_score"] for point in points],
        [point["observed_hurt_score"] for point in points],
    )
    risk["small_effect_count_abs_score_below_1e-4"] = sum(
        abs(point["observed_hurt_score"]) < 1e-4 for point in points
    )
    variants = {}
    disagreements = []
    for variant in VARIANTS:
        x = [point["log_error"]["emulator"][variant] for point in points]
        y = [point["log_error"]["hardware"][variant] for point in points]
        variants[variant] = {
            "physical_heads": len(points),
            "spearman": spearman(x, y),
            "emulator_median_relative_fro": float(np.median(np.expm1(x))) if x else None,
            "kernel_median_relative_fro": float(np.median(np.expm1(y))) if y else None,
            "kernel_p90_relative_fro": float(np.quantile(np.expm1(y), 0.9)) if y else None,
        }
        for point in points:
            emu, real = (
                point["log_error"]["emulator"][variant],
                point["log_error"]["hardware"][variant],
            )
            disagreements.append(
                {
                    "model": point["model"],
                    "layer": point["layer"],
                    "head": point["head"],
                    "variant": variant,
                    "emulator_relative_fro": math.expm1(emu),
                    "kernel_relative_fro": math.expm1(real),
                    "log_error_difference": real - emu,
                }
            )
    risk["errors"] = variants
    risk["largest_error_disagreements"] = sorted(
        disagreements, key=lambda item: -abs(item["log_error_difference"])
    )[:12]
    risk["largest_rotation_effect_disagreements"] = [
        {
            "model": point["model"],
            "layer": point["layer"],
            "head": point["head"],
            "emulator_hurt_score": point["emulator_hurt_score"],
            "kernel_hurt_score": point["observed_hurt_score"],
            "locked_predicted_hurt_score": point["predicted_hurt_score"],
        }
        for point in sorted(
            points,
            key=lambda point: -abs(point["observed_hurt_score"] - point["emulator_hurt_score"]),
        )[:12]
    ]
    return risk


def summarize_downstream(run):
    if not run["downstream"]:
        raise ValueError("FA3 full run lacks required downstream evaluation")
    groups = defaultdict(list)
    for row in run["downstream"]["rows"]:
        groups[(row["model"], row["variant"])].append(row)
    expected = {
        (model, variant) for model in ("qwen05", "qwen15") for variant in ("bf16", *VARIANTS)
    }
    if set(groups) != expected:
        raise ValueError("Downstream model/variant coverage is incomplete")
    result = {}
    for (model, variant), rows in sorted(groups.items()):
        if len(rows) != 3 or {row["text"] for row in rows} != {"alice", "moby", "pride"}:
            raise ValueError("Downstream needs the same three complete heldout texts")
        count = sum(row["tokens"] for row in rows)
        result.setdefault(model, {})[variant] = {
            "tokens": count,
            "next_token_ce": math.fsum(row["next_token_ce"] * row["tokens"] for row in rows)
            / count,
            "kl_from_bf16": math.fsum(row["kl_from_bf16"] * row["tokens"] for row in rows) / count,
            "kernel_calls_per_text": rows[0]["kernel_calls"],
        }
    for variants in result.values():
        baseline = variants["bf16"]["next_token_ce"]
        for metrics in variants.values():
            metrics["delta_ce"] = metrics["next_token_ce"] - baseline
            metrics["exp_ce_ratio"] = math.exp(metrics["delta_ce"])
    return result


def draw(points_by_backend, destination):
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    markers = {"tile": "o", "rotate": "^", "smooth_k": "s", "rotate_smooth_k": "x"}
    for row, backend in enumerate(("fa3", "sage")):
        points = points_by_backend[backend]
        absolute, effect = axes[row]
        for model, color in COLORS.items():
            subset = [point for point in points if point["model"] == model]
            for variant, marker in markers.items():
                absolute.scatter(
                    [point["log_error"]["emulator"][variant] for point in subset],
                    [point["log_error"]["hardware"][variant] for point in subset],
                    s=9,
                    alpha=0.35,
                    marker=marker,
                    color=color,
                    linewidths=0.5,
                )
            effect.scatter(
                [point["emulator_hurt_score"] for point in subset],
                [point["observed_hurt_score"] for point in subset],
                s=11,
                alpha=0.55,
                color=color,
                label=model,
            )
            effect.scatter(
                [
                    point["log_error"]["emulator"]["rotate_smooth_k"]
                    - point["log_error"]["emulator"]["smooth_k"]
                    for point in subset
                ],
                [
                    point["log_error"]["hardware"]["rotate_smooth_k"]
                    - point["log_error"]["hardware"]["smooth_k"]
                    for point in subset
                ],
                s=12,
                alpha=0.35,
                marker="x",
                color=color,
                linewidths=0.5,
            )
        absolute.set(
            xlabel="Emulator: mean log1p(relative output error)",
            ylabel="Real kernel: mean log1p(relative output error)",
            title=f"{backend.upper()}: four transformations, {len(points)} heads",
        )
        effect.set(
            xlabel="Emulator rotation harm score",
            ylabel="Real-kernel rotation harm score",
            title="Rotation effect: circles raw K, crosses smoothed K",
        )
        for axis in (absolute, effect):
            low = min(axis.get_xlim()[0], axis.get_ylim()[0])
            high = max(axis.get_xlim()[1], axis.get_ylim()[1])
            axis.plot(
                [low, high], [low, high], linestyle="--", color="#777", linewidth=0.8, zorder=0
            )
            axis.grid(alpha=0.18)
        effect.axhline(0, color="#555", linewidth=0.6)
        effect.axvline(0, color="#555", linewidth=0.6)
    labels = ("Unrotated", "Rotated", "Center K", "Rotate + center K")
    axes[0, 0].legend(
        handles=[
            Line2D([], [], color="#555", marker=marker, linestyle="none", label=label)
            for marker, label in zip(markers.values(), labels, strict=True)
        ],
        ncol=2,
        fontsize=8,
        loc="upper left",
    )
    axes[0, 1].legend(ncol=3, fontsize=8, loc="upper left")
    figure.suptitle("Real kernels versus uniform-E4 CPU emulation", fontsize=15)
    figure.text(
        0.5,
        0.01,
        "Exact original BF16 operands; FP64 causal reference; "
        "three texts averaged per physical head.\n"
        "Qwens are exhaustive; the other models mix top-ranked and uniform samples, "
        "not population estimates.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    figure.savefig(destination)
    plt.close(figure)


def write_report(results_dir):
    results_dir = Path(results_dir)
    original_points = json.loads((ROOT / "results/v2/rotation_risk.json").read_text())[
        "physical_heads"
    ]
    original = {(point["model"], point["layer"], point["head"]): point for point in original_points}
    runs, points_by_backend = {}, {}
    for backend in ("fa3", "sage"):
        runs[backend] = load_run(results_dir / f"{backend}.json.gz")
        if runs[backend]["backend"] != backend:
            raise ValueError("Result filename/backend mismatch")
        points_by_backend[backend] = physical_points(runs[backend], original)

    def identities(points):
        return {(point["model"], point["layer"], point["head"]) for point in points}

    if identities(points_by_backend["fa3"]) != identities(points_by_backend["sage"]):
        raise ValueError("Kernels did not use identical physical-head populations")
    summary = {
        "definitions": {
            "physical_unit": (
                "One physical query head, averaging log1p(relative Frobenius error) "
                "across three texts before comparisons"
            ),
            "predictor": (
                "Unchanged f756745 v2 operand-derived score; threshold0; "
                "no kernel-specific fitting or threshold tuning"
            ),
            "harm": "mean_text(log1p(kernel rotate error)-log1p(kernel tile error)) > 0",
            "spearman": (
                "Tie-aware rank correlation; null for constant ranks or fewer than two points"
            ),
            "sampling": (
                "Qwens exhaustive; other models independently top32 plus uniform32, "
                "overlapping heads measured once. Enriched overall result is descriptive; "
                "use random stratum for an unenriched sample."
            ),
            "downstream": (
                "Batch next-token teacher forcing; native GPU BF16 is the matched baseline, "
                "not CPU v2. exp(delta CE) is a batch exp-CE ratio, "
                "not streaming decoder perplexity."
            ),
            "causal_attribution": (
                "Scale granularity, probability representation, smoothing precision "
                "and accumulation differ together. Source differences and error correlations "
                "do not isolate individual causal contributions."
            ),
        },
        "kernels": {},
        "downstream_fa3": summarize_downstream(runs["fa3"]),
        "source_files": {
            backend: {
                "file": f"{backend}.json.gz",
                "sha256": __import__("hashlib")
                .sha256((results_dir / f"{backend}.json.gz").read_bytes())
                .hexdigest(),
            }
            for backend in runs
        },
    }
    for backend, points in points_by_backend.items():
        summary["kernels"][backend] = {
            "runtime": runs[backend]["runtime"],
            "adapter": runs[backend]["adapter"],
            "overall_selected": compare(points),
            "qwen_exhaustive": compare(
                [point for point in points if point["model"].startswith("qwen")]
            ),
            "non_qwen_uniform_sample": compare(
                [point for point in points if "random" in point["selected_by"]]
            ),
            "non_qwen_top_ranked": compare(
                [point for point in points if "top" in point["selected_by"]]
            ),
            "per_model": {
                model: compare([point for point in points if point["model"] == model])
                for model in COLORS
            },
            "reference_energy_max_relative_disagreement": max(
                row["reference_energy_relative_disagreement"]
                for row in runs[backend]["heads"]["rows"]
            ),
        }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (results_dir / "physical_heads.csv").open("w", newline="") as handle:
        fields = [
            "backend",
            "model",
            "layer",
            "head",
            "selected_by",
            "predicted_hurt_score",
            "emulator_hurt_score",
            "kernel_hurt_score",
        ]
        fields += [
            f"{kind}_{variant}_log_error"
            for kind in ("emulator", "hardware")
            for variant in VARIANTS
        ]
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for backend, points in points_by_backend.items():
            for point in points:
                writer.writerow(
                    {
                        "backend": backend,
                        **{
                            field: point[field]
                            for field in (
                                "model",
                                "layer",
                                "head",
                                "predicted_hurt_score",
                                "emulator_hurt_score",
                            )
                        },
                        "selected_by": "+".join(point["selected_by"]),
                        "kernel_hurt_score": point["observed_hurt_score"],
                        **{
                            f"{kind}_{variant}_log_error": point["log_error"][kind][variant]
                            for kind in ("emulator", "hardware")
                            for variant in VARIANTS
                        },
                    }
                )
    draw(points_by_backend, results_dir / "real_vs_emulator.svg")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/hardware")
    args = parser.parse_args()
    summary = write_report(args.results_dir)
    print(
        json.dumps(
            {
                backend: data["qwen_exhaustive"]["classification"]
                for backend, data in summary["kernels"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
