"""Summarize saved v2 measurements: python -m study.report --results-dir results/v2.

This module never captures operands, loads models, or generates experimental data.
Head/text summaries are descriptive; repeated texts do not become scatter points.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

VARIANTS = ("tile", "rotate", "smooth_k", "rotate_smooth_k")
DOWNSTREAM_VARIANTS = ("bf16", *VARIANTS, "smooth_kq")
TRANSFORMS = {
    "rotation": ("tile", "rotate"),
    "smoothing": ("tile", "smooth_k"),
    "rotation_after_smoothing": ("smooth_k", "rotate_smooth_k"),
    "smoothing_after_rotation": ("rotate", "rotate_smooth_k"),
    "rotation_and_smoothing": ("tile", "rotate_smooth_k"),
}
COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#34495e")
LABELS = {
    "bf16": "BF16 baseline",
    "tile": "Tile",
    "rotate": "Rotate",
    "smooth_k": "Smooth K",
    "rotate_smooth_k": "Rotate + smooth K",
    "smooth_kq": "Smooth K + Q",
}
LIMITATIONS = [
    "Heads share model weights, layers and inputs; texts and heads are dependent observations.",
    "Descriptive summaries only: no inferential confidence intervals or independence claims.",
    "Prediction points average per-text gains within each physical model/layer/query head.",
    "Actual head errors are relative Frobenius errors against an FP64 attention reference.",
    "Downstream is batch teacher forcing on held-out tokens, not streaming/generation perplexity.",
    "Full-key means and block scales can depend on later tokens in the teacher-forced batch.",
]


def _number(row, key, nonnegative=True):
    value = float(row[key])
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{key} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _integer(row, key, positive=False):
    value = _number(row, key)
    if not value.is_integer() or (positive and value == 0):
        raise ValueError(f"{key} must be an integer" + (" greater than zero" if positive else ""))
    return int(value)


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _fractions(gains):
    gains = np.asarray(gains)
    denominator = len(gains)
    helped, hurt, tied = (
        int(np.count_nonzero(gains > 0)),
        int(np.count_nonzero(gains < 0)),
        int(np.count_nonzero(gains == 0)),
    )
    return {
        "denominator": denominator,
        "helped": helped,
        "hurt": hurt,
        "tied": tied,
        "help_fraction": helped / denominator,
        "hurt_fraction": hurt / denominator,
        "tie_fraction": tied / denominator,
    }


def _gain(row, before, after, predicted=False):
    suffix = "predicted_error" if predicted else "relative_fro"
    return math.log1p(row[f"{before}_{suffix}"]) - math.log1p(row[f"{after}_{suffix}"])


def aggregate_heads(rows):
    """Return descriptive statistics and one point per physical query head.

    Error distributions use head/text rows, with a separate distribution of
    per-head arithmetic text means. Help/hurt uses mean per-text log1p gains,
    not log1p of a mean error. Ties use exact equality, without a tolerance.
    """
    if not rows:
        raise ValueError("heads.csv contains no measurements")
    groups = defaultdict(list)
    metadata = {}
    identities = set()
    for source in rows:
        row = dict(source)
        model = row["model"]
        identity = (row["family"], row["revision"])
        if model in metadata and metadata[model] != identity:
            raise ValueError(f"inconsistent family/revision for {model}")
        metadata[model] = identity
        row["layer"], row["head"] = _integer(row, "layer"), _integer(row, "head")
        key = (model, row["layer"], row["head"])
        measurement = (*key, row["text"])
        if measurement in identities:
            raise ValueError(f"duplicate head/text measurement: {measurement}")
        identities.add(measurement)
        for operand in ("q", "k"):
            field = f"{operand}_mean_energy_fraction"
            row[field] = _number(row, field)
            if row[field] > 1 + 1e-12:
                raise ValueError(f"{field} must be a fraction")
        for variant in VARIANTS:
            for suffix in ("relative_fro", "predicted_error"):
                field = f"{variant}_{suffix}"
                row[field] = _number(row, field)
        groups[key].append(row)

    points = []
    for (model, layer, head), records in sorted(groups.items()):
        records.sort(key=lambda row: row["text"])
        point = {
            "model": model,
            "family": metadata[model][0],
            "revision": metadata[model][1],
            "layer": layer,
            "head": head,
            "texts": [row["text"] for row in records],
            "text_count": len(records),
        }
        for operand in ("q", "k"):
            field = f"{operand}_mean_energy_fraction"
            point[field] = float(np.mean([row[field] for row in records]))
        for variant in VARIANTS:
            field = f"{variant}_relative_fro"
            point[field] = float(np.mean([row[field] for row in records]))
        for transform, (before, after) in TRANSFORMS.items():
            for predicted in (False, True):
                name = "predicted" if predicted else "observed"
                point[f"{transform}_{name}_gain"] = float(
                    np.mean([_gain(row, before, after, predicted) for row in records])
                )
        points.append(point)
    records = [row for key in sorted(groups) for row in groups[key]]

    def summarize(subset, head_points):
        return {
            "counts": {
                "models": len({row["model"] for row in subset}),
                "layers": len({(row["model"], row["layer"]) for row in subset}),
                "physical_heads": len(head_points),
                "texts": len({row["text"] for row in subset}),
                "head_text_measurements": len(subset),
                "prediction_datapoints_per_panel": len(head_points),
                "texts_per_head": _stats([point["text_count"] for point in head_points]),
            },
            "mean_energy_fraction": {
                operand: {
                    "head_text_measurements": _stats(
                        [row[f"{operand}_mean_energy_fraction"] for row in subset]
                    ),
                    "physical_head_text_means": _stats(
                        [point[f"{operand}_mean_energy_fraction"] for point in head_points]
                    ),
                }
                for operand in ("q", "k")
            },
            "actual_relative_fro": {
                variant: {
                    "head_text_measurements": _stats(
                        [row[f"{variant}_relative_fro"] for row in subset]
                    ),
                    "physical_head_text_means": _stats(
                        [point[f"{variant}_relative_fro"] for point in head_points]
                    ),
                }
                for variant in VARIANTS
            },
            "transforms": {
                name: {
                    "before": before,
                    "after": after,
                    "head_text_measurements": _fractions(
                        [_gain(row, before, after) for row in subset]
                    ),
                    "physical_head_mean_gains": _fractions(
                        [point[f"{name}_observed_gain"] for point in head_points]
                    ),
                }
                for name, (before, after) in TRANSFORMS.items()
            },
        }

    summary = summarize(records, points)
    summary["per_model"] = {
        model: {
            "family": metadata[model][0],
            "revision": metadata[model][1],
            **summarize(
                [row for row in records if row["model"] == model],
                [point for point in points if point["model"] == model],
            ),
        }
        for model in sorted(metadata)
    }
    return summary, points


def aggregate_downstream(rows):
    """Token-weighted CE/KL; expCE is exp(weighted CE), not mean(expCE)."""
    if not rows:
        raise ValueError("downstream.csv contains no measurements")
    groups = defaultdict(lambda: defaultdict(dict))
    metadata = {}
    for row in rows:
        model, variant, text = row["model"], row["variant"], row["text"]
        if variant not in DOWNSTREAM_VARIANTS:
            raise ValueError(f"unknown downstream variant: {variant}")
        identity = (row["family"], row["revision"])
        if model in metadata and metadata[model] != identity:
            raise ValueError(f"inconsistent family/revision for {model}")
        metadata[model] = identity
        if text in groups[model][variant]:
            raise ValueError(f"duplicate downstream measurement: {model}/{variant}/{text}")
        groups[model][variant][text] = {
            "tokens": _integer(row, "tokens", positive=True),
            **{field: _number(row, field) for field in ("ce", "exp_ce", "mean_kl")},
        }
    result = {}
    for model, variants in sorted(groups.items()):
        if set(variants) != set(DOWNSTREAM_VARIANTS):
            raise ValueError(f"downstream requires all six variants for {model}")
        baseline_tokens = {text: row["tokens"] for text, row in variants["bf16"].items()}
        aggregates = {}
        for variant in DOWNSTREAM_VARIANTS:
            records = variants[variant]
            if {text: row["tokens"] for text, row in records.items()} != baseline_tokens:
                raise ValueError(f"downstream text/token coverage differs for {model}/{variant}")
            tokens = sum(baseline_tokens.values())
            aggregate = {"tokens": tokens, "texts": len(records)}
            for field in ("ce", "mean_kl", "exp_ce"):
                name = "token_weighted_mean_exp_ce" if field == "exp_ce" else field
                aggregate[name] = (
                    math.fsum(
                        records[text][field] * records[text]["tokens"] for text in sorted(records)
                    )
                    / tokens
                )
            aggregate["exp_ce"] = math.exp(aggregate["ce"])
            aggregates[variant] = aggregate
        baseline = aggregates["bf16"]
        for aggregate in aggregates.values():
            aggregate["ce_delta_from_bf16"] = aggregate["ce"] - baseline["ce"]
            aggregate["ce_relative_change_from_bf16"] = (
                aggregate["ce_delta_from_bf16"] / baseline["ce"] if baseline["ce"] else None
            )
            aggregate["exp_ce_relative_change_from_bf16"] = math.expm1(
                aggregate["ce_delta_from_bf16"]
            )
        result[model] = {
            "family": metadata[model][0],
            "revision": metadata[model][1],
            "texts": sorted(baseline_tokens),
            "variants": aggregates,
        }
    return result


def _save(fig, path, plt):
    fig.savefig(path, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)


def draw_figures(points, downstream, results_dir):
    """Draw every aggregated physical head; SVG retains searchable text."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({point["model"] for point in points})
    colors = {model: COLORS[index % len(COLORS)] for index, model in enumerate(models)}
    with plt.rc_context(
        {
            "svg.fonttype": "none",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "savefig.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        for ax, transform in zip(axes, ("rotation", "smoothing"), strict=True):
            xfield, yfield = f"{transform}_predicted_gain", f"{transform}_observed_gain"
            all_values = [point[field] for point in points for field in (xfield, yfield)]
            lower, upper = min(0.0, min(all_values)), max(0.0, max(all_values))
            padding = (upper - lower) * 0.07 or 0.01
            limits = (lower - padding, upper + padding)
            for model in models:
                subset = [point for point in points if point["model"] == model]
                ax.scatter(
                    [point[xfield] for point in subset],
                    [point[yfield] for point in subset],
                    s=15,
                    alpha=0.6,
                    color=colors[model],
                    linewidths=0,
                    label=f"{model} ({len(subset)} heads)",
                )
            ax.plot(limits, limits, color="#222222", linestyle="--", linewidth=1, label="1:1")
            ax.axhline(0, color="#999999", linewidth=0.6)
            ax.axvline(0, color="#999999", linewidth=0.6)
            ax.set(
                xlim=limits,
                ylim=limits,
                aspect="equal",
                xlabel="Predicted gain (dimensionless log1p difference)",
                ylabel="Observed gain (dimensionless log1p difference)",
                title=f"{transform.capitalize()}: tile → {LABELS[TRANSFORMS[transform][1]]}",
            )
            ax.grid(alpha=0.15)
        axes[1].legend(loc="best", fontsize=8, framealpha=0.9)
        measurements = sum(point["text_count"] for point in points)
        texts = len({text for point in points for text in point["texts"]})
        fig.suptitle(f"All {len(points)} physical heads · {measurements} head/text measurements")
        fig.text(
            0.5,
            0.01,
            "Gain = log1p(E_tile) − log1p(E_transform); E = relative Frobenius error. "
            "Positive = helps.\n"
            f"One point per model/layer/query head; {texts} saved texts averaged within heads. "
            "Dependent observations; no confidence claims.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.1, 1, 0.94))
        _save(fig, results_dir / "prediction.svg", plt)

        columns = min(3, len(models))
        panel_rows = math.ceil(len(models) / columns)
        fig, axes = plt.subplots(
            panel_rows, columns, figsize=(4.2 * columns, 3.3 * panel_rows), squeeze=False
        )
        for ax, model in zip(axes.flat, models, strict=False):
            subset = [point for point in points if point["model"] == model]
            layers = sorted({point["layer"] for point in subset})
            for index, variant in enumerate(VARIANTS):
                values = [
                    float(
                        np.mean(
                            [
                                point[f"{variant}_relative_fro"]
                                for point in subset
                                if point["layer"] == layer
                            ]
                        )
                    )
                    * 100
                    for layer in layers
                ]
                ax.plot(
                    layers,
                    values,
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    color=COLORS[index],
                    label=LABELS[variant],
                )
            ax.set(
                title=f"{model} · {len(subset)} heads",
                xlabel="Layer (zero-based)",
                ylabel="Mean relative Frobenius error (%)",
            )
            ax.grid(alpha=0.2)
        for ax in list(axes.flat)[len(models) :]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle("Actual attention errors against FP64 by layer")
        fig.text(
            0.5,
            0.01,
            "Equal-weight text means within heads, then equal-weight means across "
            "all heads in each layer. Descriptive, dependent observations.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        _save(fig, results_dir / "layers.svg", plt)

        if downstream is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
            x = np.arange(len(DOWNSTREAM_VARIANTS))
            fields = ("ce_delta_from_bf16", "exp_ce_relative_change_from_bf16", "mean_kl")
            labels = (
                "CE − BF16 CE (nats/token)",
                "expCE change from BF16 (%)",
                "Mean KL from BF16 (nats/token)",
            )
            for ax, field, label in zip(axes, fields, labels, strict=True):
                for model, values in downstream.items():
                    ys = [values["variants"][variant][field] for variant in DOWNSTREAM_VARIANTS]
                    if field == "exp_ce_relative_change_from_bf16":
                        ys = [100 * value for value in ys]
                    ax.plot(
                        x, ys, "o-", markersize=4, linewidth=1.2, color=colors[model], label=model
                    )
                ax.axvspan(-0.25, 0.25, color="#dddddd", alpha=0.5)
                ax.axhline(0, color="#555555", linewidth=0.8)
                ax.set_xticks(
                    x, [LABELS[variant] for variant in DOWNSTREAM_VARIANTS], rotation=40, ha="right"
                )
                ax.set_ylabel(label)
                ax.grid(axis="y", alpha=0.2)
            axes[-1].legend(fontsize=8)
            fig.suptitle("Held-out batch teacher forcing · token-weighted text aggregates")
            coverage = "; ".join(
                f"{model}: {values['variants']['bf16']['tokens']} tokens / "
                f"{len(values['texts'])} texts"
                for model, values in downstream.items()
            )
            fig.text(0.5, 0.035, coverage, ha="center", fontsize=8, wrap=True)
            fig.text(
                0.5,
                0.005,
                "expCE = exp(token-weighted CE), not generation perplexity. "
                "Baseline shaded; batch calibration may use later tokens.",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.1, 1, 0.94))
            _save(fig, results_dir / "downstream.svg", plt)


def _load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/v2"))
    args = parser.parse_args(argv)
    directory = args.results_dir
    heads, points = aggregate_heads(_load_csv(directory / "heads.csv"))
    downstream_path, fit_path = directory / "downstream.csv", directory / "fit.json"
    downstream = (
        aggregate_downstream(_load_csv(downstream_path)) if downstream_path.exists() else None
    )
    if downstream is not None:
        for model, values in downstream.items():
            if model not in heads["per_model"]:
                raise ValueError(f"downstream model absent from heads.csv: {model}")
            head_model = heads["per_model"][model]
            if any(values[field] != head_model[field] for field in ("family", "revision")):
                raise ValueError(f"downstream family/revision disagrees with heads.csv: {model}")
    summary = {
        "sources": {
            "heads": "heads.csv",
            "downstream": "downstream.csv" if downstream else None,
            "fit": "fit.json" if fit_path.exists() else None,
        },
        "definitions": {
            "relative_fro": "||O_variant - O_FP64||_F / ||O_FP64||_F (dimensionless)",
            "gain": "mean_text(log1p(E_before) - log1p(E_after)); positive means lower error",
            "physical_head": "unique (model, layer, query head); texts are not independent heads",
            "layer_count": "unique (model, layer), not unique numeric layer indices",
            "p95": "NumPy linear 0.95 quantile, descriptive only",
            "tie": "exact zero gain, no tolerance",
            "downstream": "token-weighted CE and mean KL; expCE = exp(aggregated CE)",
        },
        "limitations": LIMITATIONS,
        "heads": heads,
        "downstream": downstream,
    }
    if fit_path.exists():
        with fit_path.open(encoding="utf-8") as stream:
            summary["fit"] = json.load(stream)
    draw_figures(points, downstream, directory)
    with (directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
