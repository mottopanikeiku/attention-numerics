"""Generate SVG figures and median/range tables from the committed raw CSV files."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sweep import environment

ROOT = Path("results")
COLORS = ["#34495e", "#0072b2", "#d55e00", "#cc79a7", "#009e73", "#e69f00"]
LABELS = {
    "fp32": "FP32",
    "bf16": "BF16",
    "e4_tensor": "E4M3 / tensor",
    "e5_tensor": "E5M2 / tensor",
    "e4_tile": "E4M3 / tile",
    "e4_rotate": "E4M3 / tile + rotation",
    "e4_reduced": "reduced14",
    "e4_promoted": "promote 128",
    "e4_compensated": "denominator Kahan",
    "e4_reverse": "reverse tiles",
    "e4_local": "local-max update",
    "e4_no_p_round": "FP32 probabilities",
    "e4_no_output_round": "FP32 output",
}


def load(study):
    with (ROOT / f"{study}.csv").open() as stream:
        return list(csv.DictReader(stream))


def statistic(rows, metric="relative_frobenius", percent=True):
    values = np.array([float(row[metric]) for row in rows]) * (100 if percent else 1)
    return float(np.median(values)), float(values.min()), float(values.max())


def select(rows, **kwargs):
    return [row for row in rows if all(row[key] == str(value) for key, value in kwargs.items())]


def save(fig, name):
    fig.savefig(ROOT / name, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)


def length_plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    variants = ["fp32", "bf16", "e4_tensor", "e5_tensor", "e4_tile", "e4_rotate"]
    for ax, scenario in zip(axes, ["normal", "outliers"], strict=True):
        subset = select(rows, d=64, causal=False, scenario=scenario)
        for color, variant in zip(COLORS, variants, strict=True):
            ns = sorted({int(row["n"]) for row in subset})
            stats = [statistic(select(subset, n=n, variant=variant)) for n in ns]
            median, lower, upper = np.array(stats).T
            ax.plot(ns, median, "o-", color=color, label=LABELS[variant], linewidth=1.6)
            ax.fill_between(ns, lower, upper, color=color, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ns, ["1k", "4k", "16k", "64k"])
        ax.set_xlabel("Keys in each query's context")
        ax.set_title("Gaussian σ=1" if scenario == "normal" else "One Q/K/V channel ×8")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Relative Frobenius error (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("CPU emulation · d=64, non-causal · medians and seed ranges")
    fig.text(
        0.5,
        -0.02,
        "1k: every query; longer contexts: 128 queries in four fixed blocks",
        ha="center",
        fontsize=9,
    )
    save(fig, "length.svg")


def fixes_plot(rows):
    variants = [
        "e4_tensor",
        "e4_reduced",
        "e4_promoted",
        "e4_compensated",
        "e4_reverse",
        "e4_local",
        "e4_no_p_round",
        "e4_no_output_round",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, scenario in zip(axes, ["normal", "outliers"], strict=True):
        subset = select(rows, n=65536, d=64, causal=False, scenario=scenario)
        stats = np.array([statistic(select(subset, variant=variant)) for variant in variants])
        median, lower, upper = stats.T
        ax.barh(np.arange(len(variants)), median, color="#0072b2", alpha=0.8)
        ax.errorbar(
            median,
            np.arange(len(variants)),
            xerr=[median - lower, upper - median],
            fmt="none",
            color="#222222",
            capsize=3,
        )
        ax.set_yticks(np.arange(len(variants)), [LABELS[x] for x in variants])
        ax.invert_yaxis()
        ax.set_xlabel("Relative Frobenius error (%)")
        ax.set_title("Gaussian σ=1" if scenario == "normal" else "One Q/K/V channel ×8")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("E4M3 tensor scaling · 64k keys, 128 query rows · median / seed range")
    fig.tight_layout()
    save(fig, "fixes.svg")


def tile_plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    variants = ["e4_tensor", "e4_tile", "e4_reduced", "e4_promoted"]
    for ax, d in zip(axes, [64, 128], strict=True):
        subset = select(rows, n=65536, d=d, causal=False)
        for color, variant in zip(COLORS, variants, strict=False):
            tiles = [32, 128, 512]
            stats = np.array(
                [statistic(select(subset, tile=tile, variant=variant)) for tile in tiles]
            )
            median, lower, upper = stats.T
            ax.plot(tiles, median, "o-", color=color, label=LABELS[variant])
            ax.fill_between(tiles, lower, upper, color=color, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xticks(tiles, [str(tile) for tile in tiles])
        ax.set_xlabel("Key tile / PV reduction size")
        ax.set_title(f"d={d}")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Relative Frobenius error (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Gaussian σ=1 · 64k keys, 128 non-causal queries · CPU emulation")
    fig.tight_layout()
    fig.text(
        0.5,
        -0.03,
        "Lines: seed medians; shading: observed min–max, not confidence intervals",
        ha="center",
        fontsize=9,
    )
    save(fig, "tiles.svg")


def dot_plot(rows):
    fig, ax = plt.subplots(figsize=(7, 4))
    fp32_zeros = []
    for color, variant in zip(COLORS, ["fp32", "reduced14", "promoted128"], strict=False):
        sizes = sorted({int(row["k"]) for row in rows})
        stats = np.array([statistic(select(rows, k=size, variant=variant)) for size in sizes])
        median, lower, upper = stats.T
        if variant == "fp32":
            fp32_zeros = [str(k) for k, value in zip(sizes, median, strict=True) if value == 0]
        # A logarithmic axis cannot represent zero; never invent a positive error floor.
        ax.plot(sizes, np.ma.masked_equal(median, 0), "o-", color=color, label=variant)
        ax.fill_between(
            sizes, lower, upper, where=(lower > 0) & (upper > 0), color=color, alpha=0.15
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Inner reduction dimension K (not attention sequence length)")
    ax.set_ylabel("Relative Frobenius error (%)")
    ax.set_title("Already-quantized FP8 products · reduced14 surrogate · 4×K @ K×4")
    ax.grid(alpha=0.2)
    ax.legend()
    if fp32_zeros:
        ax.text(
            0.03,
            0.30,
            f"FP32 medians: exactly 0 at K={','.join(fp32_zeros)}\n"
            "Zero points/bounds omitted on log axes",
            transform=ax.transAxes,
            fontsize=8,
        )
    fig.text(
        0.5,
        -0.03,
        "Lines: seed medians; shading: observed min–max, not confidence intervals",
        ha="center",
        fontsize=9,
    )
    save(fig, "dots.svg")


def real_plot(rows):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    variants = ["bf16", "e4_tensor", "e4_tile", "e4_rotate", "torch_sdpa_bf16"]
    names = [LABELS.get(variant, "PyTorch CPU / BF16") for variant in variants]
    heads = [(0, 0), (0, 7), (12, 0), (12, 7)]
    for offset, (layer, head) in enumerate(heads):
        values = [
            statistic(select(rows, n=1024, layer=layer, head=head, variant=variant))[0]
            for variant in variants
        ]
        ax.plot(
            np.arange(len(variants)),
            values,
            "o-",
            color=COLORS[offset],
            label=f"layer {layer}, head {head}",
            alpha=0.85,
        )
    ax.set_xticks(np.arange(len(variants)), names, rotation=15, ha="right")
    ax.set_ylabel("Relative Frobenius error (%)")
    ax.set_yscale("log")
    ax.set_title("Actual Qwen2.5-0.5B-Instruct BF16 operands · 1024 causal tokens")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.text(
        0.5,
        -0.04,
        "One public-domain text, four heads; prefix activations captured on CPU, not GPU",
        ha="center",
        fontsize=9,
    )
    save(fig, "real.svg")


def summarize(studies):
    entries = []
    keys = [
        "study",
        "n",
        "d",
        "causal",
        "scenario",
        "tile",
        "scale_multiplier",
        "variant",
        "rows_evaluated",
        "all_rows",
    ]
    for rows in studies:
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row[key] for key in keys)].append(row)
        for group, members in grouped.items():
            result = dict(zip(keys, group, strict=True))
            result["seeds"] = [int(row["seed"]) for row in members]
            for metric in ["relative_frobenius", "max_abs", "worst_row_l2"]:
                median, lower, upper = statistic(members, metric, percent=False)
                result[metric] = {"median": median, "min": lower, "max": upper}
            result["worst_rows_by_seed"] = {row["seed"]: int(row["worst_row"]) for row in members}
            entries.append(result)
    (ROOT / "summary.json").write_text(json.dumps(entries, indent=2) + "\n")
    with (ROOT / "table.csv").open("w", newline="") as stream:
        fields = [
            "study",
            "n",
            "d",
            "causal",
            "scenario",
            "variant",
            "rows_evaluated",
            "relative_percent_median",
            "relative_percent_min",
            "relative_percent_max",
            "max_abs_median",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            if (
                entry["n"] == "65536"
                and entry["d"] == "64"
                and entry["causal"] == "False"
                and entry["scenario"] in ["normal", "outliers"]
                and entry["study"] in ["length", "fixes"]
            ):
                stats = entry["relative_frobenius"]
                writer.writerow(
                    {
                        **{key: entry[key] for key in fields[:7]},
                        "relative_percent_median": 100 * stats["median"],
                        "relative_percent_min": 100 * stats["min"],
                        "relative_percent_max": 100 * stats["max"],
                        "max_abs_median": entry["max_abs"]["median"],
                    }
                )


if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "attention-numerics", "font.family": "DejaVu Sans"})
    studies = []
    for study, plot in [("length", length_plot), ("fixes", fixes_plot), ("tiles", tile_plot)]:
        rows = load(study)
        studies.append(rows)
        plot(rows)
    for study in ["softmax", "full", "full64"]:
        studies.append(load(study))
    dot_plot(load("dots"))
    real_plot(load("real"))
    summarize(studies)
    (ROOT / "machine.json").write_text(json.dumps(environment(), indent=2) + "\n")
