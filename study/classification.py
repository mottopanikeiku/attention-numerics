"""Rotation-only diagnostics: python -m study.classification --results-dir results/v2.

Descriptive physical-head statistics, not independent-observation inference. The
thresholds are fixed before prospective evaluation; this module fits nothing.
Both input tables must contain all three texts for every reported physical head.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from study.prediction import _metrics, _ranks
from study.report import _integer, _number

EVALUATION_MODELS = ("smol17", "tiny11", "olmo1")
DEVELOPMENT_MODELS = ("qwen05", "smol036")
DEFAULT_DESIGN = Path(__file__).resolve().parents[1] / "data/v2/design.json"
MATCH_FIELDS = ("family", "revision", "kv_head", "n", "d", "scale")
ERROR_FIELDS = (
    "tile_relative_fro",
    "rotate_relative_fro",
    "tile_predicted_error",
    "rotate_predicted_error",
)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _identity(source):
    row = dict(source)
    for field in ("model", "family", "revision", "text"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"{field} must be a nonempty string")
    for field in ("layer", "head", "kv_head", "n", "d"):
        row[field] = _integer(row, field, positive=field in ("n", "d"))
    row["scale"] = _number(row, "scale")
    if row["scale"] == 0:
        raise ValueError("scale must be positive")
    return (row["model"], row["layer"], row["head"], row["text"]), row


def _physical_heads(head_rows, sink_rows, design):
    heads, sinks, models = {}, {}, {}
    for source in head_rows:
        key, row = _identity(source)
        if key in heads:
            raise ValueError(f"duplicate head/text measurement: {key}")
        metadata = (row["family"], row["revision"])
        if row["model"] in models and models[row["model"]] != metadata:
            raise ValueError(f"inconsistent family/revision for {row['model']}")
        models[row["model"]] = metadata
        for field in ERROR_FIELDS:
            row[field] = _number(row, field)
        heads[key] = row
    for source in sink_rows:
        key, row = _identity(source)
        if key in sinks:
            raise ValueError(f"duplicate sink head/text measurement: {key}")
        if key not in heads:
            raise ValueError(f"sink diagnostic has no matching head/text: {key}")
        for field in MATCH_FIELDS:
            if row[field] != heads[key][field]:
                raise ValueError(f"sink {field} disagrees with head/text: {key}")
        start = _integer(row, "query_start")
        count = _integer(row, "query_count", positive=True)
        if row["n"] != 1024 or start != 128 or count != 896:
            raise ValueError("sink diagnostic must cover queries 128..1023 at n=1024")
        for field in ("prefix1_mass", "prefix4_mass"):
            row[field] = _number(row, field)
            if row[field] > 1:
                raise ValueError(f"{field} must lie in [0, 1]")
        if row["prefix1_mass"] > row["prefix4_mass"]:
            raise ValueError("prefix1_mass cannot exceed prefix4_mass")
        sinks[key] = row
    missing = heads.keys() - sinks.keys()
    if missing:
        raise ValueError(f"missing sink head/text diagnostic: {min(missing)}")

    groups = defaultdict(list)
    for key in sorted(heads):
        groups[key[:3]].append(heads[key])
    expected_texts = design.get("text_ids")
    if expected_texts is not None:
        if len(expected_texts) != 3 or len(set(expected_texts)) != 3:
            raise ValueError("design must specify exactly three distinct texts")
        expected_texts = set(expected_texts)
    points = []
    for (model, layer, head), rows in sorted(groups.items()):
        texts = {row["text"] for row in rows}
        if len(texts) != 3:
            raise ValueError(f"physical head requires all three texts: {(model, layer, head)}")
        if expected_texts is None:
            expected_texts = texts
        if texts != expected_texts:
            raise ValueError(f"physical head text set disagrees: {(model, layer, head)}")
        first = rows[0]
        for row in rows[1:]:
            if any(row[field] != first[field] for field in MATCH_FIELDS):
                raise ValueError(f"inconsistent physical-head metadata: {(model, layer, head)}")
        # Average log errors first, then subtract: never log an arithmetic error mean.
        logs = {
            field: math.fsum(math.log1p(row[field]) for row in rows) / 3 for field in ERROR_FIELDS
        }
        observed = logs["rotate_relative_fro"] - logs["tile_relative_fro"]
        predicted = logs["rotate_predicted_error"] - logs["tile_predicted_error"]
        sink_mass = (
            math.fsum(sinks[(model, layer, head, row["text"])]["prefix4_mass"] for row in rows) / 3
        )
        points.append(
            {
                "model": model,
                "layer": layer,
                "head": head,
                **{field: first[field] for field in MATCH_FIELDS},
                "texts": sorted(texts),
                "text_count": 3,
                "observed_hurt_score": observed,
                "predicted_hurt_score": predicted,
                "hurts": observed > 0,
                "helps": observed < 0,
                "ties": observed == 0,
                "predict_hurt": predicted > 0,
                "mean_prefix4_mass": sink_mass,
                "prefix_sink": sink_mass >= 0.5,
            }
        )
    return points, sorted(expected_texts) if expected_texts is not None else []


def _summary(points):
    n = len(points)
    positive = sum(point["hurts"] for point in points)
    negative = n - positive  # Exact observed ties belong to the non-hurt class.
    helps = sum(point["helps"] for point in points)
    ties = sum(point["ties"] for point in points)
    tp = sum(point["hurts"] and point["predict_hurt"] for point in points)
    fp = sum(not point["hurts"] and point["predict_hurt"] for point in points)
    fn, tn = positive - tp, negative - fp
    recall, specificity = _ratio(tp, positive), _ratio(tn, negative)
    auc = None
    if positive and negative:
        ranks = _ranks([point["predicted_hurt_score"] for point in points])
        positive_rank_sum = math.fsum(
            float(rank) for rank, point in zip(ranks, points, strict=True) if point["hurts"]
        )
        auc = (positive_rank_sum - positive * (positive - 1) / 2) / (positive * negative)
    sinks = [point for point in points if point["prefix_sink"]]
    sink_hurts = sum(point["hurts"] for point in sinks)
    return {
        "physical_heads": n,
        "head_text_rows": sum(point["text_count"] for point in points),
        "hurts": {"count": positive, "fraction": _ratio(positive, n)},
        "helps": {"count": helps, "fraction": _ratio(helps, n)},
        "ties": {"count": ties, "fraction": _ratio(ties, n)},
        "classification": {
            "threshold": 0.0,
            "comparison": "predicted_hurt_score > threshold",
            "positive_class": "rotation hurts",
            "negative_class": "rotation helps or ties",
            "majority_class": (
                None
                if not n
                else "both"
                if positive == negative
                else "hurt"
                if positive > negative
                else "non_hurt"
            ),
            "majority_class_accuracy": _ratio(max(positive, negative), n),
            "accuracy": _ratio(tp + tn, n),
            "balanced_accuracy": (recall + specificity) / 2 if positive and negative else None,
            "roc_auc": auc,
            "precision": _ratio(tp, tp + fp),
            "recall": recall,
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "specificity": specificity,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "denominators": {
                "accuracy": n,
                "majority_class_accuracy": n,
                "recall": positive,
                "specificity": negative,
                "precision": tp + fp,
                "f1": 2 * tp + fp + fn,
                "roc_auc_positive_negative_pairs": positive * negative,
                "balanced_accuracy": "mean(recall, specificity); both classes required",
            },
        },
        "log_gain": _metrics(
            [-point["observed_hurt_score"] for point in points],
            [-point["predicted_hurt_score"] for point in points],
        ),
        "sink_concentration": {
            "sink_heads": len(sinks),
            "non_sink_heads": n - len(sinks),
            "sink_fraction": _ratio(len(sinks), n),
            "hurting_sink_heads": sink_hurts,
            "hurting_non_sink_heads": positive - sink_hurts,
            "sink_hurt_rate": _ratio(sink_hurts, len(sinks)),
            "non_sink_hurt_rate": _ratio(positive - sink_hurts, n - len(sinks)),
            "fraction_of_hurting_heads_that_are_sinks": _ratio(sink_hurts, positive),
            "denominators": {
                "sink_fraction": n,
                "sink_hurt_rate": len(sinks),
                "non_sink_hurt_rate": n - len(sinks),
                "fraction_of_hurting_heads_that_are_sinks": positive,
            },
        },
    }


def _layers(points):
    grouped = defaultdict(list)
    for point in points:
        grouped[point["layer"]].append(point)
    layers = [
        {
            "layer": layer,
            "physical_heads": len(rows),
            "hurting_heads": sum(row["hurts"] for row in rows),
            "hurt_fraction": _ratio(sum(row["hurts"] for row in rows), len(rows)),
        }
        for layer, rows in sorted(grouped.items())
    ]
    top = sorted(layers, key=lambda row: (-row["hurting_heads"], row["layer"]))[:3]
    total = sum(row["hurting_heads"] for row in layers)
    return {
        "per_layer": layers,
        "top3_layers": [row["layer"] for row in top],
        "top3_hurting_heads": sum(row["hurting_heads"] for row in top),
        "hurting_heads_denominator": total,
        "top3_share_of_hurting_heads": _ratio(sum(row["hurting_heads"] for row in top), total),
    }


def rotation_report(head_rows, sink_rows, design):
    """Return all-head diagnostics without calibration, threshold search, or sampling.

    design supplies evaluation_models, development_models, calibration_family and
    optionally text_ids. Exactly three distinct, identical text sets are required
    for every physical query head.
    Undefined metrics use None; observed/predicted exact zeros are non-hurt.
    """
    evaluation = list(design.get("evaluation_models", EVALUATION_MODELS))
    development = list(design.get("development_models", DEVELOPMENT_MODELS))
    if set(evaluation) & set(development):
        raise ValueError("evaluation and development models must be disjoint")
    if len(set(evaluation)) != len(evaluation) or len(set(development)) != len(development):
        raise ValueError("model groups must not contain duplicates")
    points, texts = _physical_heads(head_rows, sink_rows, design)
    present = {point["model"] for point in points}
    per_model = {}
    for model in sorted(present):
        subset = [point for point in points if point["model"] == model]
        per_model[model] = {
            "family": subset[0]["family"],
            "revision": subset[0]["revision"],
            **_summary(subset),
            "layer_concentration": _layers(subset),
        }
    return {
        "sources": {"heads": "heads.csv", "sinks": "sinks.csv", "design": "data/v2/design.json"},
        "metadata": {
            "calibration_family": design.get("calibration_family", "qwen"),
            "calibration_models": list(design.get("calibration_models", ("qwen05", "qwen15"))),
            "evaluation_models": evaluation,
            "development_models": development,
            "evaluation_models_present": sorted(present & set(evaluation)),
            "evaluation_models_missing": sorted(set(evaluation) - present),
            "evaluation_complete": set(evaluation) <= present,
            "development_models_present": sorted(present & set(development)),
            "other_models_present": sorted(present - set(evaluation) - set(development)),
            "texts": texts,
            "required_texts_per_head": 3,
            "head_text_completeness": "three matched texts per reported head; identical text sets",
            "coverage": "all supplied heads; architecture-wide head coverage is not inferred",
            "evaluation_status": (
                "Prospective evaluation models designated before their outputs were analyzed. "
                "qwen05 and smol036 are development/pilot models, not untouched evaluation. "
                "Only designated evaluation models enter untouched_evaluation; absent models "
                "have no measurements and are listed explicitly."
            ),
            "limitations": [
                "Descriptive statistics only: texts, layers and GQA query heads are dependent; "
                "no confidence intervals or independent-observation significance claims.",
                "Prefix-sink is an operational diagnostic, not a paper-defined permanent "
                "head identity or a causal explanation of downstream harm.",
                "Local attention-output error only; no GPU, decoding or runtime-performance "
                "claims. Native BF16 reference uses Transformers SDPA, not eager.",
            ],
        },
        "definitions": {
            "physical_head": "unique (model, layer, query head); texts are averaged, not counted",
            "error": "relative Frobenius attention-output error versus ideal causal FP64",
            "observed_hurt_score": "mean_text(log1p(E_rotate)) - mean_text(log1p(E_tile))",
            "predicted_hurt_score": (
                "mean_text(log1p(E_rotate_pred)) - mean_text(log1p(E_tile_pred))"
            ),
            "hurt": "observed_hurt_score > 0; helps < 0; ties == 0 exactly, no tolerance",
            "predict_hurt": "predicted_hurt_score > 0; fixed threshold, no tuning",
            "fractions": "hurt/help/tie counts divided by physical_heads in that group",
            "roc_auc": "P(score_hurt > score_nonhurt) + 0.5 P(equal), all positive-negative pairs",
            "log_gain": "negative hurt score; existing rank/R2 helper, parameter-free, no refit",
            "r2": "1 - sum((observed_gain-predicted_gain)^2)/sum((observed_gain-mean)^2)",
            "prefix_sink": "mean_text(prefix4_mass) >= 0.5; fixed operational threshold",
            "sink_reference": "ideal causal FP64 probabilities from native captured BF16 "
            "post-RoPE Q/K; mean prefix masses over queries 128..1023 inclusive",
            "top3_layers": "three largest per-model hurt counts (or all if fewer); "
            "ties ordered by layer index; share denominator is all hurting heads",
            "undefined": "null for zero denominators, empty groups, constant rank/target "
            "variance, or AUC/balanced accuracy without both binary classes",
        },
        "overall": _summary(points),
        "per_model": per_model,
        "untouched_evaluation": _summary([p for p in points if p["model"] in evaluation]),
        "development": _summary([p for p in points if p["model"] in development]),
        "physical_heads": points,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/v2"))
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    args = parser.parse_args(argv)
    directory = args.results_dir
    with args.design.open(encoding="utf-8") as stream:
        design = json.load(stream)
    tables = []
    for name in ("heads.csv", "sinks.csv"):
        with (directory / name).open(newline="", encoding="utf-8") as stream:
            tables.append(list(csv.DictReader(stream)))
    result = rotation_report(*tables, design)
    with (directory / "rotation_risk.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
