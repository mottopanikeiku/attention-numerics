"""Validate complete multiple-choice runs and report paired accuracy changes.

python -m study.accuracy.report --results-dir results/accuracy
"""

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("bf16", "tile", "rotate", "smooth_k", "rotate_smooth_k")
LIMITATIONS = [
    "Zero-shot standard harness prompts, without chat templates; not a chat evaluation.",
    "ARC-Challenge uses its full test split; HellaSwag and MMLU use fixed 2,000-item "
    "samples, uniform and subject-stratified respectively, not their full test populations.",
    "Intervals resample these items, not models, hardware, prompts or dataset selection.",
    "No multiple-comparison correction; harm flags are not confirmatory population proof.",
    "These multiple-choice results do not measure generation quality or perplexity.",
    "Full-sequence quantization scales and key means can use later tokens; these are "
    "batch multiple-choice scores, not streaming-generation measurements.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_plan(plan):
    """Use the committed plan, never infer coverage from the available runs."""
    _require(isinstance(plan, dict), "Plan must be an object")
    models = plan.get("models")
    count = plan.get("planned_model_count", 6)
    _require(
        _integer(count, 1) and isinstance(models, list) and len(models) == count,
        f"Plan needs {count} planned models",
    )
    keys = []
    for model in models:
        _require(isinstance(model, dict), "Plan model must be an object")
        key, revision = model.get("key"), model.get("revision")
        _require(
            isinstance(key, str) and key and Path(key).name == key and key not in (".", ".."),
            "Invalid model key",
        )
        _require(isinstance(revision, str) and revision, "Missing model revision")
        keys.append(key)
    _require(len(set(keys)) == len(keys), "Duplicate plan model")
    tasks = plan.get("tasks")
    _require(isinstance(tasks, dict) and len(tasks) == 3, "Plan needs three tasks")
    for task, spec in tasks.items():
        _require(isinstance(task, str) and task, "Invalid task key")
        _require(isinstance(spec, dict), "Task specification must be an object")
        _require(_integer(spec.get("count"), 1), "Task count must be positive")
        _require(spec.get("metric") in ("acc", "acc_norm"), "Unknown primary metric")
    _require(
        {task: spec["metric"] for task, spec in tasks.items()}
        == {"arc_challenge": "acc_norm", "hellaswag": "acc_norm", "mmlu": "acc"},
        "Plan must name ARC/HellaSwag acc_norm and MMLU acc",
    )
    _require(plan.get("variants") == list(VARIANTS), "Plan variants differ from the five variants")
    bootstrap, rule = plan.get("bootstrap"), plan.get("harm_rule")
    _require(
        isinstance(bootstrap, dict)
        and bootstrap.get("replicates") == 20000
        and bootstrap.get("seed") == 20261007
        and bootstrap.get("interval", 0.95) == 0.95,
        "Plan bootstrap differs from the fixed 20,000-replicate 95% interval and seed",
    )
    _require(
        isinstance(rule, dict)
        and rule.get("minimum_drop") == 0.02
        and rule.get("interval_upper_below") == 0,
        "Plan harm rule differs from the fixed 2pp and upper < 0 rule",
    )


def validate_manifest(plan, manifest):
    """Return frozen items keyed by task and item ID, rejecting duplicate IDs."""
    _require(isinstance(manifest, dict), "Item manifest must be an object")
    _require(
        type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
        "Unknown manifest schema_version",
    )
    items = manifest.get("items")
    _require(isinstance(items, list), "Item manifest needs an items list")
    indexed = {}
    counts = Counter()
    for item in items:
        _require(isinstance(item, dict), "Manifest item must be an object")
        task, item_id = item.get("task"), item.get("item_id")
        _require(isinstance(task, str) and task in plan["tasks"], "Unknown manifest task")
        _require(isinstance(item_id, str) and item_id, "Invalid manifest item ID")
        identity = (task, item_id)
        _require(identity not in indexed, f"Duplicate manifest item: {identity}")
        choices = item.get("choices")
        _require(
            isinstance(choices, list)
            and len(choices) >= 2
            and all(isinstance(choice, str) and len(choice) > 0 for choice in choices),
            f"Invalid manifest choices: {identity}",
        )
        _require(
            _integer(item.get("gold")) and item["gold"] < len(choices),
            f"Invalid manifest gold: {identity}",
        )
        _require(
            isinstance(item.get("context"), str) and item["context"],
            f"Invalid manifest context: {identity}",
        )
        _require(
            item.get("metric") == plan["tasks"][task]["metric"],
            f"Manifest primary metric mismatch: {identity}",
        )
        _require(
            item.get("normalization_lengths") == [len(choice) for choice in choices],
            f"Manifest character normalization lengths mismatch: {identity}",
        )
        indexed[identity] = item
        counts[task] += 1
    _require(
        counts == {task: spec["count"] for task, spec in plan["tasks"].items()},
        "Manifest task coverage differs from plan counts",
    )
    return indexed


def validate_run(run, model, plan, items, plan_sha256, item_manifest_sha256):
    """Require one correctly scored row per frozen item and variant."""
    _require(isinstance(run, dict), "Raw run must be an object")
    _require(
        type(run.get("schema_version")) is int and run["schema_version"] == 1,
        "Unknown raw schema_version",
    )
    _require(run.get("model") == model["key"], "Raw model differs from plan")
    _require(run.get("revision") == model["revision"], "Raw revision differs from plan")
    _require(run.get("plan_sha256") == plan_sha256, "Raw plan hash mismatch")
    _require(
        run.get("item_manifest_sha256") == item_manifest_sha256, "Raw item manifest hash mismatch"
    )
    _require(run.get("variants") == plan["variants"], "Raw variants differ from plan")
    _require(isinstance(run.get("runtime"), dict), "Missing raw runtime metadata")
    _require(isinstance(run.get("adapter"), dict), "Missing raw adapter metadata")
    _require(isinstance(run.get("rows"), list), "Missing raw rows")
    indexed = {}
    for row in run["rows"]:
        _require(isinstance(row, dict), "Raw row must be an object")
        task, item_id, variant = row.get("task"), row.get("item_id"), row.get("variant")
        _require(isinstance(task, str) and isinstance(item_id, str), "Invalid raw item ID")
        _require(isinstance(variant, str) and variant in VARIANTS, "Unknown raw variant")
        identity = (task, item_id)
        _require(identity in items, f"Unknown raw item: {identity}")
        key = (*identity, variant)
        _require(key not in indexed, f"Duplicate raw row: {key}")
        item = items[identity]
        nchoices = len(item["choices"])
        _require(row.get("model") == model["key"], "Row model mismatch")
        _require(
            _integer(row.get("gold")) and row["gold"] == item["gold"], f"Row gold mismatch: {key}"
        )
        _require(
            _integer(row.get("prediction")) and row["prediction"] < nchoices,
            f"Invalid prediction: {key}",
        )
        _require(
            type(row.get("correct")) is bool
            and row["correct"] == (row["prediction"] == row["gold"]),
            f"Correctness disagrees with gold/prediction: {key}",
        )
        for field in ("choice_log_likelihoods", "choice_scores"):
            values = row.get(field)
            _require(
                isinstance(values, list)
                and len(values) == nchoices
                and all(_finite(value) for value in values),
                f"Invalid finite {field}: {key}",
            )
        for field in ("normalization_lengths", "choice_token_counts", "sequence_lengths"):
            values = row.get(field)
            _require(
                isinstance(values, list)
                and len(values) == nchoices
                and all(_integer(value, 1) for value in values),
                f"Invalid {field}: {key}",
            )
        _require(_integer(row.get("context_tokens"), 1), f"Invalid context_tokens: {key}")
        _require(
            all(
                total == row["context_tokens"] + tokens - 1
                for total, tokens in zip(
                    row["sequence_lengths"], row["choice_token_counts"], strict=True
                )
            ),
            f"Sequence lengths disagree with context and continuation: {key}",
        )
        lengths = [len(choice) for choice in item["choices"]]
        _require(
            row["normalization_lengths"] == lengths,
            f"Character normalization lengths mismatch: {key}",
        )
        likelihoods = row["choice_log_likelihoods"]
        expected_scores = (
            [ll / length for ll, length in zip(likelihoods, lengths, strict=True)]
            if plan["tasks"][task]["metric"] == "acc_norm"
            else likelihoods
        )
        _require(
            all(
                math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                for actual, expected in zip(row["choice_scores"], expected_scores, strict=True)
            ),
            f"Scores disagree with primary metric normalization: {key}",
        )
        _require(
            row["prediction"] == int(np.argmax(row["choice_scores"]))
            and row["prediction"] == int(np.argmax(expected_scores)),
            f"Prediction is not first score argmax: {key}",
        )
        indexed[key] = row
    expected = {(*identity, variant) for identity in items for variant in VARIANTS}
    _require(
        set(indexed) == expected,
        f"Incomplete raw coverage for {model['key']}: "
        f"missing {len(expected - indexed.keys())}, extra {len(indexed.keys() - expected)}",
    )
    # The coverage check gives each variant exactly the BF16 IDs; lengths must also
    # describe the same model-tokenized choices, independent of the attention kernel.
    for identity in items:
        baseline = indexed[(*identity, "bf16")]
        for variant in VARIANTS[1:]:
            row = indexed[(*identity, variant)]
            for field in ("context_tokens", "choice_token_counts", "sequence_lengths"):
                _require(
                    row[field] == baseline[field], f"Variant tokenization mismatch: {identity}"
                )
    return indexed


def stream_seed(seed, model, task, variant):
    """Stable independent streams, unaffected by iteration order or Python hash salt."""
    payload = json.dumps([seed, model, task, variant], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def paired_change(baseline, candidate, seed, replicates=20000):
    """Percentile item bootstrap via exact (-1, 0, +1) multinomial counts.

    Each draw is the category counts of n sampled paired items. This is identical
    in distribution to resampling item indices and needs only replicates x 3 space.
    """
    baseline, candidate = np.asarray(baseline), np.asarray(candidate)
    _require(
        baseline.ndim == candidate.ndim == 1
        and baseline.size == candidate.size
        and baseline.size > 0,
        "Paired samples must be nonempty, equal-length vectors",
    )
    _require(baseline.dtype.kind == candidate.dtype.kind == "b", "Paired samples must be boolean")
    _require(_integer(replicates, 1), "Replicates must be positive")
    difference = candidate.astype(np.int8) - baseline.astype(np.int8)
    n = difference.size
    counts = np.bincount(difference + 1, minlength=3)
    draws = np.random.default_rng(seed).multinomial(n, counts / n, size=replicates)
    deltas = (draws[:, 2] - draws[:, 0]) / n
    low, high = np.quantile(deltas, [0.025, 0.975], method="linear")
    return {
        "denominator": int(n),
        "baseline_accuracy": float(baseline.mean()),
        "accuracy": float(candidate.mean()),
        "delta": int(counts[2] - counts[0]) / n,
        "ci95": [float(low), float(high)],
        "beneficial": int(counts[2]),
        "harmed": int(counts[0]),
        "tied": int(counts[1]),
    }


def summarize(plan, items, runs):
    expected_models = {model["key"] for model in plan["models"]}
    _require(set(runs) == expected_models, "Incomplete model coverage")
    rows = []
    bootstrap = plan["bootstrap"]
    rule = plan["harm_rule"]
    for model in plan["models"]:
        key = model["key"]
        for task, spec in plan["tasks"].items():
            identities = sorted(identity for identity in items if identity[0] == task)
            baseline = [runs[key][(*identity, "bf16")]["correct"] for identity in identities]
            for variant in plan["variants"]:
                seed = stream_seed(bootstrap["seed"], key, task, variant)
                candidate = [runs[key][(*identity, variant)]["correct"] for identity in identities]
                change = paired_change(baseline, candidate, seed, bootstrap["replicates"])
                change.update(
                    {
                        "model": key,
                        "revision": model["revision"],
                        "task": task,
                        "variant": variant,
                        "primary_metric": spec["metric"],
                        "bootstrap_seed": seed,
                        "harm_flag": change["delta"] <= -rule["minimum_drop"]
                        and change["ci95"][1] < rule["interval_upper_below"],
                    }
                )
                rows.append(change)
    return {
        "schema_version": 1,
        "complete": True,
        "baseline": "bf16",
        "difference_units": "accuracy fraction; multiply by 100 for percentage points",
        "interval_method": "paired item percentile bootstrap via exact multinomial categories",
        "bootstrap": bootstrap,
        "harm_rule": rule,
        "primary_metrics": {task: spec["metric"] for task, spec in plan["tasks"].items()},
        "denominators": {task: spec["count"] for task, spec in plan["tasks"].items()},
        "sample_limitations": LIMITATIONS,
        "rows": rows,
    }


def paired_figure(summary):
    """Small deterministic SVG: every model/task/nonbaseline variant, no raster axes."""
    rows = [row for row in summary["rows"] if row["variant"] != "bf16"]
    low = min(-3, *(100 * row["ci95"][0] for row in rows))
    high = max(1, *(100 * row["ci95"][1] for row in rows))
    step = max(1, math.ceil((high - low) / 10))
    low, high = math.floor(low / step) * step, math.ceil(high / step) * step
    left, right, top, spacing = 490, 950, 102, 23
    height = top + spacing * len(rows) + 70

    def x(value):
        return left + (value - low) / (high - low) * (right - left)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" '
        f'viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Paired multiple-choice accuracy changes from BF16</title>',
        '<desc id="desc">Points show changes in percentage points; lines show 95% paired '
        "item bootstrap intervals. Dashed orange line marks minus two percentage points. "
        "Red markers meet both the drop threshold and interval upper bound below zero.</desc>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#222">',
        '<text x="20" y="28" font-size="19">'
        "Accuracy change from BF16, all planned comparisons</text>",
        '<text x="20" y="50">95% paired bootstrap intervals; '
        "red = drop ≥ 2pp and upper &lt; 0</text>",
        '<text x="20" y="77">Model / task / variant</text>',
        '<text x="990" y="77">Δ (pp) / n</text>',
    ]
    bottom = top + (len(rows) - 1) * spacing + 12
    for tick in range(low, high + 1, step):
        at = x(tick)
        svg.append(f'<line x1="{at:.2f}" x2="{at:.2f}" y1="87" y2="{bottom}" stroke="#ececec"/>')
        svg.append(f'<text x="{at:.2f}" y="{bottom + 24}" text-anchor="middle">{tick}</text>')
    for threshold, color, dash in ((0, "#222", ""), (-2, "#bb6500", ' stroke-dasharray="5 4"')):
        at = x(threshold)
        svg.append(
            f'<line x1="{at:.2f}" x2="{at:.2f}" y1="87" y2="{bottom}" stroke="{color}"{dash}/>'
        )
    previous = None
    for index, row in enumerate(rows):
        y = top + index * spacing
        if previous is not None and row["model"] != previous:
            svg.append(f'<line x1="20" x2="1095" y1="{y - 14}" y2="{y - 14}" stroke="#bbb"/>')
        previous = row["model"]
        label = escape(f"{row['model']} / {row['task']} / {row['variant']}")
        color = "#b52332" if row["harm_flag"] else "#176b8a"
        a, b = (x(100 * value) for value in row["ci95"])
        point = x(100 * row["delta"])
        svg.extend(
            [
                f'<text x="20" y="{y + 4}">{label}</text>',
                f'<path d="M {a:.2f} {y} H {b:.2f} M {a:.2f} {y - 4} V {y + 4} '
                f'M {b:.2f} {y - 4} V {y + 4}" stroke="{color}" fill="none"/>',
                f'<circle cx="{point:.2f}" cy="{y}" r="3" fill="{color}"/>',
                f'<text x="990" y="{y + 4}">{100 * row["delta"]:+.2f} / '
                f"{row['denominator']}</text>",
            ]
        )
    svg.append(
        f'<text x="720" y="{bottom + 49}" text-anchor="middle">'
        "Accuracy change (percentage points; negative is worse)</text></g></svg>"
    )
    return "\n".join(svg) + "\n"


def _load(path, compressed=False):
    raw = Path(path).read_bytes()
    payload = gzip.decompress(raw) if compressed else raw
    return json.loads(payload), hashlib.sha256(raw).hexdigest()


def build_report(results_dir, plan_path, items_path, additional_plans=()):
    """Write summary and figure only after every planned raw file passes validation."""
    results_dir = Path(results_dir)
    plan, plan_hash = _load(plan_path)
    manifest, items_hash = _load(items_path, compressed=True)
    validate_plan(plan)
    items = validate_manifest(plan, manifest)
    expected_files = {f"{model['key']}.json.gz" for model in plan["models"]}
    pilot_files = {f"pilot-{model['key']}.json.gz" for model in plan["models"]}
    actual_files = {
        path.name for path in results_dir.glob("*.json.gz") if path.name not in pilot_files
    }
    _require(
        actual_files == expected_files,
        f"Incomplete model file coverage: missing {sorted(expected_files - actual_files)}, "
        f"extra {sorted(actual_files - expected_files)}",
    )
    runs, evidence = {}, []
    for model in plan["models"]:
        filename = f"{model['key']}.json.gz"
        run, raw_hash = _load(results_dir / filename, compressed=True)
        runs[model["key"]] = validate_run(run, model, plan, items, plan_hash, items_hash)
        evidence.append(
            {
                "model": model["key"],
                "path": filename,
                "sha256": raw_hash,
                "runtime": run["runtime"],
                "adapter": run["adapter"],
                "plan_sha256": plan_hash,
            }
        )
    summary = summarize(plan, items, runs)
    summary.update(
        {"plan_sha256": plan_hash, "item_manifest_sha256": items_hash, "primary_evidence": evidence}
    )
    for additional_path in additional_plans:
        additional, additional_hash = _load(additional_path)
        _require(
            additional.get("extends_plan_sha256") == plan_hash,
            "Additional plan must explicitly extend the original plan hash",
        )
        for field in (
            "variants",
            "tasks",
            "sample_seed",
            "special_tokens",
            "harness",
            "scoring",
            "bootstrap",
            "harm_rule",
            "kernel",
            "item_manifest",
        ):
            _require(additional.get(field) == plan.get(field), f"Additional plan changed {field}")
        subdirectory = additional.get("results_subdirectory")
        _require(
            isinstance(subdirectory, str)
            and subdirectory not in ("", ".", "..")
            and Path(subdirectory).name == subdirectory,
            "Additional plan must name its result subdirectory",
        )
        extra = build_report(results_dir / subdirectory, additional_path, items_path)
        existing_models = {row["model"] for row in summary["rows"]}
        extra_models = {row["model"] for row in extra["rows"]}
        _require(existing_models.isdisjoint(extra_models), "Repeated model across plans")
        summary["rows"].extend(extra["rows"])
        for entry in extra["primary_evidence"]:
            summary["primary_evidence"].append({**entry, "path": f"{subdirectory}/{entry['path']}"})
        summary.setdefault("additional_plans", []).append(
            {"path": Path(additional_path).name, "sha256": additional_hash}
        )
    figure = paired_figure(summary)
    _require(len(figure.encode()) < 400000, "Paired change figure exceeds 400KB")
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (results_dir / "summary.json").write_text(payload)
    (results_dir / "paired-change.svg").write_text(figure)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "accuracy")
    parser.add_argument("--plan", type=Path, default=ROOT / "data" / "accuracy" / "plan.json")
    parser.add_argument("--items", type=Path, default=ROOT / "data" / "accuracy" / "items.json.gz")
    parser.add_argument("--additional-plan", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        build_report(args.results_dir, args.plan, args.items, args.additional_plan)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
