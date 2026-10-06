"""Analyze BF16 captures without loading weights; checkpoint whole layers only.

Run ``python -m study.head_table --model KEY --work-dir PATH --layers 4
--seconds 540``. Each external MODEL/head_table/layer_NN.csv contains every
query head of every pinned text. Repeating the command completes the next layers,
then rebuilds the portable results/v2/heads.csv from available complete chunks.
``python -m study.head_table --fit`` evaluates that possibly partial table with
qwen-only calibration. Neither full-K calibration nor these causal batch errors
are measurements of streaming/generation perplexity.
"""

import argparse
import csv
import fcntl
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from attention import reference
from study.attention import apply_attention
from study.data import ROOT, model_spec, models, texts
from study.prediction import VARIANTS, predict_head, summarize_fit

CAPTURE_TOKENS = 1024
RESULTS_DIR = ROOT / "results/v2"
DESIGN_PATH = ROOT / "data/v2/design.json"
IDENTITY_FIELDS = ("model", "family", "revision", "text", "layer", "head", "kv_head")
INTEGER_FIELDS = {"layer", "head", "kv_head", "n", "d", "value_dimension"}
PREFIX_FIELDS = (*IDENTITY_FIELDS, "n", "d", "scale")


def _atomic_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fieldnames(row):
    return [*PREFIX_FIELDS, *sorted(set(row) - set(PREFIX_FIELDS))]


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"invalid CSV header: {path}")
        rows = []
        for source in reader:
            if None in source or any(value is None or value == "" for value in source.values()):
                raise ValueError(f"incomplete CSV row: {path}")
            row = {}
            for key, value in source.items():
                if key in IDENTITY_FIELDS[:4]:
                    row[key] = value
                elif key in INTEGER_FIELDS or key.endswith("_nonfinite"):
                    row[key] = int(value)
                else:
                    row[key] = float(value)
                if isinstance(row[key], int | float) and not math.isfinite(row[key]):
                    raise ValueError(f"nonfinite CSV field {key}: {path}")
            rows.append(row)
    return fields, rows


def _dimensions(spec):
    config = spec["config"]
    return (
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
        config["num_hidden_layers"],
    )


def _validate_chunk(path, spec, layer, text_keys):
    fields, rows = _read_csv(path)
    required = {
        *PREFIX_FIELDS,
        "reference_nonfinite",
        *(
            f"{variant}_{suffix}"
            for variant in VARIANTS
            for suffix in (
                "predicted_relative_mse",
                "predicted_error",
                "relative_fro",
                "max_abs",
                "output_nonfinite",
            )
        ),
    }
    if not required.issubset(fields):
        raise ValueError(f"missing head-table columns: {path}")
    heads, kvheads, d, _ = _dimensions(spec)
    expected = {(text, head) for text in text_keys for head in range(heads)}
    seen = set()
    for row in rows:
        identity = (row["text"], row["head"])
        if identity in seen or identity not in expected:
            raise ValueError(f"duplicate or unexpected head/text in layer chunk: {path}")
        seen.add(identity)
        if (
            any(
                row[key] != spec[key if key != "model" else "key"]
                for key in (
                    "model",
                    "family",
                    "revision",
                )
            )
            or row["layer"] != layer
        ):
            raise ValueError(f"layer chunk identity differs from pinned model: {path}")
        if (
            row["kv_head"] != row["head"] // (heads // kvheads)
            or row["n"] != CAPTURE_TOKENS
            or row["d"] != d
            or row["scale"] <= 0
        ):
            raise ValueError(f"layer chunk dimensions or GQA mapping differ: {path}")
        if any(row[key] != 0 for key in fields if key.endswith("_nonfinite")):
            raise ValueError(f"nonfinite outputs recorded in layer chunk: {path}")
        for variant in VARIANTS:
            if any(
                row[f"{variant}_{suffix}"] < 0
                for suffix in (
                    "predicted_relative_mse",
                    "predicted_error",
                    "relative_fro",
                    "max_abs",
                )
            ):
                raise ValueError(f"negative error in layer chunk: {path}")
    if seen != expected:
        raise ValueError(
            f"incomplete layer chunk: {path}; expected {len(expected)} rows, got {len(rows)}"
        )
    rows.sort(key=lambda row: (row["text"], row["head"]))
    return fields, rows


def _load_capture(path, spec):
    if not path.is_file():
        raise FileNotFoundError(f"missing capture: {path}")
    heads, kvheads, d, _ = _dimensions(spec)
    shapes = ((heads, CAPTURE_TOKENS, d), (kvheads, CAPTURE_TOKENS, d))
    with np.load(path, allow_pickle=False) as capture:
        tensors = []
        for name, shape in zip(("q", "k", "v"), (shapes[0], shapes[1], shapes[1]), strict=True):
            array = capture[name]
            if array.dtype != np.uint16 or array.shape != shape:
                raise ValueError(f"{path}: {name} must contain uint16 BF16 bits with shape {shape}")
            tensor = torch.from_numpy(array).view(torch.bfloat16)
            if any(not torch.isfinite(head).all().item() for head in tensor):
                raise ValueError(f"{path}: nonfinite {name}")
            tensors.append(tensor)
        stored_scale = capture["scale"]
        if stored_scale.shape != () or stored_scale.dtype != np.float64:
            raise ValueError(f"{path}: scale must be a float64 scalar")
        scale = float(stored_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{path}: scale must be finite and positive")
    return (*tensors, scale)


def _capture_rows(path, spec, layer, text):
    q, k, v, scale = _load_capture(path, spec)
    heads, kvheads, n, d = q.shape[0], k.shape[0], q.shape[1], q.shape[2]
    group = heads // kvheads
    # Bound reference/output operands to <=8 heads and approximately 8 MiB.
    # Never expand the full layer, create N*N probabilities, or sample heads.
    chunk = max(1, min(8, (8 * 1024 * 1024) // (n * d * 8)))
    for kv_head in range(kvheads):
        kn, vn = k[kv_head].float().numpy(), v[kv_head].float().numpy()
        first_head = kv_head * group
        for first in range(first_head, first_head + group, chunk):
            last = min(first + chunk, first_head + group)
            rows, references = [], []
            for head in range(first, last):
                qn = q[head].float().numpy()
                expected = reference(qn, kn, vn, causal=True, scale=scale)
                if not np.isfinite(expected).all():
                    raise ValueError(f"{path}: nonfinite FP64 reference for head {head}")
                # Only original operands and independent reference enter prediction;
                # measured FP8 outputs do not exist yet.
                features = predict_head(qn, kn, vn, reference_output=expected, scale=scale)
                if not all(np.isfinite(value) for value in features.values()):
                    raise ValueError(f"{path}: nonfinite predictor feature for head {head}")
                rows.append(
                    {
                        "model": spec["key"],
                        "family": spec["family"],
                        "revision": spec["revision"],
                        "text": text,
                        "layer": layer,
                        "head": head,
                        "kv_head": kv_head,
                        **features,
                        "reference_nonfinite": 0,
                    }
                )
                references.append(expected)
            # Every slice stays inside one KV group, preserving original GQA.
            for variant in VARIANTS:
                actual = (
                    apply_attention(
                        q[first:last].unsqueeze(0),
                        k[kv_head : kv_head + 1].unsqueeze(0),
                        v[kv_head : kv_head + 1].unsqueeze(0),
                        variant,
                        scale=scale,
                    )[0]
                    .float()
                    .numpy()
                )
                if not np.isfinite(actual).all():
                    raise ValueError(f"{path}: nonfinite {variant} output in heads {first}:{last}")
                for offset, (row, expected) in enumerate(zip(rows, references, strict=True)):
                    difference = actual[offset].astype(np.float64) - expected
                    norm, error = float(np.linalg.norm(expected)), float(np.linalg.norm(difference))
                    if norm == 0 and error != 0:
                        raise ValueError(
                            f"{path}: relative error undefined for zero reference norm"
                        )
                    row[f"{variant}_relative_fro"] = error / norm if norm else 0.0
                    row[f"{variant}_max_abs"] = float(np.max(np.abs(difference)))
                    row[f"{variant}_output_nonfinite"] = 0
                del actual
            yield from rows


def combine_tables(work_dir, results_dir=RESULTS_DIR):
    """Rebuild the wide table from whole-layer chunks, including partial models.

    Only one layer's rows are resident at a time. An external advisory lock
    serializes rebuilds by independent model processes sharing the work directory.
    """
    work_dir, results_dir = Path(work_dir), Path(results_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    text_keys = sorted(text["key"] for text in texts())
    with (work_dir / "head_table.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        chunks, fields = [], None
        for spec in sorted(models(), key=lambda item: item["key"]):
            directory = work_dir / spec["key"] / "head_table"
            for path in sorted(directory.glob("layer_*.csv")):
                try:
                    layer = int(path.stem.removeprefix("layer_"))
                except ValueError as error:
                    raise ValueError(f"invalid layer chunk filename: {path.name}") from error
                if path.name != f"layer_{layer:02d}.csv" or not 0 <= layer < _dimensions(spec)[3]:
                    raise ValueError(f"unexpected layer chunk: {path}")
                current, _ = _validate_chunk(path, spec, layer, text_keys)
                if fields is None:
                    fields = current
                elif current != fields:
                    raise ValueError(f"inconsistent head-table columns: {path}")
                chunks.append((spec, layer, path))
        count = 0

        def combined_rows():
            nonlocal count
            for spec, layer, path in sorted(chunks, key=lambda item: (item[0]["key"], item[1])):
                _, rows = _validate_chunk(path, spec, layer, text_keys)
                count += len(rows)
                yield from rows

        _atomic_csv(results_dir / "heads.csv", fields or PREFIX_FIELDS, combined_rows())
    return {"combined_layers": len(chunks), "combined_rows": count}


def analyze_model(model, work_dir, layers=4, seconds=540, results_dir=RESULTS_DIR):
    """Complete at most `layers` new layers; check the deadline at boundaries.

    Missing pending captures raise explicitly. A layer interrupted during analysis
    has no checkpoint; earlier complete chunks survive and remain in heads.csv.
    Counts describe this model and the combined table, not timing measurements.
    """
    if layers <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("layers and seconds must be positive")
    start = time.monotonic()
    spec = model_spec(model)
    total = _dimensions(spec)[3]
    text_keys = sorted(text["key"] for text in texts())
    directory = Path(work_dir) / model
    chunks = directory / "head_table"
    chunks.mkdir(parents=True, exist_ok=True)
    complete = []
    for layer in range(total):
        path = chunks / f"layer_{layer:02d}.csv"
        if path.exists():
            _validate_chunk(path, spec, layer, text_keys)
            complete.append(layer)
    counts = combine_tables(work_dir, results_dir)
    new_layers = 0
    stop_reason = "complete"
    for layer in range(total):
        if layer in complete:
            continue
        if new_layers >= layers:
            stop_reason = "layer_limit"
            break
        if time.monotonic() - start >= seconds:
            stop_reason = "deadline"
            break
        captures = [directory / "capture" / f"layer_{layer:02d}_{text}.npz" for text in text_keys]
        missing = [str(path) for path in captures if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing captures for pending layer: " + ", ".join(missing))
        rows = []
        for text, path in zip(text_keys, captures, strict=True):
            rows.extend(_capture_rows(path, spec, layer, text))
        _atomic_csv(chunks / f"layer_{layer:02d}.csv", _fieldnames(rows[0]), rows)
        complete.append(layer)
        new_layers += 1
        counts = combine_tables(work_dir, results_dir)
    return {
        "model": model,
        "total_layers": total,
        "completed_layers": len(complete),
        "new_layers": new_layers,
        "remaining_layers": total - len(complete),
        "completed_rows": len(complete) * len(text_keys) * _dimensions(spec)[0],
        "complete": len(complete) == total,
        "stop_reason": stop_reason,
        **counts,
    }


def fit_table(results_dir=RESULTS_DIR):
    """Fit available wide rows in relative-MSE units; only qwen trains calibration."""
    results_dir = Path(results_dir)
    _, wide = _read_csv(results_dir / "heads.csv")
    long = []
    for row in wide:
        identity = {key: row[key] for key in (*IDENTITY_FIELDS, "n", "d", "scale")}
        for variant in VARIANTS:
            long.append(
                {
                    **identity,
                    "variant": variant,
                    "predicted_mse": row[f"{variant}_predicted_relative_mse"],
                    "observed_mse": row[f"{variant}_relative_fro"] ** 2,
                }
            )
    design = json.loads(DESIGN_PATH.read_text())
    summary = summarize_fit(
        long,
        train_family=design["calibration_family"],
        development_models=design["development_models"],
        evaluation_models=design["evaluation_models"],
    )
    # summarize_fit's reported identities omit revision/KV fields; retain full
    # evaluated input identities here without changing the predictor's contract.
    summary["input_table"] = {
        "path": "heads.csv",
        "head_text_rows": len(wide),
        "variant_rows": len(long),
        "identity_fields": list(IDENTITY_FIELDS),
        "models": sorted({(row["model"], row["family"], row["revision"]) for row in wide}),
        "completed_layers": len({(row["model"], row["layer"]) for row in wide}),
    }
    content = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=results_dir,
            prefix=".fit.json.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, results_dir / "fit.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[spec["key"] for spec in models()])
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--layers", type=int, default=4, help="Maximum newly completed layers")
    parser.add_argument(
        "--seconds", type=float, default=540, help="Deadline checked between layers"
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--fit", action="store_true", help="Fit the available combined table")
    args = parser.parse_args(argv)
    result = {}
    if args.model is not None:
        if args.work_dir is None:
            parser.error("--model requires --work-dir")
        result = analyze_model(
            args.model,
            args.work_dir,
            args.layers,
            args.seconds,
            args.results_dir,
        )
    elif not args.fit:
        parser.error("provide --model and --work-dir, or --fit")
    if args.fit:
        summary = fit_table(args.results_dir)
        result["fit"] = summary["input_table"]
        result["calibration"] = summary["calibration"]
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
