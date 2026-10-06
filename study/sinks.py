"""Mathematical FP64 prefix-mass diagnostics from native post-RoPE BF16 captures.

Run ``python -m study.sinks --model KEY --work-dir PATH --layers 4 --seconds 520``.
Every completed layer includes all query heads and all three pinned texts. The
external MODEL/sinks/layer_NN.csv chunks rebuild results/v2/sinks.csv on resume.
No weights, V tensors, or native attention output errors enter this calculation.

Production means cover queries 128..1023 (an explicit 128-query warmup), with
full causal keys and the captured scaling. The classifier's operational
prefix-sink label uses a physical head's mean prefix4_mass over three texts >=0.5.
This chosen descriptive diagnostic is neither a permanent paper-defined head
identity nor a causal explanation of rotation harm. Heads/texts/layers, including
shared GQA keys, are dependent; the table is not an independent-sample estimate.
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

from study.data import ROOT, model_spec, models, texts

CAPTURE_TOKENS = 1024
QUERY_START = 128
SOFTMAX_ROWS = 32
RESULTS_DIR = ROOT / "results/v2"
IDENTITY_FIELDS = ("model", "family", "revision", "text")
INTEGER_FIELDS = ("layer", "head", "kv_head", "n", "d", "query_start", "query_count")
FIELDS = (
    *IDENTITY_FIELDS,
    "layer",
    "head",
    "kv_head",
    "n",
    "d",
    "scale",
    "query_start",
    "query_count",
    "prefix1_mass",
    "prefix4_mass",
)


def _fp64(array):
    """Expand BF16 storage bits exactly, or accept already numeric test operands."""
    array = np.asarray(array)
    if array.dtype == np.uint16:
        # BF16 is the upper half of an IEEE binary32 word, including subnormals.
        words = np.left_shift(array.astype(np.uint32), 16)
        return words.view(np.float32).astype(np.float64)
    if array.dtype.kind != "f":
        raise ValueError("Q/K must be floating-point operands or uint16 BF16 bits")
    return array.astype(np.float64, copy=False)


def prefix_mass(q, k, scale, query_start=128, prefix_tokens=4):
    """Return query_start/count and mean prefix1_mass/prefix4_mass for one head.

    Q/K are [N,D] arrays, either uint16 BF16 storage bits or floating-point values.
    Scores and softmax use FP64 throughout; at most 32 query rows are resident.
    Short unit inputs may choose any query_start in [0,N). ``prefix4_mass`` names
    the requested prefix width (four by default); early causal rows automatically
    include only prefix tokens already visible to that query.
    """
    q, k = _fp64(q), _fp64(k)
    if q.ndim != 2 or k.shape != q.shape or min(q.shape) <= 0:
        raise ValueError("Q/K must have matching nonempty [N,D] shapes")
    n = q.shape[0]
    if not isinstance(query_start, int | np.integer) or not 0 <= query_start < n:
        raise ValueError("query_start must be an integer in [0,N)")
    if not isinstance(prefix_tokens, int | np.integer) or prefix_tokens < 1:
        raise ValueError("prefix_tokens must be a positive integer")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError("nonfinite Q/K operands cannot define finite probabilities")
    prefix1, prefix4 = 0.0, 0.0
    for start in range(query_start, n, SOFTMAX_ROWS):
        stop = min(start + SOFTMAX_ROWS, n)
        # Full causal history, never just a prefix or a key-window approximation.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            probabilities = q[start:stop] @ k[:stop].T
            probabilities *= scale
            if not np.isfinite(probabilities).all():
                raise ValueError(f"nonfinite probabilities: nonfinite FP64 scores at query {start}")
            positions = np.arange(start, stop)[:, None]
            probabilities[np.arange(stop)[None, :] > positions] = -np.inf
            probabilities -= probabilities.max(axis=1, keepdims=True)
            np.exp(probabilities, out=probabilities)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"nonfinite probabilities at query {start}")
        prefix1 += float(probabilities[:, 0].sum())
        prefix4 += float(probabilities[:, :prefix_tokens].sum())
    count = n - query_start
    return {
        "query_start": int(query_start),
        "query_count": int(count),
        "prefix1_mass": prefix1 / count,
        "prefix4_mass": prefix4 / count,
    }


def _dimensions(spec):
    config = spec["config"]
    heads, kvheads = config["num_attention_heads"], config["num_key_value_heads"]
    d, layers = config["head_dim"], config["num_hidden_layers"]
    if min(heads, kvheads, d, layers) <= 0 or heads % kvheads:
        raise ValueError(f"invalid pinned head dimensions or GQA grouping: {spec['key']}")
    return heads, kvheads, d, layers


def _atomic_csv(path, rows):
    """Use the head-table convention: same-directory temporary and atomic replace."""
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
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_chunk(path, spec, layer, text_keys):
    heads, kvheads, d, _ = _dimensions(spec)
    expected = {(text, head) for text in text_keys for head in range(heads)}
    seen, rows = set(), []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"invalid sinks CSV header: {path}")
        for source in reader:
            if None in source or any(value is None or value == "" for value in source.values()):
                raise ValueError(f"incomplete CSV row: {path}")
            row = {key: source[key] for key in IDENTITY_FIELDS}
            for key in INTEGER_FIELDS:
                row[key] = int(source[key])
            for key in ("scale", "prefix1_mass", "prefix4_mass"):
                row[key] = float(source[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f"nonfinite CSV field {key}: {path}")
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
                or row["query_start"] != QUERY_START
                or row["query_count"] != CAPTURE_TOKENS - QUERY_START
            ):
                raise ValueError(f"layer chunk dimensions, warmup or GQA mapping differ: {path}")
            if not 0 <= row["prefix1_mass"] <= row["prefix4_mass"] <= 1 + 1e-12:
                raise ValueError(f"invalid prefix probabilities in layer chunk: {path}")
            rows.append(row)
    if seen != expected:
        raise ValueError(
            f"incomplete layer chunk: {path}; expected {len(expected)} rows, got {len(rows)}"
        )
    rows.sort(key=lambda row: (row["text"], row["head"]))
    return rows


def _load_capture(path, spec):
    if not path.is_file():
        raise FileNotFoundError(f"missing capture: {path}")
    heads, kvheads, d, _ = _dimensions(spec)
    with np.load(path, allow_pickle=False) as capture:
        arrays = []
        for name, count in (("q", heads), ("k", kvheads)):
            if name not in capture:
                raise ValueError(f"{path}: missing captured {name}")
            array = capture[name]
            shape = (count, CAPTURE_TOKENS, d)
            if array.dtype != np.uint16 or array.shape != shape:
                raise ValueError(f"{path}: {name} must contain uint16 BF16 bits with shape {shape}")
            # Exponent all-ones identifies both NaN and infinity without expanding a layer.
            if np.any(np.bitwise_and(array, 0x7F80) == 0x7F80):
                raise ValueError(f"{path}: nonfinite {name}; probabilities are undefined")
            arrays.append(array)
        if "scale" not in capture:
            raise ValueError(f"{path}: missing captured scale")
        stored_scale = capture["scale"]
        if stored_scale.shape != () or stored_scale.dtype != np.float64:
            raise ValueError(f"{path}: scale must be a float64 scalar")
        scale = float(stored_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{path}: scale must be finite and positive")
    return (*arrays, scale)


def _capture_rows(path, spec, layer, text):
    q, k, scale = _load_capture(path, spec)
    group = q.shape[0] // k.shape[0]
    for kv_head in range(k.shape[0]):
        # Expand one key head, reused by its query group, not a full FP64 layer.
        keys = _fp64(k[kv_head])
        for head in range(kv_head * group, (kv_head + 1) * group):
            try:
                masses = prefix_mass(q[head], keys, scale, query_start=QUERY_START)
            except ValueError as error:
                raise ValueError(f"{path}: head {head}: {error}") from error
            yield {
                "model": spec["key"],
                "family": spec["family"],
                "revision": spec["revision"],
                "text": text,
                "layer": layer,
                "head": head,
                "kv_head": kv_head,
                "n": q.shape[1],
                "d": q.shape[2],
                "scale": scale,
                **masses,
            }


def combine_tables(work_dir, results_dir=RESULTS_DIR):
    """Atomically rebuild all available complete layers, ordered without duplicates."""
    work_dir, results_dir = Path(work_dir), Path(results_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    text_keys = sorted(text["key"] for text in texts())
    with (work_dir / "sinks.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        chunks = []
        for spec in sorted(models(), key=lambda item: item["key"]):
            for path in sorted((work_dir / spec["key"] / "sinks").glob("layer_*.csv")):
                try:
                    layer = int(path.stem.removeprefix("layer_"))
                except ValueError as error:
                    raise ValueError(f"invalid layer chunk filename: {path.name}") from error
                if path.name != f"layer_{layer:02d}.csv" or not 0 <= layer < _dimensions(spec)[3]:
                    raise ValueError(f"unexpected layer chunk: {path}")
                _validate_chunk(path, spec, layer, text_keys)
                chunks.append((spec, layer, path))
        count = 0

        def combined_rows():
            nonlocal count
            for spec, layer, path in sorted(chunks, key=lambda item: (item[0]["key"], item[1])):
                rows = _validate_chunk(path, spec, layer, text_keys)
                count += len(rows)
                yield from rows

        _atomic_csv(results_dir / "sinks.csv", combined_rows())
    return {"combined_layers": len(chunks), "combined_rows": count}


def analyze_model(model, work_dir, layers=4, seconds=520, results_dir=RESULTS_DIR):
    """Resume whole layers; limits are checked only between complete layers.

    Missing pending captures raise explicitly. Previously completed layers stay
    published; an interrupted or invalid layer never receives a CSV checkpoint.
    Returned counts are bookkeeping, not runtime-performance measurements.
    """
    if layers <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("layers and seconds must be positive")
    start = time.monotonic()
    spec = model_spec(model)
    total = _dimensions(spec)[3]
    text_keys = sorted(text["key"] for text in texts())
    if len(text_keys) != 3 or len(set(text_keys)) != 3:
        raise ValueError("sink diagnostics require exactly three distinct pinned texts")
    directory = Path(work_dir) / model
    chunks = directory / "sinks"
    chunks.mkdir(parents=True, exist_ok=True)
    complete = set()
    for layer in range(total):
        path = chunks / f"layer_{layer:02d}.csv"
        if path.exists():
            _validate_chunk(path, spec, layer, text_keys)
            complete.add(layer)
    counts = combine_tables(work_dir, results_dir)
    new_layers, stop_reason = 0, "complete"
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
        _atomic_csv(chunks / f"layer_{layer:02d}.csv", rows)
        complete.add(layer)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=[spec["key"] for spec in models()])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4, help="Maximum newly completed layers")
    parser.add_argument(
        "--seconds", type=float, default=520, help="Deadline checked between layers"
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    result = analyze_model(args.model, args.work_dir, args.layers, args.seconds, args.results_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
