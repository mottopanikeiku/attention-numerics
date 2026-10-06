"""Reproduce the pinned study in resumable, bounded-duration invocations."""

import argparse
import json
import time
from pathlib import Path

import torch
from huggingface_hub import try_to_load_from_cache

from study.classification import main as classification_main
from study.data import ROOT, download, models, work_directory
from study.downstream import combine, score
from study.head_table import analyze_model, fit_table
from study.report import main as report_main
from study.run import capture, downstream
from study.sinks import analyze_model as analyze_sinks
from study.validate import fast, native_qwen, storage


def inputs_ready(spec, cache):
    marker = cache / spec["key"] / "verified_revision.txt"
    present = all(
        isinstance(
            try_to_load_from_cache(spec["model_id"], file["path"], revision=spec["revision"]), str
        )
        for file in spec["files"]
    )
    return present and marker.exists() and marker.read_text().strip() == spec["revision"]


def reproduce(selected, cache, seconds=540, layers=4):
    deadline = time.monotonic() + seconds
    validation = ROOT / "results/v2/validation"
    storage(validation)
    statuses = []
    for spec in selected:
        key = spec["key"]
        if time.monotonic() >= deadline:
            return {"complete": False, "models_finished": statuses, "next_model": key}
        if not inputs_ready(spec, cache):
            download(key)
            marker = cache / key / "verified_revision.txt"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(spec["revision"] + "\n")
        native_marker = cache / key / "native_validated.txt"
        if key == "qwen05" and not native_marker.exists():
            native_qwen(validation)
            native_marker.write_text(spec["revision"] + "\n")
        for stage in (capture, analyze_model, analyze_sinks, downstream):
            while time.monotonic() < deadline:
                state = stage(
                    key, cache, layers=layers, seconds=max(1, deadline - time.monotonic())
                )
                if state["complete"]:
                    break
            else:
                return {
                    "complete": False,
                    "models_finished": statuses,
                    "next_model": key,
                    "next_stage": stage.__name__,
                }
        if time.monotonic() >= deadline:
            return {
                "complete": False,
                "models_finished": statuses,
                "next_model": key,
                "next_stage": "score",
            }
        if not (cache / key / "downstream/metrics.csv").exists():
            score(key, cache)
        fast_marker = cache / key / "fast_validated.txt"
        if not fast_marker.exists():
            fast(key, cache, validation)
            fast_marker.write_text(spec["revision"] + "\n")
        statuses.append(key)
    combine(cache, ROOT / "results/v2/downstream.csv")
    fit_table()
    report_main(["--results-dir", str(ROOT / "results/v2")])
    classification_main(["--results-dir", str(ROOT / "results/v2")])
    return {"complete": True, "models_finished": statuses}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["all", *[spec["key"] for spec in models()]], default="all"
    )
    parser.add_argument("--work-dir", type=Path, default=work_directory())
    parser.add_argument("--seconds", type=float, default=540)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.seconds, args.layers, args.threads) <= 0:
        parser.error("seconds, layers and threads must be positive")
    torch.set_num_threads(args.threads)
    selected = (
        models()
        if args.model == "all"
        else [spec for spec in models() if spec["key"] == args.model]
    )
    print(json.dumps(reproduce(selected, args.work_dir, args.seconds, args.layers)), flush=True)
