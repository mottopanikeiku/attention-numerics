"""Reproducible error experiments; no runtime claims are collected here."""

import argparse
import csv
import json
import os
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

import ml_dtypes
import numpy as np

from attention import Config, emulate, matmul, metrics, quantize, reference

CONFIGS = {
    "fp32": Config(storage="fp32", output="fp32"),
    "fp32_rotate": Config(storage="fp32", output="fp32", rotate=True),
    "bf16": Config(),
    "e4_tensor": Config(storage="e4m3"),
    "e5_tensor": Config(storage="e5m2"),
    "e4_tile": Config(storage="e4m3", scaling="tile"),
    "e4_rotate": Config(storage="e4m3", scaling="tile", rotate=True),
    "e4_rotate_no_p_round": Config(storage="e4m3", scaling="tile", rotate=True, probability="fp32"),
    "e4_reduced": Config(storage="e4m3", accumulator="reduced14"),
    "e4_promoted": Config(storage="e4m3", accumulator="reduced14", promote=128),
    "e4_compensated": Config(storage="e4m3", compensated=True),
    "e4_reverse": Config(storage="e4m3", order="reverse"),
    "e4_local": Config(storage="e4m3", update="local"),
    "e4_no_p_round": Config(storage="e4m3", probability="fp32"),
    "e4_no_output_round": Config(storage="e4m3", output="fp32"),
    "fp32_compensated": Config(storage="fp32", output="fp32", compensated=True),
    "fp32_reverse": Config(storage="fp32", output="fp32", order="reverse"),
    "fp32_local": Config(storage="fp32", output="fp32", update="local"),
}
SCENARIOS = {"flat": 0.25, "normal": 1.0, "sharp": 2.0, "outliers": 1.0}


def make_inputs(n, d, seed, scenario):
    """Independent standard-normal Q/K/V, scaled together; one outlier channel."""
    rng = np.random.default_rng(seed)
    q, k, v = (rng.standard_normal((n, d), dtype=np.float32) for _ in range(3))
    scale = SCENARIOS[scenario]
    q *= scale
    k *= scale
    v *= scale
    if scenario == "outliers":
        # A persistent feature outlier, not sparse token corruption.
        q[:, 0] *= 8
        k[:, 0] *= 8
        v[:, 0] *= 8
    return q, k, v


def query_rows(n, full=False):
    if full or n <= 1024:
        return np.arange(n)
    # Four complete 32-row query tiles, spanning beginning and end of the context.
    starts = [0, (n // 4 // 32) * 32, (n // 2 // 32) * 32, n - 32]
    return np.unique(np.concatenate([np.arange(start, start + 32) for start in starts]))


def study_cases(study):
    if study == "length":
        names = ["fp32", "bf16", "e4_tensor", "e5_tensor", "e4_tile", "e4_rotate"]
        for n in [1024, 4096, 16384, 65536]:
            for d in [64, 128]:
                for causal in [False, True]:
                    for scenario in SCENARIOS:
                        yield n, d, causal, scenario, 128, 1.0, names
    elif study == "fixes":
        names = [
            "e4_tensor",
            "e4_reduced",
            "e4_promoted",
            "e4_compensated",
            "e4_reverse",
            "e4_local",
            "e4_no_p_round",
            "e4_no_output_round",
        ]
        for n in [4096, 65536]:
            for d in [64, 128]:
                for causal in [False, True]:
                    for scenario in ["normal", "outliers"]:
                        yield n, d, causal, scenario, 128, 1.0, names
    elif study == "tiles":
        for n in [4096, 65536]:
            for d in [64, 128]:
                for causal in [False, True]:
                    for tile in [32, 128, 512]:
                        yield (
                            n,
                            d,
                            causal,
                            "normal",
                            tile,
                            1.0,
                            ["e4_tensor", "e4_tile", "e4_reduced", "e4_promoted"],
                        )
    elif study == "softmax":
        for scenario in ["flat", "normal", "sharp"]:
            for multiplier in [0.5, 1.0, 2.0]:
                yield (
                    65536,
                    64,
                    False,
                    scenario,
                    128,
                    multiplier,
                    [
                        "fp32",
                        "fp32_compensated",
                        "fp32_reverse",
                        "fp32_local",
                        "e4_tensor",
                        "e4_compensated",
                    ],
                )
    elif study == "full":
        for d in [64, 128]:
            for causal in [False, True]:
                yield (
                    4096,
                    d,
                    causal,
                    "normal",
                    128,
                    1.0,
                    ["fp32", "bf16", "e4_tensor", "e4_rotate"],
                )
    elif study == "full64":
        yield 65536, 64, False, "normal", 128, 1.0, ["fp32", "bf16", "e4_tensor"]
    elif study == "smoke":
        yield 64, 64, True, "normal", 16, 1.0, list(CONFIGS)
    else:
        raise ValueError(f"unknown study {study}")


def cpu_model(cpuinfo=Path("/proc/cpuinfo")):
    """x86 Linux names the CPU in /proc/cpuinfo; ARM Linux and macOS do not."""
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def environment():
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "ml_dtypes": ml_dtypes.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_model(),
        "numpy_build": np.__config__.show(mode="dicts"),
        "threads": {
            name: os.getenv(name)
            for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
        },
        "command": sys.argv,
        "timings": "not measured",
    }


def run(study, destination, seeds):
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{study}.csv"
    with path.open("w", newline="") as stream:
        writer = None
        for n, d, causal, scenario, tile, multiplier, names in study_cases(study):
            for seed in seeds:
                q, k, v = make_inputs(n, d, seed, scenario)
                rows = query_rows(n, full=study in ("full", "full64"))
                scale = multiplier / np.sqrt(d)
                expected = reference(q, k, v, rows=rows, causal=causal, scale=scale)
                for name in names:
                    cfg = replace(CONFIGS[name], tile=tile, causal=causal, scale=float(scale))
                    actual = emulate(q, k, v, cfg, rows)
                    entry = {
                        "study": study,
                        "n": n,
                        "d": d,
                        "causal": causal,
                        "scenario": scenario,
                        "input_std": SCENARIOS[scenario],
                        "outlier_multiplier": 8 if scenario == "outliers" else 1,
                        "tile": tile,
                        "scale_multiplier": multiplier,
                        "seed": seed,
                        "variant": name,
                        "rows_evaluated": len(rows),
                        "all_rows": len(rows) == n,
                        **metrics(actual, expected, rows),
                        **{f"cfg_{key}": value for key, value in asdict(cfg).items()},
                    }
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(entry))
                        writer.writeheader()
                    writer.writerow(entry)
                    stream.flush()
                print(f"{study}: N={n} d={d} causal={causal} {scenario} seed={seed}", flush=True)
    metadata = environment()
    metadata["reference"] = (
        "float64 chunked two-pass attention of original generated float32 inputs"
    )
    metadata["seeds"] = seeds
    metadata["query_rows"] = (
        "all query rows"
        if study in ("full", "full64")
        else "all at N<=1024; otherwise four complete 32-row blocks at 0,N/4,N/2,N-32"
    )
    (destination / f"{study}.json").write_text(json.dumps(metadata, indent=2) + "\n")


def dot_study(destination, seeds):
    """Isolate accumulation from input quantization at long reduction dimensions."""
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "dots.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["k", "seed", "variant", "max_abs", "relative_frobenius"]
        )
        writer.writeheader()
        for size in [64, 128, 512, 4096, 16384, 65536]:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                a, _ = quantize(rng.standard_normal((4, size), dtype=np.float32), "e4m3")
                b, _ = quantize(rng.standard_normal((size, 4), dtype=np.float32), "e4m3")
                expected = a.astype(np.float64) @ b.astype(np.float64)
                for name, accumulator, promote in [
                    ("fp32", "fp32", 0),
                    ("reduced14", "reduced14", 0),
                    ("promoted128", "reduced14", 128),
                ]:
                    result = metrics(matmul(a, b, accumulator, promote), expected)
                    writer.writerow(
                        {
                            "k": size,
                            "seed": seed,
                            "variant": name,
                            **{key: result[key] for key in ["max_abs", "relative_frobenius"]},
                        }
                    )
    metadata = environment()
    metadata["seeds"] = seeds
    metadata["reference"] = "float64 matmul of already-quantized FP8 storage values; not attention"
    (destination / "dots.json").write_text(json.dumps(metadata, indent=2) + "\n")


def denominator_study(destination):
    """Known-answer construction isolates denominator error from numerator error."""
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "denominator.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["n", "tile", "variant", "actual", "exact", "max_abs", "relative_frobenius"],
        )
        writer.writeheader()
        for n in [1024, 4096, 16384, 65536]:
            q = np.ones((1, 1), dtype=np.float32)
            k = np.full((n, 1), -16, dtype=np.float32)
            k[0] = 0
            v = np.zeros_like(k)
            v[0] = 1
            expected = np.array([[1 / (1 + (n - 1) * np.exp(-16.0))]])
            for tile in [1, 128]:
                for name in ["fp32", "fp32_compensated"]:
                    actual = emulate(q, k, v, replace(CONFIGS[name], tile=tile))
                    result = metrics(actual, expected)
                    writer.writerow(
                        {
                            "n": n,
                            "tile": tile,
                            "variant": name,
                            "actual": actual.item(),
                            "exact": expected.item(),
                            **{key: result[key] for key in ["max_abs", "relative_frobenius"]},
                        }
                    )
    metadata = environment()
    metadata["reference"] = "analytic 1/(1+(N-1)*exp(-16)) in float64; numerator exactly 1"
    (destination / "denominator.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=[
            "length",
            "fixes",
            "tiles",
            "softmax",
            "full",
            "full64",
            "smoke",
            "dots",
            "denominator",
        ],
        default="length",
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 17, 29])
    args = parser.parse_args()
    if args.study == "dots":
        dot_study(args.output, args.seeds)
    elif args.study == "denominator":
        denominator_study(args.output)
    else:
        run(args.study, args.output, args.seeds)
