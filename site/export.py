"""Export existing measurements and Python FP8 bytes: uv run python site/export.py."""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attention import raw_storage_cast  # noqa: E402


def export_heads():
    groups = defaultdict(list)
    with (ROOT / "results/v2/heads.csv").open() as source:
        for row in csv.DictReader(source):
            groups[(row["model"], int(row["layer"]), int(row["head"]))].append(row)
    points = []
    for (model, layer, head), rows in sorted(groups.items()):
        if len(rows) != 3 or len({row["text"] for row in rows}) != 3:
            raise ValueError("Each physical head must have three distinct texts")
        logs = {
            field: math.fsum(math.log1p(float(row[field])) for row in rows) / 3
            for field in (
                "tile_relative_fro",
                "rotate_relative_fro",
                "tile_predicted_error",
                "rotate_predicted_error",
            )
        }
        points.append(
            [
                model,
                layer,
                head,
                logs["rotate_predicted_error"] - logs["tile_predicted_error"],
                logs["rotate_relative_fro"] - logs["tile_relative_fro"],
                math.fsum(float(row["k_mean_energy_fraction"]) for row in rows) / 3,
                math.expm1(logs["tile_relative_fro"]),
                math.expm1(logs["rotate_relative_fro"]),
            ]
        )
    if len(points) != 2880 or sum(point[4] > 0 for point in points) != 250:
        raise ValueError("Unexpected study counts; update the site claims before exporting")
    return {
        "columns": [
            "model",
            "layer",
            "head",
            "predicted",
            "observed",
            "keyMeanEnergy",
            "tileError",
            "rotateError",
        ],
        "source": "results/v2/heads.csv",
        "definition": "mean_text(log1p(rotate_error)) - mean_text(log1p(tile_error))",
        "points": points,
    }


def export_fixture():
    # Every BF16 bit pattern, including signed zero, infinity and signed NaN.
    bits = np.arange(65536, dtype=np.uint32) << 16
    values = bits.view(np.float32)
    # Also probe exact ties and adjacent FP32 values at every positive FP8 boundary.
    codes = np.arange(127, dtype=np.uint8)
    import ml_dtypes

    levels = codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    midpoints = (levels[:-1] + levels[1:]) / 2
    edges = np.concatenate(
        [np.nextafter(midpoints, -np.inf), midpoints, np.nextafter(midpoints, np.inf)]
    )
    edges = np.concatenate([edges, -edges, np.array([464, -464], dtype=np.float32)])
    values = np.concatenate([values, edges])
    raw = raw_storage_cast(values, "e4m3").view(np.uint8)
    saturated = raw_storage_cast(np.clip(values, -448, 448), "e4m3").view(np.uint8)
    return {
        "source": "attention.raw_storage_cast (float32 input, round-to-nearest-even)",
        "columns": ["float32_bits", "raw_e4m3_byte", "saturating_e4m3_byte"],
        "cases": np.column_stack([values.view(np.uint32), raw, saturated]).tolist(),
    }


if __name__ == "__main__":
    for name, data in [("heads.json", export_heads()), ("fp8-fixture.json", export_fixture())]:
        (ROOT / "site" / name).write_text(json.dumps(data, separators=(",", ":")) + "\n")
