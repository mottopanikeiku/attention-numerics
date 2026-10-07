"""Select hardware cases before outcomes and package exact v2 BF16 operands.

python -m study.hardware.inputs --capture-cache PATH --output PATH
The large bundle stays outside Git; selection/provenance stay in data/hardware.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEXTS = ("alice", "moby", "pride")
VARIANTS = ("tile", "rotate", "smooth_k", "rotate_smooth_k")
SEED = 20261007


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_heads(rows, top=32, sample=32, seed=SEED):
    """Top locked-predictor ranks plus an independent uniform physical-head sample."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], int(row["layer"]), int(row["head"]))].append(row)
    by_model = defaultdict(list)
    for (model, layer, head), cases in sorted(groups.items()):
        if len(cases) != 3 or {row["text"] for row in cases} != set(TEXTS):
            raise ValueError(f"Expected three texts for {model}/{layer}/{head}")
        score = (
            math.fsum(
                math.log1p(float(row["rotate_predicted_error"]))
                - math.log1p(float(row["tile_predicted_error"]))
                for row in cases
            )
            / 3
        )
        by_model[model].append({"layer": layer, "head": head, "predicted_hurt_score": score})
    selections = {}
    for model, points in sorted(by_model.items()):
        if model in ("qwen05", "qwen15"):
            selections[model] = [dict(point, selected_by=["all"]) for point in points]
            continue
        ordered = sorted(
            points,
            key=lambda point: (-point["predicted_hurt_score"], point["layer"], point["head"]),
        )
        top_ids = {(point["layer"], point["head"]) for point in ordered[:top]}
        # Model-specific stable seed; adding another model cannot change a sample.
        model_seed = int.from_bytes(hashlib.sha256(f"{seed}:{model}".encode()).digest()[:8], "big")
        random_points = random.Random(model_seed).sample(points, min(sample, len(points)))
        random_ids = {(point["layer"], point["head"]) for point in random_points}
        selected = []
        for point in points:
            identity = (point["layer"], point["head"])
            tags = (["top"] if identity in top_ids else []) + (
                ["random"] if identity in random_ids else []
            )
            if tags:
                selected.append(dict(point, selected_by=tags))
        selections[model] = selected
    return selections


def build_inputs(capture_cache, output, *, top=32, sample=32, seed=SEED):
    capture_cache, output = Path(capture_cache), Path(output)
    source_table = ROOT / "results/v2/heads.csv"
    with source_table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selections = select_heads(rows, top, sample, seed)
    selected_ids = {
        (model, point["layer"], point["head"])
        for model, points in selections.items()
        for point in points
    }
    cases = [
        row for row in rows if (row["model"], int(row["layer"]), int(row["head"])) in selected_ids
    ]
    plan = {
        "schema_version": 1,
        "source_head_table_sha256": sha256(source_table),
        "predictor_lock_commit": "f756745",
        "score": "mean_over_three_texts(log1p(rotate_predicted_error)-log1p(tile_predicted_error))",
        "classification_threshold": 0,
        "top_count_per_non_qwen_model": top,
        "uniform_sample_per_non_qwen_model": sample,
        "sample_seed": seed,
        "sampling": (
            "Independent uniform sample from the entire model; "
            "overlaps with top ranks are retained as both labels and evaluated once."
        ),
        "population_warning": (
            "Qwen coverage is exhaustive. Top-rank enrichment is not a population estimate; "
            "report random and top strata separately."
        ),
        "texts": list(TEXTS),
        "tokens": 1024,
        "variants": list(VARIANTS),
        "rotation_seed": 1729,
        "models": selections,
    }
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in cases:
        grouped[(row["model"], int(row["layer"]), row["text"])].append(row)
    captures = {}
    for model in selections:
        manifest = json.loads((ROOT / f"results/v2/captures/{model}.json").read_text())
        if not manifest["complete"]:
            raise ValueError(f"Incomplete published capture: {model}")
        captures[model] = {(item["layer"], item["text"]): item for item in manifest["files"]}
    files = []
    for (model, layer, text), group in sorted(grouped.items()):
        provenance = captures[model][(layer, text)]
        source = capture_cache / provenance["file"]
        if sha256(source) != provenance["sha256"]:
            raise ValueError(f"Source capture SHA256 mismatch: {model}/{layer}/{text}")
        heads = sorted(int(row["head"]) for row in group)
        row_by_head = {int(row["head"]): row for row in group}
        kv_heads = sorted({int(row["kv_head"]) for row in group})
        kv_lookup = {head: index for index, head in enumerate(kv_heads)}
        mapping = np.array(
            [kv_lookup[int(row_by_head[head]["kv_head"])] for head in heads], dtype=np.int64
        )
        relative = f"operands/{model}/layer_{layer:02d}_{text}.npz"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with np.load(source, allow_pickle=False) as source_npz:
            if any(source_npz[key].dtype != np.uint16 for key in ("q", "k", "v")):
                raise ValueError("Expected exact uint16 BF16 storage")
            np.savez(
                destination,
                q=source_npz["q"][heads],
                k=source_npz["k"][kv_heads],
                v=source_npz["v"][kv_heads],
                kv_mapping=mapping,
                scale=source_npz["scale"],
                heads=np.array(heads, dtype=np.int64),
            )
        files.append(
            {
                "model": model,
                "layer": layer,
                "text": text,
                "file": relative,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "source_file": provenance["file"],
                "source_sha256": provenance["sha256"],
                "heads": heads,
                "kv_heads": kv_heads,
                "scale": float(group[0]["scale"]),
            }
        )
    for model in ("qwen05", "qwen15"):
        for text in TEXTS:
            destination = output / f"tokens/{model}/{text}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(capture_cache / model / "tokens" / f"{text}.npz", destination)
    metadata = {"selection": plan, "files": files, "head_text_rows": cases}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copyfile(ROOT / "data/v2/models.json", output / "models.json")
    archive = output.parent / f"{output.name}.tar"
    with tarfile.open(archive, "w") as handle:
        for source in sorted(output.rglob("*")):
            if source.is_file():
                handle.add(source, arcname=str(source.relative_to(output)), recursive=False)
    plan["bundle"] = {
        "file": archive.name,
        "sha256": sha256(archive),
        "bytes": archive.stat().st_size,
        "operand_files": len(files),
        "head_text_rows": len(cases),
    }
    tracked = ROOT / "data/hardware/selection.json"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text(json.dumps(plan, indent=2) + "\n")
    publish_manifest(output, plan, files)
    print(
        json.dumps(
            {"bundle": str(archive), **plan["bundle"], "physical_heads": len(selected_ids)},
            indent=2,
        )
    )
    return plan


def publish_manifest(output, plan, files):
    """Publish every bundled member's hash, not the large private operand arrays."""
    output = Path(output)
    members = [
        {
            "file": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    public = {"bundle": plan["bundle"], "captures": files, "members": members}
    (ROOT / "data/hardware/operands.json").write_text(json.dumps(public, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=32)
    parser.add_argument("--sample", type=int, default=32)
    args = parser.parse_args()
    if args.top < 1 or args.sample < 1:
        parser.error("Sample sizes must be positive")
    build_inputs(args.capture_cache, args.output, top=args.top, sample=args.sample)


if __name__ == "__main__":
    main()
