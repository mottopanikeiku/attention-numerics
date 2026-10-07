"""Ephemeral CPU staging and real-H100 multiple-choice accuracy runs.

Set ATTENTION_GPU=none for weights/prepare; H100 for pilot/full.
Resource environment values must match the booked container allocation.
"""

import base64
import hashlib
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/project")
GPU = os.environ.get("ATTENTION_GPU", "none")
MINUTES = int(os.environ.get("ATTENTION_MINUTES", "5"))
CORES = int(os.environ.get("ATTENTION_CORES", "4"))
MEM_GIB = int(os.environ.get("ATTENTION_MEM_GIB", "16"))
if GPU not in ("none", "H100") or min(MINUTES, CORES, MEM_GIB) < 1:
    raise ValueError("Invalid accuracy resource configuration")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.9.1+cu128", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy==2.2.6",
        "transformers==4.57.6",
        "huggingface-hub==0.36.2",
        "safetensors==0.7.0",
        "accelerate==1.12.0",
    )
    .run_commands(
        "python -m venv /opt/fa3-acquire",
        "/opt/fa3-acquire/bin/python -m pip install huggingface-hub==1.10.0",
    )
    .add_local_file(ROOT / "study/hardware/acquire_fa3.py", "/opt/acquire_fa3.py", copy=True)
    .run_commands("/opt/fa3-acquire/bin/python /opt/acquire_fa3.py")
    .env({"ATTENTION_FA3_ROOT": "/opt/fa3/build/torch-stable-abi29-cu128-x86_64-linux"})
    .pip_install("datasets==3.6.0", "sentencepiece==0.2.1", "matplotlib==3.10.8")
    .env(
        {
            "PYTHONPATH": "/project",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "HF_HOME": "/volume/hf",
            "HF_DATASETS_CACHE": "/volume/datasets",
            "ATTENTION_GPU": GPU,
            "ATTENTION_MINUTES": str(MINUTES),
        }
    )
    .add_local_dir(ROOT / "study", "/project/study")
    .add_local_dir(ROOT / "data/accuracy", "/project/data/accuracy")
)
app = modal.App("attention-numerics-day-accuracy")
volume = modal.Volume.from_name("attention-numerics-day-models", create_if_missing=True)


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@app.function(
    image=image,
    gpu=None if GPU == "none" else GPU,
    cpu=CORES,
    memory=MEM_GIB * 1024,
    timeout=MINUTES * 60,
    max_containers=1,
    volumes={"/volume": volume},
)
def experiment(mode="weights", models="", token_budget=16384, pilot_items=8):
    import time

    plan = json.loads(Path("/project/data/accuracy/plan.json").read_text())
    selected = models.split(",") if models else [entry["key"] for entry in plan["models"]]
    if any(key not in {entry["key"] for entry in plan["models"]} for key in selected):
        raise ValueError("Model selection is not in the published plan")
    started = time.monotonic()
    if mode == "weights":
        if GPU != "none":
            raise ValueError("Download weights in a CPU container, never on paid GPU time")
        from huggingface_hub import snapshot_download

        records = {}
        for entry in plan["models"]:
            if entry["key"] not in selected:
                continue
            target = Path("/volume/checkpoints") / entry["key"]
            snapshot_download(
                entry["repo_id"],
                revision=entry["revision"],
                local_dir=target,
                allow_patterns=[
                    "*.safetensors", "*.safetensors.index.json", "config.json",
                    "generation_config.json", "tokenizer*", "special_tokens_map.json",
                    "vocab.json", "merges.txt", "LICENSE*", "README.md",
                ],
                max_workers=4,
            )
            config = json.loads((target / "config.json").read_text())
            dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
            if dim > 256 or dim % 16 or dim & (dim - 1):
                raise ValueError(f"{entry['key']} has incompatible FA3/Hadamard head dimension")
            files = [path for path in target.iterdir() if path.is_file()]
            if not any(path.suffix == ".safetensors" for path in files):
                raise ValueError("No native safetensors checkpoint downloaded")
            records[entry["key"]] = {
                "repo_id": entry["repo_id"], "revision": entry["revision"],
                "directory": f"checkpoints/{entry['key']}", "license": entry["license"],
                "head_dimension": dim, "layers": config["num_hidden_layers"],
                "sliding_window": config.get("sliding_window"),
                "files": {path.name: {"bytes": path.stat().st_size, "sha256": _digest(path)}
                          for path in sorted(files)},
            }
            volume.commit()
            print(f"Staged and hashed {entry['key']}", flush=True)
        manifest = {"models": records, "volume": "attention-numerics-day-models",
                    "elapsed_seconds_for_sizing": time.monotonic() - started}
        Path("/volume/models.json").write_text(json.dumps(manifest, indent=2) + "\n")
        volume.commit()
        return json.dumps({"mode": mode, "manifest": manifest}, allow_nan=False)
    if mode == "prepare":
        if GPU != "none":
            raise ValueError("Prepare datasets on CPU")
        from study.accuracy.data import prepare_items

        destination = Path("/volume/evaluation")
        metadata = prepare_items(destination)
        volume.commit()
        return json.dumps({"mode": mode, "manifest": metadata,
                           "files": {path.name: base64.b64encode(path.read_bytes()).decode()
                                     for path in destination.iterdir() if path.is_file()}},
                          allow_nan=False)
    if mode not in ("pilot", "full") or GPU != "H100":
        raise ValueError("Accuracy modes require real H100; no fallback")
    from study.accuracy.worker import run

    result = run(
        Path("/project/data/accuracy/plan.json"),
        Path("/project/data/accuracy/items.json.gz"),
        Path("/volume"),
        selected,
        pilot=mode == "pilot",
        pilot_items=pilot_items,
        token_budget=token_budget,
        commit_callback=volume.commit,
    )
    return json.dumps(result, allow_nan=False)


@app.local_entrypoint()
def main(mode: str = "weights", output: str = "results/accuracy", models: str = "",
         token_budget: int = 16384, pilot_items: int = 8):
    result = json.loads(experiment.remote(mode, models, token_budget, pilot_items))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for name, encoded in result.pop("files", {}).items():
        (destination / name).write_bytes(base64.b64decode(encoded))
    (destination / f"{mode}-metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    if mode == "weights":
        (destination / "models.json").write_text(json.dumps(result["manifest"], indent=2) + "\n")
    print(f"Saved {mode} records to {destination}")
