"""Ephemeral, single-container real-kernel runs; no deployment or secrets.

Set ATTENTION_BACKEND=fa3|sage, ATTENTION_GPU=H100|L4|none,
ATTENTION_MINUTES, ATTENTION_CORES and ATTENTION_MEM_GIB to the booked limits.
ATTENTION_INPUT_BUNDLE uploads exact cached operands in CPU-only upload mode;
numerical runs read the pinned bundle from the private Modal Volume.
"""

import gzip
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/project")
BACKEND = os.environ.get("ATTENTION_BACKEND", "fa3")
GPU = os.environ.get("ATTENTION_GPU", "H100" if BACKEND == "fa3" else "L4")
MINUTES = int(os.environ.get("ATTENTION_MINUTES", "5"))
CORES = int(os.environ.get("ATTENTION_CORES", "4"))
MEM_GIB = int(os.environ.get("ATTENTION_MEM_GIB", "16"))
if BACKEND not in ("fa3", "sage") or MINUTES < 1 or CORES < 1 or MEM_GIB < 1:
    raise ValueError("Invalid backend/resource configuration")
if GPU not in (("H100", "none") if BACKEND == "fa3" else ("L4", "L40S", "none")):
    raise ValueError("GPU does not match the selected kernel")

if BACKEND == "fa3":
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
    )
else:
    image = (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
        .apt_install("git", "build-essential")
        .pip_install("torch==2.6.0+cu124", index_url="https://download.pytorch.org/whl/cu124")
        .pip_install(
            "numpy==2.2.6",
            "setuptools==75.8.0",
            "wheel==0.45.1",
            "packaging==24.2",
            "ninja==1.11.1.3",
        )
        .env(
            {
                "TORCH_CUDA_ARCH_LIST": "8.9",
                "SAGEATTN_SKIP_CUDA_BUILD": "0",
                "EXT_PARALLEL": "1",
                "MAX_JOBS": "2",
                "OMP_NUM_THREADS": "2",
                "NVCC_APPEND_FLAGS": "--threads 2",
                "CC": "gcc",
                "CXX": "g++",
            }
        )
        .run_commands(
            "python -m pip install --no-build-isolation --no-deps "
            "git+https://github.com/thu-ml/SageAttention.git@eb615cf6cf4d221338033340ee2de1c37fbdba4a"
        )
    )
image = image.env(
    {
        "PYTHONPATH": "/project",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "ATTENTION_BACKEND": BACKEND,
        "ATTENTION_GPU": GPU,
        "ATTENTION_MINUTES": str(MINUTES),
        "ATTENTION_CORES": str(CORES),
        "ATTENTION_MEM_GIB": str(MEM_GIB),
    }
)
image = image.add_local_dir(ROOT / "study", "/project/study")
image = image.add_local_file(
    ROOT / "data/hardware/selection.json", "/project/data/hardware/selection.json"
)
if os.environ.get("ATTENTION_INPUT_BUNDLE"):
    image = image.add_local_file(os.environ["ATTENTION_INPUT_BUNDLE"], "/inputs/bundle.tar")

app = modal.App("attention-numerics-real-kernels")
volume = modal.Volume.from_name("attention-numerics-inputs", create_if_missing=True)


@app.function(
    image=image,
    gpu=None if GPU == "none" else GPU,
    cpu=CORES,
    memory=MEM_GIB * 1024,
    timeout=MINUTES * 60,
    max_containers=1,
    volumes={"/volume": volume},
)
def experiment(mode="pilot", downstream=True):
    import importlib
    import sys

    sys.path.insert(0, "/project")
    if mode == "upload":
        import hashlib
        import shutil

        source = Path("/inputs/bundle.tar")
        plan = json.loads(Path("/project/data/hardware/selection.json").read_text())
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != plan["bundle"]["sha256"]:
            raise ValueError("Upload bundle SHA256 disagrees with committed selection")
        shutil.copyfile(source, "/volume/night2-inputs.tar")
        volume.commit()
        return json.dumps(
            {"mode": mode, "bundle": plan["bundle"], "volume": "attention-numerics-inputs"}
        )
    if mode == "image":
        adapter = importlib.import_module(f"study.hardware.{BACKEND}")
        adapter._load_kernel()
        return json.dumps({"backend": BACKEND, "mode": mode, "adapter": adapter.describe()})
    if GPU == "none":
        raise ValueError("A real GPU is required for numerical experiments")
    from study.hardware.worker import run

    result = run(BACKEND, mode=mode, downstream=downstream, limit_seconds=MINUTES * 60)
    # A JSON boundary does not require Torch/NumPy on the SDK-only client.
    return json.dumps(result, allow_nan=False)


@app.local_entrypoint()
def main(mode="pilot", output="results/hardware/pilot.json", downstream=True):
    if mode not in ("image", "upload", "pilot", "full"):
        raise ValueError("mode must be image, upload, pilot or full")
    result = json.loads(experiment.remote(mode=mode, downstream=downstream))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if destination.suffix == ".gz":
        with destination.open("wb") as handle:
            handle.write(gzip.compress(encoded, mtime=0))
    else:
        destination.write_bytes(encoded)
    print(f"Saved {BACKEND} {mode} result to {destination}")
