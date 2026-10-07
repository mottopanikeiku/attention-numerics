"""GPU experiment body: exact archived operands, FP64 errors and Qwen losses."""

import gc
import hashlib
import importlib
import inspect
import json
import platform
import subprocess
import tarfile
import time
from importlib import metadata
from pathlib import Path
from types import FunctionType, MethodType

import numpy as np
import torch
import torch.nn.functional as F

from .common import VARIANTS, error_metrics, exact_attention

ROOT = Path("/project")
INPUTS = Path("/inputs/unpacked")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs():
    plan = json.loads((ROOT / "data/hardware/selection.json").read_text())
    archive = Path("/volume/night2-inputs.tar")
    if not archive.is_file():
        raise ValueError(
            "Prepare the committed capture pipeline's operands and run Modal upload mode first"
        )
    if sha256(archive) != plan["bundle"]["sha256"]:
        raise ValueError("Hardware input bundle SHA256 mismatch")
    INPUTS.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        handle.extractall(INPUTS, filter="data")
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    if manifest["selection"]["models"] != plan["models"]:
        raise ValueError("Bundled and published hardware selection disagree")
    return plan, manifest


def runtime():
    properties = torch.cuda.get_device_properties(0)
    versions = {}
    for name in (
        "torch",
        "numpy",
        "transformers",
        "huggingface-hub",
        "sageattention",
        "safetensors",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "packages": versions,
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "gpu_memory_bytes": properties.total_memory,
        "driver_query": driver,
        "cpu_threads": torch.get_num_threads(),
        "tf32": False,
        "bf16_reduced_precision_reduction": False,
        "timing_note": (
            "Elapsed times size the paid run and bound cost; "
            "they are not latency or speed benchmarks."
        ),
    }


def operands(item):
    path = INPUTS / item["file"]
    if sha256(path) != item["sha256"]:
        raise ValueError(f"Packed operand digest mismatch: {item['file']}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {}
        for name in ("q", "k", "v"):
            if data[name].dtype != np.uint16:
                raise ValueError("Expected uint16 BF16 operand bits")
            arrays[name] = (
                torch.from_numpy(data[name].copy()).view(torch.bfloat16).unsqueeze(0).cuda()
            )
        mapping = torch.from_numpy(data["kv_mapping"].copy()).cuda()
        scale = float(data["scale"])
    # A selected subset need not consist of equal-size GQA groups. Explicitly
    # restore each selected query's original KV index, not a new inferred group.
    return (
        arrays["q"],
        arrays["k"].index_select(1, mapping),
        arrays["v"].index_select(1, mapping),
        scale,
    )


def kernel_trace(adapter, q, k, v, scale):
    # The trace provides observed CUDA names, not just the Python API label.
    adapter.apply_attention(q, k, v, "tile", scale=scale)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        adapter.apply_attention(q, k, v, "tile", scale=scale)
        torch.cuda.synchronize()
    names = sorted(
        {event.name for event in profile.events() if str(event.device_type).endswith("CUDA")}
    )
    if not names:
        raise RuntimeError("Profiler did not observe any actual CUDA kernel events")
    return names


def measure_heads(adapter, plan, manifest, *, pilot=False, deadline=None):
    source = {
        (row["model"], int(row["layer"]), int(row["head"]), row["text"]): row
        for row in manifest["head_text_rows"]
    }
    tags = {
        (model, point["layer"], point["head"]): point["selected_by"]
        for model, points in plan["models"].items()
        for point in points
    }
    files = manifest["files"]
    if pilot:
        first = {}
        for item in files:
            first.setdefault(item["model"], item)
        files = list(first.values())
    rows, diagnostics, trace = [], [], None
    started = time.monotonic()
    for index, item in enumerate(files):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("Insufficient booked time; no partial full result is published")
        q, k, v, scale = operands(item)
        if trace is None:
            trace = kernel_trace(adapter, q, k, v, scale)
        reference = exact_attention(q, k, v, scale=scale)
        reference_energy = reference.square().mean(dim=(-2, -1))[0].cpu().tolist()
        native = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        results = {"bf16": error_metrics(native, reference)}
        for variant in VARIANTS:
            observed = {}
            output = adapter.apply_attention(q, k, v, variant, scale=scale, diagnostics=observed)
            results[variant] = error_metrics(output, reference)
            diagnostics.append({"file": item["file"], "variant": variant, "values": observed})
            del output
        for position, head in enumerate(item["heads"]):
            identity = (item["model"], item["layer"], head, item["text"])
            old = source[identity]
            old_energy = float(old["reference_output_mean_square"])
            energy_disagreement = abs(reference_energy[position] - old_energy) / old_energy
            if energy_disagreement > 1e-9:
                raise ValueError(
                    f"FP64 reference disagrees with v2 energy: {identity}: {energy_disagreement}"
                )
            rows.append(
                {
                    **{name: old[name] for name in ("model", "family", "revision", "text")},
                    "layer": item["layer"],
                    "head": head,
                    "kv_head": int(old["kv_head"]),
                    "n": int(old["n"]),
                    "d": int(old["d"]),
                    "scale": scale,
                    "selected_by": tags[identity[:3]],
                    "source_capture_sha256": item["source_sha256"],
                    "packed_operand_sha256": item["sha256"],
                    "reference_energy_relative_disagreement": energy_disagreement,
                    "emulator": {
                        variant: {
                            "relative_fro": float(old[f"{variant}_relative_fro"]),
                            "predicted_error": float(old[f"{variant}_predicted_error"]),
                        }
                        for variant in VARIANTS
                    },
                    "hardware": {
                        variant: {
                            "relative_fro": values["relative_fro"][0][position],
                            "max_abs": values["max_abs"][0][position],
                            "output_nonfinite": 0,
                        }
                        for variant, values in results.items()
                    },
                }
            )
        del q, k, v, reference, native, results
        if (index + 1) % 30 == 0:
            print(
                f"Measured {index + 1}/{len(files)} operand files, {len(rows)} head/text cases",
                flush=True,
            )
    return {
        "rows": rows,
        "quantization_diagnostics": diagnostics,
        "cuda_kernel_trace": trace,
        "operand_files": len(files),
        "elapsed_seconds_for_sizing": time.monotonic() - started,
    }


def install_attention(model, adapter, variant, calls):
    """Rebind native attention's lookup per instance, preserving projection/RoPE."""
    original = []
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        forward = inspect.unwrap(type(attention).forward)

        def intercepted(module, query, key, value, attention_mask, scaling, **kwargs):
            if kwargs.get("dropout", 0) != 0:
                raise ValueError("Downstream kernel comparison is inference-only")
            output = adapter.apply_attention(query, key, value, variant, scale=scaling)
            calls.append(module.layer_idx)
            return output.transpose(1, 2).contiguous().to(query.dtype), None

        globals_copy = dict(forward.__globals__, ALL_ATTENTION_FUNCTIONS={"sdpa": intercepted})
        rebound = FunctionType(
            forward.__code__,
            globals_copy,
            forward.__name__,
            forward.__defaults__,
            forward.__closure__,
        )
        rebound.__kwdefaults__ = forward.__kwdefaults__
        attention.forward = MethodType(rebound, attention)
        original.append((index, attention))
    return original


def remove_attention(original):
    for _, attention in original:
        del attention.forward


@torch.inference_mode()
def distribution_metrics(logits, baseline, labels, row_chunk=128):
    """Native BF16 vocabulary projection, FP64 CE and KL(BF16 baseline||variant)."""
    if logits.dtype != torch.bfloat16 or baseline.dtype != torch.bfloat16:
        raise ValueError("Expected native BF16 projected logits")
    ce, kl = 0.0, 0.0
    for first in range(0, labels.shape[1], row_chunk):
        stop = min(first + row_chunk, labels.shape[1])
        alt = logits[:, first:stop].double()
        base = baseline[:, first:stop].double()
        alt_log = alt - torch.logsumexp(alt, dim=-1, keepdim=True)
        base_log = base - torch.logsumexp(base, dim=-1, keepdim=True)
        ce += (-alt_log.gather(-1, labels[:, first:stop].unsqueeze(-1))).sum().item()
        kl += (base_log.exp() * (base_log - alt_log)).sum().item()
    count = labels.numel()
    if not np.isfinite(ce) or not np.isfinite(kl) or kl < -1e-9:
        raise FloatingPointError("Invalid downstream CE/KL")
    return {"tokens": count, "next_token_ce": ce / count, "kl_from_bf16": kl / count}


@torch.inference_mode()
def measure_downstream(adapter, *, pilot=False, deadline=None):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    pins = {
        item["key"]: item for item in json.loads((INPUTS / "models.json").read_text())["models"]
    }
    models = ("qwen05",) if pilot else ("qwen05", "qwen15")
    texts = ("alice",) if pilot else ("alice", "moby", "pride")
    variants = ("tile", "rotate") if pilot else VARIANTS
    rows = []
    started = time.monotonic()
    for key in models:
        pin = pins[key]
        snapshot = snapshot_download(
            pin["model_id"],
            revision=pin["revision"],
            allow_patterns=["*.json", "*.safetensors"],
            max_workers=2,
        )
        # Check the actual pinned weight bytes rather than trusting the cache key.
        checked_weights = []
        for item in pin["files"]:
            if item["role"] == "weights":
                path = Path(snapshot) / item["path"]
                observed = sha256(path)
                if path.stat().st_size != item["bytes"] or observed != item["sha256"]:
                    raise ValueError(f"Pinned model weight mismatch: {key}/{item['path']}")
                checked_weights.append(
                    {"file": item["path"], "bytes": item["bytes"], "sha256": observed}
                )
        model = (
            AutoModelForCausalLM.from_pretrained(
                snapshot,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
            .eval()
            .cuda()
        )
        expected_layers = set(range(len(model.model.layers)))
        for text in texts:
            with np.load(INPUTS / f"tokens/{key}/{text}.npz", allow_pickle=False) as data:
                heldout = torch.from_numpy(data["heldout"].copy()).long().cuda().unsqueeze(0)
            if heldout.shape != (1, 1025):
                raise ValueError("Heldout tokens must be the exact v2 1025-token window")
            inputs, labels = heldout[:, :-1], heldout[:, 1:]
            baseline = model(input_ids=inputs, use_cache=False).logits
            if baseline.dtype != torch.bfloat16 or not torch.isfinite(baseline).all():
                raise ValueError("Invalid native BF16 baseline logits")
            common = {
                "model": key,
                "text": text,
                "model_id": pin["model_id"],
                "revision": pin["revision"],
                "window": [1024, 2049],
                "attention_layers": len(model.model.layers),
                "weight_files": checked_weights,
            }
            base_metrics = distribution_metrics(baseline, baseline, labels)
            rows.append({**common, "variant": "bf16", **base_metrics, "kernel_calls": 0})
            for variant in variants:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(
                        "Insufficient booked time for complete downstream evaluation"
                    )
                calls = []
                original = install_attention(model, adapter, variant, calls)
                try:
                    logits = model(input_ids=inputs, use_cache=False).logits
                finally:
                    remove_attention(original)
                if set(calls) != expected_layers or len(calls) != len(expected_layers):
                    raise ValueError(
                        "Real FP8 kernel did not run exactly once in every decoder layer"
                    )
                metrics = distribution_metrics(logits, baseline, labels)
                rows.append(
                    {
                        **common,
                        "variant": variant,
                        **metrics,
                        "kernel_calls": len(calls),
                        "delta_ce": metrics["next_token_ce"] - base_metrics["next_token_ce"],
                    }
                )
                del logits
            del baseline, inputs, labels, heldout
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "rows": rows,
        "elapsed_seconds_for_sizing": time.monotonic() - started,
        "semantics": (
            "Batch teacher forcing with reset positions; no streaming/generation claim. "
            "Full-sequence quantization scales and K means may see later batch tokens."
        ),
    }


def run(backend, *, mode="pilot", downstream=True, limit_seconds=300):
    if mode not in ("pilot", "full"):
        raise ValueError("Numerical mode must be pilot or full")
    started = time.monotonic()
    deadline = started + limit_seconds - 20
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    adapter = importlib.import_module(f"study.hardware.{backend}")
    plan, manifest = load_inputs()
    head_results = measure_heads(adapter, plan, manifest, pilot=mode == "pilot", deadline=deadline)
    downstream_results = None
    if backend == "fa3" and downstream:
        downstream_results = measure_downstream(adapter, pilot=mode == "pilot", deadline=deadline)
    return {
        "schema_version": 1,
        "backend": backend,
        "mode": mode,
        "complete": True,
        "runtime": runtime(),
        "adapter": adapter.describe(),
        "selection": plan,
        "reference": (
            "FP64 original QK, explicit causal softmax, FP64 PV; BF16 inputs expanded exactly"
        ),
        "heads": head_results,
        "downstream": downstream_results,
        "elapsed_seconds_for_sizing": time.monotonic() - started,
        "cost_note": (
            "Wall-time budget upper bound is recorded separately; "
            "this is not a performance benchmark."
        ),
    }
