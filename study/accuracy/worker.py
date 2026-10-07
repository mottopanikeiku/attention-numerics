"""Native BF16 models with only attention replaced; exact-length MC batches.

The vocabulary projection is positionwise: projecting only continuation prediction
positions preserves the model while avoiding unused full-context vocabulary logits.
Weights and tokenized fixed items are prepared on CPU before GPU allocation.
"""

import base64
import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from study.hardware import fa3
from study.hardware.worker import install_attention, remove_attention


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_gzip(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(gzip.compress(json.dumps(payload, allow_nan=False).encode(), mtime=0))
    temporary.replace(path)


def read_gzip(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def encode_items(items, tokenizer, max_length):
    from .data import encode_choice

    rows = []
    for index, item in enumerate(items):
        for choice_index, choice in enumerate(item["choices"]):
            context, continuation = encode_choice(
                tokenizer, item["context"], choice, tokenizer.eos_token_id, add_bos_token=None
            )
            if not context or not continuation:
                raise ValueError("Every scored choice needs context and continuation tokens")
            tokens = context + continuation
            if len(tokens) - 1 > max_length:
                raise ValueError(
                    f"Fixed item {item['item_id']} exceeds model context; no truncation"
                )
            rows.append(
                {
                    "item_index": index,
                    "choice_index": choice_index,
                    "input_ids": tokens[:-1],
                    "continuation_ids": continuation,
                    "context_tokens": len(context),
                    "sequence_length": len(tokens) - 1,
                }
            )
    return rows


def prepare_tokens(plan_path, items_path, volume_path):
    """Tokenizer-only CPU stage; never load model tensors."""
    from transformers import AutoTokenizer

    volume_path = Path(volume_path)
    plan = json.loads(Path(plan_path).read_text())
    items = read_gzip(items_path)["items"]
    metadata = {
        "plan_sha256": digest(plan_path),
        "item_manifest_sha256": digest(items_path),
        "models": {},
    }
    for entry in plan["models"]:
        checkpoint = volume_path / "checkpoints" / entry["key"]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        config = json.loads((checkpoint / "config.json").read_text())
        maximum = config["max_position_embeddings"]
        if config.get("sliding_window") is not None:
            maximum = min(maximum, config["sliding_window"])
        rows = encode_items(items, tokenizer, maximum)
        payload = {
            "model": entry["key"],
            "revision": entry["revision"],
            "plan_sha256": metadata["plan_sha256"],
            "item_manifest_sha256": metadata["item_manifest_sha256"],
            "rows": rows,
        }
        path = volume_path / "tokens" / f"{entry['key']}.json.gz"
        write_gzip(path, payload)
        lengths = [row["sequence_length"] for row in rows]
        metadata["models"][entry["key"]] = {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "choices": len(rows),
            "input_tokens_per_variant": sum(lengths),
            "maximum_sequence_length": max(lengths),
            "tokenizer_class": type(tokenizer).__name__,
            "bos_token_id": tokenizer.bos_token_id,
            "special_tokens": "tokenizer-native default, as harness add_bos_token=None",
        }
        print(
            f"Tokenized {entry['key']}: {len(rows)} choices, {sum(lengths)} input tokens",
            flush=True,
        )
    (volume_path / "tokens.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def batches(rows, token_budget=16384, maximum_batch=256):
    if token_budget < 1 or maximum_batch < 1:
        raise ValueError("Batch limits must be positive")
    groups = defaultdict(list)
    for row in rows:
        if row["sequence_length"] != len(row["input_ids"]) or row["sequence_length"] < 1:
            raise ValueError("Encoded choice sequence length is invalid")
        groups[row["sequence_length"]].append(row)
    for length in sorted(groups, reverse=True):
        size = min(maximum_batch, max(1, token_budget // length))
        for first in range(0, len(groups[length]), size):
            yield groups[length][first : first + size]


def prediction_positions(batch):
    row_indices, positions, targets, counts = [], [], [], []
    for index, row in enumerate(batch):
        count = len(row["continuation_ids"])
        start = row["context_tokens"] - 1
        if start < 0 or start + count != row["sequence_length"]:
            raise ValueError("Continuation prediction positions disagree with encoded sequence")
        row_indices.extend([index] * count)
        positions.extend(range(start, start + count))
        targets.extend(row["continuation_ids"])
        counts.append(count)
    return row_indices, positions, targets, counts


@torch.inference_mode()
def continuation_scores(hidden, projection, batch):
    """Positionwise native vocabulary projection with independent token sums."""
    if hidden.dtype != torch.bfloat16:
        raise ValueError("Native model hidden states must stay BF16")
    row_indices, positions, targets, counts = prediction_positions(batch)
    selected = hidden[
        torch.tensor(row_indices, device=hidden.device),
        torch.tensor(positions, device=hidden.device),
    ]
    log_probabilities = []
    for first in range(0, len(targets), 256):
        stop = min(first + 256, len(targets))
        logits = projection(selected[first:stop])
        if logits.dtype != torch.bfloat16:
            raise ValueError("Native vocabulary projection must stay BF16")
        values = logits.float()
        target = torch.tensor(targets[first:stop], device=values.device).unsqueeze(-1)
        log_prob = values.gather(-1, target)[:, 0] - torch.logsumexp(values, dim=-1)
        if not bool(torch.isfinite(log_prob).all()):
            raise FloatingPointError("Nonfinite native choice log-likelihood")
        log_probabilities.extend(log_prob.double().cpu().tolist())
    scores, offset = [], 0
    for count in counts:
        scores.append(math.fsum(log_probabilities[offset : offset + count]))
        offset += count
    return scores


@torch.inference_mode()
def score_batch(model, batch):
    inputs = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device="cuda")
    output = model.model(input_ids=inputs, use_cache=False, return_dict=True)
    return continuation_scores(output.last_hidden_state, model.lm_head, batch)


def runtime():
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "python": platform.python_version(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "scoring": "native BF16 vocabulary projection; FP32 logsumexp; FP64 ordered token sum",
        "batching": "same exact sequence length, no padding, causal, no KV cache",
        "elapsed_interpretation": "sizing and cost only; not a latency benchmark",
    }


def _pilot_indices(items, count):
    if count < 1:
        raise ValueError("Pilot item count must be positive")
    groups = defaultdict(list)
    for index, item in enumerate(items):
        groups[item["task"]].append(index)
    return {index for values in groups.values() for index in values[:count]}


def _cuda_trace(profiler):
    names = sorted(
        {
            event.name
            for event in profiler.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        }
    )
    if not any("flash" in name.lower() or "fwdkernel" in name.lower() for name in names):
        raise RuntimeError("Profiler did not record a real FA3 forward kernel")
    return names


@torch.inference_mode()
def run(
    plan_path,
    items_path,
    volume_path,
    selected_models,
    *,
    pilot=False,
    pilot_items=8,
    token_budget=16384,
    commit_callback=None,
):
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise ValueError("Accuracy experiment requires real H100 sm90, not a fallback")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    volume_path = Path(volume_path)
    plan = json.loads(Path(plan_path).read_text())
    items = read_gzip(items_path)["items"]
    plan_sha, items_sha = digest(plan_path), digest(items_path)
    token_manifest = json.loads((volume_path / "tokens.json").read_text())
    if (token_manifest["plan_sha256"], token_manifest["item_manifest_sha256"]) != (
        plan_sha,
        items_sha,
    ):
        raise ValueError("CPU-tokenized inputs do not match the published plan/items")
    indices = _pilot_indices(items, pilot_items) if pilot else set(range(len(items)))
    files, sizing = {}, {}
    for entry in plan["models"]:
        key = entry["key"]
        if key not in selected_models:
            continue
        completed_path = volume_path / "accuracy-results" / f"{key}.json.gz"
        if not pilot and completed_path.is_file():
            cached = read_gzip(completed_path)
            expected = {
                (item["item_id"], variant) for item in items for variant in plan["variants"]
            }
            observed = {(row["item_id"], row["variant"]) for row in cached["rows"]}
            if (
                cached["plan_sha256"] != plan_sha
                or cached["item_manifest_sha256"] != items_sha
                or cached["revision"] != entry["revision"]
                or observed != expected
                or len(cached["rows"]) != len(expected)
            ):
                raise ValueError("Completed model does not match the fixed accuracy inputs")
            files[f"{key}.json.gz"] = base64.b64encode(completed_path.read_bytes()).decode()
            sizing[key] = {
                "items_per_variant": len(items),
                "execution": cached["execution"],
                "total_elapsed_seconds_for_sizing": cached["total_elapsed_seconds_for_sizing"],
                "full_input_tokens_per_variant": (
                    token_manifest["models"][key]["input_tokens_per_variant"]
                ),
                "reused_complete_model": True,
            }
            continue
        token_path = volume_path / "tokens" / f"{key}.json.gz"
        if digest(token_path) != token_manifest["models"][key]["sha256"]:
            raise ValueError("Tokenized choices changed after CPU preparation")
        encoded = read_gzip(token_path)
        if encoded["revision"] != entry["revision"]:
            raise ValueError("Token checkpoint revision disagrees with plan")
        chosen = [row for row in encoded["rows"] if row["item_index"] in indices]
        choices_by_item = defaultdict(list)
        for row in chosen:
            choices_by_item[row["item_index"]].append(row)
        checkpoint = volume_path / "checkpoints" / key
        model_started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        layer_count = len(model.model.layers)
        calls, all_rows, variant_metadata, trace = [], [], {}, []
        for variant in plan["variants"]:
            stored = volume_path / "accuracy-checkpoints" / key / f"{variant}.json.gz"
            if not pilot and stored.is_file():
                cached = read_gzip(stored)
                if (cached["plan_sha256"], cached["item_manifest_sha256"]) != (plan_sha, items_sha):
                    raise ValueError("Variant checkpoint belongs to different fixed inputs")
                all_rows.extend(cached["rows"])
                variant_metadata[variant] = cached["execution"]
                trace.extend(cached.get("cuda_kernel_trace", []))
                continue
            originals = install_attention(model, fa3, variant, calls) if variant != "bf16" else []
            started = time.monotonic()
            scores = {}
            number_batches, processed_tokens = 0, 0
            try:
                for batch in batches(chosen, token_budget):
                    calls.clear()
                    capture = variant == "tile" and not trace
                    if capture:
                        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA]
                        ) as prof:
                            values = score_batch(model, batch)
                        trace = _cuda_trace(prof)
                    else:
                        values = score_batch(model, batch)
                    if variant != "bf16" and calls != list(range(layer_count)):
                        raise RuntimeError("Native FA3 did not execute exactly once in every layer")
                    for row, value in zip(batch, values, strict=True):
                        identity = (row["item_index"], row["choice_index"])
                        if identity in scores:
                            raise ValueError("Duplicate choice scored")
                        scores[identity] = value
                    processed_tokens += sum(row["sequence_length"] for row in batch)
                    number_batches += 1
                rows = []
                for index in sorted(indices):
                    item = items[index]
                    item_choices = choices_by_item[index]
                    likelihoods = [scores[index, choice] for choice in range(len(item["choices"]))]
                    score_values = (
                        [
                            value / length
                            for value, length in zip(
                                likelihoods, item["normalization_lengths"], strict=True
                            )
                        ]
                        if item["metric"] == "acc_norm"
                        else likelihoods
                    )
                    prediction = int(np.argmax(score_values))
                    rows.append(
                        {
                            "model": key,
                            "task": item["task"],
                            "item_id": item["item_id"],
                            "variant": variant,
                            "gold": item["gold"],
                            "prediction": prediction,
                            "correct": prediction == item["gold"],
                            "choice_log_likelihoods": likelihoods,
                            "choice_scores": score_values,
                            "normalization_lengths": item["normalization_lengths"],
                            "choice_token_counts": [
                                len(row["continuation_ids"]) for row in item_choices
                            ],
                            "context_tokens": item_choices[0]["context_tokens"],
                            "sequence_lengths": [row["sequence_length"] for row in item_choices],
                        }
                    )
            finally:
                remove_attention(originals)
            execution = {
                "batches": number_batches,
                "input_tokens": processed_tokens,
                "calls_per_batch": 0 if variant == "bf16" else layer_count,
                "all_layer_call_order_checked": variant != "bf16",
                "elapsed_seconds_for_sizing": time.monotonic() - started,
                "token_budget": token_budget,
            }
            variant_metadata[variant] = execution
            all_rows.extend(rows)
            if not pilot:
                write_gzip(
                    stored,
                    {
                        "plan_sha256": plan_sha,
                        "item_manifest_sha256": items_sha,
                        "rows": rows,
                        "execution": execution,
                        "cuda_kernel_trace": trace if variant == "tile" else [],
                    },
                )
                if commit_callback is not None:
                    commit_callback()
            print(
                f"{key} {variant}: {len(rows)} items, {number_batches} exact-length batches",
                flush=True,
            )
        if len(all_rows) != len(indices) * len(plan["variants"]):
            raise ValueError(
                "Missing item/variant results; never publish a partial model as complete"
            )
        result = {
            "schema_version": 1,
            "model": key,
            "revision": entry["revision"],
            "mode": "pilot" if pilot else "full",
            "plan_sha256": plan_sha,
            "item_manifest_sha256": items_sha,
            "variants": plan["variants"],
            "runtime": runtime(),
            "adapter": fa3.describe(),
            "rows": all_rows,
            "cuda_kernel_trace": sorted(set(trace)),
            "execution": variant_metadata,
            "total_elapsed_seconds_for_sizing": time.monotonic() - model_started,
        }
        output_name = f"{'pilot-' if pilot else ''}{key}.json.gz"
        output_path = volume_path / "accuracy-results" / output_name
        write_gzip(output_path, result)
        files[output_name] = base64.b64encode(output_path.read_bytes()).decode()
        sizing[key] = {
            "items_per_variant": len(indices),
            "execution": variant_metadata,
            "total_elapsed_seconds_for_sizing": result["total_elapsed_seconds_for_sizing"],
            "full_input_tokens_per_variant": token_manifest["models"][key][
                "input_tokens_per_variant"
            ],
        }
        if commit_callback is not None:
            commit_callback()
        del model, encoded, chosen, result
        torch.cuda.empty_cache()
    return {
        "mode": "pilot" if pilot else "full",
        "files": files,
        "sizing": sizing,
        "plan_sha256": plan_sha,
        "item_manifest_sha256": items_sha,
        "volume": "attention-numerics-day-models",
    }
