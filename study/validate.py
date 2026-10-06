"""Measured CPU conversion and captured-operand parity checks."""

import argparse
import gc
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import torch

from attention import Config, emulate, raw_storage_cast
from study.attention import apply_attention
from study.data import ROOT, model_spec, snapshot, texts, work_directory


def save_record(directory, name, record):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def errors(actual, expected):
    difference = actual.double() - expected.double()
    denominator = torch.linalg.vector_norm(expected.double())
    if not denominator:
        raise ValueError("relative parity error is undefined for a zero reference")
    return {
        "relative_fro": float(torch.linalg.vector_norm(difference) / denominator),
        "max_abs": float(difference.abs().max()),
        "bit_exact": bool(
            torch.equal(
                actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
            )
        ),
        "elements": actual.numel(),
    }


def storage(directory):
    words = np.arange(65536, dtype=np.uint16)
    values = words.view(ml_dtypes.bfloat16).astype(np.float32)
    native = torch.from_numpy(words.copy()).view(torch.bfloat16).float()
    cases = {}
    for mode in ("raw", "saturating"):
        source = native if mode == "raw" else native.clamp(-448, 448)
        wanted = source.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        emulated = raw_storage_cast(
            values if mode == "raw" else np.clip(values, -448, 448), "e4m3"
        ).view(np.uint8)
        wrong = np.flatnonzero(wanted != emulated)
        cases[mode] = {
            "patterns": len(words),
            "mismatches": len(wrong),
            "first_mismatches": [
                {
                    "bf16_hex": f"{int(words[i]):04x}",
                    "torch_byte": int(wanted[i]),
                    "emulator_byte": int(emulated[i]),
                }
                for i in wrong[:10]
            ],
            "nan_outputs": int(np.sum((wanted & 127) == 127)),
            "negative_zero_outputs": int(np.sum(wanted == 128)),
        }
    record = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "ml_dtypes": ml_dtypes.__version__,
        "device": "cpu",
        "format": "float8_e4m3fn",
        "input_domain": "All 65536 BF16 storage patterns, including signed NaNs and infinities.",
        "cases": cases,
        "limitation": "CPU conversion only; does not validate a GPU GEMM or accumulator.",
    }
    save_record(directory, "storage.json", record)
    if any(case["mismatches"] for case in cases.values()):
        raise AssertionError("E4M3 storage-byte mismatch")


def native_qwen(directory):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from study.stream import LayerStream

    spec = model_spec("qwen05")
    location = snapshot("qwen05")
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True)
    text = texts()[0]
    ids = torch.tensor(
        tokenizer((ROOT / text["path"]).read_text(), add_special_tokens=False)["input_ids"][:64]
    ).unsqueeze(0)
    model = AutoModelForCausalLM.from_pretrained(
        location, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval()
    layers = []
    handles = [
        layer.register_forward_hook(
            lambda module, inputs, output: layers.append(output.detach().clone())
        )
        for layer in model.model.layers
    ]
    with torch.inference_mode():
        expected = model(ids, use_cache=False, output_hidden_states=True)
    for handle in handles:
        handle.remove()
    del model
    gc.collect()
    stream = LayerStream(location)
    checked = []
    with torch.inference_mode():
        hidden = stream.embed(ids)
        checked.append({"stage": "embedding", **errors(hidden, expected.hidden_states[0])})
        for index in range(stream.layer_count):
            hidden = stream.forward_layer(index, hidden)
            checked.append({"stage": f"layer_{index}", **errors(hidden, layers[index])})
        stream.release_current()
        hidden = stream.finish(hidden)
        checked.append({"stage": "final_norm", **errors(hidden, expected.hidden_states[-1])})
        for start, stop in ((0, 127), (500, 691), (150000, stream.config.vocab_size)):
            output = stream.project_logits(hidden, start, stop)
            checked.append(
                {
                    "stage": f"logits_{start}_{stop}",
                    **errors(output, expected.logits[..., start:stop]),
                }
            )
    stream.close()
    record = {
        "model": "qwen05",
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "text": text["key"],
        "tokens": ids.shape[1],
        "dtype": "bf16",
        "device": "cpu",
        "torch": torch.__version__,
        "attention_implementation": "sdpa",
        "checks": checked,
        "relative_fro_tolerance": 0.003,
        "scope": "Native Transformers whole-model forward versus one-layer stream; no FP8.",
    }
    save_record(directory, "native_qwen05.json", record)
    if any(check["relative_fro"] > 0.003 for check in checked):
        raise AssertionError("native layer-stream parity failed")


def fast(key, cache, directory):
    spec = model_spec(key)
    layer_count = spec["config"]["num_hidden_layers"]
    checked = []
    for layer in sorted({0, layer_count // 2, layer_count - 1}):
        for text in (texts()[0],):
            path = cache / key / "capture" / f"layer_{layer:02d}_{text['key']}.npz"
            with np.load(path, allow_pickle=False) as captured:
                q, k, v = [
                    torch.from_numpy(captured[name].copy()).view(torch.bfloat16)
                    for name in ("q", "k", "v")
                ]
                scale = float(captured["scale"])
            heads, kvheads = q.shape[0], k.shape[0]
            for head in sorted({0, heads - 1}):
                kvhead = head // (heads // kvheads)
                a, b, c = q[head], k[kvhead], v[kvhead]
                for variant in ("tile", "rotate", "smooth_k", "rotate_smooth_k", "smooth_kq"):
                    config = Config(
                        storage="e4m3",
                        scaling="tile",
                        causal=True,
                        scale=scale,
                        rotate="rotate" in variant,
                        smooth_k="smooth_k" in variant,
                        smooth_q=variant == "smooth_kq",
                    )
                    wanted = torch.from_numpy(
                        emulate(a.float().numpy(), b.float().numpy(), c.float().numpy(), config)
                    )
                    actual = apply_attention(
                        a[None, None], b[None, None], c[None, None], variant, scale=scale
                    )[0, 0].float()
                    checked.append(
                        {
                            "layer": layer,
                            "head": head,
                            "kv_head": kvhead,
                            "text": text["key"],
                            "variant": variant,
                            **errors(actual, wanted),
                        }
                    )
    record = {
        "model": key,
        "revision": spec["revision"],
        "torch": torch.__version__,
        "tokens": 1024,
        "relative_fro_tolerance": 0.003,
        "checks": checked,
        "selection": (
            "Alice; first/middle/last layer; first/last query head; all five FP8 variants."
        ),
        "scope": "Torch fast surrogate versus independent full NumPy emulator, not native BF16.",
    }
    save_record(directory, f"fast_{key}.json", record)
    if any(check["relative_fro"] > 0.003 for check in checked):
        raise AssertionError("captured-head fast/full parity failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("storage", "native", "fast"))
    parser.add_argument("--model", default="qwen05")
    parser.add_argument("--work-dir", type=Path, default=work_directory())
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/v2/validation")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "storage":
        storage(args.output_dir)
    elif args.command == "native":
        native_qwen(args.output_dir)
    else:
        fast(args.model, args.work_dir, args.output_dir)
