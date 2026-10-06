"""Resume short layer-streaming capture and teacher-forced downstream stages."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from study.attention import VARIANTS
from study.data import ROOT, digest, model_spec, prepare_tokens, snapshot, texts, work_directory
from study.stream import LayerStream


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def bits(tensor):
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Expected actual BF16 tensors, received {tensor.dtype}")
    return tensor.detach().contiguous().view(torch.uint16).numpy()


def ensure_tokens(key, cache):
    if not all((cache / key / "tokens" / f"{text['key']}.npz").exists() for text in texts()):
        prepare_tokens(key, cache)


def token_arrays(key, cache, name):
    arrays = {}
    for text in texts():
        with np.load(cache / key / "tokens" / f"{text['key']}.npz", allow_pickle=False) as record:
            arrays[text["key"]] = torch.from_numpy(record[name].copy()).long().unsqueeze(0)
    return arrays


def load_state(path):
    with np.load(path, allow_pickle=False) as record:
        layer, finished = int(record["next_layer"]), bool(record["finished"])
        states = {
            name: torch.from_numpy(record[name].copy()).view(torch.bfloat16)
            for name in record.files
            if name not in {"next_layer", "finished"}
        }
    return layer, finished, states


def save_state(path, layer, finished, states):
    atomic_npz(
        path,
        next_layer=np.asarray(layer),
        finished=np.asarray(finished),
        **{name: bits(value) for name, value in states.items()},
    )


@torch.inference_mode()
def capture(key, cache, layers=4, seconds=540):
    start = time.monotonic()
    spec = model_spec(key)
    ensure_tokens(key, cache)
    stream = LayerStream(snapshot(key))
    state_path = cache / key / "capture/state.npz"
    resuming = state_path.exists()
    if resuming:
        next_layer, _, states = load_state(state_path)
    else:
        next_layer = 0
        states = {
            text: stream.embed(ids) for text, ids in token_arrays(key, cache, "capture").items()
        }
    if stream.layer_count != spec["config"]["num_hidden_layers"]:
        raise ValueError("Pinned and loaded layer counts disagree")
    manifest_path = ROOT / "results/v2/captures" / f"{key}.json"
    previous = (
        json.loads(manifest_path.read_text())
        if resuming and manifest_path.exists()
        else {"files": []}
    )
    files = {(item["layer"], item["text"]): item for item in previous["files"]}
    completed = 0
    for index in range(next_layer, stream.layer_count):
        if completed >= layers or (completed and time.monotonic() - start >= seconds):
            break
        for text_key in states:
            called = []

            def retain(layer, q, k, v, scale, text_key=text_key, called=called):
                if called:
                    raise ValueError("A decoder layer invoked capture more than once")
                called.append(True)
                filename = f"{key}/capture/layer_{layer:02d}_{text_key}.npz"
                target = cache / filename
                atomic_npz(
                    target,
                    q=bits(q[0]),
                    k=bits(k[0]),
                    v=bits(v[0]),
                    scale=np.asarray(scale, dtype=np.float64),
                )
                files[(layer, text_key)] = {
                    "layer": layer,
                    "text": text_key,
                    "file": filename,
                    "sha256": digest(target),
                    "q_shape": list(q.shape[1:]),
                    "k_shape": list(k.shape[1:]),
                    "v_shape": list(v.shape[1:]),
                    "dtype": "bf16",
                    "scale": float(scale),
                }

            states[text_key] = stream.forward_layer(index, states[text_key], capture=retain)
            if not called:
                raise ValueError("Native decoder did not invoke capture")
        next_layer = index + 1
        save_state(state_path, next_layer, False, states)
        manifest = {
            "model": key,
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "family": spec["family"],
            "tokens": next(iter(states.values())).shape[1],
            "next_layer": next_layer,
            "total_layers": stream.layer_count,
            "complete": next_layer == stream.layer_count,
            "files": [files[identity] for identity in sorted(files)],
            "capture_point": "Actual native BF16 post-RoPE Q/K, pre-GQA-duplication K/V.",
            "attention_baseline": "Unmodified Transformers SDPA BF16 attention.",
            "storage": "External NPZ captures; manifest filenames are relative to work-dir.",
        }
        atomic_json(manifest_path, manifest)
        completed += 1
        print(
            json.dumps(
                {
                    "stage": "capture",
                    "model": key,
                    "next_layer": next_layer,
                    "total_layers": stream.layer_count,
                }
            ),
            flush=True,
        )
    stream.close()
    return {
        "model": key,
        "new_layers": completed,
        "next_layer": next_layer,
        "total_layers": stream.layer_count,
        "complete": next_layer == stream.layer_count,
    }


@torch.inference_mode()
def downstream(key, cache, layers=4, seconds=540):
    start = time.monotonic()
    ensure_tokens(key, cache)
    stream = LayerStream(snapshot(key))
    state_path = cache / key / "downstream/state.npz"
    if state_path.exists():
        next_layer, finished, states = load_state(state_path)
    else:
        next_layer, finished = 0, False
        states = {}
        for text, ids in token_arrays(key, cache, "heldout").items():
            embedded = stream.embed(ids[:, :-1])
            # Native decoder operations do not mutate their input hidden state.
            states.update({f"{variant}__{text}": embedded for variant in VARIANTS})
    completed = 0
    if not finished:
        for index in range(next_layer, stream.layer_count):
            if completed >= layers or (completed and time.monotonic() - start >= seconds):
                break
            # One current decoder is reused across all variants and texts.
            # Each forward uses B=1 to bound native dense-softmax scratch memory.
            for name, hidden in states.items():
                variant = name.split("__", 1)[0]
                states[name] = stream.forward_layer(index, hidden, variant=variant)
                if not torch.isfinite(states[name]).all():
                    raise ValueError(f"Nonfinite downstream hidden state: {key}/{index}/{name}")
            next_layer = index + 1
            save_state(state_path, next_layer, False, states)
            completed += 1
            print(
                json.dumps(
                    {
                        "stage": "downstream",
                        "model": key,
                        "next_layer": next_layer,
                        "total_layers": stream.layer_count,
                    }
                ),
                flush=True,
            )
        if next_layer == stream.layer_count:
            stream.release_current()
            states = {name: stream.finish(hidden) for name, hidden in states.items()}
            finished = True
            save_state(state_path, next_layer, True, states)
    stream.close()
    return {
        "model": key,
        "new_layers": completed,
        "next_layer": next_layer,
        "total_layers": stream.layer_count,
        "complete": finished,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "downstream"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", type=Path, default=work_directory())
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=540)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.layers < 1 or args.seconds <= 0 or args.threads < 1:
        parser.error("layers, seconds and threads must be positive")
    torch.set_num_threads(args.threads)
    function = capture if args.stage == "capture" else downstream
    print(json.dumps(function(args.model, args.work_dir, args.layers, args.seconds)), flush=True)
