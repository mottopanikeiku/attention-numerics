"""Capture actual post-RoPE BF16 Q/K/V from a pinned small Qwen checkpoint on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main(output, length):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    snapshot = Path(snapshot_download(MODEL, revision=REVISION))
    # Verify the full download against the committed manifest on subsequent runs.
    files = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]
    hashes = {name: sha256(snapshot / name) for name in files}
    manifest_path = Path("data/model-hashes.json")
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())
        if recorded["revision"] != REVISION or recorded["files"] != hashes:
            raise ValueError("download differs from pinned checkpoint manifest")
    else:
        manifest_path.write_text(
            json.dumps({"model": MODEL, "revision": REVISION, "files": hashes}, indent=2) + "\n"
        )
    text_path = Path("data/alice.txt")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    inputs = tokenizer(text_path.read_text(), return_tensors="pt", add_special_tokens=False)
    if inputs["input_ids"].shape[1] < length:
        raise ValueError("excerpt does not contain the requested number of tokens")
    inputs = {key: tensor[:, :length] for key, tensor in inputs.items()}
    model = AutoModel.from_pretrained(
        snapshot, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
    )
    model.eval()
    arrays = {"input_ids": inputs["input_ids"].numpy()}
    original = modeling_qwen2.eager_attention_forward

    class CapturedLastLayer(Exception):
        pass

    def capture(module, query, key, value, attention_mask, **kwargs):
        if module.layer_idx in [0, 12]:
            if not all(tensor.dtype == torch.bfloat16 for tensor in [query, key, value]):
                raise ValueError("capture operands are not BF16")
            for head in [0, 7]:
                kv_head = head // module.num_key_value_groups
                prefix = f"layer{module.layer_idx}_head{head}"
                arrays[f"{prefix}_q"] = query[0, head].float().numpy().copy()
                arrays[f"{prefix}_k"] = key[0, kv_head].float().numpy().copy()
                arrays[f"{prefix}_v"] = value[0, kv_head].float().numpy().copy()
            if module.layer_idx == 12:
                raise CapturedLastLayer
        return original(module, query, key, value, attention_mask, **kwargs)

    modeling_qwen2.eager_attention_forward = capture
    try:
        with torch.inference_mode():
            try:
                model(**inputs, use_cache=False)
            except CapturedLastLayer:
                pass
    finally:
        modeling_qwen2.eager_attention_forward = original
    if len(arrays) != 13:
        raise ValueError("missing requested layer/head captures")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata = {
        "model": MODEL,
        "revision": REVISION,
        "checkpoint_hashes": hashes,
        "capture_sha256": sha256(output),
        "text_sha256": sha256(text_path),
        "text_source": (
            "Lewis Carroll, Alice's Adventures in Wonderland, chapter I, public domain in the US; "
            "https://www.gutenberg.org/files/11/11-0.txt"
        ),
        "tokens": length,
        "layers_zero_based": [0, 12],
        "query_heads_zero_based": [0, 7],
        "kv_heads_zero_based": [0, 1],
        "head_dimension": 64,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": "bfloat16 CPU model forward; captured arrays expanded exactly to float32",
        "position": "after rotary embedding, before grouped-query K/V repetition and attention",
        "preceding_layers": (
            "unmodified Transformers eager attention in BF16; stopped before layer-12 attention"
        ),
        "model_license": "Apache-2.0",
        "threads": 1,
        "timings": "not measured",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/qwen-qkv.npz"))
    parser.add_argument("--tokens", type=int, default=1024)
    args = parser.parse_args()
    main(args.output, args.tokens)
