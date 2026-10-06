"""Evaluate committed real Q/K/V, with PyTorch CPU SDPA as a separate baseline."""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from attention import emulate, metrics, reference
from capture import sha256
from sweep import CONFIGS, environment


def main(path, destination):
    metadata = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != metadata["capture_sha256"]:
        raise ValueError("capture checksum mismatch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = np.load(path)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "real.csv").open("w", newline="") as stream:
        writer = None
        for layer in metadata["layers_zero_based"]:
            for head in metadata["query_heads_zero_based"]:
                prefix = f"layer{layer}_head{head}"
                for n in [128, 512, metadata["tokens"]]:
                    q, k, v = (data[f"{prefix}_{name}"][:n] for name in ["q", "k", "v"])
                    expected = reference(q, k, v, causal=True)
                    for name in [
                        "fp32",
                        "fp32_rotate",
                        "bf16",
                        "e4_tensor",
                        "e5_tensor",
                        "e4_tile",
                        "e4_rotate",
                        "e4_no_p_round",
                        "e4_rotate_no_p_round",
                        "e4_reduced",
                        "e4_promoted",
                        "torch_sdpa_fp32",
                        "torch_sdpa_bf16",
                    ]:
                        if name.startswith("torch_sdpa"):
                            dtype = torch.bfloat16 if name.endswith("bf16") else torch.float32
                            operands = [
                                torch.from_numpy(x.copy()).to(dtype)[None, None] for x in [q, k, v]
                            ]
                            with torch.inference_mode():
                                actual = (
                                    functional.scaled_dot_product_attention(
                                        *operands, is_causal=True
                                    )[0, 0]
                                    .float()
                                    .numpy()
                                )
                        else:
                            actual = emulate(q, k, v, replace(CONFIGS[name], causal=True))
                        entry = {
                            "model": metadata["model"],
                            "layer": layer,
                            "head": head,
                            "n": n,
                            "d": q.shape[1],
                            "variant": name,
                            "rows_evaluated": n,
                            "all_rows": True,
                            **metrics(actual, expected),
                        }
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(entry))
                            writer.writeheader()
                        writer.writerow(entry)
                        stream.flush()
                    print(f"Real operands: layer={layer} head={head} N={n}", flush=True)
    summary = environment()
    summary["capture"] = metadata
    summary["torch"] = torch.__version__
    summary["cpu_baseline"] = (
        "torch.nn.functional.scaled_dot_product_attention; causal=True; all queries; "
        "BF16 and FP32; not a GPU run"
    )
    (destination / "real.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/qwen-qkv.npz"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    main(args.input, args.output)
