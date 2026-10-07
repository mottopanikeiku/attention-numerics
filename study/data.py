"""Pinned model/text inputs and portable cache records for the cross-model study."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download, try_to_load_from_cache

ROOT = Path(__file__).resolve().parents[1]


def models():
    return json.loads((ROOT / "data/v2/models.json").read_text())["models"]


def texts():
    return json.loads((ROOT / "data/v2/texts.json").read_text())["texts"]


def model_spec(key):
    return next(model for model in models() if model["key"] == key)


def work_directory():
    return Path(
        os.environ.get("ATTENTION_NUMERICS_CACHE", Path.home() / ".cache/attention-numerics")
    )


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path, spec):
    size = path.stat().st_size
    if size != spec["bytes"]:
        raise ValueError(f"wrong size: {spec['path']}")
    sha256 = digest(path)
    if spec.get("sha256"):
        if sha256 != spec["sha256"]:
            raise ValueError(f"checksum mismatch: {spec['path']}")
    elif spec.get("git_blob_sha1"):
        check = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                check.update(chunk)
        if check.hexdigest() != spec["git_blob_sha1"]:
            raise ValueError(f"git-blob mismatch: {spec['path']}")
    else:
        raise ValueError(f"missing input checksum: {spec['path']}")
    return {"path": spec["path"], "bytes": size, "sha256": sha256}


def snapshot(key, offline=True):
    spec = model_spec(key)
    return Path(
        snapshot_download(
            spec["model_id"],
            revision=spec["revision"],
            allow_patterns=[file["path"] for file in spec["files"]],
            local_files_only=offline,
            max_workers=2,
        )
    )


def download(key, destination=Path("results/v2/downloads")):
    spec = model_spec(key)
    present = {
        file["path"]: isinstance(
            try_to_load_from_cache(spec["model_id"], file["path"], revision=spec["revision"]), str
        )
        for file in spec["files"]
    }
    location = snapshot(key, offline=False)
    checked = [verify_file(location / file["path"], file) for file in spec["files"]]
    weight_files = [file for file in spec["files"] if file["role"] == "weights"]
    result = {
        "model": key,
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "files": checked,
        "weight_bytes": sum(file["bytes"] for file in weight_files),
        "new_weight_bytes": sum(
            file["bytes"] for file in weight_files if not present[file["path"]]
        ),
        "cache_location": "$HF_HOME/hub/"
        + "models--"
        + spec["model_id"].replace("/", "--")
        + "/snapshots/"
        + spec["revision"],
    }
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{key}.json"
    # A resumed download must not erase its original accounting record.
    if target.exists():
        prior = json.loads(target.read_text())
        result["new_weight_bytes"] = max(prior["new_weight_bytes"], result["new_weight_bytes"])
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"model": key, "verified_files": len(checked), "weight_bytes": result["weight_bytes"]},
            sort_keys=True,
        ),
        flush=True,
    )
    return location


def prepare_tokens(key, cache):
    import numpy as np
    from transformers import AutoTokenizer

    spec = model_spec(key)
    tokenizer = AutoTokenizer.from_pretrained(snapshot(key), local_files_only=True)
    directory = cache / key / "tokens"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for text in texts():
        path = ROOT / text["path"]
        if digest(path) != text["sha256"]:
            raise ValueError(f"text checksum mismatch: {text['key']}")
        ids = tokenizer(path.read_text(), add_special_tokens=False)["input_ids"]
        if len(ids) < 2049:
            raise ValueError(f"{key}/{text['key']} has only {len(ids)} tokens")
        np.savez(
            directory / f"{text['key']}.npz",
            capture=np.asarray(ids[:1024], dtype=np.int64),
            heldout=np.asarray(ids[1024:2049], dtype=np.int64),
        )
        records.append(
            {
                "text": text["key"],
                "sha256": text["sha256"],
                "tokens_available": len(ids),
                "capture_token_interval": [0, 1024],
                "heldout_token_interval": [1024, 2049],
            }
        )
    output = ROOT / "results/v2/tokens"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{key}.json").write_text(
        json.dumps(
            {
                "model": key,
                "model_id": spec["model_id"],
                "revision": spec["revision"],
                "add_special_tokens": False,
                "records": records,
                "heldout_scope": (
                    "Disjoint from capture/fit; no guarantee of exclusion from pretraining."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"model": key, "texts": records}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "tokens"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", type=Path, default=work_directory())
    args = parser.parse_args()
    if args.command == "download":
        download(args.model)
    else:
        prepare_tokens(args.model, args.work_dir)
