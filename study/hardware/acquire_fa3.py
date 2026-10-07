"""Image-build helper; run in the isolated huggingface-hub1.10 environment."""

import base64
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REVISION = "7cb368cf8278b583132eb72cbf312d54586df2e2"
VARIANT = "torch-stable-abi29-cu128-x86_64-linux"


def main():
    snapshot_download(
        "kernels-community/flash-attn3",
        repo_type="kernel",
        revision=REVISION,
        allow_patterns=[f"build/{VARIANT}/*"],
        local_dir="/opt/fa3",
        max_workers=2,
    )
    root = Path("/opt/fa3/build") / VARIANT
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata["name"] != "flash-attn3" or metadata["version"] != 1:
        raise ValueError("Unexpected FA3 build metadata")
    if metadata["id"] != "_flash_attn3_cuda_8a730d9":
        raise ValueError("Unexpected FA3 extension identifier")
    if metadata["digest"]["algorithm"] != "sha256":
        raise ValueError("Unexpected FA3 digest algorithm")
    for name, expected in metadata["digest"]["files"].items():
        digest = hashlib.sha256()
        with (root / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if base64.b64encode(digest.digest()).decode() != expected:
            raise ValueError(f"FA3 artifact digest mismatch: {name}")
    extension = root / "_flash_attn3_cuda_8a730d9.abi3.so"
    if extension.stat().st_size != 802030296:
        raise ValueError("Unexpected FA3 extension size")
    print(json.dumps({"revision": REVISION, "variant": VARIANT, "metadata": metadata}))


if __name__ == "__main__":
    main()
