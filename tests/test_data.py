import hashlib
import json
from pathlib import Path

import numpy as np

from attention import cast


def test_real_capture_contains_exact_bf16_operands():
    path = Path("data/qwen-qkv.npz")
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["capture_sha256"]
    assert (
        hashlib.sha256(Path("data/alice.txt").read_bytes()).hexdigest() == metadata["text_sha256"]
    )
    with np.load(path) as arrays:
        assert arrays["input_ids"].shape == (1, 1024)
        for layer in [0, 12]:
            for head in [0, 7]:
                for name in ["q", "k", "v"]:
                    values = arrays[f"layer{layer}_head{head}_{name}"]
                    assert values.shape == (1024, 64)
                    assert values.dtype == np.float32
                    assert np.all(np.isfinite(values))
                    np.testing.assert_array_equal(cast(values, "bf16"), values)
