"""Selective CPU safetensors reads; no whole-checkpoint materialization.

Layer tensors are copied into owned storage before their file mapping is closed.
Embedding rows and output-vocabulary intervals are selected in the source dtype,
so an F32 checkpoint never requires a full converted embedding or language head.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open


class SafeTensorWeights:
    """Read a local Transformers single-file or indexed sharded checkpoint."""

    def __init__(self, snapshot: Path):
        self.snapshot = Path(snapshot)
        index = self.snapshot / "model.safetensors.index.json"
        single = self.snapshot / "model.safetensors"
        if index.is_file():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        elif single.is_file():
            with safe_open(single, framework="pt", device="cpu") as handle:
                self.weight_map = dict.fromkeys(handle.keys(), single.name)
        else:
            raise FileNotFoundError(f"No safetensors checkpoint in {self.snapshot}")

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def load_module(self, module: torch.nn.Module, prefix: str, dtype: torch.dtype) -> None:
        """Populate a meta module with exactly its own weights using assign=True."""
        state = {}
        for local_name in module.state_dict():
            name = prefix + local_name
            shard = self.weight_map[name]
            # Close each mapping immediately: converted F32 source pages must
            # not accumulate while the remaining layer tensors are loaded.
            with safe_open(self.snapshot / shard, framework="pt", device="cpu") as handle:
                source = handle.get_tensor(name)
                # copy=True also detaches already-BF16 tensors from their mapping.
                state[local_name] = source.to(dtype=dtype, copy=True)
                del source
        module.load_state_dict(state, strict=True, assign=True)
        module.requires_grad_(False)
        module.eval()

    def rows(self, name: str, indices: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Gather token rows before conversion, coalescing adjacent unique IDs."""
        unique, inverse = torch.unique(indices.reshape(-1), sorted=True, return_inverse=True)
        ids = unique.tolist()
        with safe_open(
            self.snapshot / self.weight_map[name], framework="pt", device="cpu"
        ) as handle:
            view = handle.get_slice(name)
            # An empty slice obtains the row dimensions without reading the table.
            row_shape = view[0:0].shape[1:]
            selected = torch.empty((len(ids), *row_shape), dtype=dtype)
            offset = 0
            while offset < len(ids):
                stop = offset + 1
                while stop < len(ids) and ids[stop] == ids[stop - 1] + 1:
                    stop += 1
                if ids[offset] < 0:
                    raise IndexError("Embedding token IDs must be nonnegative")
                source = view[ids[offset] : ids[stop - 1] + 1]
                # copy_ converts directly into the selected-row destination.
                selected[offset:stop].copy_(source)
                del source
                offset = stop
        return selected.index_select(0, inverse).reshape(*indices.shape, *row_shape)

    def interval(self, name: str, start: int, stop: int, dtype: torch.dtype) -> torch.Tensor:
        """Read an owned first-axis interval, converting only the selected slice."""
        with safe_open(
            self.snapshot / self.weight_map[name], framework="pt", device="cpu"
        ) as handle:
            source = handle.get_slice(name)[start:stop]
            result = source.to(dtype=dtype, copy=True)
            del source
        return result
