import gzip
import importlib
import json

import numpy as np
import pytest

pytest.importorskip("torch")
torch = importlib.import_module("torch")
worker = importlib.import_module("study.accuracy.worker")


def choice(length, context=2, index=0):
    count = length - context + 1
    return {
        "item_index": index,
        "choice_index": 0,
        "input_ids": list(range(length)),
        "continuation_ids": [1] * count,
        "context_tokens": context,
        "sequence_length": length,
    }


def test_exact_length_batching_never_pads_or_loses_choices():
    rows = [choice(length, index=index) for index, length in enumerate([4, 3, 4, 4, 3, 9])]
    batches = list(worker.batches(rows, token_budget=8, maximum_batch=2))
    assert [len(batch[0]["input_ids"]) for batch in batches] == [9, 4, 4, 3]
    assert sorted(row["item_index"] for batch in batches for row in batch) == list(range(6))
    assert all(len({row["sequence_length"] for row in batch}) == 1 for batch in batches)
    assert all(len(batch) <= 2 for batch in batches)


def test_prediction_positions_include_first_and_last_choice_targets():
    batch = [choice(4, context=3), choice(4, context=2)]
    row, position, targets, counts = worker.prediction_positions(batch)
    assert row == [0, 0, 1, 1, 1]
    assert position == [2, 3, 1, 2, 3]
    assert targets == [1] * 5
    assert counts == [2, 3]
    assert max(position) == 3  # Forward excludes the final target token itself.
    broken = {**batch[0], "context_tokens": 2}
    with pytest.raises(ValueError, match="prediction positions"):
        worker.prediction_positions([broken])


def test_continuation_only_projection_matches_independent_all_position_reference():
    hidden = (torch.arange(32).reshape(2, 4, 4) / 32).to(torch.bfloat16)
    weights = torch.tensor([[1, 0, -1, 0], [0, 1, 0, -1], [1, 1, 1, 1]], dtype=torch.bfloat16) / 4
    batch = [choice(4, context=3), choice(4, context=2)]
    observed = worker.continuation_scores(hidden, lambda states: states @ weights.T, batch)
    # Compute every position's projected distribution independently in NumPy.
    dense = (hidden.float().numpy() @ weights.float().numpy().T).astype(np.float32)
    maximum = dense.max(axis=-1)
    denominator = np.log(np.exp(dense - maximum[..., None]).sum(axis=-1)) + maximum
    expected = [
        sum(
            float(dense[row, position, 1] - denominator[row, position])
            for position in range(record["context_tokens"] - 1, 4)
        )
        for row, record in enumerate(batch)
    ]
    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=0)


def test_nonfinite_projection_and_wrong_hidden_precision_are_rejected():
    batch = [choice(2)]
    hidden = torch.zeros((1, 2, 3), dtype=torch.bfloat16)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        worker.continuation_scores(
            hidden,
            lambda states: torch.full((len(states), 3), float("nan"), dtype=torch.bfloat16),
            batch,
        )
    with pytest.raises(ValueError, match="hidden states"):
        worker.continuation_scores(hidden.float(), lambda states: states, batch)


def test_checkpoints_are_complete_json_and_repeatable_bytes(tmp_path):
    path = tmp_path / "checkpoint.json.gz"
    payload = {"rows": [{"value": 2.5}], "complete": True}
    worker.write_gzip(path, payload)
    first = path.read_bytes()
    worker.write_gzip(path, payload)
    assert path.read_bytes() == first
    assert json.loads(gzip.decompress(first)) == payload
    assert not path.with_suffix(".gz.tmp").exists()
