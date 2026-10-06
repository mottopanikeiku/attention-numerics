"""Real tiny-model checks for checkpointing and full-vocabulary metrics."""

import hashlib
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from study import downstream as likelihood
from study import run
from study.attention import VARIANTS
from study.data import verify_file


@pytest.fixture
def tiny_study(tmp_path, monkeypatch):
    location = tmp_path / "model"
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(49)
    model = Qwen2ForCausalLM(config).eval()
    model.save_pretrained(location)
    spec = {
        "key": "qwen05",
        "model_id": "local-tiny-qwen",
        "revision": "local-test",
        "family": "qwen",
        "config": {"num_hidden_layers": 2},
    }
    books = [{"key": key} for key in ("alice", "pride", "moby")]
    cache = tmp_path / "cache"
    token_dir = cache / "qwen05/tokens"
    token_dir.mkdir(parents=True)
    for index, text in enumerate(books):
        ids = (np.arange(12, dtype=np.int64) + 3 * index) % 97
        np.savez(token_dir / f"{text['key']}.npz", capture=ids[:-1], heldout=ids)
    for module in (run, likelihood):
        monkeypatch.setattr(module, "snapshot", lambda key: location)
        monkeypatch.setattr(module, "model_spec", lambda key: spec)
        monkeypatch.setattr(module, "texts", lambda: books)
    monkeypatch.setattr(run, "ROOT", tmp_path / "published")
    monkeypatch.setattr(likelihood, "models", lambda: [spec])
    native = Qwen2ForCausalLM.from_pretrained(
        location, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval()
    return location, cache, native, books


def test_capture_resumes_without_duplicating_layer_records(tiny_study):
    _, cache, _, books = tiny_study
    first = run.capture("qwen05", cache, layers=1)
    assert first["next_layer"] == 1 and not first["complete"]
    second = run.capture("qwen05", cache, layers=1)
    assert second["complete"] and second["new_layers"] == 1
    again = run.capture("qwen05", cache, layers=1)
    assert again["complete"] and again["new_layers"] == 0
    manifest = json.loads((run.ROOT / "results/v2/captures/qwen05.json").read_text())
    assert manifest["tokens"] == 11 and len(manifest["files"]) == 6
    assert {(item["layer"], item["text"]) for item in manifest["files"]} == {
        (layer, text["key"]) for layer in range(2) for text in books
    }
    with np.load(cache / "qwen05/capture/layer_00_alice.npz") as captured:
        assert captured["q"].dtype == np.uint16
        assert captured["q"].shape == (4, 11, 8)
        assert captured["k"].shape == captured["v"].shape == (2, 11, 8)
        assert float(captured["scale"]) == 8**-0.5


def test_chunked_metrics_match_full_vocab_distribution(tiny_study, tmp_path):
    _, cache, native, books = tiny_study
    first = run.downstream("qwen05", cache, layers=1)
    assert first["next_layer"] == 1 and not first["complete"]
    second = run.downstream("qwen05", cache, layers=1)
    assert second["complete"]
    results = tmp_path / "results"
    records = likelihood.score("qwen05", cache, results_dir=results, vocab_chunk=13)
    layer, finished, states = run.load_state(cache / "qwen05/downstream/state.npz")
    assert layer == 2 and finished
    names = [f"{variant}__{text['key']}" for variant in VARIANTS for text in books]
    hidden = torch.cat([states[name] for name in names], dim=0)
    with torch.inference_mode():
        logits = F.linear(hidden, native.lm_head.weight).float().double().reshape(6, 3, 11, 97)
        logp = logits.log_softmax(dim=-1)
        labels = run.token_arrays("qwen05", cache, "heldout")
        targets = torch.cat([labels[text["key"]][:, 1:] for text in books], dim=0)
        expected_ce = -logp.gather(-1, targets[None, :, :, None].expand(6, -1, -1, -1)).mean(
            dim=(-1, -2)
        )
        expected_kl = (logp[0].exp()[None] * (logp[0][None] - logp)).sum(dim=-1).mean(dim=-1)
    for record in records:
        variant = VARIANTS.index(record["variant"])
        book = [text["key"] for text in books].index(record["text"])
        assert record["tokens"] == 11
        assert record["ce"] == pytest.approx(float(expected_ce[variant, book]), abs=2e-5)
        assert record["mean_kl"] == pytest.approx(float(expected_kl[variant, book]), abs=2e-7)
    assert all(row["mean_kl"] == 0 for row in records if row["variant"] == "bf16")
    assert len(records) == 18
    assert likelihood.combine(cache, results / "downstream.csv") == 18
    # The actual native model's BF16 branch matches the stored reference state.
    ids = labels["alice"][:, :-1]
    with torch.inference_mode():
        expected_hidden = native.model(ids, use_cache=False).last_hidden_state
    torch.testing.assert_close(states["bf16__alice"], expected_hidden, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ("sha256", "git_blob"))
def test_input_hash_validation_uses_actual_bytes(tmp_path, kind):
    payload = b"pinned input\n"
    path = tmp_path / "input.txt"
    path.write_bytes(payload)
    spec = {"path": "input.txt", "bytes": len(payload)}
    if kind == "sha256":
        spec["sha256"] = hashlib.sha256(payload).hexdigest()
    else:
        spec["git_blob_sha1"] = hashlib.sha1(
            f"blob {len(payload)}\0".encode() + payload
        ).hexdigest()
    checked = verify_file(path, spec)
    assert checked["sha256"] == hashlib.sha256(payload).hexdigest()
    path.write_bytes(b"changed data")
    with pytest.raises(ValueError):
        verify_file(path, spec)


def test_new_cache_does_not_inherit_prior_run_capture_entries(tiny_study):
    _, cache, _, books = tiny_study
    run.capture("qwen05", cache, layers=2)
    fresh = cache.parent / "fresh-cache"
    token_dir = fresh / "qwen05/tokens"
    token_dir.mkdir(parents=True)
    for index, text in enumerate(books):
        ids = (np.arange(12, dtype=np.int64) + 3 * index) % 97
        np.savez(token_dir / f"{text['key']}.npz", capture=ids[:-1], heldout=ids)
    run.capture("qwen05", fresh, layers=1)
    manifest = json.loads((run.ROOT / "results/v2/captures/qwen05.json").read_text())
    assert manifest["next_layer"] == 1
    assert len(manifest["files"]) == 3
    assert {record["layer"] for record in manifest["files"]} == {0}
    resumed = run.capture("qwen05", cache, layers=1)
    restored = json.loads((run.ROOT / "results/v2/captures/qwen05.json").read_text())
    assert resumed["complete"] and resumed["new_layers"] == 0
    assert restored["next_layer"] == 2 and len(restored["files"]) == 6
    assert {record["layer"] for record in restored["files"]} == {0, 1}


@pytest.mark.parametrize("destination", ("private", "public"))
def test_capture_recovers_from_interrupted_manifest_publication(
    tiny_study, monkeypatch, destination
):
    _, cache, _, _ = tiny_study
    target = (
        cache / "qwen05/capture/manifest.json"
        if destination == "private"
        else run.ROOT / "results/v2/captures/qwen05.json"
    )
    original = run.atomic_json

    def interrupt(path, value):
        if path == target:
            raise RuntimeError("interrupted publication")
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(run, "atomic_json", interrupt)
        with pytest.raises(RuntimeError, match="interrupted publication"):
            run.capture("qwen05", cache, layers=1)
    result = run.capture("qwen05", cache, layers=2)
    manifest = json.loads((run.ROOT / "results/v2/captures/qwen05.json").read_text())
    private = json.loads((cache / "qwen05/capture/manifest.json").read_text())
    assert result["complete"] and manifest["complete"]
    assert manifest == private
    assert {(record["layer"], record["text"]) for record in manifest["files"]} == {
        (layer, text) for layer in range(2) for text in ("alice", "pride", "moby")
    }


def test_score_commits_metadata_before_completion_marker(tiny_study, tmp_path, monkeypatch):
    _, cache, _, _ = tiny_study
    run.downstream("qwen05", cache, layers=2)
    results = tmp_path / "scores"

    def interrupt(path, value):
        raise RuntimeError("interrupted metadata")

    with monkeypatch.context() as patch:
        patch.setattr(likelihood, "atomic_json", interrupt)
        with pytest.raises(RuntimeError, match="interrupted metadata"):
            likelihood.score("qwen05", cache, results_dir=results, vocab_chunk=13)
    assert not (cache / "qwen05/downstream/metrics.csv").exists()
    likelihood.score("qwen05", cache, results_dir=results, vocab_chunk=13)
    metadata = json.loads((results / "downstream_metadata/qwen05.json").read_text())
    assert metadata["revision"] == "local-test" and metadata["tokens_per_text"] == 11
    assert (cache / "qwen05/downstream/metrics.csv").exists()
