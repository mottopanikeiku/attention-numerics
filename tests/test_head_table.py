"""Small real BF16-storage captures, not model-derived study measurements."""

import csv
import json
from importlib import import_module

import numpy as np
import pytest

from attention import reference

torch = pytest.importorskip("torch")
head_table = import_module("study.head_table")


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def captures(tmp_path, monkeypatch):
    specs = [
        {
            "key": key,
            "family": family,
            "revision": revision,
            "config": {
                "num_hidden_layers": 2,
                "num_attention_heads": 18,
                "num_key_value_heads": 2,
                "head_dim": 8,
            },
        }
        for key, family, revision in (
            ("qwen-small", "qwen", "qwen-pinned"),
            ("olmo-small", "olmo", "olmo-pinned"),
        )
    ]
    text_keys = ["alice", "pride", "moby"]
    monkeypatch.setattr(head_table, "models", lambda: specs)
    monkeypatch.setattr(
        head_table, "model_spec", lambda key: next(s for s in specs if s["key"] == key)
    )
    monkeypatch.setattr(head_table, "texts", lambda: [{"key": key} for key in text_keys])
    monkeypatch.setattr(head_table, "CAPTURE_TOKENS", 11)
    work, results = tmp_path / "external", tmp_path / "results"
    design_path = tmp_path / "design.json"
    design_path.write_text(
        json.dumps(
            {
                "calibration_family": "qwen",
                "development_models": ["qwen-small"],
                "evaluation_models": ["olmo-small"],
            }
        )
    )
    monkeypatch.setattr(head_table, "DESIGN_PATH", design_path)
    operands = {}
    rng = np.random.default_rng(172)
    for spec in specs:
        directory = work / spec["key"] / "capture"
        directory.mkdir(parents=True)
        for layer in range(2):
            for text in text_keys:
                q = rng.normal(size=(18, 11, 8)).astype(np.float32)
                k = rng.normal(size=(2, 11, 8)).astype(np.float32)
                v = rng.normal(size=(2, 11, 8)).astype(np.float32)
                # Distinct KV heads and query means expose mapping mistakes.
                k[1] += 3
                v[1] = v[1] * 2 + 1
                q[9:] += 0.75
                tensors = [torch.from_numpy(x).to(torch.bfloat16) for x in (q, k, v)]
                scale = np.float64(0.3)
                operands[(spec["key"], layer, text)] = (*tensors, float(scale))
                np.savez(
                    directory / f"layer_{layer:02d}_{text}.npz",
                    **{
                        name: tensor.view(torch.uint16).numpy()
                        for name, tensor in zip(
                            ("q", "k", "v"),
                            tensors,
                            strict=True,
                        )
                    },
                    scale=scale,
                )
    return work, results, specs, text_keys, operands


def test_all_texts_heads_and_gqa_against_independent_reference(captures, monkeypatch):
    work, results, specs, text_keys, operands = captures
    original_apply, original_predict = head_table.apply_attention, head_table.predict_head
    events = []

    def predict(q, k, v, **kwargs):
        # Predictor receives the independent reference, never a measured output.
        expected = reference(q, k, v, causal=True, scale=kwargs["scale"])
        np.testing.assert_array_equal(kwargs["reference_output"], expected)
        events.append("predict")
        return original_predict(q, k, v, **kwargs)

    def apply(q, k, v, variant, **kwargs):
        assert q.shape[1] <= 8
        assert k.shape[1] == v.shape[1] == 1
        if variant == head_table.VARIANTS[0]:
            assert events[-q.shape[1] :] == ["predict"] * q.shape[1]
        events.append(variant)
        return original_apply(q, k, v, variant, **kwargs)

    monkeypatch.setattr(head_table, "predict_head", predict)
    monkeypatch.setattr(head_table, "apply_attention", apply)
    status = head_table.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    assert status["completed_layers"] == status["new_layers"] == 1
    assert status["remaining_layers"] == 1
    assert status["completed_rows"] == status["combined_rows"] == 3 * 18
    assert status["stop_reason"] == "layer_limit"
    rows = _rows(results / "heads.csv")
    assert {(row["text"], int(row["head"])) for row in rows} == {
        (text, head) for text in text_keys for head in range(18)
    }
    for text in text_keys:
        q, k, v, scale = operands[(specs[0]["key"], 0, text)]
        actual = {
            variant: original_apply(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), variant, scale=scale
            )[0]
            .float()
            .numpy()
            for variant in head_table.VARIANTS
        }
        for row in (row for row in rows if row["text"] == text):
            head = int(row["head"])
            assert int(row["kv_head"]) == head // 9
            assert int(row["n"]) == 11 and int(row["d"]) == 8
            assert float(row["scale"]) == scale
            assert row["revision"] == specs[0]["revision"]
            expected = reference(
                q[head].float().numpy(),
                k[head // 9].float().numpy(),
                v[head // 9].float().numpy(),
                causal=True,
                scale=scale,
            )
            features = original_predict(
                q[head].float().numpy(),
                k[head // 9].float().numpy(),
                v[head // 9].float().numpy(),
                reference_output=expected,
                scale=scale,
            )
            for key, value in features.items():
                assert float(row[key]) == value
            assert int(row["reference_nonfinite"]) == 0
            for variant in head_table.VARIANTS:
                difference = actual[variant][head].astype(np.float64) - expected
                assert float(row[f"{variant}_relative_fro"]) == pytest.approx(
                    np.linalg.norm(difference) / np.linalg.norm(expected),
                    abs=1e-12,
                )
                assert float(row[f"{variant}_max_abs"]) == pytest.approx(np.max(np.abs(difference)))
                assert int(row[f"{variant}_output_nonfinite"]) == 0


def test_resume_and_cross_model_combination_are_idempotent(captures, monkeypatch):
    work, results, specs, _, _ = captures
    first = head_table.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    second = head_table.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    assert first["completed_layers"] == 1
    assert second["complete"] and second["new_layers"] == 1
    # Two-model checkpoint exists even when the second model has pending layers.
    checkpoint = head_table.analyze_model(specs[1]["key"], work, layers=1, results_dir=results)
    assert checkpoint["combined_layers"] == 3
    assert checkpoint["combined_rows"] == 3 * 3 * 18
    before = (results / "heads.csv").read_bytes()

    def no_analysis(*args, **kwargs):
        pytest.fail("completed layers must not be analyzed again")

    monkeypatch.setattr(head_table, "_capture_rows", no_analysis)
    resumed = head_table.analyze_model(specs[0]["key"], work, results_dir=results)
    assert resumed["new_layers"] == 0 and resumed["complete"]
    assert (results / "heads.csv").read_bytes() == before
    assert head_table.combine_tables(work, results)["combined_rows"] == 162
    assert (results / "heads.csv").read_bytes() == before
    rows = _rows(results / "heads.csv")
    identities = [(row["model"], int(row["layer"]), row["text"], int(row["head"])) for row in rows]
    assert identities == sorted(identities)
    assert len(identities) == len(set(identities))
    assert str(work) not in before.decode() and str(results) not in before.decode()


def test_deadline_is_checked_only_between_whole_layers(captures, monkeypatch):
    work, results, specs, _, _ = captures
    clock = iter([0.0, 0.0, 600.0])
    monkeypatch.setattr(head_table.time, "monotonic", lambda: next(clock))
    status = head_table.analyze_model(
        specs[0]["key"], work, layers=2, seconds=540, results_dir=results
    )
    assert status["stop_reason"] == "deadline"
    assert status["completed_layers"] == status["new_layers"] == 1
    assert len(_rows(results / "heads.csv")) == 54
    assert not (work / specs[0]["key"] / "head_table/layer_01.csv").exists()


def test_missing_capture_raises_without_partial_layer(captures):
    work, results, specs, _, _ = captures
    missing = work / specs[0]["key"] / "capture/layer_01_pride.npz"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="layer_01_pride"):
        head_table.analyze_model(specs[0]["key"], work, layers=2, results_dir=results)
    assert (work / specs[0]["key"] / "head_table/layer_00.csv").is_file()
    assert not (work / specs[0]["key"] / "head_table/layer_01.csv").exists()
    assert len(_rows(results / "heads.csv")) == 54


def test_nonfinite_output_raises_instead_of_recording_fake_errors(captures, monkeypatch):
    work, results, specs, _, _ = captures

    def nonfinite(q, *args, **kwargs):
        return torch.full_like(q, float("nan"))

    monkeypatch.setattr(head_table, "apply_attention", nonfinite)
    with pytest.raises(ValueError, match="nonfinite tile output"):
        head_table.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    assert not list((work / specs[0]["key"] / "head_table").glob("layer_*.csv"))
    assert _rows(results / "heads.csv") == []


def test_incomplete_or_duplicate_chunk_is_not_resumed(captures):
    work, results, specs, _, _ = captures
    head_table.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    chunk = work / specs[0]["key"] / "head_table/layer_00.csv"
    rows = _rows(chunk)
    _write_rows(chunk, rows[:-1])
    with pytest.raises(ValueError, match="incomplete layer chunk"):
        head_table.analyze_model(specs[0]["key"], work, results_dir=results)
    _write_rows(chunk, [*rows, rows[0]])
    with pytest.raises(ValueError, match="duplicate or unexpected"):
        head_table.analyze_model(specs[0]["key"], work, results_dir=results)


def test_fit_uses_relative_mse_and_qwen_only_training(captures, monkeypatch):
    work, results, specs, _, _ = captures
    for spec in specs:
        head_table.analyze_model(spec["key"], work, layers=1, results_dir=results)
    wide = _rows(results / "heads.csv")
    for row in wide:
        for variant in head_table.VARIANTS:
            row[f"{variant}_relative_fro"] = "0.5" if row["family"] == "qwen" else "0.75"
            row[f"{variant}_predicted_relative_mse"] = "0.04"
            # Deliberately inconsistent: this convenience feature must not be
            # mistaken for predicted MSE by the CSV-to-long conversion.
            row[f"{variant}_predicted_error"] = "123"
    _write_rows(results / "heads.csv", wide)
    original_summarize = head_table.summarize_fit
    captured = []

    def summarize(rows, **kwargs):
        assert kwargs["train_family"] == "qwen"
        assert kwargs["evaluation_models"] == ["olmo-small"]
        captured.extend(rows)
        return original_summarize(rows, **kwargs)

    monkeypatch.setattr(head_table, "summarize_fit", summarize)
    summary = head_table.fit_table(results)
    assert len(captured) == 2 * 3 * 18 * 4
    assert all(row["predicted_mse"] == 0.04 for row in captured)
    assert all(
        row["observed_mse"] == (0.25 if row["family"] == "qwen" else 0.5625) for row in captured
    )
    assert all(
        row["revision"] == next(s["revision"] for s in specs if s["key"] == row["model"])
        for row in captured
    )
    assert all(row["kv_head"] == row["head"] // 9 for row in captured)
    assert summary["calibration"]["train_models"] == ["qwen-small"]
    assert summary["calibration"]["heldout_families"] == ["olmo"]
    assert summary["calibration"]["train_rows"] == summary["calibration"]["heldout_rows"] == 216
    assert summary["input_table"]["head_text_rows"] == 108
    saved = json.loads((results / "fit.json").read_text())
    assert saved["input_table"]["path"] == "heads.csv"
    assert saved["calibration"] == summary["calibration"]
    intercept = summary["calibration"]["intercept"]
    for row in wide:
        if row["family"] == "olmo":
            for variant in head_table.VARIANTS:
                row[f"{variant}_relative_fro"] = "8"
    _write_rows(results / "heads.csv", wide)
    assert head_table.fit_table(results)["calibration"]["intercept"] == intercept


def test_module_entrypoint_returns_completion_counts_and_fit(captures, capsys):
    work, results, specs, _, _ = captures
    status = head_table.main(
        [
            "--model",
            specs[0]["key"],
            "--work-dir",
            str(work),
            "--layers",
            "1",
            "--seconds",
            "540",
            "--results-dir",
            str(results),
            "--fit",
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["new_layers"] == status["new_layers"] == 1
    assert printed["remaining_layers"] == 1
    assert printed["fit"]["head_text_rows"] == 54
    head_table.main(["--fit", "--results-dir", str(results)])
    printed = json.loads(capsys.readouterr().out)
    assert printed["calibration"]["train_family"] == "qwen"


@pytest.mark.parametrize(
    "defect, message",
    [
        ("dtype", "uint16 BF16 bits"),
        ("shape", "uint16 BF16 bits"),
        ("scale", "float64 scalar"),
        ("nonfinite", "nonfinite k"),
    ],
)
def test_invalid_capture_is_rejected_before_checkpoint(captures, defect, message):
    work, results, specs, _, operands = captures
    model = specs[0]["key"]
    q, k, v, scale = operands[(model, 0, "alice")]
    arrays = {
        name: tensor.view(torch.uint16).numpy().copy()
        for name, tensor in zip(("q", "k", "v"), (q, k, v), strict=True)
    }
    stored_scale = np.float64(scale)
    if defect == "dtype":
        arrays["q"] = q.float().numpy()
    elif defect == "shape":
        arrays["q"] = arrays["q"][:, :-1]
    elif defect == "scale":
        stored_scale = np.float32(scale)
    else:
        corrupt = k.clone()
        corrupt[0, 0, 0] = float("inf")
        arrays["k"] = corrupt.view(torch.uint16).numpy()
    np.savez(work / model / "capture/layer_00_alice.npz", **arrays, scale=stored_scale)
    with pytest.raises(ValueError, match=message):
        head_table.analyze_model(model, work, layers=1, results_dir=results)
    assert not list((work / model / "head_table").glob("layer_*.csv"))
