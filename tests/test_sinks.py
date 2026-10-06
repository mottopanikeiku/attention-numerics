"""Synthetic BF16 captures and independent dense mathematical sink references."""

import csv
import json
import struct

import numpy as np
import pytest

from study import sinks


def _bits(values):
    """Construct finite BF16 storage by truncating binary32, without a model."""
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _decode(bits):
    # Independent scalar IEEE decoder, not the implementation's NumPy word view.
    return np.asarray(
        [
            struct.unpack(">f", struct.pack(">I", int(word) << 16))[0]
            for word in np.asarray(bits).flat
        ],
        dtype=np.float64,
    ).reshape(bits.shape)


def _dense(q, k, scale, start, prefix=4):
    """Independent full-matrix oracle; dense allocation is restricted to tests."""
    scores = np.einsum("id,jd->ij", q, k) * scale
    mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    scores[mask] = -np.inf
    weights = np.exp(scores - np.max(scores, axis=1)[:, None])
    probabilities = weights / np.sum(weights, axis=1)[:, None]
    return probabilities[start:, 0].mean(), probabilities[start:, :prefix].sum(axis=1).mean()


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sinks.FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("start", [0, 3, 7, 74])
def test_prefix_mass_matches_independent_dense_bf16_softmax(start, monkeypatch):
    rng = np.random.default_rng(821)
    q, k = _bits(rng.normal(size=(75, 5))), _bits(rng.normal(size=(75, 5)))
    # Signed zero and the smallest signed BF16 subnormals must expand exactly.
    q[0, :4] = [0, 0x8000, 1, 0x8001]
    k[1, :4] = [1, 0x8001, 0x3F80, 0xBF80]
    q64, k64 = _decode(q), _decode(k)
    np.testing.assert_array_equal(sinks._fp64(q), q64)
    assert np.signbit(sinks._fp64(q)[0, 1])
    expected1, expected4 = _dense(q64, k64, 0.375, start)
    exp = np.exp
    blocks = []

    def bounded_exp(array, **kwargs):
        blocks.append(array.shape)
        assert array.shape[0] <= 32 and array.shape[1] <= 75
        return exp(array, **kwargs)

    monkeypatch.setattr(sinks.np, "exp", bounded_exp)
    actual = sinks.prefix_mass(q, k, 0.375, query_start=start)
    assert actual["query_start"] == start and actual["query_count"] == 75 - start
    assert actual["prefix1_mass"] == pytest.approx(expected1, abs=2e-15)
    assert actual["prefix4_mass"] == pytest.approx(expected4, abs=2e-15)
    assert len(blocks) == (75 - start + 31) // 32
    assert all(rows < 75 for rows, _ in blocks)


@pytest.mark.parametrize("start", [0, 2, 4, 128])
def test_uniform_causal_prefix_has_analytic_mass(start):
    n = 137
    q, k = np.zeros((n, 2), dtype=np.uint16), _bits(np.ones((n, 2)))
    actual = sinks.prefix_mass(q, k, 0.7, query_start=start)
    counts = np.arange(start + 1, n + 1, dtype=np.float64)
    assert actual["query_count"] == n - start
    assert actual["prefix1_mass"] == pytest.approx(np.mean(1 / counts), abs=2e-16)
    assert actual["prefix4_mass"] == pytest.approx(np.mean(np.minimum(4, counts) / counts))


def test_constant_key_offset_cancels_for_every_causal_query():
    rng = np.random.default_rng(16)
    # Exactly representable binary fractions avoid changing BF16 values on shift.
    q = rng.integers(-4, 5, size=(43, 3)).astype(np.float64) / 4
    k = rng.integers(-4, 5, size=(43, 3)).astype(np.float64) / 4
    shifted = k + np.asarray([1, -2, 0.5])
    original = sinks.prefix_mass(_bits(q), _bits(k), 0.375, query_start=2)
    translated = sinks.prefix_mass(_bits(q), _bits(shifted), 0.375, query_start=2)
    assert translated["prefix1_mass"] == pytest.approx(original["prefix1_mass"], abs=2e-16)
    assert translated["prefix4_mass"] == pytest.approx(original["prefix4_mass"], abs=2e-16)


def test_default_warmup_excludes_first_128_queries():
    q, k = _bits(np.ones((161, 1))), _bits(np.zeros((161, 1)))
    k[0, 0] = _bits(np.asarray([2.0]))[0]
    expected1, expected4 = _dense(_decode(q), _decode(k), 0.25, 128)
    actual = sinks.prefix_mass(q, k, 0.25)
    assert actual["query_start"] == 128 and actual["query_count"] == 33
    assert actual["prefix1_mass"] == pytest.approx(expected1)
    assert actual["prefix4_mass"] == pytest.approx(expected4)
    assert actual["prefix4_mass"] < sinks.prefix_mass(q, k, 0.25, query_start=0)["prefix4_mass"]


def test_requested_prefix_width_and_short_input_start():
    q = np.zeros((5, 2), dtype=np.uint16)
    actual = sinks.prefix_mass(q, q, 1.0, query_start=0, prefix_tokens=2)
    assert actual["prefix4_mass"] == pytest.approx(np.mean([1, 1, 2 / 3, 2 / 4, 2 / 5]))
    with pytest.raises(ValueError, match="query_start"):
        sinks.prefix_mass(q, q, 1.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_operands_raise_instead_of_reporting_sink_mass(bad):
    q, k = np.ones((5, 2)), np.ones((5, 2))
    k[-1, 0] = bad
    with pytest.raises(ValueError, match="nonfinite.*probabilities"):
        sinks.prefix_mass(q, k, 1.0, query_start=0)
    with pytest.raises(ValueError, match="nonfinite.*probabilities"):
        sinks.prefix_mass(_bits(q), _bits(k), 1.0, query_start=0)


def test_nonfinite_fp64_scores_raise_explicitly():
    operands = np.full((5, 2), 1e308)
    with pytest.raises(ValueError, match="nonfinite probabilities"):
        sinks.prefix_mass(operands, operands, 1.0, query_start=0)


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_invalid_scale_is_not_replaced_with_default(scale):
    operands = np.ones((5, 2))
    with pytest.raises(ValueError, match="scale"):
        sinks.prefix_mass(operands, operands, scale, query_start=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_start": -1},
        {"query_start": 5},
        {"query_start": 0.5},
        {"query_start": 0, "prefix_tokens": 0},
        {"query_start": 0, "prefix_tokens": 1.5},
    ],
)
def test_invalid_query_interval_or_prefix_is_rejected(kwargs):
    with pytest.raises(ValueError):
        sinks.prefix_mass(np.ones((5, 2)), np.ones((5, 2)), 1.0, **kwargs)


@pytest.fixture
def captures(tmp_path, monkeypatch):
    specs = [
        {
            "key": key,
            "family": family,
            "revision": f"{key}-pinned",
            "config": {
                "num_hidden_layers": 2,
                "num_attention_heads": 6,
                "num_key_value_heads": 2,
                "head_dim": 3,
            },
        }
        for key, family in (("qwen-small", "qwen"), ("olmo-small", "olmo"))
    ]
    books = ["pride", "alice", "moby"]
    monkeypatch.setattr(sinks, "models", lambda: specs)
    monkeypatch.setattr(sinks, "model_spec", lambda key: next(s for s in specs if s["key"] == key))
    monkeypatch.setattr(sinks, "texts", lambda: [{"key": key} for key in books])
    # Only fixture-based table tests shorten the interval. Production stays 1024/128.
    monkeypatch.setattr(sinks, "CAPTURE_TOKENS", 39)
    monkeypatch.setattr(sinks, "QUERY_START", 4)
    work, results = tmp_path / "external", tmp_path / "results"
    operands = {}
    rng = np.random.default_rng(237)
    for spec in specs:
        directory = work / spec["key"] / "capture"
        directory.mkdir(parents=True)
        for layer in range(2):
            for text in books:
                q, k = rng.normal(size=(6, 39, 3)), rng.normal(size=(2, 39, 3))
                q[3:] += 0.25
                k[1, :4] += 2
                qbits, kbits = _bits(q), _bits(k)
                operands[(spec["key"], layer, text)] = (_decode(qbits), _decode(kbits))
                np.savez(
                    directory / f"layer_{layer:02d}_{text}.npz",
                    q=qbits,
                    k=kbits,
                    v=np.zeros_like(kbits),
                    scale=np.float64(0.375),
                )
    return work, results, specs, books, operands


def test_all_heads_texts_gqa_and_scale_match_dense_reference(captures):
    work, results, specs, books, operands = captures
    status = sinks.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    assert status["new_layers"] == status["completed_layers"] == 1
    assert status["remaining_layers"] == 1 and status["stop_reason"] == "layer_limit"
    assert status["combined_rows"] == status["completed_rows"] == 18
    rows = _rows(results / "sinks.csv")
    assert {(row["text"], int(row["head"])) for row in rows} == {
        (text, head) for text in books for head in range(6)
    }
    assert list(rows[0]) == list(sinks.FIELDS)
    for row in rows:
        head = int(row["head"])
        assert int(row["kv_head"]) == head // 3
        assert row["model"] == specs[0]["key"] and row["revision"] == specs[0]["revision"]
        assert int(row["n"]) == 39 and int(row["d"]) == 3 and float(row["scale"]) == 0.375
        assert int(row["query_start"]) == 4 and int(row["query_count"]) == 35
        q, k = operands[(specs[0]["key"], 0, row["text"])]
        expected1, expected4 = _dense(q[head], k[head // 3], 0.375, 4)
        assert float(row["prefix1_mass"]) == pytest.approx(expected1, abs=2e-15)
        assert float(row["prefix4_mass"]) == pytest.approx(expected4, abs=2e-15)


def test_resume_and_cross_model_combination_have_stable_unique_rows(captures, monkeypatch):
    work, results, specs, _, _ = captures
    first = sinks.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    second = sinks.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    assert first["completed_layers"] == 1
    assert second["complete"] and second["new_layers"] == 1
    partial = sinks.analyze_model(specs[1]["key"], work, layers=1, results_dir=results)
    assert partial["combined_layers"] == 3 and partial["combined_rows"] == 54
    before = (results / "sinks.csv").read_bytes()

    def no_analysis(*args, **kwargs):
        pytest.fail("resuming completed layers must not read their captures again")

    monkeypatch.setattr(sinks, "_capture_rows", no_analysis)
    for path in (work / specs[0]["key"] / "capture").glob("*.npz"):
        path.unlink()
    resumed = sinks.analyze_model(specs[0]["key"], work, results_dir=results)
    assert resumed["complete"] and resumed["new_layers"] == 0
    assert resumed["completed_rows"] == 36
    assert (results / "sinks.csv").read_bytes() == before
    assert sinks.combine_tables(work, results) == {"combined_layers": 3, "combined_rows": 54}
    assert (results / "sinks.csv").read_bytes() == before
    rows = _rows(results / "sinks.csv")
    identities = [(r["model"], int(r["layer"]), r["text"], int(r["head"])) for r in rows]
    assert identities == sorted(identities) and len(identities) == len(set(identities))
    assert str(work) not in before.decode() and str(results) not in before.decode()


def test_missing_capture_preserves_only_prior_complete_layers(captures):
    work, results, specs, _, _ = captures
    missing = work / specs[0]["key"] / "capture/layer_01_pride.npz"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="layer_01_pride"):
        sinks.analyze_model(specs[0]["key"], work, layers=2, results_dir=results)
    assert (work / specs[0]["key"] / "sinks/layer_00.csv").is_file()
    assert not (work / specs[0]["key"] / "sinks/layer_01.csv").exists()
    assert len(_rows(results / "sinks.csv")) == 18


def test_incomplete_and_duplicate_chunks_cannot_resume_or_replace_table(captures):
    work, results, specs, _, _ = captures
    sinks.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    published = (results / "sinks.csv").read_bytes()
    chunk = work / specs[0]["key"] / "sinks/layer_00.csv"
    rows = _rows(chunk)
    for malformed, message in (
        (rows[:-1], "incomplete layer chunk"),
        ([*rows, rows[0]], "duplicate or unexpected"),
    ):
        _write_rows(chunk, malformed)
        with pytest.raises(ValueError, match=message):
            sinks.analyze_model(specs[0]["key"], work, results_dir=results)
        with pytest.raises(ValueError, match=message):
            sinks.combine_tables(work, results)
        assert (results / "sinks.csv").read_bytes() == published


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "wrong-pin"),
        ("kv_head", "1"),
        ("query_start", "0"),
        ("query_count", "39"),
        ("prefix4_mass", "nan"),
        ("prefix4_mass", "1.5"),
    ],
)
def test_invalid_chunk_identity_interval_or_mass_is_rejected(captures, field, value):
    work, results, specs, _, _ = captures
    sinks.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    chunk = work / specs[0]["key"] / "sinks/layer_00.csv"
    rows = _rows(chunk)
    rows[0][field] = value
    _write_rows(chunk, rows)
    with pytest.raises(ValueError):
        sinks.combine_tables(work, results)


def test_invalid_capture_never_checkpoints_partial_layer(captures):
    work, results, specs, _, _ = captures
    path = work / specs[0]["key"] / "capture/layer_00_pride.npz"
    with np.load(path) as capture:
        q, k, scale = capture["q"], capture["k"], capture["scale"]
    k[1, -1, 0] = 0x7F80
    np.savez(path, q=q, k=k, scale=scale)
    with pytest.raises(ValueError, match="nonfinite k"):
        sinks.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    assert not (work / specs[0]["key"] / "sinks/layer_00.csv").exists()
    assert _rows(results / "sinks.csv") == []


def test_diagnostic_ignores_value_tensors_and_native_attention_outputs(captures):
    work, results, specs, _, _ = captures
    for path in (work / specs[0]["key"] / "capture").glob("*.npz"):
        with np.load(path) as capture:
            q, k, scale = capture["q"], capture["k"], capture["scale"]
        np.savez(path, q=q, k=k, v=np.full(k.shape, 0x7FC0, dtype=np.uint16), scale=scale)
    status = sinks.analyze_model(specs[0]["key"], work, layers=1, results_dir=results)
    assert status["new_layers"] == 1 and status["combined_rows"] == 18


def test_production_cli_requires_1024_token_captures(captures, monkeypatch):
    work, results, specs, _, _ = captures
    monkeypatch.setattr(sinks, "CAPTURE_TOKENS", 1024)
    monkeypatch.setattr(sinks, "QUERY_START", 128)
    with pytest.raises(ValueError, match=r"shape \(6, 1024, 3\)"):
        sinks.main(
            [
                "--model",
                specs[0]["key"],
                "--work-dir",
                str(work),
                "--layers",
                "4",
                "--seconds",
                "520",
                "--results-dir",
                str(results),
            ]
        )
    assert not (work / specs[0]["key"] / "sinks/layer_00.csv").exists()


def test_cli_spaced_flags_return_resume_counts(captures, capsys):
    work, results, specs, _, _ = captures
    status = sinks.main(
        [
            "--model",
            specs[0]["key"],
            "--work-dir",
            str(work),
            "--layers",
            "1",
            "--seconds",
            "520",
            "--results-dir",
            str(results),
        ]
    )
    assert json.loads(capsys.readouterr().out) == status
    assert status["new_layers"] == 1 and status["combined_rows"] == 18


def test_deadline_stops_only_between_complete_layers(captures, monkeypatch):
    work, results, specs, _, _ = captures
    clock = iter([0.0, 0.0, 521.0])
    monkeypatch.setattr(sinks.time, "monotonic", lambda: next(clock))
    status = sinks.analyze_model(specs[0]["key"], work, layers=2, seconds=520, results_dir=results)
    assert status["stop_reason"] == "deadline" and status["new_layers"] == 1
    assert len(_rows(results / "sinks.csv")) == 18
    assert not (work / specs[0]["key"] / "sinks/layer_01.csv").exists()
