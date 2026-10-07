"""Synthetic fixtures test arithmetic only; they are not published measurements."""

import csv
import json
import math

import numpy as np
import pytest

from study import report


def head_row(model="model-a", layer=0, head=0, text="text-a", tile=0.2, rotate=0.1):
    row = {
        "model": model,
        "family": "test-family",
        "revision": "test-revision",
        "text": text,
        "layer": layer,
        "head": head,
        "q_mean_energy_fraction": 0.25,
        "k_mean_energy_fraction": 0.75,
    }
    errors = {"tile": tile, "rotate": rotate, "smooth_k": tile / 2, "rotate_smooth_k": rotate / 2}
    for variant, error in errors.items():
        row[f"{variant}_relative_fro"] = error
        row[f"{variant}_predicted_error"] = error * 1.5
    return row


def head_fixture():
    rows = []
    for index, (tile, rotate) in enumerate(((0.1, 0.2), (0.4, 0.2), (0.9, 0.3))):
        row = head_row(text=f"text-{index}", tile=tile, rotate=rotate)
        row["q_mean_energy_fraction"] = (index + 1) / 5
        row["k_mean_energy_fraction"] = (index + 2) / 5
        rows.append(row)
        rows.append(head_row(head=1, text=f"text-{index}", tile=tile, rotate=tile + 1))
        rows.append(head_row(head=2, text=f"text-{index}", tile=tile, rotate=tile))
    return rows


def downstream_fixture():
    rows = []
    for index, variant in enumerate(report.DOWNSTREAM_VARIANTS):
        for text, tokens, ce, kl in (("short", 1, 1.0, 0.02), ("long", 3, 3.0, 0.06)):
            ce += index * 0.1
            rows.append(
                {
                    "model": "model-a",
                    "family": "test-family",
                    "revision": "test-revision",
                    "variant": variant,
                    "text": text,
                    "tokens": tokens,
                    "ce": ce,
                    "exp_ce": math.exp(ce),
                    "mean_kl": 0 if index == 0 else kl * index,
                }
            )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_predictions_average_per_text_gains_not_errors_or_independent_texts():
    summary, points = report.aggregate_heads(head_fixture())
    assert summary["counts"]["physical_heads"] == 3
    assert summary["counts"]["prediction_datapoints_per_panel"] == 3
    assert summary["counts"]["head_text_measurements"] == 9
    assert summary["counts"]["texts"] == 3
    assert summary["counts"]["texts_per_head"]["min"] == 3
    assert len(points) == 3
    point = points[0]
    before, after = np.array([0.1, 0.4, 0.9]), np.array([0.2, 0.2, 0.3])
    expected = float(np.mean(np.log1p(before) - np.log1p(after)))
    assert point["rotation_observed_gain"] == pytest.approx(expected)
    assert point["rotation_predicted_gain"] == pytest.approx(
        np.mean(np.log1p(before * 1.5) - np.log1p(after * 1.5))
    )
    assert expected != pytest.approx(math.log1p(before.mean()) - math.log1p(after.mean()))
    assert point["texts"] == ["text-0", "text-1", "text-2"]


def test_transform_fractions_have_exact_physical_head_and_row_denominators():
    summary, _ = report.aggregate_heads(head_fixture())
    transform = summary["transforms"]["rotation"]
    physical = transform["physical_head_mean_gains"]
    assert physical["denominator"] == 3
    assert (physical["helped"], physical["hurt"], physical["tied"]) == (1, 1, 1)
    assert physical["help_fraction"] == pytest.approx(1 / 3)
    measured = transform["head_text_measurements"]
    assert measured["denominator"] == 9
    assert (measured["helped"], measured["hurt"], measured["tied"]) == (2, 4, 3)
    assert measured["hurt_fraction"] == pytest.approx(4 / 9)


def test_actual_error_and_energy_summaries_are_descriptive_saved_values():
    rows = head_fixture()
    summary, _ = report.aggregate_heads(rows)
    errors = [row["tile_relative_fro"] for row in rows]
    stats = summary["actual_relative_fro"]["tile"]["head_text_measurements"]
    assert stats["n"] == 9
    assert stats["mean"] == pytest.approx(np.mean(errors))
    assert stats["median"] == pytest.approx(np.median(errors))
    assert stats["p95"] == pytest.approx(np.quantile(errors, 0.95))
    assert (stats["min"], stats["max"]) == (0.1, 0.9)
    for operand in ("q", "k"):
        values = [row[f"{operand}_mean_energy_fraction"] for row in rows]
        energy = summary["per_model"]["model-a"]["mean_energy_fraction"][operand]
        assert energy["head_text_measurements"]["median"] == pytest.approx(np.median(values))
        assert energy["head_text_measurements"]["min"] == min(values)
        assert energy["head_text_measurements"]["max"] == max(values)
        assert energy["physical_head_text_means"]["n"] == 3


def test_head_and_layer_identities_include_model_and_all_heads_are_retained():
    rows = [
        head_row(model=model, layer=layer, head=head, text=text)
        for model in ("model-a", "model-b")
        for layer in (0, 1)
        for head in range(31)
        for text in ("text-a", "text-b", "text-c")
    ]
    summary, points = report.aggregate_heads(rows)
    assert summary["counts"]["models"] == 2
    assert summary["counts"]["layers"] == 4
    assert summary["counts"]["physical_heads"] == 124
    assert summary["counts"]["head_text_measurements"] == 372
    assert len(points) == 124
    assert len({(point["model"], point["layer"], point["head"]) for point in points}) == 124
    assert summary["per_model"]["model-b"]["counts"]["layers"] == 2


def test_aggregation_is_stable_under_input_order_and_handles_actual_text_coverage():
    rows = head_fixture()
    assert report.aggregate_heads(rows) == report.aggregate_heads(list(reversed(rows)))
    summary, points = report.aggregate_heads(rows[:2])
    assert summary["counts"]["texts"] == 1
    assert summary["counts"]["physical_heads"] == 2
    assert all(point["text_count"] == 1 for point in points)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tile_relative_fro", -0.1),
        ("rotate_predicted_error", float("nan")),
        ("q_mean_energy_fraction", 2),
        ("layer", 0.5),
    ],
)
def test_invalid_head_measurements_are_not_silently_dropped(field, value):
    row = head_row()
    row[field] = value
    with pytest.raises(ValueError):
        report.aggregate_heads([row])


def test_duplicate_head_text_and_mixed_revision_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        report.aggregate_heads([head_row(), head_row()])
    other = head_row(text="text-b")
    other["revision"] = "other-revision"
    with pytest.raises(ValueError, match="revision"):
        report.aggregate_heads([head_row(), other])
    with pytest.raises(ValueError, match="no measurements"):
        report.aggregate_heads([])


def test_downstream_uses_token_weighted_ce_kl_and_exp_of_aggregated_ce():
    rows = downstream_fixture()
    result = report.aggregate_downstream(rows)
    assert result == report.aggregate_downstream(list(reversed(rows)))
    variants = result["model-a"]["variants"]
    assert tuple(variants) == report.DOWNSTREAM_VARIANTS
    baseline = variants["bf16"]
    assert baseline["tokens"] == 4
    assert baseline["texts"] == 2
    assert baseline["ce"] == pytest.approx(2.5)
    assert baseline["exp_ce"] == pytest.approx(math.exp(2.5))
    assert baseline["token_weighted_mean_exp_ce"] == pytest.approx(
        (math.exp(1) + 3 * math.exp(3)) / 4
    )
    assert baseline["exp_ce"] != pytest.approx(baseline["token_weighted_mean_exp_ce"])
    assert baseline["ce_delta_from_bf16"] == 0
    assert baseline["exp_ce_relative_change_from_bf16"] == 0
    assert baseline["mean_kl"] == 0
    for index, variant in enumerate(report.DOWNSTREAM_VARIANTS):
        values = variants[variant]
        assert values["ce_delta_from_bf16"] == pytest.approx(index * 0.1)
        assert values["ce_relative_change_from_bf16"] == pytest.approx(index * 0.1 / 2.5)
        assert values["exp_ce_relative_change_from_bf16"] == pytest.approx(math.expm1(index * 0.1))
        assert values["mean_kl"] == pytest.approx(index * 0.05)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_variant",
        "missing_text",
        "unequal_tokens",
        "duplicate",
        "zero_tokens",
        "unknown_variant",
    ],
)
def test_downstream_comparisons_require_matched_saved_coverage(problem):
    rows = downstream_fixture()
    if problem == "missing_variant":
        rows = [row for row in rows if row["variant"] != "smooth_kq"]
    elif problem == "missing_text":
        rows.pop()
    elif problem == "unequal_tokens":
        rows[-1]["tokens"] += 1
    elif problem == "duplicate":
        rows.append(dict(rows[-1]))
    elif problem == "zero_tokens":
        rows[-1]["tokens"] = 0
    else:
        rows[-1]["variant"] = "not-a-variant"
    with pytest.raises(ValueError):
        report.aggregate_downstream(rows)


def test_summary_command_reads_saved_tables_and_optional_fit_without_plotting(
    tmp_path, monkeypatch
):
    write_csv(tmp_path / "heads.csv", head_fixture())
    write_csv(tmp_path / "downstream.csv", downstream_fixture())
    fit = {"fixture_only": {"n": 9, "parameter_free": "synthetic test metadata"}}
    (tmp_path / "fit.json").write_text(json.dumps(fit), encoding="utf-8")
    calls = []
    monkeypatch.setattr(report, "draw_figures", lambda *args: calls.append(args))
    report.main(["--results-dir", str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["sources"] == {
        "heads": "heads.csv",
        "downstream": "downstream.csv",
        "fit": "fit.json",
    }
    assert summary["fit"] == fit
    assert summary["heads"]["counts"]["physical_heads"] == 3
    assert summary["downstream"]["model-a"]["variants"]["bf16"]["tokens"] == 4
    assert any("dependent" in label for label in summary["limitations"])
    assert any("not streaming/generation" in label for label in summary["limitations"])
    assert str(tmp_path) not in json.dumps(summary)
    assert len(calls) == 1
    assert len(calls[0][0]) == 3


def test_summary_command_without_optional_files_records_absence(tmp_path, monkeypatch):
    write_csv(tmp_path / "heads.csv", [head_row()])
    monkeypatch.setattr(report, "draw_figures", lambda *args: None)
    report.main(["--results-dir", str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["downstream"] is None
    assert summary["sources"]["downstream"] is None
    assert summary["sources"]["fit"] is None
    assert "fit" not in summary


def test_summary_command_rejects_cross_table_revision_mismatch(tmp_path, monkeypatch):
    write_csv(tmp_path / "heads.csv", [head_row()])
    rows = downstream_fixture()
    for row in rows:
        row["revision"] = "wrong-revision"
    write_csv(tmp_path / "downstream.csv", rows)
    monkeypatch.setattr(report, "draw_figures", lambda *args: None)
    with pytest.raises(ValueError, match="disagrees"):
        report.main(["--results-dir", str(tmp_path)])
