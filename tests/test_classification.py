import csv
import json
import math

import pytest

from study.classification import main, rotation_report

TEXTS = ("alice", "moby", "pride")


def _tables(specs):
    """Specs: (model, layer, head, observed score, predicted score, prefix4 mass)."""
    heads, sinks = [], []
    for model, layer, head, observed, predicted, mass in specs:
        for text in TEXTS:
            identity = {
                "model": model,
                "family": "qwen" if model.startswith("qwen") else "smol",
                "revision": f"revision-{model}",
                "text": text,
                "layer": layer,
                "head": head,
                "kv_head": head // 2,
                "n": 1024,
                "d": 64,
                "scale": 0.125,
            }
            heads.append(
                {
                    **identity,
                    "tile_relative_fro": math.expm1(max(-observed, 0)),
                    "rotate_relative_fro": math.expm1(max(observed, 0)),
                    "tile_predicted_error": math.expm1(max(-predicted, 0)),
                    "rotate_predicted_error": math.expm1(max(predicted, 0)),
                }
            )
            sinks.append(
                {
                    **identity,
                    "query_start": 128,
                    "query_count": 896,
                    "prefix1_mass": mass / 2,
                    "prefix4_mass": mass,
                }
            )
    return heads, sinks


def _report(specs, design=None):
    return rotation_report(*_tables(specs), {} if design is None else design)


def test_imbalance_majority_trap_is_visible():
    result = _report([("qwen05", 0, head, 1 if head == 0 else -1, 0, 0.1) for head in range(10)])
    overall = result["overall"]
    classifier = overall["classification"]
    assert overall["physical_heads"] == 10
    assert overall["head_text_rows"] == 30
    assert overall["hurts"] == {"count": 1, "fraction": 0.1}
    assert overall["helps"] == {"count": 9, "fraction": 0.9}
    assert classifier["majority_class"] == "non_hurt"
    assert classifier["majority_class_accuracy"] == pytest.approx(0.9)
    assert classifier["accuracy"] == pytest.approx(0.9)
    assert classifier["balanced_accuracy"] == pytest.approx(0.5)
    assert classifier["roc_auc"] == pytest.approx(0.5)
    assert classifier["recall"] == 0
    assert classifier["precision"] is None
    assert classifier["f1"] == 0
    assert classifier["specificity"] == 1
    assert classifier["confusion"] == {"tp": 0, "fp": 0, "fn": 1, "tn": 9}
    assert classifier["denominators"]["roc_auc_positive_negative_pairs"] == 9


def test_continuous_auc_gives_half_credit_for_ties_and_zero_threshold_is_strict():
    result = _report(
        [
            ("qwen05", 0, 0, 1, 1, 0.1),
            ("qwen05", 0, 1, 1, 0, 0.1),
            ("qwen05", 0, 2, -1, 0, 0.1),
            ("qwen05", 0, 3, -1, -1, 0.1),
        ]
    )
    classifier = result["overall"]["classification"]
    # Positive-negative score comparisons: 1 + 1 + 0.5 + 1 = 3.5 of 4.
    assert classifier["roc_auc"] == pytest.approx(0.875)
    assert classifier["threshold"] == 0
    assert classifier["accuracy"] == pytest.approx(0.75)
    assert classifier["balanced_accuracy"] == pytest.approx(0.75)
    assert classifier["precision"] == 1
    assert classifier["recall"] == pytest.approx(0.5)
    assert classifier["f1"] == pytest.approx(2 / 3)
    assert classifier["specificity"] == 1
    assert classifier["confusion"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 2}
    assert classifier["denominators"]["precision"] == 1
    assert classifier["denominators"]["recall"] == 2
    assert not result["physical_heads"][1]["predict_hurt"]
    assert not result["physical_heads"][2]["predict_hurt"]


def test_observed_ties_are_reported_separately_and_are_binary_non_hurt():
    result = _report(
        [
            ("qwen05", 0, 0, 0, 1, 0.1),
            ("qwen05", 0, 1, 1, 0, 0.1),
            ("qwen05", 0, 2, -1, -1, 0.1),
        ]
    )
    overall = result["overall"]
    for outcome in ("hurts", "helps", "ties"):
        assert overall[outcome]["count"] == 1
        assert overall[outcome]["fraction"] == pytest.approx(1 / 3)
    classifier = overall["classification"]
    assert classifier["confusion"] == {"tp": 0, "fp": 1, "fn": 1, "tn": 1}
    assert classifier["denominators"]["specificity"] == 2
    assert classifier["precision"] == classifier["recall"] == classifier["f1"] == 0
    assert classifier["balanced_accuracy"] == pytest.approx(0.25)


@pytest.mark.parametrize("observed,predicted", [(1, 1), (-1, -1), (0, 0)])
def test_single_class_undefined_metrics_are_null(observed, predicted):
    result = _report([("qwen05", 0, 0, observed, predicted, 0.1)])
    overall = result["overall"]
    classifier = overall["classification"]
    assert classifier["roc_auc"] is None
    assert classifier["balanced_accuracy"] is None
    assert classifier["accuracy"] == classifier["majority_class_accuracy"] == 1
    assert overall["log_gain"]["r2"] is None
    assert overall["log_gain"]["spearman"] is None
    if observed > 0:
        assert classifier["recall"] == classifier["precision"] == classifier["f1"] == 1
        assert classifier["specificity"] is None
    else:
        assert classifier["recall"] is None
        assert classifier["precision"] is None
        assert classifier["f1"] is None
        assert classifier["specificity"] == 1
        assert overall["sink_concentration"]["fraction_of_hurting_heads_that_are_sinks"] is None
        assert (
            result["per_model"]["qwen05"]["layer_concentration"]["top3_share_of_hurting_heads"]
            is None
        )
    json.dumps(result, allow_nan=False)


def test_empty_groups_have_null_fractions_and_metrics_not_fake_perfect_scores():
    result = rotation_report([], [], {})
    assert result["per_model"] == {}
    assert result["physical_heads"] == []
    assert not result["metadata"]["evaluation_complete"]
    for key in ("overall", "untouched_evaluation", "development"):
        summary = result[key]
        assert summary["physical_heads"] == summary["head_text_rows"] == 0
        for outcome in ("hurts", "helps", "ties"):
            assert summary[outcome] == {"count": 0, "fraction": None}
        for metric in (
            "majority_class_accuracy",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "specificity",
        ):
            assert summary["classification"][metric] is None
        assert summary["log_gain"]["n"] == 0
        assert summary["log_gain"]["r2"] is None
    json.dumps(result, allow_nan=False)


def test_physical_head_averages_log_errors_before_subtracting_and_averages_sink_mass():
    heads, sinks = _tables([("qwen05", 0, 0, 0, 0, 0.1)])
    for index, row in enumerate(heads):
        row["tile_relative_fro"] = 1000 if index == 0 else 0
        row["rotate_relative_fro"] = 10
        row["tile_predicted_error"] = 1000 if index == 0 else 0
        row["rotate_predicted_error"] = 1
    for row, mass in zip(sinks, (0.3, 0.6, 0.6), strict=True):
        row["prefix4_mass"] = mass
        row["prefix1_mass"] = mass / 2
    result = rotation_report(heads, sinks, {"text_ids": list(TEXTS)})
    point = result["physical_heads"][0]
    assert point["observed_hurt_score"] == pytest.approx(math.log1p(10) - math.log1p(1000) / 3)
    assert point["predicted_hurt_score"] == pytest.approx(math.log1p(1) - math.log1p(1000) / 3)
    assert point["hurts"] and not point["predict_hurt"]
    assert point["mean_prefix4_mass"] == pytest.approx(0.5)
    assert point["prefix_sink"]
    assert result["overall"]["physical_heads"] == 1
    assert result["overall"]["head_text_rows"] == 3
    assert result["overall"]["classification"]["confusion"]["fn"] == 1
    assert result == rotation_report(
        list(reversed(heads)), list(reversed(sinks)), {"text_ids": list(TEXTS)}
    )


def test_matching_requires_every_sink_head_text_and_rejects_extra_or_duplicate_rows():
    heads, sinks = _tables([("qwen05", 0, 0, 1, 1, 0.6)])
    with pytest.raises(ValueError, match="missing sink"):
        rotation_report(heads, sinks[:-1], {})
    with pytest.raises(ValueError, match="no matching head/text"):
        rotation_report(heads, [*sinks, {**sinks[0], "head": 99}], {})
    with pytest.raises(ValueError, match="duplicate sink"):
        rotation_report(heads, [*sinks, sinks[0]], {})
    with pytest.raises(ValueError, match="duplicate head/text"):
        rotation_report([*heads, heads[0]], sinks, {})


@pytest.mark.parametrize(
    "field,value",
    [
        ("family", "other"),
        ("revision", "other-revision"),
        ("kv_head", 3),
        ("n", 512),
        ("d", 128),
        ("scale", 0.25),
    ],
)
def test_sink_metadata_must_match_head_measurements(field, value):
    heads, sinks = _tables([("qwen05", 0, 0, 1, 1, 0.6)])
    sinks[0][field] = value
    with pytest.raises(ValueError, match=f"sink {field} disagrees"):
        rotation_report(heads, sinks, {})


def test_each_physical_head_requires_three_identical_texts():
    heads, sinks = _tables([("qwen05", 0, 0, 1, 1, 0.6)])
    with pytest.raises(ValueError, match="all three texts"):
        rotation_report(heads[:-1], sinks[:-1], {})
    with pytest.raises(ValueError, match="text set disagrees"):
        rotation_report(heads, sinks, {"text_ids": ["alice", "moby", "other"]})
    heads, sinks = _tables([("qwen05", 0, 0, 1, 1, 0.6), ("qwen05", 0, 1, 1, 1, 0.6)])
    heads[-1]["text"] = sinks[-1]["text"] = "other"
    with pytest.raises(ValueError, match="text set disagrees"):
        rotation_report(heads, sinks, {})


@pytest.mark.parametrize(
    "field,value",
    [
        ("query_start", 127),
        ("query_count", 895),
        ("prefix4_mass", 1.01),
        ("prefix4_mass", -0.01),
        ("prefix1_mass", 0.9),
    ],
)
def test_sink_window_and_probability_contract(field, value):
    heads, sinks = _tables([("qwen05", 0, 0, 1, 1, 0.6)])
    sinks[0][field] = value
    with pytest.raises(ValueError):
        rotation_report(heads, sinks, {})


@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_nonfinite_or_negative_errors_are_rejected(value):
    heads, sinks = _tables([("qwen05", 0, 0, 1, 1, 0.6)])
    heads[0]["rotate_relative_fro"] = value
    with pytest.raises(ValueError, match="finite.*nonnegative"):
        rotation_report(heads, sinks, {})


def test_layer_and_sink_concentration_uses_physical_head_counts():
    result = _report(
        [
            ("qwen05", 0, 0, 1, 1, 0.6),
            ("qwen05", 0, 1, 1, 1, 0.6),
            ("qwen05", 0, 2, 1, 1, 0.1),
            ("qwen05", 0, 3, -1, -1, 0.1),
            ("qwen05", 1, 0, 1, 1, 0.1),
            ("qwen05", 1, 1, 1, 1, 0.6),
            ("qwen05", 1, 2, -1, -1, 0.1),
            ("qwen05", 2, 0, 1, 1, 0.1),
            ("qwen05", 2, 1, 0, 0, 0.6),
            ("qwen05", 3, 0, 1, 1, 0.6),
            ("qwen05", 3, 1, -1, -1, 0.1),
        ]
    )
    summary = result["per_model"]["qwen05"]
    assert summary["physical_heads"] == 11
    layers = summary["layer_concentration"]
    assert layers["per_layer"] == [
        {"layer": 0, "physical_heads": 4, "hurting_heads": 3, "hurt_fraction": 0.75},
        {"layer": 1, "physical_heads": 3, "hurting_heads": 2, "hurt_fraction": 2 / 3},
        {"layer": 2, "physical_heads": 2, "hurting_heads": 1, "hurt_fraction": 0.5},
        {"layer": 3, "physical_heads": 2, "hurting_heads": 1, "hurt_fraction": 0.5},
    ]
    assert layers["top3_layers"] == [0, 1, 2]
    assert layers["top3_hurting_heads"] == 6
    assert layers["hurting_heads_denominator"] == 7
    assert layers["top3_share_of_hurting_heads"] == pytest.approx(6 / 7)
    sinks = summary["sink_concentration"]
    assert sinks["sink_heads"] == 5
    assert sinks["non_sink_heads"] == 6
    assert sinks["hurting_sink_heads"] == 4
    assert sinks["hurting_non_sink_heads"] == 3
    assert sinks["sink_hurt_rate"] == pytest.approx(4 / 5)
    assert sinks["non_sink_hurt_rate"] == pytest.approx(3 / 6)
    assert sinks["fraction_of_hurting_heads_that_are_sinks"] == pytest.approx(4 / 7)
    assert sinks["denominators"]["fraction_of_hurting_heads_that_are_sinks"] == 7


def test_rank_and_r2_metrics_use_fixed_predictions_without_calibration():
    result = _report(
        [("qwen05", 0, head, score, score, 0.1) for head, score in enumerate((-1, 0, 1))]
    )
    metrics = result["overall"]["log_gain"]
    assert metrics["n"] == 3
    assert metrics["spearman"] == pytest.approx(1)
    assert metrics["r2"] == pytest.approx(1)
    assert metrics["mae"] == 0
    assert metrics["sign_accuracy"] == 1
    assert "calibration" not in result
    reversed_prediction = _report(
        [("qwen05", 0, head, score, -score, 0.1) for head, score in enumerate((-1, 0, 1))]
    )
    reversed_metrics = reversed_prediction["overall"]["log_gain"]
    assert reversed_metrics["spearman"] == pytest.approx(-1)
    assert reversed_metrics["r2"] == pytest.approx(-3)
    assert reversed_metrics["mae"] == pytest.approx(4 / 3)


def test_untouched_evaluation_excludes_both_development_models_and_tracks_missing():
    specs = [
        ("qwen05", 0, 0, 1, -1, 0.6),
        ("smol036", 0, 0, 1, -1, 0.1),
        ("qwen15", 0, 0, 1, -1, 0.1),
        ("smol17", 0, 0, 1, 1, 0.6),
        ("tiny11", 0, 0, -1, -1, 0.1),
        ("olmo1", 0, 0, 0, 0, 0.1),
    ]
    result = _report(specs)
    evaluation = result["untouched_evaluation"]
    assert evaluation["physical_heads"] == 3
    assert evaluation["classification"]["accuracy"] == 1
    assert evaluation["classification"]["balanced_accuracy"] == 1
    assert evaluation["classification"]["roc_auc"] == 1
    assert result["development"]["physical_heads"] == 2
    assert result["development"]["classification"]["accuracy"] == 0
    assert result["metadata"]["other_models_present"] == ["qwen15"]
    assert result["metadata"]["evaluation_complete"]
    assert result["metadata"]["evaluation_models_missing"] == []
    development_only = _report(specs[:2])
    assert development_only["untouched_evaluation"]["physical_heads"] == 0
    assert development_only["untouched_evaluation"]["classification"]["accuracy"] is None
    assert development_only["metadata"]["evaluation_models_missing"] == [
        "olmo1",
        "smol17",
        "tiny11",
    ]
    # A change in pilot targets cannot change prospective evaluation diagnostics.
    changed = [
        (*spec[:3], -spec[3], spec[4], spec[5]) if spec[0] in ("qwen05", "smol036") else spec
        for spec in specs
    ]
    assert _report(changed)["untouched_evaluation"] == evaluation


def test_design_groups_are_honored_and_cannot_overlap():
    specs = [("custom", 0, 0, 1, 1, 0.6)]
    result = _report(
        specs,
        {
            "evaluation_models": ["custom"],
            "development_models": [],
            "calibration_family": "qwen",
            "text_ids": list(TEXTS),
        },
    )
    assert result["untouched_evaluation"]["physical_heads"] == 1
    assert result["metadata"]["evaluation_models"] == ["custom"]
    with pytest.raises(ValueError, match="disjoint"):
        _report(specs, {"evaluation_models": ["custom"], "development_models": ["custom"]})


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_cli_reads_actual_wide_csvs_and_design_and_writes_json(tmp_path):
    heads, sinks = _tables([("qwen05", 0, 0, 1, 1, 0.6), ("smol17", 0, 0, -1, -1, 0.1)])
    _write_csv(tmp_path / "heads.csv", heads)
    _write_csv(tmp_path / "sinks.csv", sinks)
    design = {
        "evaluation_models": ["smol17", "tiny11", "olmo1"],
        "development_models": ["qwen05", "smol036"],
        "calibration_family": "qwen",
    }
    design_path = tmp_path / "pinned-design.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")
    main(["--results-dir", str(tmp_path), "--design", str(design_path)])
    result = json.loads((tmp_path / "rotation_risk.json").read_text(encoding="utf-8"))
    assert result == rotation_report(heads, sinks, design)
    assert result["overall"]["physical_heads"] == 2
    assert result["overall"]["head_text_rows"] == 6
    assert result["metadata"]["evaluation_models_present"] == ["smol17"]
    assert result["metadata"]["evaluation_models_missing"] == ["olmo1", "tiny11"]


def test_cli_requires_sinks_instead_of_assuming_every_head_is_non_sink(tmp_path):
    heads, _ = _tables([("qwen05", 0, 0, 1, 1, 0.6)])
    _write_csv(tmp_path / "heads.csv", heads)
    design_path = tmp_path / "pinned-design.json"
    design_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        main(["--results-dir", str(tmp_path), "--design", str(design_path)])
    assert not (tmp_path / "rotation_risk.json").exists()


def test_cli_defaults_to_pinned_design_not_a_results_directory_design(tmp_path, monkeypatch):
    import study.classification as classification

    heads, sinks = _tables([("smol17", 0, 0, 1, 1, 0.6)])
    directory = tmp_path / "results"
    directory.mkdir()
    _write_csv(directory / "heads.csv", heads)
    _write_csv(directory / "sinks.csv", sinks)
    design_path = tmp_path / "data" / "v2" / "design.json"
    design_path.parent.mkdir(parents=True)
    design_path.write_text(json.dumps({"text_ids": list(TEXTS)}), encoding="utf-8")
    monkeypatch.setattr(classification, "DEFAULT_DESIGN", design_path)
    main(["--results-dir", str(directory)])
    result = json.loads((directory / "rotation_risk.json").read_text(encoding="utf-8"))
    assert result["sources"]["design"] == "data/v2/design.json"
    assert result["untouched_evaluation"]["physical_heads"] == 1
    assert not (directory / "design.json").exists()
