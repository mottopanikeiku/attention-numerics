"""Synthetic CPU statistics tests; hardware evidence comes only from Modal runs."""

import copy
import math

import pytest

from study.hardware.common import VARIANTS
from study.hardware.report import compare, physical_points, spearman, summarize_downstream


def risk_fixture():
    rows, original, selected = [], {}, []
    for head in range(2):
        predicted_rotate = 0.3 if head == 0 else 0.02
        score = math.log1p(predicted_rotate) - math.log1p(0.1)
        original[("qwen05", 0, head)] = {
            "model": "qwen05",
            "layer": 0,
            "head": head,
            "revision": "abc",
            "predicted_hurt_score": score,
            "prefix_sink": head == 0,
        }
        selected.append({"layer": 0, "head": head, "selected_by": ["all"]})
        for text in ("alice", "moby", "pride"):
            rows.append(
                {
                    "model": "qwen05",
                    "layer": 0,
                    "head": head,
                    "revision": "abc",
                    "text": text,
                    "selected_by": ["all"],
                    "emulator": {
                        variant: {
                            "relative_fro": predicted_rotate if variant == "rotate" else 0.1,
                            "predicted_error": predicted_rotate if variant == "rotate" else 0.1,
                        }
                        for variant in VARIANTS
                    },
                    "hardware": {
                        variant: {
                            "relative_fro": (0.08 if head == 0 else 0.2)
                            if variant == "rotate"
                            else 0.1
                        }
                        for variant in (*VARIANTS, "bf16")
                    },
                }
            )
    run = {"selection": {"models": {"qwen05": selected}}, "heads": {"rows": rows}}
    return run, original


def test_hardware_risk_does_not_refit_or_hide_reversed_classifier():
    run, original = risk_fixture()
    points = physical_points(run, original)
    summary = compare(points)
    assert summary["physical_heads"] == 2
    assert summary["head_text_rows"] == 6
    assert summary["hurts"]["count"] == 1
    assert summary["classification"]["roc_auc"] == 0
    assert summary["classification"]["accuracy"] == 0
    assert summary["classification"]["balanced_accuracy"] == 0
    assert summary["emulator_harm_disagreement"]["count"] == 2
    assert summary["emulator_rotation_effect_spearman"] == pytest.approx(-1)
    assert summary["classification"]["threshold"] == 0
    assert "sink_concentration" not in summary


def test_risk_requires_all_texts_and_exact_published_predictor():
    run, original = risk_fixture()
    missing = copy.deepcopy(run)
    missing["heads"]["rows"].pop()
    with pytest.raises(ValueError, match="text coverage"):
        physical_points(missing, original)
    changed = copy.deepcopy(run)
    changed["heads"]["rows"][0]["emulator"]["rotate"]["predicted_error"] += 0.01
    with pytest.raises(ValueError, match="published lock"):
        physical_points(changed, original)
    changed = copy.deepcopy(run)
    changed["selection"]["models"]["qwen05"].pop()
    with pytest.raises(ValueError, match="coverage"):
        physical_points(changed, original)


def test_auc_remains_undefined_for_one_real_kernel_class():
    run, original = risk_fixture()
    for row in run["heads"]["rows"]:
        row["hardware"]["rotate"]["relative_fro"] = 0.05
    summary = compare(physical_points(run, original))
    assert summary["classification"]["roc_auc"] is None
    assert summary["classification"]["balanced_accuracy"] is None
    assert summary["classification"]["majority_class_accuracy"] == 1


def test_spearman_uses_tied_ranks_and_reports_undefined():
    assert spearman([0, 0, 2], [9, 9, 10]) == pytest.approx(1)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1], [2]) is None


def test_downstream_uses_token_weighted_ce_then_exponentiates():
    rows = []
    for model in ("qwen05", "qwen15"):
        for variant in ("bf16", *VARIANTS):
            for index, text in enumerate(("alice", "moby", "pride")):
                rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "text": text,
                        "tokens": index + 1,
                        "next_token_ce": 2 + index + (0.2 if variant != "bf16" else 0),
                        "kl_from_bf16": 0.1 if variant != "bf16" else 0,
                        "kernel_calls": 24 if variant != "bf16" else 0,
                    }
                )
    summary = summarize_downstream({"downstream": {"rows": rows}})
    assert summary["qwen05"]["bf16"]["next_token_ce"] == pytest.approx((2 + 6 + 12) / 6)
    assert summary["qwen05"]["rotate"]["exp_ce_ratio"] == pytest.approx(math.exp(0.2))
    with pytest.raises(ValueError, match="coverage"):
        summarize_downstream({"downstream": {"rows": rows[:-3]}})
