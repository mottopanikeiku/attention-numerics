"""The README accuracy numbers must match the committed summary."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANS = ("plan.json", "qwen14-plan.json", "smol17-plan.json")


def _summary():
    return json.loads((ROOT / "results/accuracy/summary.json").read_text())["rows"]


def _pp(value):
    return f"{100 * value:.2f}".replace("-", "−")


def test_readme_accuracy_table_matches_summary():
    names = {}
    for plan in PLANS:
        for model in json.loads((ROOT / "data/accuracy" / plan).read_text())["models"]:
            names[model["repo_id"].split("/")[1]] = model["key"]
    rows = _summary()
    expected = {}
    for name, key in names.items():
        model_rows = [row for row in rows if row["model"] == key]
        baseline = {row["task"]: row["accuracy"] for row in model_rows if row["variant"] == "bf16"}
        expected[name] = [
            *(_pp(baseline[task]) for task in ("arc_challenge", "hellaswag", "mmlu")),
            str(sum(row["harm_flag"] for row in model_rows)),
        ]
    table = {}
    for line in (ROOT / "README.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] in names:
            table[cells[0]] = cells[1:]
    assert table == expected


def test_readme_headline_changes_match_summary():
    readme = (ROOT / "README.md").read_text()
    rows = {(row["model"], row["task"], row["variant"]): row for row in _summary()}
    rotate = rows[("qwen7", "hellaswag", "rotate")]
    low, high = (_pp(value) for value in rotate["ci95"])
    assert f"**{_pp(-rotate['delta'])} percentage points**" in readme
    assert f"**[{low}, {high}]**" in readme
    centered = [row for row in rows.values() if row["variant"] in ("smooth_k", "rotate_smooth_k")]
    assert sum(row["harm_flag"] for row in centered) == 0
    assert f"**zero ≥2pp harm flags across {len(centered)} comparisons**" in readme
