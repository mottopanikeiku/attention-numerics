import gzip
import hashlib
import json
import random
import sys
from types import SimpleNamespace

import pytest

from study.accuracy import data


def test_arc_exact_prompt_variable_choices_numeric_gold_and_char_lengths():
    item = data.render_item(
        "arc_challenge",
        {
            "id": "Mercury_SC_1",
            "question": "Which?  ",
            "choices": {"label": ["1", "2", "3"], "text": ["é", "long answer", " x "]},
            "answerKey": "3",
        },
        7,
    )
    assert item == {
        "item_id": "arc_challenge:test:Mercury_SC_1",
        "task": "arc_challenge",
        "context": "Question: Which?  \nAnswer:",
        "choices": ["é", "long answer", " x "],
        "gold": 2,
        "normalization_lengths": [1, 11, 3],
        "metric": "acc_norm",
        "source_index": 7,
    }


def test_arc_five_choices_letter_labels():
    item = data.render_item(
        "arc_challenge",
        {
            "id": "five",
            "question": "Q",
            "choices": {
                "label": list("ABCDE"),
                "text": ["one", "two", "three", "four", "five"],
            },
            "answerKey": "E",
        },
        0,
    )
    assert item["gold"] == 4
    assert len(item["choices"]) == 5


def test_hellaswag_exact_preprocessing_and_capitalize():
    item = data.render_item(
        "hellaswag",
        {
            "ind": 91,
            "activity_label": "Making tea",
            "ctx_a": "Water [discard] boils.",
            "ctx_b": "tHEN POUR",
            "endings": ["  Cup [title] fills [note].  ", "Tea cools."],
            "label": "1",
        },
        3,
    )
    assert item["context"] == "Making tea: Water boils. Then pour"
    assert item["choices"] == ["Cup. fills .", "Tea cools."]
    assert item["normalization_lengths"] == [12, 10]
    assert item["gold"] == 1
    assert item["item_id"] == "hellaswag:validation:3"
    assert item["metric"] == "acc_norm"


def test_hellaswag_does_not_collapse_all_whitespace():
    assert data._hellaswag_preprocess("  a    b [x] c  ") == "a  b c"
    assert data._hellaswag_preprocess("a\n[x]\nb") == "a\n\nb"


def test_mmlu_zero_shot_includes_subject_description_and_letter_continuations():
    item = data.render_item(
        "mmlu",
        {
            "subject": "high_school_us_history",
            "question": "  Who? \n",
            "choices": ["Alice", "Bob", "Carol", "Dan"],
            "answer": 1,
        },
        13,
        "high_school_us_history",
    )
    assert item["context"] == (
        "The following are multiple choice questions (with answers) about "
        "high school us history.\n\nWho?\nA. Alice\nB. Bob\nC. Carol\nD. Dan\nAnswer:"
    )
    assert item["choices"] == ["A", "B", "C", "D"]
    assert item["normalization_lengths"] == [1, 1, 1, 1]
    assert item["gold"] == 1
    assert item["metric"] == "acc"
    assert item["subject"] == "high_school_us_history"
    assert item["item_id"] == "mmlu:test:high_school_us_history:13"


@pytest.mark.parametrize("subject", [None, "unrecognized", "anatomy"])
def test_mmlu_rejects_wrong_subject(subject):
    with pytest.raises(ValueError, match="subject"):
        data.render_item(
            "mmlu",
            {
                "subject": "abstract_algebra",
                "question": "Q",
                "choices": list("abcd"),
                "answer": 0,
            },
            0,
            subject,
        )


def test_largest_remainder_known_result_and_alphabetical_ties():
    assert data.proportional_allocation({"c": 2, "a": 5, "b": 3}, 4) == {
        "a": 2,
        "b": 1,
        "c": 1,
    }
    assert data.proportional_allocation({"c": 1, "b": 1, "a": 1}, 2) == {
        "a": 1,
        "b": 1,
        "c": 0,
    }
    assert data.proportional_allocation({"a": 3, "b": 7}, 10) == {"a": 3, "b": 7}
    assert data.proportional_allocation({"a": 3, "b": 7}, 0) == {"a": 0, "b": 0}


@pytest.mark.parametrize("counts,size", [({}, 1), ({"a": 0}, 0), ({"a": 2}, 3), ({"a": 2}, -1)])
def test_invalid_stratification_rejected(counts, size):
    with pytest.raises(ValueError):
        data.proportional_allocation(counts, size)


def test_subject_seeds_use_stable_sha256_not_python_hash():
    expected = int(hashlib.sha256(b"20261007:mmlu:anatomy").hexdigest(), 16)
    assert data.subject_seed("anatomy") == expected
    assert data.subject_seed("astronomy") != expected


class CharacterTokenizer:
    bos_token_id = None

    def __init__(self, native_bos=False):
        self.native_bos = native_bos
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append((text, kwargs))
        tokens = [ord(c) for c in text]
        return ([999] if kwargs.get("add_special_tokens", self.native_bos) else []) + tokens

    def decode(self, ids):
        return "<BOS>" if ids == [999] else "<EOS>"


def test_encoding_moves_all_context_trailing_whitespace_and_adds_delimiter_once():
    tokenizer = CharacterTokenizer()
    context_ids, continuation_ids = data.encode_choice(tokenizer, "Answer: \t\n", "blue", 1000)
    assert context_ids == [ord(c) for c in "Answer:"]
    assert continuation_ids == [ord(c) for c in " \t\n blue"]
    assert tokenizer.calls == [("Answer: \t\n blue", {}), ("Answer:", {})]


def test_encoding_is_joint_not_separately_encoded_choice():
    class JointTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            assert kwargs == {}
            return {"Q:": [10], "Q: yes": [10, 20], " yes": [30]}[text]

    assert data.encode_choice(JointTokenizer(), "Q:", "yes", 1000) == ([10], [20])


@pytest.mark.parametrize(
    "setting,native_bos,expected_bos,kwargs",
    [
        (None, False, False, {}),
        (None, True, True, {}),
        (False, True, False, {"add_special_tokens": False}),
        (True, False, True, {"add_special_tokens": True}),
    ],
)
def test_encoding_native_default_and_explicit_special_token_policy(
    setting, native_bos, expected_bos, kwargs
):
    tokenizer = CharacterTokenizer(native_bos)
    context_ids, continuation_ids = data.encode_choice(tokenizer, "Q:", "A", 1000, setting)
    assert context_ids == ([999] if expected_bos else []) + [81, 58]
    assert continuation_ids == [32, 65]
    assert [call[1] for call in tokenizer.calls] == [kwargs, kwargs]


def test_empty_context_prefers_bos_over_eot_and_disables_special_tokens():
    tokenizer = CharacterTokenizer(True)
    tokenizer.bos_token_id = 999
    assert data.encode_choice(tokenizer, "", "A", 1000) == ([999], [32, 65])
    assert tokenizer.calls == [(" A", {"add_special_tokens": False})]
    tokenizer.bos_token_id = None
    assert data.encode_choice(tokenizer, "", "A", 1000) == ([1000], [32, 65])


def test_existing_bos_prefix_disables_duplicate_special_tokens():
    tokenizer = CharacterTokenizer(True)
    tokenizer.bos_token_id = 999
    data.encode_choice(tokenizer, "<BOS>Q:", "A", 1000)
    assert all(kwargs == {"add_special_tokens": False} for _, kwargs in tokenizer.calls)


def test_empty_context_reuses_leading_prefix_token():
    class PrefixTokenizer(CharacterTokenizer):
        bos_token_id = 999

        def encode(self, text, **kwargs):
            assert text == " A"
            assert kwargs == {"add_special_tokens": False}
            return [999, 65]

    assert data.encode_choice(PrefixTokenizer(), "", "A", 1000) == ([999], [65])


def test_encoding_rejects_cross_boundary_merge():
    class MergingTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            return {"a": [1], "a b": [2, 3]}[text]

    with pytest.raises(ValueError, match="boundary"):
        data.encode_choice(MergingTokenizer(), "a", "b", 1000)


@pytest.mark.parametrize("context,choice", [("Q:", ""), ("Q:", " \t"), (" \t", "A")])
def test_encoding_rejects_empty_choice_or_whitespace_only_context(context, choice):
    with pytest.raises(ValueError):
        data.encode_choice(CharacterTokenizer(), context, choice, 1000)


def test_encoding_rejects_zero_token_continuation():
    class EmptyTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            return [81] if text else []

    with pytest.raises(ValueError, match="continuation tokens"):
        data.encode_choice(EmptyTokenizer(), "Q", "A", 1000)


@pytest.fixture(scope="module")
def full_sources():
    arc = [
        {
            "id": f"arc-{i}",
            "question": "Q",
            "choices": {"label": ["1", "2"], "text": ["a", "bb"]},
            "answerKey": "2",
        }
        for i in range(1172)
    ]
    hella = [
        {
            "ind": i % 2,
            "activity_label": "Act",
            "ctx_a": "A",
            "ctx_b": "b",
            "endings": ["one", "two", "three", "four"],
            "label": str(i % 4),
        }
        for i in range(10042)
    ]
    mmlu = {
        subject: [
            {"subject": subject, "question": f"Q {i}", "choices": list("abcd"), "answer": i % 4}
            for i in range(247 if s < 20 else 246)
        ]
        for s, subject in enumerate(data.MMLU_SUBJECTS)
    }
    return {"arc_challenge": {"ARC-Challenge": arc}, "hellaswag": {"default": hella}, "mmlu": mmlu}


def test_prepare_exact_counts_ids_pinning_hashes_and_deterministic_gzip(
    monkeypatch, tmp_path, full_sources
):
    monkeypatch.setattr(data, "_load_sources", lambda: full_sources)
    first = data.prepare_items(tmp_path / "one")
    second = data.prepare_items(tmp_path / "two")
    compressed = (tmp_path / "one/items.json.gz").read_bytes()
    assert compressed == (tmp_path / "two/items.json.gz").read_bytes()
    assert compressed[4:8] == b"\0\0\0\0"
    assert first == second == json.loads((tmp_path / "one/selection.json").read_bytes())
    assert first["item_manifest_sha256"] == hashlib.sha256(compressed).hexdigest()
    payload = gzip.decompress(compressed)
    assert first["items_json_sha256"] == hashlib.sha256(payload).hexdigest()
    manifest = json.loads(payload)
    assert manifest["schema_version"] == 1
    items = manifest["items"]
    assert len(items) == len(set(first["item_ids"])) == 5172
    assert first["counts"] == {"arc_challenge": 1172, "hellaswag": 2000, "mmlu": 2000}
    assert [item["item_id"] for item in items] == first["item_ids"]
    assert first["harness_revision"] == "ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9"
    assert (
        first["datasets"]["arc_challenge"]["revision"] == "210d026faf9955653af8916fad021475a3f00453"
    )
    assert first["datasets"]["hellaswag"]["revision"] == "218ec52e09a7e7462a5400043bb9a69a41d06b76"
    assert first["datasets"]["mmlu"]["revision"] == "c30699e8356da336a370243923dbaf21066bb9fe"
    expected_indices = sorted(random.Random(20261007).sample(range(10042), 2000))
    hella = first["datasets"]["hellaswag"]
    assert hella["configs"]["default"]["selected_source_indices"] == expected_indices
    assert hella["item_ids"] == [f"hellaswag:validation:{i}" for i in expected_indices]
    source_rows = hella["configs"]["default"]["source_rows"]
    assert len(source_rows) == 10042
    expected_raw = json.dumps(
        full_sources["hellaswag"]["default"][0],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert source_rows[0]["raw_row_sha256"] == hashlib.sha256(expected_raw).hexdigest()
    mmlu = first["datasets"]["mmlu"]
    assert len(mmlu["configs"]) == 57
    assert sum(config["selected_count"] for config in mmlu["configs"].values()) == 2000
    assert sum(config["source_count"] for config in mmlu["configs"].values()) == 14042
    for subject, config in mmlu["configs"].items():
        expected_seed = int(hashlib.sha256(f"20261007:mmlu:{subject}".encode()).hexdigest(), 16)
        assert config["sampling_seed"] == expected_seed
        assert config["selected_source_indices"] == sorted(
            random.Random(expected_seed).sample(
                range(config["source_count"]), config["selected_count"]
            )
        )
        assert config["source_rows"][0]["item_id"] == f"mmlu:test:{subject}:0"
    selected_mmlu = [
        (item["subject"], item["source_index"]) for item in items if item["task"] == "mmlu"
    ]
    assert selected_mmlu == sorted(selected_mmlu)


def test_prepare_rejects_partial_sources_without_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "_load_sources", lambda: {"arc_challenge": {"ARC-Challenge": []}})
    with pytest.raises(ValueError, match="all three"):
        data.prepare_items(tmp_path)
    assert not (tmp_path / "items.json.gz").exists()


def test_prepare_rejects_duplicate_native_ids(monkeypatch, tmp_path, full_sources):
    sources = {
        **full_sources,
        "arc_challenge": {
            "ARC-Challenge": [
                {**row, "id": "duplicate"} for row in full_sources["arc_challenge"]["ARC-Challenge"]
            ]
        },
    }
    monkeypatch.setattr(data, "_load_sources", lambda: sources)
    with pytest.raises(ValueError, match="Duplicate source IDs"):
        data.prepare_items(tmp_path)
    assert not (tmp_path / "items.json.gz").exists()


def test_source_loader_uses_resolved_pins_and_harness_subject_configs(monkeypatch):
    calls = []

    class Api:
        def dataset_info(self, repo_id, revision):
            calls.append(("resolve", repo_id, revision))
            return SimpleNamespace(sha=revision)

    def load_dataset(repo_id, config, *, split, revision):
        calls.append(("load", repo_id, config, split, revision))
        return []

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    sources = data._load_sources()
    assert set(sources["mmlu"]) == set(data.MMLU_SUBJECTS)
    assert len([call for call in calls if call[0] == "load"]) == 59
    for spec in data.DATASETS.values():
        assert ("resolve", spec["repo_id"], spec["revision"]) in calls
        assert all(
            call[3:] == (spec["split"], spec["revision"])
            for call in calls
            if call[0] == "load" and call[1] == spec["repo_id"]
        )


def test_source_loader_rejects_changed_resolved_revision(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: []))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: SimpleNamespace(
                dataset_info=lambda *a, **kw: SimpleNamespace(sha="changed")
            )
        ),
    )
    with pytest.raises(ValueError, match="revision mismatch"):
        data._load_sources()
