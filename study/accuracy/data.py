"""Pinned, zero-shot lm-evaluation-harness benchmark items (no model loading).

Rendering and pair tokenization follow the cited EleutherAI harness sources;
this module is a small independent implementation, not a harness dependency.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

SEED = 20261007
HARNESS_REVISION = "ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9"
_HARNESS = f"https://github.com/EleutherAI/lm-evaluation-harness/blob/{HARNESS_REVISION}"
SOURCE_CITATIONS = {
    "arc_challenge": [
        f"{_HARNESS}/lm_eval/tasks/arc/arc_challenge.yaml",
        f"{_HARNESS}/lm_eval/tasks/arc/arc_easy.yaml",
    ],
    "hellaswag": [
        f"{_HARNESS}/lm_eval/tasks/hellaswag/hellaswag.yaml",
        f"{_HARNESS}/lm_eval/tasks/hellaswag/utils.py",
    ],
    "mmlu": [
        f"{_HARNESS}/lm_eval/tasks/mmlu/default/_default_template_yaml",
        f"{_HARNESS}/lm_eval/tasks/mmlu/_generate_configs.py",
        f"{_HARNESS}/lm_eval/tasks/mmlu/default/mmlu_abstract_algebra.yaml",
    ],
    "scoring": [f"{_HARNESS}/lm_eval/api/task.py#L1454-L1652"],
    "tokenization": [
        f"{_HARNESS}/lm_eval/api/model.py#L357-L451",
        f"{_HARNESS}/lm_eval/models/huggingface.py#L859-L881",
        f"{_HARNESS}/lm_eval/models/utils.py#L878-L883",
    ],
}
# Revisions resolved from the public HF dataset APIs before any GPU accuracy run.
DATASETS = {
    "arc_challenge": {
        "repo_id": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "split": "test",
        "source_count": 1172,
        "selected_count": 1172,
    },
    "hellaswag": {
        "repo_id": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "split": "validation",
        "source_count": 10042,
        "selected_count": 2000,
    },
    "mmlu": {
        "repo_id": "cais/mmlu",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "split": "test",
        "source_count": 14042,
        "selected_count": 2000,
    },
}
MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _hellaswag_preprocess(text: str) -> str:
    # Match the harness's single replace, not arbitrary whitespace collapsing.
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def render_item(task: str, row: dict, source_index: int, subject: str | None = None) -> dict:
    """Render one source row; choices exclude the harness's space delimiter."""
    if task == "arc_challenge":
        choices = list(row["choices"]["text"])
        gold = list(row["choices"]["label"]).index(row["answerKey"])
        context = f"Question: {row['question']}\nAnswer:"
        item_id = f"arc_challenge:test:{row['id']}"
        metric = "acc_norm"
    elif task == "hellaswag":
        context = _hellaswag_preprocess(
            row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize()
        )
        choices = [_hellaswag_preprocess(ending) for ending in row["endings"]]
        gold = int(row["label"])
        item_id = f"hellaswag:validation:{source_index}"
        metric = "acc_norm"
    elif task == "mmlu":
        if subject not in MMLU_SUBJECTS or row["subject"] != subject:
            raise ValueError("MMLU row must belong to its pinned subject configuration")
        if len(row["choices"]) != 4:
            raise ValueError("MMLU requires four choices")
        description = (
            "The following are multiple choice questions (with answers) about "
            + subject.replace("_", " ")
            + ".\n\n"
        )
        context = (
            description
            + row["question"].strip()
            + "\n"
            + "\n".join(
                f"{letter}. {answer}" for letter, answer in zip("ABCD", row["choices"], strict=True)
            )
            + "\nAnswer:"
        )
        choices = list("ABCD")
        gold = int(row["answer"])
        item_id = f"mmlu:test:{subject}:{source_index}"
        metric = "acc"
    else:
        raise ValueError(f"Unknown task: {task}")
    if len(choices) < 2 or any(not isinstance(c, str) or not c.strip() for c in choices):
        raise ValueError(f"Empty or invalid choice in {item_id}")
    if not 0 <= gold < len(choices):
        raise ValueError(f"Invalid gold in {item_id}")
    item = {
        "item_id": item_id,
        "task": task,
        "context": context,
        "choices": choices,
        "gold": gold,
        "normalization_lengths": [len(c) for c in choices],
        "metric": metric,
        "source_index": source_index,
    }
    if subject is not None:
        item["subject"] = subject
    return item


def proportional_allocation(counts: dict[str, int], sample_size: int) -> dict[str, int]:
    """Largest remainder; exact integer arithmetic, alphabetical subject ties."""
    total = sum(counts.values())
    if not counts or any(n <= 0 for n in counts.values()) or not 0 <= sample_size <= total:
        raise ValueError("Invalid stratum sizes or sample size")
    allocation = {subject: sample_size * n // total for subject, n in sorted(counts.items())}
    remaining = sample_size - sum(allocation.values())
    ranked = sorted(counts, key=lambda s: (-(sample_size * counts[s] % total), s))
    for subject in ranked[:remaining]:
        allocation[subject] += 1
    return allocation


def subject_seed(subject: str) -> int:
    """Seed is the full SHA256 digest of UTF-8 '<seed>:mmlu:<subject>'."""
    return int.from_bytes(hashlib.sha256(f"{SEED}:mmlu:{subject}".encode()).digest(), "big")


def _load_sources() -> dict[str, dict[str, list[dict]]]:
    # Heavy dependencies and downloads occur only when explicitly preparing items.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    sources = {}
    for task, spec in DATASETS.items():
        resolved = api.dataset_info(spec["repo_id"], revision=spec["revision"]).sha
        if resolved != spec["revision"]:
            raise ValueError(f"Dataset revision mismatch for {task}: {resolved}")
        configs = (
            MMLU_SUBJECTS
            if task == "mmlu"
            else (("ARC-Challenge",) if task == "arc_challenge" else ("default",))
        )
        sources[task] = {
            config: list(
                load_dataset(spec["repo_id"], config, split=spec["split"], revision=resolved)
            )
            for config in configs
        }
    return sources


def prepare_items(destination: str | Path) -> dict:
    """Fetch pinned sources, write deterministic items/selection, return selection.

    ``items.json.gz`` contains ``{'schema_version': 1, 'items': [...]}``.
    ``selection.json`` records every source row's ID/hash and selected IDs; its
    ``item_manifest_sha256`` hashes the exact compressed bytes consumed by runners.
    No tokenizers, models, or GPU operations are involved.
    """
    sources = _load_sources()
    if set(sources) != set(DATASETS):
        raise ValueError("Preparation requires all three tasks")
    items = []
    selection = {
        "schema_version": 1,
        "seed": SEED,
        "harness_revision": HARNESS_REVISION,
        "num_fewshot": 0,
        "apply_chat_template": False,
        "target_delimiter": " ",
        "normalization": "Python len(choice), excluding target delimiter; only acc_norm divides",
        "source_citations": SOURCE_CITATIONS,
        "datasets": {},
        "sampling_algorithm": {
            "arc_challenge": "All test rows in source order",
            "hellaswag": "sorted(random.Random(seed).sample(range(source_count), 2000))",
            "mmlu": "Proportional subject counts, integer largest remainder; alphabetical ties; "
            "sorted(random.Random(subject_seed).sample(range(subject_count), allocation)); "
            "alphabetical subject order, then source index",
            "subject_seed": "int.from_bytes(SHA256(UTF8('<seed>:mmlu:<subject>')), 'big')",
            "raw_row_hash": (
                "SHA256 of sorted-key compact UTF-8 JSON, ensure_ascii=False, allow_nan=False"
            ),
            "dataset_raw_rows_hash": (
                "SHA256 of concatenated raw-row JSON bytes, each followed by LF, "
                "alphabetical config order then source index"
            ),
        },
    }
    for task, spec in DATASETS.items():
        configs = sources[task]
        expected_configs = (
            set(MMLU_SUBJECTS)
            if task == "mmlu"
            else {"ARC-Challenge" if task == "arc_challenge" else "default"}
        )
        if set(configs) != expected_configs:
            raise ValueError(f"Unexpected source configurations for {task}")
        counts = {config: len(rows) for config, rows in configs.items()}
        if sum(counts.values()) != spec["source_count"]:
            raise ValueError(f"Unexpected source count for {task}: {sum(counts.values())}")
        allocations = (
            proportional_allocation(counts, spec["selected_count"])
            if task == "mmlu"
            else {next(iter(configs)): spec["selected_count"]}
        )
        dataset_meta = {
            **spec,
            "api_citation": f"https://huggingface.co/api/datasets/{spec['repo_id']}",
            "source_citation": f"https://huggingface.co/datasets/{spec['repo_id']}/tree/{spec['revision']}",
            "configs": {},
            "item_ids": [],
        }
        dataset_hash = hashlib.sha256()
        for config, rows in sorted(configs.items()):
            if task == "arc_challenge":
                indices = list(range(len(rows)))
            else:
                seed = subject_seed(config) if task == "mmlu" else SEED
                indices = sorted(random.Random(seed).sample(range(len(rows)), allocations[config]))
            selected = set(indices)
            source_rows = []
            for index, row in enumerate(rows):
                item = render_item(task, row, index, config if task == "mmlu" else None)
                raw_bytes = _json_bytes(row)
                dataset_hash.update(raw_bytes)
                dataset_hash.update(b"\n")
                source_rows.append(
                    {
                        "item_id": item["item_id"],
                        "source_index": index,
                        "raw_row_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    }
                )
                if index in selected:
                    items.append(item)
                    dataset_meta["item_ids"].append(item["item_id"])
            source_ids = [row["item_id"] for row in source_rows]
            if len(set(source_ids)) != len(source_ids):
                raise ValueError(f"Duplicate source IDs for {task}/{config}")
            dataset_meta["configs"][config] = {
                "source_count": len(rows),
                "selected_count": len(indices),
                "selected_source_indices": indices,
                "source_rows": source_rows,
                **({"sampling_seed": subject_seed(config)} if task == "mmlu" else {}),
            }
        dataset_meta["raw_rows_sha256"] = dataset_hash.hexdigest()
        selection["datasets"][task] = dataset_meta
    ids = [item["item_id"] for item in items]
    if len(ids) != len(set(ids)) or len(items) != 5172:
        raise ValueError("Preparation requires 5,172 unique selected items")
    selection["item_ids"] = ids
    selection["counts"] = {task: spec["selected_count"] for task, spec in DATASETS.items()}
    payload = _json_bytes({"schema_version": 1, "items": items})
    selection["items_json_sha256"] = hashlib.sha256(payload).hexdigest()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # An empty filename prevents destination-specific gzip headers; zero mtime.
    with (destination / "items.json.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(payload)
    selection["item_manifest_sha256"] = hashlib.sha256(
        (destination / "items.json.gz").read_bytes()
    ).hexdigest()
    (destination / "selection.json").write_bytes(_json_bytes(selection) + b"\n")
    return selection


def encode_choice(
    tokenizer: Any,
    context: str,
    choice: str,
    eot_token_id: int,
    add_bos_token: bool | None = None,
) -> tuple[list[int], list[int]]:
    """Encode a raw choice with the harness's one-space target delimiter.

    The pin's HFLM default is None: leave encode kwargs empty and delegate special
    tokens to the tokenizer. False/True explicitly set add_special_tokens. Prefix
    selection prefers tokenizer.bos_token_id, falling back to eot_token_id, as in
    HFLM. Cross-boundary token merges are rejected instead of silently scoring a
    suffix with a different context prefix.
    """
    if not isinstance(context, str) or not isinstance(choice, str) or not choice.strip():
        raise ValueError("Context must be text and choice must be nonempty text")
    prefix_token_id = getattr(tokenizer, "bos_token_id", None)
    if prefix_token_id is None:
        prefix_token_id = eot_token_id
    continuation = " " + choice
    if context == "":
        continuation_ids = list(tokenizer.encode(continuation, add_special_tokens=False))
        if not continuation_ids:
            raise ValueError("Choice has no continuation tokens")
        if continuation_ids[0] == prefix_token_id:
            context_ids, continuation_ids = continuation_ids[:1], continuation_ids[1:]
        else:
            context_ids = [prefix_token_id]
    else:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        if not context:
            raise ValueError("Whitespace-only context has an unsupported token boundary")
        # HFLM disables special tokens for an already present prefix token.
        prefix = tokenizer.decode([prefix_token_id])
        kwargs = {} if add_bos_token is None else {"add_special_tokens": add_bos_token}
        if prefix is not None and context.startswith(prefix):
            kwargs["add_special_tokens"] = False
        whole_ids = list(tokenizer.encode(context + continuation, **kwargs))
        context_ids = list(tokenizer.encode(context, **kwargs))
        if not context_ids or whole_ids[: len(context_ids)] != context_ids:
            raise ValueError("Unsupported context/continuation token boundary")
        continuation_ids = whole_ids[len(context_ids) :]
    if not continuation_ids:
        raise ValueError("Choice has no continuation tokens")
    return context_ids, continuation_ids
