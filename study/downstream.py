"""Full-vocabulary batch teacher-forced CE, exp(CE), and BF16-reference KL."""

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import torch

from study.attention import VARIANTS
from study.data import ROOT, model_spec, models, snapshot, texts, work_directory
from study.run import atomic_json, load_state, token_arrays
from study.stream import LayerStream

FIELDS = ("model", "family", "revision", "text", "variant", "tokens", "ce", "exp_ce", "mean_kl")


def combine(cache, destination):
    records = []
    for model in models():
        path = cache / model["key"] / "downstream/metrics.csv"
        if path.exists():
            with path.open(newline="") as stream:
                records.extend(csv.DictReader(stream))
    records.sort(key=lambda row: (row["model"], row["text"], VARIANTS.index(row["variant"])))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(destination)
    return len(records)


@torch.inference_mode()
def score(key, cache, results_dir=ROOT / "results/v2", vocab_chunk=1024):
    spec = model_spec(key)
    layer, finished, states = load_state(cache / key / "downstream/state.npz")
    if not finished or layer != spec["config"]["num_hidden_layers"]:
        raise ValueError("All downstream decoder layers and the final norm must be complete")
    text_keys = [text["key"] for text in texts()]
    names = [f"{variant}__{text}" for variant in VARIANTS for text in text_keys]
    if set(states) != set(names):
        raise ValueError("Downstream checkpoint variant/text coverage is incomplete")
    hidden = torch.cat([states[name] for name in names], dim=0)
    del states
    gc.collect()
    sequences = token_arrays(key, cache, "heldout")
    labels = torch.cat([sequences[text][:, 1:] for text in text_keys], dim=0)
    variants, books, tokens = len(VARIANTS), len(text_keys), labels.shape[-1]
    stream = LayerStream(snapshot(key))
    vocab = stream.config.vocab_size
    if labels.min() < 0 or labels.max() >= vocab:
        raise ValueError("Tokenizer produced target IDs outside the model vocabulary")
    logsum = torch.full((variants, books, tokens), -torch.inf, dtype=torch.float64)
    targets = torch.full_like(logsum, torch.nan)
    for start in range(0, vocab, vocab_chunk):
        stop = min(start + vocab_chunk, vocab)
        logits = (
            stream.project_logits(hidden, start, stop)
            .double()
            .reshape(variants, books, tokens, stop - start)
        )
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite projected logits")
        logsum = torch.logaddexp(logsum, torch.logsumexp(logits, dim=-1))
        for book in range(books):
            selected = (labels[book] >= start) & (labels[book] < stop)
            targets[:, book, selected] = logits[:, book, selected, labels[book, selected] - start]
    if not torch.isfinite(targets).all():
        raise ValueError("A next-token target was not projected")
    ce = (logsum - targets).mean(dim=-1)
    kl_tokens = torch.zeros_like(logsum)
    for start in range(0, vocab, vocab_chunk):
        stop = min(start + vocab_chunk, vocab)
        logits = (
            stream.project_logits(hidden, start, stop)
            .double()
            .reshape(variants, books, tokens, stop - start)
        )
        reference_logp = logits[0] - logsum[0, :, :, None]
        reference_p = reference_logp.exp()
        # Contract one variant at a time instead of allocating a full
        # [variants,books,tokens,vocab_chunk] error/probability product.
        for variant in range(variants):
            logp = logits[variant] - logsum[variant, :, :, None]
            kl_tokens[variant] += (reference_p * (reference_logp - logp)).sum(dim=-1)
    stream.close()
    mean_kl = kl_tokens.mean(dim=-1)
    if not torch.isfinite(ce).all() or not torch.isfinite(mean_kl).all():
        raise ValueError("Nonfinite likelihood metric")
    if torch.any(mean_kl < -1e-10):
        raise ValueError("Negative KL exceeds float64 roundoff")
    records = []
    for variant_index, variant in enumerate(VARIANTS):
        for book, text in enumerate(text_keys):
            loss = float(ce[variant_index, book])
            records.append(
                {
                    "model": key,
                    "family": spec["family"],
                    "revision": spec["revision"],
                    "text": text,
                    "variant": variant,
                    "tokens": tokens,
                    "ce": loss,
                    "exp_ce": math.exp(loss),
                    "mean_kl": float(mean_kl[variant_index, book]),
                }
            )
    metadata = {
        "model": key,
        "revision": spec["revision"],
        "tokens_per_text": tokens,
        "texts": text_keys,
        "variants": list(VARIANTS),
        "vocabulary": vocab,
        "vocabulary_chunk": vocab_chunk,
        "torch": torch.__version__,
        "device": "cpu",
        "weights_and_hidden": "bf16",
        "projection": "Native BF16 linear, then FP32 logits.",
        "metric_reduction": "FP64 full-vocabulary logsumexp and KL(BF16 || variant).",
        "attention_reference": "Unmodified native Transformers SDPA BF16 attention.",
        "evaluation": "Batch teacher forcing with raw text, no chat template, and reset context.",
        "limitation": (
            "Full-K centering and current-block scale calibration can depend on later tokens "
            "in the batch. Not streaming-decoder perplexity or generation."
        ),
    }
    atomic_json(results_dir / "downstream_metadata" / f"{key}.json", metadata)
    target = cache / key / "downstream/metrics.csv"
    temporary = target.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(target)
    count = combine(cache, results_dir / "downstream.csv")
    print(
        json.dumps(
            {
                "model": key,
                "rows": len(records),
                "combined_rows": count,
                "ce": {variant: float(ce[i].mean()) for i, variant in enumerate(VARIANTS)},
                "mean_kl": {
                    variant: float(mean_kl[i].mean()) for i, variant in enumerate(VARIANTS)
                },
            }
        ),
        flush=True,
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", type=Path, default=work_directory())
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/v2")
    parser.add_argument("--vocab-chunk", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.vocab_chunk < 1 or args.threads < 1:
        parser.error("vocab-chunk and threads must be positive")
    torch.set_num_threads(args.threads)
    score(args.model, args.work_dir, args.results_dir, args.vocab_chunk)
