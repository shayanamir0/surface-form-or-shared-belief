from __future__ import annotations

import json
import math
import random
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REWARDBENCH_DATASET = "allenai/reward-bench"
REWARDBENCH_CHAT_SUBSETS = frozenset(
    {
        "alpacaeval-easy",
        "alpacaeval-hard",
        "alpacaeval-length",
        "mt-bench-easy",
        "mt-bench-med",
    }
)
REWARDBENCH_CHAT_HARD_SUBSETS = frozenset(
    {
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    }
)

CHAOS_MNLI_URL = (
    "https://huggingface.co/datasets/metaeval/chaos-mnli-ambiguity/"
    "resolve/main/chaos_mnli.jsonl"
)
LABEL_MAP = {
    "e": "entailment",
    "n": "neutral",
    "c": "contradiction",
    "entailment": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
}


def download_chaos_mnli(destination: Path, *, url: str = CHAOS_MNLI_URL) -> Path:
    """Download the canonical ChaosNLI-MNLI JSONL file if it is not cached."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        destination.write_bytes(response.read())
    return destination


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_chaos_item(raw: dict[str, Any]) -> dict[str, Any]:
    example = raw.get("example", raw)
    counts = raw.get("label_count")
    if counts is None:
        counter = raw["label_counter"]
        counts = [counter.get("e", 0), counter.get("n", 0), counter.get("c", 0)]

    total = sum(counts)
    if total <= 0:
        raise ValueError(f"Item {raw.get('uid')} has no human labels")
    probabilities = [count / total for count in counts]
    entropy = raw.get(
        "entropy",
        -sum(probability * math.log2(probability) for probability in probabilities if probability),
    )
    gold = LABEL_MAP[raw["majority_label"].lower()]
    return {
        "item_id": str(raw.get("uid", example.get("uid"))),
        "premise": example["premise"],
        "hypothesis": example["hypothesis"],
        "gold_label": gold,
        "human_label_counts": counts,
        "human_entropy": float(entropy),
    }


def entropy_stratified_sample(
    records: list[dict[str, Any]],
    *,
    n: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample approximately equal counts from empirical entropy terciles."""
    if n > len(records):
        raise ValueError(f"Requested {n} items from only {len(records)} records")

    ordered = sorted(records, key=lambda item: (item["human_entropy"], item["item_id"]))
    bins = [ordered[: len(ordered) // 3], ordered[len(ordered) // 3 : 2 * len(ordered) // 3]]
    bins.append(ordered[2 * len(ordered) // 3 :])

    base, remainder = divmod(n, 3)
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for index, bucket in enumerate(bins):
        count = base + (1 if index < remainder else 0)
        sample.extend(rng.sample(bucket, count))

    sample.sort(key=lambda item: item["item_id"])
    return sample


def prepare_chaos_mnli(
    raw_path: Path,
    output_path: Path,
    *,
    n: int = 500,
    seed: int = 42,
    dev_output_path: Path | None = None,
    dev_n: int = 40,
) -> list[dict[str, Any]]:
    download_chaos_mnli(raw_path)
    normalized = [normalize_chaos_item(item) for item in read_jsonl(raw_path)]
    sampled = entropy_stratified_sample(normalized, n=n, seed=seed)
    write_jsonl(output_path, sampled)
    if dev_output_path is not None:
        test_ids = {item["item_id"] for item in sampled}
        remaining = [item for item in normalized if item["item_id"] not in test_ids]
        dev = entropy_stratified_sample(remaining, n=dev_n, seed=seed + 1)
        write_jsonl(dev_output_path, dev)
    return sampled


def normalize_rewardbench_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize one RewardBench v1 Chat or Chat Hard preference pair."""
    subset = str(raw["subset"])
    if subset in REWARDBENCH_CHAT_SUBSETS:
        category = "chat"
    elif subset in REWARDBENCH_CHAT_HARD_SUBSETS:
        category = "chat_hard"
    else:
        raise ValueError(f"RewardBench subset is outside scope: {subset}")

    prompt = str(raw["prompt"]).strip()
    chosen = str(raw["chosen"]).strip()
    rejected = str(raw["rejected"]).strip()
    if not prompt or not chosen or not rejected:
        raise ValueError(f"RewardBench item {raw.get('id')} has empty text")
    return {
        "item_id": f"{subset}:{raw['id']}",
        "task": "preference",
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "subset": subset,
        "category": category,
    }


def category_stratified_sample(
    records: list[dict[str, Any]],
    *,
    n: int = 300,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample approximately equal counts from RewardBench Chat and Chat Hard."""
    categories = ("chat", "chat_hard")
    buckets = {
        category: [record for record in records if record["category"] == category]
        for category in categories
    }
    if n > sum(len(bucket) for bucket in buckets.values()):
        raise ValueError(f"Requested {n} items from only {sum(map(len, buckets.values()))}")

    base, remainder = divmod(n, len(categories))
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for index, category in enumerate(categories):
        count = base + (1 if index < remainder else 0)
        bucket = buckets[category]
        if count > len(bucket):
            raise ValueError(f"Requested {count} {category} items from only {len(bucket)}")
        sampled.extend(rng.sample(bucket, count))
    sampled.sort(key=lambda item: item["item_id"])
    return sampled


def prepare_rewardbench(
    output_path: Path,
    *,
    n: int = 300,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Download and prepare the fixed RewardBench Chat/Chat Hard sample."""
    from datasets import load_dataset

    raw_records = load_dataset(REWARDBENCH_DATASET, split="filtered")
    allowed = REWARDBENCH_CHAT_SUBSETS | REWARDBENCH_CHAT_HARD_SUBSETS
    normalized = [
        normalize_rewardbench_item(dict(record))
        for record in raw_records
        if str(record["subset"]) in allowed
    ]
    sampled = category_stratified_sample(normalized, n=n, seed=seed)
    write_jsonl(output_path, sampled)
    return sampled
