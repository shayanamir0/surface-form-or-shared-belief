import pytest

from behavioural_test.data import (
    category_stratified_sample,
    entropy_stratified_sample,
    normalize_chaos_item,
    normalize_rewardbench_item,
)
from behavioural_test.prompts import parse_nli_label, parse_preference_label


def test_normalize_chaos_item() -> None:
    raw = {
        "uid": "x",
        "label_counter": {"e": 76, "n": 20, "c": 4},
        "majority_label": "e",
        "example": {"premise": "P", "hypothesis": "H"},
    }
    item = normalize_chaos_item(raw)
    assert item["gold_label"] == "entailment"
    assert item["human_label_counts"] == [76, 20, 4]
    assert item["human_entropy"] > 0


def test_entropy_stratified_sample_is_seeded() -> None:
    records = [
        {"item_id": str(index), "human_entropy": index / 30}
        for index in range(30)
    ]
    first = entropy_stratified_sample(records, n=15, seed=42)
    second = entropy_stratified_sample(records, n=15, seed=42)
    assert first == second
    assert len(first) == 15


def test_rewardbench_normalization_and_balanced_sample() -> None:
    raw = {
        "id": 7,
        "subset": "alpacaeval-easy",
        "prompt": "Help me.",
        "chosen": "Useful answer.",
        "rejected": "Bad answer.",
    }
    normalized = normalize_rewardbench_item(raw)
    assert normalized["item_id"] == "alpacaeval-easy:7"
    assert normalized["category"] == "chat"
    records = [
        {**normalized, "item_id": f"chat-{index}", "category": "chat"}
        for index in range(10)
    ] + [
        {**normalized, "item_id": f"hard-{index}", "category": "chat_hard"}
        for index in range(10)
    ]
    sample = category_stratified_sample(records, n=10, seed=42)
    assert sum(record["category"] == "chat" for record in sample) == 5
    assert sum(record["category"] == "chat_hard" for record in sample) == 5
    assert sample == category_stratified_sample(records, n=10, seed=42)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("entailment", "entailment"),
        ("Neutral.", "neutral"),
        ("The label is contradiction.", "contradiction"),
    ],
)
def test_parse_nli_label(text: str, expected: str) -> None:
    assert parse_nli_label(text) == expected


def test_parse_nli_label_rejects_ambiguous_output() -> None:
    with pytest.raises(ValueError):
        parse_nli_label("neutral or contradiction")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", "A"),
        ("[[B]]", "B"),
        (
            "A\n\nResponse A is more comprehensive.\n\nResponse B is shorter.",
            "A",
        ),
        ("B... no — A", "A"),
        ("A, wait, B", "B"),
    ],
)
def test_parse_preference_label(text: str, expected: str) -> None:
    assert parse_preference_label(text) == expected


def test_parse_preference_label_rejects_ambiguous_output() -> None:
    with pytest.raises(ValueError):
        parse_preference_label("A or B")
