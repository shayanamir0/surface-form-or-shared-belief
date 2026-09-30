from __future__ import annotations

import re
from typing import Any

NLI_LABELS = ("entailment", "neutral", "contradiction")
PREFERENCE_LABELS = ("A", "B")

BASELINE_INSTRUCTIONS = """You are an impartial natural-language inference judge.
Classify the relationship between the premise and hypothesis using exactly one label:
entailment, neutral, or contradiction.

Definitions:
- entailment: the premise makes the hypothesis definitely true.
- contradiction: the premise makes the hypothesis definitely false.
- neutral: neither entailment nor contradiction follows.

Return only the label, with no explanation."""

BASELINE_PREFERENCE_INSTRUCTIONS = """You are an impartial evaluator of AI assistant responses.
Given a user prompt and two responses, choose the response that is more helpful, correct,
relevant, well-written, and safe. Follow the user's intent; do not prefer an answer merely
because it is longer or appears first.

Return only A or B, with no explanation."""


def baseline_nli_prompt(item: dict[str, Any]) -> str:
    return (
        f"Premise: {item['premise']}\n"
        f"Hypothesis: {item['hypothesis']}\n"
        "Label:"
    )


def parse_nli_label(text: str) -> str:
    """Strictly parse one NLI label while tolerating harmless punctuation."""
    normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    matches = [label for label in NLI_LABELS if re.search(rf"\b{label}\b", normalized)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one NLI label, got {text!r}")
    return matches[0]


def baseline_preference_prompt(
    *,
    prompt: str,
    option_a: str,
    option_b: str,
) -> str:
    return (
        f"User prompt:\n{prompt}\n\n"
        f"Response A:\n{option_a}\n\n"
        f"Response B:\n{option_b}\n\n"
        "Better response:"
    )


def parse_preference_label(text: str) -> str:
    """Strictly parse one A/B preference while tolerating harmless wrappers."""
    stripped = text.strip()
    first_line = stripped.splitlines()[0].strip() if stripped else ""
    first_token = re.sub(r"[^A-Za-z]+", "", first_line).upper()
    if first_token in PREFERENCE_LABELS:
        return first_token
    normalized = re.sub(r"[^A-Za-z]+", " ", stripped.upper()).strip()
    matches = [label for label in PREFERENCE_LABELS if re.search(rf"\b{label}\b", normalized)]
    if len(matches) == 1:
        return matches[0]
    tokens = re.findall(r"\b[AB]\b", normalized)
    if (
        len(set(tokens)) == 2
        and len(tokens) >= 2
        and re.search(r"\b(NO|WAIT|ACTUALLY|CORRECTION|INSTEAD)\b", normalized)
    ):
        return tokens[-1]
    raise ValueError(f"Expected exactly one preference label, got {text!r}")
