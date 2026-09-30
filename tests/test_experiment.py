import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from behavioural_test.data import read_jsonl, write_jsonl
from behavioural_test.experiment import (
    MODEL_SPECS,
    estimate_cost,
    generate_views,
    judgment_token_cap,
    preference_order,
    run_condition,
)


class FakeOpenAI:
    def __init__(self, output: str = "entailment") -> None:
        self.responses = self
        self.output = output
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="openai-response",
            model=kwargs["model"],
            output_text=self.output,
            usage=SimpleNamespace(input_tokens=100, output_tokens=2),
        )


class FakeAnthropic:
    def __init__(self, output: str = "entailment") -> None:
        self.messages = self
        self.output = output

    async def create(self, **kwargs):
        return SimpleNamespace(
            id="anthropic-message",
            model=kwargs["model"],
            content=[SimpleNamespace(type="text", text=self.output)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=2),
        )


def _write_dataset(path: Path) -> None:
    write_jsonl(
        path,
        [
            {
                "item_id": f"item-{index}",
                "premise": "A dog is running.",
                "hypothesis": "An animal is running.",
                "gold_label": "entailment",
                "human_entropy": 0.1,
            }
            for index in range(2)
        ],
    )


def _write_preference_dataset(path: Path) -> None:
    write_jsonl(
        path,
        [
            {
                "item_id": f"pref-{index}",
                "task": "preference",
                "prompt": f"Question {index}",
                "chosen": "Helpful response.",
                "rejected": "Unhelpful response.",
                "subset": "alpacaeval-easy",
                "category": "chat",
            }
            for index in range(2)
        ],
    )


def test_two_provider_run_is_resumable(tmp_path: Path) -> None:
    dataset = tmp_path / "items.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset)

    kwargs = {
        "dataset_path": dataset,
        "output_path": output,
        "condition": "C0",
        "model_keys": ["openai-luna", "anthropic-haiku"],
        "openai_client": FakeOpenAI(),
        "anthropic_client": FakeAnthropic(),
    }
    assert asyncio.run(run_condition(**kwargs)) == 4
    assert asyncio.run(run_condition(**kwargs)) == 0

    records = read_jsonl(output)
    assert len(records) == 4
    assert {record["provider"] for record in records} == {"openai", "anthropic"}
    assert all(record["prediction"] == "entailment" for record in records)
    assert all(record["cost_usd"] > 0 for record in records)


def test_unknown_model_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "items.jsonl"
    _write_dataset(dataset)
    with pytest.raises(ValueError, match="Unknown model keys"):
        asyncio.run(
            run_condition(
                dataset_path=dataset,
                output_path=tmp_path / "results.jsonl",
                condition="C0",
                model_keys=["not-a-model"],
            )
        )


def test_cost_estimate_has_all_six_judges() -> None:
    estimate = estimate_cost(items=500)
    assert len(estimate["by_model_usd"]) == 6
    assert estimate["total_usd"] > 0
    json.dumps(estimate)


def test_adaptive_thinking_models_have_room_to_emit_a_label() -> None:
    assert judgment_token_cap(MODEL_SPECS["anthropic-opus"]) == 512
    assert judgment_token_cap(MODEL_SPECS["anthropic-sonnet"]) == 512
    assert judgment_token_cap(MODEL_SPECS["anthropic-haiku"]) == 128


@pytest.mark.parametrize("condition", ["C2", "C2b", "C3"])
def test_conditions_with_auxiliary_data(tmp_path: Path, condition: str) -> None:
    dataset = tmp_path / "items.jsonl"
    dev = tmp_path / "dev.jsonl"
    views = tmp_path / "views.jsonl"
    _write_dataset(dataset)
    write_jsonl(
        dev,
        [
            {
                "item_id": f"dev-{index}",
                "premise": "A cat sleeps.",
                "hypothesis": "An animal sleeps.",
                "gold_label": "entailment",
            }
            for index in range(4)
        ],
    )
    write_jsonl(
        views,
        [
            {
                "item_id": f"item-{item_index}",
                "views": [
                    {"premise": "A dog runs.", "hypothesis": "An animal runs."},
                    {"premise": "A canine runs.", "hypothesis": "An animal moves."},
                ],
            }
            for item_index in range(2)
        ],
    )
    created = asyncio.run(
        run_condition(
            dataset_path=dataset,
            output_path=tmp_path / f"{condition}.jsonl",
            condition=condition,
            model_keys=["openai-luna", "anthropic-haiku"],
            dev_dataset_path=dev,
            views_path=views,
            openai_client=FakeOpenAI(),
            anthropic_client=FakeAnthropic(),
        )
    )
    assert created == 4


@pytest.mark.parametrize("condition", ["C0", "C2", "C2b"])
def test_preference_conditions_are_resumable_and_log_order(
    tmp_path: Path, condition: str
) -> None:
    dataset = tmp_path / "preferences.jsonl"
    views = tmp_path / "preference_views.jsonl"
    output = tmp_path / f"{condition}.jsonl"
    _write_preference_dataset(dataset)
    write_jsonl(
        views,
        [
            {
                "item_id": f"pref-{item_index}",
                "task": "preference",
                "views": [{"prompt": "Rewritten one."}, {"prompt": "Rewritten two."}],
            }
            for item_index in range(2)
        ],
    )
    kwargs = {
        "dataset_path": dataset,
        "output_path": output,
        "condition": condition,
        "task": "preference",
        "model_keys": ["openai-luna", "anthropic-haiku"],
        "views_path": views,
        "openai_client": FakeOpenAI("A"),
        "anthropic_client": FakeAnthropic("A"),
    }
    assert asyncio.run(run_condition(**kwargs)) == 4
    assert asyncio.run(run_condition(**kwargs)) == 0
    records = read_jsonl(output)
    assert all(record["task"] == "preference" for record in records)
    assert all(record["prediction"] == "A" for record in records)
    for item_id in {record["item_id"] for record in records}:
        item_records = [record for record in records if record["item_id"] == item_id]
        assert len({record["ab_order"] for record in item_records}) == 1
        assert len({record["gold_label"] for record in item_records}) == 1


def test_preference_rejects_out_of_scope_condition(tmp_path: Path) -> None:
    dataset = tmp_path / "preferences.jsonl"
    _write_preference_dataset(dataset)
    with pytest.raises(ValueError, match="supports only C0, C2, and C2b"):
        asyncio.run(
            run_condition(
                dataset_path=dataset,
                output_path=tmp_path / "results.jsonl",
                condition="C1",
                task="preference",
                model_keys=["openai-luna"],
                openai_client=FakeOpenAI("A"),
            )
        )


def test_preference_order_is_deterministic() -> None:
    assert preference_order("item-1") == preference_order("item-1")
    order, option_a, option_b = preference_order("item-1")
    assert order in {"chosen_first", "rejected_first"}
    assert {option_a, option_b} == {"chosen", "rejected"}


def test_nli_v2_view_generation_uses_strict_prompt_and_reasoning(tmp_path: Path) -> None:
    dataset = tmp_path / "items.jsonl"
    output = tmp_path / "views_v2.jsonl"
    _write_dataset(dataset)
    payload = json.dumps(
        {
            "views": [
                {"premise": "A dog is running.", "hypothesis": "An animal is running."},
                {"premise": "A canine is running.", "hypothesis": "An animal is running."},
            ]
        }
    )
    client = FakeOpenAI(payload)
    created = asyncio.run(
        generate_views(
            dataset_path=dataset,
            output_path=output,
            view_count=2,
            limit=1,
            reasoning_effort="low",
            client=client,
        )
    )
    assert created == 1
    assert client.calls[0]["reasoning"] == {"effort": "low"}
    assert "Preserve ambiguity and unresolved coreference" in client.calls[0]["input"]
    assert "The NLI relation must remain entailment" in client.calls[0]["input"]
    record = read_jsonl(output)[0]
    assert record["prompt_version"] == "nli-v2"
    assert record["reasoning_effort"] == "low"
