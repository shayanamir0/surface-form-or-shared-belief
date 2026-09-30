from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from .data import read_jsonl
from .prompts import (
    BASELINE_INSTRUCTIONS,
    BASELINE_PREFERENCE_INSTRUCTIONS,
    baseline_nli_prompt,
    baseline_preference_prompt,
    parse_nli_label,
    parse_preference_label,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    model: str
    input_usd_per_million: float
    output_usd_per_million: float
    effort: str | None = None


# Current first-party model IDs and standard API prices, verified 2026-08-10.
MODEL_SPECS = {
    spec.key: spec
    for spec in (
        ModelSpec("openai-sol", "openai", "gpt-5.6-sol", 5.00, 30.00, "none"),
        ModelSpec("openai-terra", "openai", "gpt-5.6-terra", 2.00, 12.00, "none"),
        ModelSpec("openai-luna", "openai", "gpt-5.6-luna", 0.20, 1.20, "none"),
        ModelSpec("anthropic-opus", "anthropic", "claude-opus-5", 5.00, 25.00, "low"),
        ModelSpec("anthropic-sonnet", "anthropic", "claude-sonnet-5", 2.00, 10.00, "low"),
        ModelSpec(
            "anthropic-haiku",
            "anthropic",
            "claude-haiku-4-5-20251001",
            1.00,
            5.00,
        ),
    )
}
DEFAULT_MODEL_KEYS = tuple(MODEL_SPECS)
CONDITIONS = ("C0", "C1", "C2", "C2b", "C3")

RUBRICS = (
    "Use only what is logically guaranteed by the premise; do not add world knowledge.",
    "Actively search for a counterexample before choosing entailment or contradiction.",
    "Translate the pair into a minimal formal statement, then classify it.",
    "Pay special attention to quantifiers, negation, modality, and scope.",
    "First decide whether both sentences can be true together, then whether the "
    "premise forces the hypothesis.",
    "Read literally and treat unstated details as unknown rather than false.",
)


@dataclass(frozen=True)
class Judgment:
    item_id: str
    task: str
    provider: str
    model_key: str
    model: str
    resolved_model: str | None
    condition: str
    repeat: int
    prediction: str
    gold_label: str
    raw_output: str
    response_id: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    ab_order: str | None = None
    option_a_source: str | None = None
    option_b_source: str | None = None
    view_index: int | None = None


def select_models(keys: list[str] | tuple[str, ...]) -> list[ModelSpec]:
    unknown = sorted(set(keys) - MODEL_SPECS.keys())
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Choose from {sorted(MODEL_SPECS)}")
    return [MODEL_SPECS[key] for key in keys]


def token_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * spec.input_usd_per_million
        + output_tokens * spec.output_usd_per_million
    ) / 1_000_000


def judgment_token_cap(spec: ModelSpec) -> int:
    # Opus 5 and Sonnet 5 may spend part of max_tokens on adaptive thinking before
    # emitting the label. A 128-token cap occasionally leaves no text block.
    return 512 if spec.provider == "anthropic" and spec.effort is not None else 128


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record["item_id"]),
        str(record["model_key"]),
        str(record["condition"]),
        int(record.get("repeat", 0)),
    )


def _load_completed(path: Path) -> set[tuple[str, str, str, int]]:
    return {_record_key(record) for record in read_jsonl(path)} if path.exists() else set()


def _text_from_anthropic(message: Any) -> str:
    texts = [block.text for block in message.content if getattr(block, "type", None) == "text"]
    if not texts:
        raise ValueError("Anthropic response contained no text block")
    return "\n".join(texts)


async def _judge_openai(
    client: AsyncOpenAI,
    spec: ModelSpec,
    *,
    instructions: str,
    prompt: str,
    max_tokens: int,
) -> tuple[str, str | None, str | None, int, int]:
    response = await client.responses.create(
        model=spec.model,
        instructions=instructions,
        input=prompt,
        max_output_tokens=max_tokens,
        reasoning={"effort": spec.effort},
        store=False,
    )
    usage = response.usage
    return (
        response.output_text,
        getattr(response, "id", None),
        getattr(response, "model", None),
        int(usage.input_tokens),
        int(usage.output_tokens),
    )


async def _judge_anthropic(
    client: AsyncAnthropic,
    spec: ModelSpec,
    *,
    instructions: str,
    prompt: str,
    max_tokens: int,
) -> tuple[str, str | None, str | None, int, int]:
    kwargs: dict[str, Any] = {}
    if spec.effort is not None:
        kwargs["output_config"] = {"effort": spec.effort}
    message = await client.messages.create(
        model=spec.model,
        max_tokens=max_tokens,
        system=instructions,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return (
        _text_from_anthropic(message),
        getattr(message, "id", None),
        getattr(message, "model", None),
        int(message.usage.input_tokens),
        int(message.usage.output_tokens),
    )


def _examples_for_judge(dev_items: list[dict[str, Any]], judge_index: int) -> str:
    if len(dev_items) < 4:
        raise ValueError("C3 requires at least four held-out development items")
    examples = random.Random(42 + judge_index).sample(dev_items, 4)
    rendered = []
    for index, example in enumerate(examples, start=1):
        rendered.append(
            f"Example {index}:\n{baseline_nli_prompt(example)} {example['gold_label']}"
        )
    return "\n\n".join(rendered)


def infer_task(item: dict[str, Any]) -> str:
    task = item.get("task")
    if task in {"nli", "preference"}:
        return str(task)
    if {"premise", "hypothesis", "gold_label"} <= item.keys():
        return "nli"
    if {"prompt", "chosen", "rejected"} <= item.keys():
        return "preference"
    raise ValueError(f"Cannot infer task from item fields: {sorted(item)}")


def _resolve_task(items: list[dict[str, Any]], task: str | None) -> str:
    if not items:
        raise ValueError("Dataset is empty")
    resolved = task or infer_task(items[0])
    if resolved not in {"nli", "preference"}:
        raise ValueError("task must be 'nli' or 'preference'")
    mismatched = [item["item_id"] for item in items if infer_task(item) != resolved]
    if mismatched:
        raise ValueError(f"Dataset mixes tasks; first mismatch={mismatched[0]}")
    return resolved


def preference_order(item_id: str, *, seed: int = 42) -> tuple[str, str, str]:
    """Return deterministic A/B source order, invariant across judges and conditions."""
    digest = hashlib.sha256(f"rewardbench-ab|{seed}|{item_id}".encode()).digest()
    if int.from_bytes(digest[:8], "big") % 2 == 0:
        return "chosen_first", "chosen", "rejected"
    return "rejected_first", "rejected", "chosen"


def _condition_prompt(
    item: dict[str, Any],
    *,
    task: str,
    condition: str,
    judge_index: int,
    dev_items: list[dict[str, Any]] | None,
    views: dict[str, list[dict[str, str]]] | None,
) -> tuple[str, str]:
    if task == "preference":
        if condition not in {"C0", "C2", "C2b"}:
            raise ValueError(f"Preference task supports only C0, C2, and C2b; got {condition}")
        prompt_item = item
        if condition in {"C2", "C2b"}:
            if views is None or item["item_id"] not in views:
                raise ValueError(f"Missing generated views for item {item['item_id']}")
            candidates = views[item["item_id"]]
            view_index = judge_index if condition == "C2" else 0
            if view_index >= len(candidates):
                raise ValueError(f"Item {item['item_id']} has only {len(candidates)} views")
            prompt_item = {**item, **candidates[view_index]}
        _, option_a_source, option_b_source = preference_order(str(item["item_id"]))
        return (
            BASELINE_PREFERENCE_INSTRUCTIONS,
            baseline_preference_prompt(
                prompt=str(prompt_item["prompt"]),
                option_a=str(item[option_a_source]),
                option_b=str(item[option_b_source]),
            ),
        )

    instructions = BASELINE_INSTRUCTIONS
    prompt_item = item
    prefix = ""
    if condition == "C1":
        instructions = f"{instructions}\n\nAdditional judging rule:\n{RUBRICS[judge_index]}"
    elif condition in {"C2", "C2b"}:
        if views is None or item["item_id"] not in views:
            raise ValueError(f"Missing generated views for item {item['item_id']}")
        candidates = views[item["item_id"]]
        view_index = judge_index if condition == "C2" else 0
        if view_index >= len(candidates):
            raise ValueError(f"Item {item['item_id']} has only {len(candidates)} views")
        prompt_item = {**item, **candidates[view_index]}
    elif condition == "C3":
        if dev_items is None:
            raise ValueError("C3 requires --dev-dataset")
        prefix = _examples_for_judge(dev_items, judge_index) + "\n\nNow classify the target:\n"
    elif condition != "C0":
        raise ValueError(f"Unsupported condition {condition}")
    return instructions, prefix + baseline_nli_prompt(prompt_item)


async def _request_with_retries(
    *,
    spec: ModelSpec,
    openai_client: AsyncOpenAI | None,
    anthropic_client: AsyncAnthropic | None,
    instructions: str,
    prompt: str,
    retries: int,
    max_tokens: int,
    validator: Callable[[str], Any] | None = None,
) -> tuple[str, str | None, str | None, int, int]:
    last_error: Exception | None = None
    total_input_tokens = 0
    total_output_tokens = 0
    for attempt in range(retries + 1):
        try:
            if spec.provider == "openai":
                if openai_client is None:
                    raise RuntimeError("OPENAI_API_KEY is required")
                result = await _judge_openai(
                    openai_client,
                    spec,
                    instructions=instructions,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
            else:
                if anthropic_client is None:
                    raise RuntimeError("ANTHROPIC_API_KEY is required")
                result = await _judge_anthropic(
                    anthropic_client,
                    spec,
                    instructions=instructions,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
            total_input_tokens += result[3]
            total_output_tokens += result[4]
            if validator is not None:
                validator(result[0])
            return (
                result[0],
                result[1],
                result[2],
                total_input_tokens,
                total_output_tokens,
            )
        except Exception as error:
            last_error = error
            if attempt < retries:
                await asyncio.sleep(min(20.0, 2**attempt + random.random()))
    raise RuntimeError(f"Failed model={spec.key} after {retries + 1} attempts") from last_error


async def run_condition(
    *,
    dataset_path: Path,
    output_path: Path,
    condition: str,
    task: str | None = None,
    model_keys: list[str] | tuple[str, ...] = DEFAULT_MODEL_KEYS,
    dev_dataset_path: Path | None = None,
    views_path: Path | None = None,
    limit: int | None = None,
    concurrency: int = 6,
    retries: int = 4,
    openai_client: AsyncOpenAI | None = None,
    anthropic_client: AsyncAnthropic | None = None,
) -> int:
    if condition not in CONDITIONS:
        raise ValueError(f"Condition must be one of {CONDITIONS}")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    specs = select_models(model_keys)
    items = read_jsonl(dataset_path)
    if limit is not None:
        items = items[:limit]
    resolved_task = _resolve_task(items, task)
    if resolved_task == "preference" and condition not in {"C0", "C2", "C2b"}:
        raise ValueError(f"Preference task supports only C0, C2, and C2b; got {condition}")
    dev_items = (
        read_jsonl(dev_dataset_path) if condition == "C3" and dev_dataset_path else None
    )
    view_records = (
        read_jsonl(views_path) if condition in {"C2", "C2b"} and views_path else []
    )
    views = {record["item_id"]: record["views"] for record in view_records} or None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(output_path)
    pending = [
        (item, spec, judge_index, repeat)
        for item in items
        for judge_index, spec in enumerate(specs)
        for repeat in (0,)
        if (item["item_id"], spec.key, condition, repeat) not in completed
    ]
    if not pending:
        return 0

    owns_openai = openai_client is None and any(spec.provider == "openai" for spec in specs)
    owns_anthropic = anthropic_client is None and any(
        spec.provider == "anthropic" for spec in specs
    )
    oa_client = openai_client or (AsyncOpenAI() if owns_openai else None)
    an_client = anthropic_client or (AsyncAnthropic() if owns_anthropic else None)
    validator = parse_preference_label if resolved_task == "preference" else parse_nli_label
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def run_one(
        item: dict[str, Any], spec: ModelSpec, judge_index: int, repeat: int
    ) -> None:
        instructions, prompt = _condition_prompt(
            item,
            task=resolved_task,
            condition=condition,
            judge_index=judge_index,
            dev_items=dev_items,
            views=views,
        )
        async with semaphore:
            raw, response_id, resolved_model, input_tokens, output_tokens = (
                await _request_with_retries(
                    spec=spec,
                    openai_client=oa_client,
                    anthropic_client=an_client,
                    instructions=instructions,
                    prompt=prompt,
                    retries=retries,
                    max_tokens=judgment_token_cap(spec),
                    validator=validator,
                )
            )
        if resolved_task == "preference":
            parsed_prediction = parse_preference_label(raw)
            ab_order, option_a_source, option_b_source = preference_order(str(item["item_id"]))
            gold_label = "A" if option_a_source == "chosen" else "B"
            view_index = judge_index if condition == "C2" else (0 if condition == "C2b" else None)
        else:
            parsed_prediction = parse_nli_label(raw)
            gold_label = str(item["gold_label"])
            ab_order = None
            option_a_source = None
            option_b_source = None
            view_index = judge_index if condition == "C2" else (0 if condition == "C2b" else None)
        judgment = Judgment(
            item_id=item["item_id"],
            task=resolved_task,
            provider=spec.provider,
            model_key=spec.key,
            model=spec.model,
            resolved_model=resolved_model,
            condition=condition,
            repeat=repeat,
            prediction=parsed_prediction,
            gold_label=gold_label,
            raw_output=raw,
            response_id=response_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=token_cost(spec, input_tokens, output_tokens),
            ab_order=ab_order,
            option_a_source=option_a_source,
            option_b_source=option_b_source,
            view_index=view_index,
        )
        line = json.dumps(asdict(judgment), ensure_ascii=False, sort_keys=True)
        async with write_lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    try:
        async with asyncio.TaskGroup() as group:
            for item, spec, judge_index, repeat in pending:
                group.create_task(run_one(item, spec, judge_index, repeat))
    finally:
        if owns_openai and oa_client is not None:
            await oa_client.close()
        if owns_anthropic and an_client is not None:
            await an_client.close()
    return len(pending)


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Response did not contain a JSON object")
    return json.loads(cleaned[start : end + 1])


def _view_generation_prompt(
    item: dict[str, Any],
    *,
    task: str,
    view_count: int,
) -> tuple[str, str, str]:
    if task == "preference":
        return (
            "You rewrite user requests without changing their intent or constraints.",
            f"""Create exactly {view_count} meaning-preserving rewrites of this user prompt.
Preserve every request, constraint, ambiguity, temporal detail, modality, reference,
quantifier, normative term, and safety-relevant detail. Do not answer the prompt.
Before returning, silently verify that each rewrite has the same intent and constraints.
Return JSON only: {{"views":[{{"prompt":"..."}}, ...]}}

User prompt: {item["prompt"]}""",
            "preference-v2",
        )

    return (
        "You rewrite NLI sentences while preserving their exact truth conditions.",
        f"""Create exactly {view_count} meaning-preserving rewrites of this NLI item.

Hard constraints:
- Preserve the premise's truth conditions and the hypothesis's truth conditions separately.
- Do not add, remove, infer, explain, or make explicit any fact absent from the original.
- Preserve ambiguity and unresolved coreference; do not resolve who, what, or where a
  phrase refers to.
- Preserve temporal force exactly (for example, immediately is not soon; during is not by).
- Preserve modality and certainty exactly (may, must, can, did, seemed, and stated are
  not interchangeable).
- Preserve every quantifier, negation, comparison, presupposition, and scope relation.
- Preserve normative force exactly (too long is not merely very long; allowed is not below a limit).
- The NLI relation must remain {item["gold_label"]}.

Before returning, silently classify every rewritten pair and revise any candidate whose
relation or sentence-level meaning differs from the original.
Return JSON only: {{"views":[{{"premise":"...","hypothesis":"..."}}, ...]}}

Premise: {item["premise"]}
Hypothesis: {item["hypothesis"]}""",
        "nli-v2",
    )


async def generate_views(
    *,
    dataset_path: Path,
    output_path: Path,
    task: str | None = None,
    view_count: int = 6,
    limit: int | None = None,
    concurrency: int = 4,
    retries: int = 4,
    reasoning_effort: str = "low",
    client: AsyncOpenAI | None = None,
) -> int:
    """Generate C2 views with the cheapest current OpenAI model, resumably."""
    items = read_jsonl(dataset_path)
    if limit is not None:
        items = items[:limit]
    resolved_task = _resolve_task(items, task)
    completed = (
        {record["item_id"] for record in read_jsonl(output_path)} if output_path.exists() else set()
    )
    pending = [item for item in items if item["item_id"] not in completed]
    if not pending:
        return 0

    if reasoning_effort not in {"none", "low", "medium", "high"}:
        raise ValueError("reasoning_effort must be one of: none, low, medium, high")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = replace(MODEL_SPECS["openai-luna"], effort=reasoning_effort)
    owns_client = client is None
    api_client = client or AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    max_view_tokens = 12_000 if resolved_task == "preference" else 2_000

    async def run_one(item: dict[str, Any]) -> None:
        instructions, prompt, prompt_version = _view_generation_prompt(
            item,
            task=resolved_task,
            view_count=view_count,
        )
        async with semaphore:
            raw, response_id, resolved_model, input_tokens, output_tokens = (
                await _request_with_retries(
                    spec=spec,
                    openai_client=api_client,
                    anthropic_client=None,
                    instructions=instructions,
                    prompt=prompt,
                    retries=retries,
                    max_tokens=max_view_tokens,
                    validator=_extract_json_object,
                )
            )
        payload = _extract_json_object(raw)
        views = payload.get("views")
        if not isinstance(views, list) or len(views) != view_count:
            raise ValueError(f"Expected {view_count} views for item {item['item_id']}")
        for view in views:
            valid = (
                isinstance(view, dict)
                and (
                    bool(view.get("prompt"))
                    if resolved_task == "preference"
                    else bool(view.get("premise")) and bool(view.get("hypothesis"))
                )
            )
            if not valid:
                raise ValueError(f"Malformed view for item {item['item_id']}")
        record = {
            "item_id": item["item_id"],
            "task": resolved_task,
            "prompt_version": prompt_version,
            "reasoning_effort": reasoning_effort,
            "model": spec.model,
            "resolved_model": resolved_model,
            "response_id": response_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": token_cost(spec, input_tokens, output_tokens),
            "views": views,
        }
        async with write_lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()

    try:
        async with asyncio.TaskGroup() as group:
            for item in pending:
                group.create_task(run_one(item))
    finally:
        if owns_client:
            await api_client.close()
    return len(pending)


def cost_report(paths: list[Path]) -> dict[str, Any]:
    by_model: dict[str, dict[str, float | int]] = {}
    total = 0.0
    for path in paths:
        for record in read_jsonl(path):
            model = str(record.get("model_key", record.get("model", "unknown")))
            bucket = by_model.setdefault(
                model, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            )
            bucket["requests"] += 1
            bucket["input_tokens"] += int(record.get("input_tokens", 0))
            bucket["output_tokens"] += int(record.get("output_tokens", 0))
            bucket["cost_usd"] += float(record.get("cost_usd", 0.0))
            total += float(record.get("cost_usd", 0.0))
    return {"by_model": by_model, "total_cost_usd": total}


def create_view_audit(
    *,
    dataset_path: Path,
    views_path: Path,
    output_path: Path,
    task: str | None = None,
    items: int = 50,
    seed: int = 42,
) -> int:
    dataset = {record["item_id"]: record for record in read_jsonl(dataset_path)}
    resolved_task = _resolve_task(list(dataset.values()), task)
    view_records = read_jsonl(views_path)
    if items > len(view_records):
        raise ValueError(f"Requested {items} audit items from {len(view_records)} views")
    selected = random.Random(seed).sample(view_records, items)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_task == "preference":
        fields = (
            "item_id",
            "subset",
            "view_index",
            "original_prompt",
            "rewritten_prompt",
            "preserves_intent",
            "preserves_preference",
            "notes",
        )
    else:
        fields = (
            "item_id",
            "view_index",
            "original_premise",
            "original_hypothesis",
            "rewritten_premise",
            "rewritten_hypothesis",
            "preserves_label",
            "notes",
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in selected:
            original = dataset[record["item_id"]]
            for index, view in enumerate(record["views"]):
                if resolved_task == "preference":
                    row = {
                        "item_id": record["item_id"],
                        "subset": original["subset"],
                        "view_index": index,
                        "original_prompt": original["prompt"],
                        "rewritten_prompt": view["prompt"],
                        "preserves_intent": "",
                        "preserves_preference": "",
                        "notes": "",
                    }
                else:
                    row = {
                        "item_id": record["item_id"],
                        "view_index": index,
                        "original_premise": original["premise"],
                        "original_hypothesis": original["hypothesis"],
                        "rewritten_premise": view["premise"],
                        "rewritten_hypothesis": view["hypothesis"],
                        "preserves_label": "",
                        "notes": "",
                    }
                writer.writerow(row)
    return sum(len(record["views"]) for record in selected)


def estimate_cost(
    *,
    items: int = 500,
    conditions: tuple[str, ...] | None = None,
    task: str = "nli",
) -> dict[str, Any]:
    if task == "preference":
        conditions = conditions or ("C0", "C2", "C2b")
        estimated_input = {condition: 1_200 for condition in conditions}
        view_input_tokens = 800
    elif task == "nli":
        conditions = conditions or ("C0", "C1", "C2", "C2b", "C3")
        estimated_input = {"C0": 180, "C1": 210, "C2": 180, "C2b": 180, "C3": 430}
        view_input_tokens = 180
    else:
        raise ValueError("task must be 'nli' or 'preference'")
    output_tokens = 8
    by_model: dict[str, float] = {}
    for spec in MODEL_SPECS.values():
        by_model[spec.key] = sum(
            token_cost(spec, items * estimated_input[condition], items * output_tokens)
            for condition in conditions
        )
    view_generation = token_cost(
        MODEL_SPECS["openai-luna"],
        items * view_input_tokens,
        items * 350,
    )
    maximum_by_model = {
        spec.key: sum(
            token_cost(
                spec,
                items * estimated_input[condition],
                items * judgment_token_cap(spec),
            )
            for condition in conditions
        )
        for spec in MODEL_SPECS.values()
    }
    maximum_view_output_tokens = 12_000 if task == "preference" else 2_000
    maximum_view_generation = token_cost(
        MODEL_SPECS["openai-luna"],
        items * view_input_tokens,
        items * maximum_view_output_tokens,
    )
    return {
        "assumptions": {
            "items": items,
            "task": task,
            "conditions": list(conditions),
            "input_tokens_per_request": {
                condition: estimated_input[condition] for condition in conditions
            },
            "output_tokens_per_judgment": output_tokens,
            "generated_views_per_item": 6,
        },
        "by_model_usd": by_model,
        "view_generation_usd": view_generation,
        "total_usd": sum(by_model.values()) + view_generation,
        "maximum_if_every_response_hits_cap_usd": (
            sum(maximum_by_model.values()) + maximum_view_generation
        ),
    }
