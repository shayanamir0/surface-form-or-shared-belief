from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .analysis import (
    behavioral_invariance,
    compare_conditions,
    evaluate_primary_endpoints,
    summarize_panel,
)
from .data import prepare_chaos_mnli, prepare_rewardbench, read_jsonl
from .experiment import (
    CONDITIONS,
    DEFAULT_MODEL_KEYS,
    MODEL_SPECS,
    cost_report,
    create_view_audit,
    estimate_cost,
    generate_views,
    run_condition,
)


def _prepare(args: argparse.Namespace) -> None:
    records = prepare_chaos_mnli(
        args.raw,
        args.output,
        n=args.items,
        seed=args.seed,
        dev_output_path=args.dev_output,
        dev_n=args.dev_items,
    )
    print(f"Wrote {len(records)} entropy-stratified items to {args.output}")
    print(f"Wrote {args.dev_items} held-out exemplars to {args.dev_output}")


def _prepare_rewardbench(args: argparse.Namespace) -> None:
    records = prepare_rewardbench(args.output, n=args.items, seed=args.seed)
    print(f"Wrote {len(records)} RewardBench Chat/Chat-Hard items to {args.output}")


def _task_path(
    supplied: Path | None,
    task: str,
    *,
    nli: str,
    preference: str,
) -> Path:
    return supplied or Path(preference if task == "preference" else nli)


def _run(args: argparse.Namespace) -> None:
    dataset = _task_path(
        args.dataset,
        args.task,
        nli="data/processed/chaos_mnli_500.jsonl",
        preference="data/processed/rewardbench_300.jsonl",
    )
    views = _task_path(
        args.views,
        args.task,
        nli="data/processed/c2_views.jsonl",
        preference="data/processed/rewardbench_c2_views.jsonl",
    )
    default_output = (
        f"results/d2_{args.condition.lower()}.jsonl"
        if args.task == "preference"
        else f"results/{args.condition.lower()}.jsonl"
    )
    created = asyncio.run(
        run_condition(
            dataset_path=dataset,
            output_path=args.output or Path(default_output),
            condition=args.condition,
            task=args.task,
            model_keys=args.models,
            dev_dataset_path=args.dev_dataset,
            views_path=views,
            limit=args.limit,
            concurrency=args.concurrency,
            retries=args.retries,
        )
    )
    print(f"Wrote {created} new {args.condition} judgments")


def _generate_views(args: argparse.Namespace) -> None:
    dataset = _task_path(
        args.dataset,
        args.task,
        nli="data/processed/chaos_mnli_500.jsonl",
        preference="data/processed/rewardbench_300.jsonl",
    )
    output = _task_path(
        args.output,
        args.task,
        nli="data/processed/c2_views.jsonl",
        preference="data/processed/rewardbench_c2_views.jsonl",
    )
    created = asyncio.run(
        generate_views(
            dataset_path=dataset,
            output_path=output,
            task=args.task,
            view_count=args.view_count,
            limit=args.limit,
            concurrency=args.concurrency,
            retries=args.retries,
            reasoning_effort=args.reasoning_effort,
        )
    )
    print(f"Wrote views for {created} new items to {output}")


def _audit_views(args: argparse.Namespace) -> None:
    dataset = _task_path(
        args.dataset,
        args.task,
        nli="data/processed/chaos_mnli_500.jsonl",
        preference="data/processed/rewardbench_300.jsonl",
    )
    views = _task_path(
        args.views,
        args.task,
        nli="data/processed/c2_views.jsonl",
        preference="data/processed/rewardbench_c2_views.jsonl",
    )
    output = _task_path(
        args.output,
        args.task,
        nli="results/c2_view_audit.csv",
        preference="results/d2_c2_view_audit.csv",
    )
    rows = create_view_audit(
        dataset_path=dataset,
        views_path=views,
        output_path=output,
        task=args.task,
        items=args.items or (30 if args.task == "preference" else 50),
        seed=args.seed,
    )
    print(f"Wrote {rows} manual-audit rows to {output}")


def _analyze(args: argparse.Namespace) -> None:
    records = [
        record
        for record in read_jsonl(args.input)
        if record.get("condition", "C0") == args.condition
    ]
    summary = summarize_panel(
        records,
        bootstrap_resamples=args.bootstrap,
        seed=args.seed,
    )
    payload = summary.to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


def _compare(args: argparse.Namespace) -> None:
    payload = compare_conditions(
        read_jsonl(args.baseline),
        read_jsonl(args.intervention),
        bootstrap_resamples=args.bootstrap,
        noise_simulations=args.noise_simulations,
        seed=args.seed,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


def _estimate_cost(args: argparse.Namespace) -> None:
    items = args.items or (300 if args.task == "preference" else 500)
    print(json.dumps(estimate_cost(items=items, task=args.task), indent=2, sort_keys=True))


def _cost_report(args: argparse.Namespace) -> None:
    print(json.dumps(cost_report(args.inputs), indent=2, sort_keys=True))


def _endpoints(args: argparse.Namespace) -> None:
    paths = {
        "d1": (args.d1_baseline, args.d1_c2, args.d1_c2b),
        "d2": (args.d2_baseline, args.d2_c2, args.d2_c2b),
    }
    comparisons: dict[str, dict[str, dict]] = {}
    payload: dict[str, object] = {}
    for dataset, (baseline_path, c2_path, c2b_path) in paths.items():
        baseline = read_jsonl(baseline_path)
        c2 = read_jsonl(c2_path)
        c2b = read_jsonl(c2b_path)
        c2_comparison = compare_conditions(
            baseline,
            c2,
            bootstrap_resamples=args.bootstrap,
            noise_simulations=args.noise_simulations,
            seed=args.seed,
        )
        c2b_comparison = compare_conditions(
            baseline,
            c2b,
            bootstrap_resamples=args.bootstrap,
            noise_simulations=args.noise_simulations,
            seed=args.seed + 10,
        )
        comparisons[dataset] = {
            "c2_vs_c0": c2_comparison,
            "c2b_vs_c0": c2b_comparison,
        }
        payload[dataset] = {
            "c0_summary": summarize_panel(baseline).to_dict(),
            "c2_vs_c0": c2_comparison,
            "c2b_vs_c0": c2b_comparison,
            "c2_invariance": behavioral_invariance(baseline, c2).to_dict(),
        }
    payload["primary_endpoints"] = evaluate_primary_endpoints(comparisons)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="behavioural-test",
        description="Run and analyze LLM judge-panel independence experiments.",
    )
    subparsers = parser.add_subparsers(required=True)

    prepare = subparsers.add_parser("prepare-chaosnli", help="Prepare the D1 sample")
    prepare.add_argument("--raw", type=Path, default=Path("data/raw/chaos_mnli.jsonl"))
    prepare.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chaos_mnli_500.jsonl"),
    )
    prepare.add_argument("--items", type=int, default=500)
    prepare.add_argument(
        "--dev-output",
        type=Path,
        default=Path("data/processed/chaos_mnli_dev_40.jsonl"),
    )
    prepare.add_argument("--dev-items", type=int, default=40)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=_prepare)

    rewardbench = subparsers.add_parser(
        "prepare-rewardbench", help="Prepare the D2 RewardBench sample"
    )
    rewardbench.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/rewardbench_300.jsonl"),
    )
    rewardbench.add_argument("--items", type=int, default=300)
    rewardbench.add_argument("--seed", type=int, default=42)
    rewardbench.set_defaults(func=_prepare_rewardbench)

    views = subparsers.add_parser("generate-views", help="Generate C2/C2b paraphrases")
    views.add_argument("--task", choices=("nli", "preference"), default="nli")
    views.add_argument("--dataset", type=Path)
    views.add_argument("--output", type=Path)
    views.add_argument("--view-count", type=int, default=6)
    views.add_argument("--limit", type=int)
    views.add_argument("--concurrency", type=int, default=4)
    views.add_argument("--retries", type=int, default=4)
    views.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="low",
    )
    views.set_defaults(func=_generate_views)

    audit = subparsers.add_parser("audit-views", help="Create the manual C2 fidelity audit")
    audit.add_argument("--task", choices=("nli", "preference"), default="nli")
    audit.add_argument("--dataset", type=Path)
    audit.add_argument("--views", type=Path)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--items", type=int)
    audit.add_argument("--seed", type=int, default=42)
    audit.set_defaults(func=_audit_views)

    run = subparsers.add_parser("run", help="Run one experimental condition")
    run.add_argument("--task", choices=("nli", "preference"), default="nli")
    run.add_argument("--dataset", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--condition", choices=CONDITIONS, required=True)
    run.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=list(DEFAULT_MODEL_KEYS),
    )
    run.add_argument(
        "--dev-dataset",
        type=Path,
        default=Path("data/processed/chaos_mnli_dev_40.jsonl"),
    )
    run.add_argument("--views", type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--concurrency", type=int, default=6)
    run.add_argument("--retries", type=int, default=4)
    run.set_defaults(func=_run)

    analyze = subparsers.add_parser("analyze", help="Analyze a complete panel JSONL")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--condition", default="C0")
    analyze.add_argument("--bootstrap", type=int, default=10_000)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.set_defaults(func=_analyze)

    compare = subparsers.add_parser(
        "compare", help="Paired comparison with noise-matched control"
    )
    compare.add_argument("--baseline", type=Path, default=Path("results/c0.jsonl"))
    compare.add_argument("--intervention", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--bootstrap", type=int, default=10_000)
    compare.add_argument("--noise-simulations", type=int, default=1_000)
    compare.add_argument("--seed", type=int, default=42)
    compare.set_defaults(func=_compare)

    endpoints = subparsers.add_parser(
        "endpoints", help="Evaluate preregistered D1/D2 endpoints"
    )
    endpoints.add_argument("--d1-baseline", type=Path, default=Path("results/c0.jsonl"))
    endpoints.add_argument("--d1-c2", type=Path, default=Path("results/c2.jsonl"))
    endpoints.add_argument("--d1-c2b", type=Path, default=Path("results/c2b.jsonl"))
    endpoints.add_argument("--d2-baseline", type=Path, default=Path("results/d2_c0.jsonl"))
    endpoints.add_argument("--d2-c2", type=Path, default=Path("results/d2_c2.jsonl"))
    endpoints.add_argument("--d2-c2b", type=Path, default=Path("results/d2_c2b.jsonl"))
    endpoints.add_argument("--output", type=Path, default=Path("results/endpoints.json"))
    endpoints.add_argument("--bootstrap", type=int, default=10_000)
    endpoints.add_argument("--noise-simulations", type=int, default=1_000)
    endpoints.add_argument("--seed", type=int, default=42)
    endpoints.set_defaults(func=_endpoints)

    estimate = subparsers.add_parser("estimate-cost", help="Estimate one task run")
    estimate.add_argument("--task", choices=("nli", "preference"), default="nli")
    estimate.add_argument("--items", type=int)
    estimate.set_defaults(func=_estimate_cost)

    report = subparsers.add_parser("cost-report", help="Sum exact cost from API usage")
    report.add_argument("inputs", type=Path, nargs="+")
    report.set_defaults(func=_cost_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
