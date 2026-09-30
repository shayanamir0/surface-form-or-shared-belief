from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PanelSummary:
    items: int
    judges: int
    mean_phi: float
    kish_n_eff: float
    eigen_n_eff: float
    panel_accuracy: float
    best_individual_accuracy: float
    panel_lift: float
    within_openai_mean_phi: float | None = None
    within_anthropic_mean_phi: float | None = None
    cross_provider_mean_phi: float | None = None
    kish_ci_low: float | None = None
    kish_ci_high: float | None = None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class InvarianceSummary:
    items: int
    judges: int
    label_stability_rate: float
    unanimous_wrong_count: int
    unanimous_wrong_persistence: float | None
    unanimous_correct_count: int
    unanimous_correct_any_flip_rate: float | None
    unanimous_correct_panel_flip_rate: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def phi_matrix(errors: np.ndarray) -> np.ndarray:
    """Return judge-by-judge phi correlations from judge-by-item binary errors."""
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2 or errors.shape[0] < 2 or errors.shape[1] < 2:
        raise ValueError("errors must have shape (at least 2 judges, at least 2 items)")
    variances = errors.var(axis=1)
    if np.any(variances == 0):
        indices = np.flatnonzero(variances == 0).tolist()
        raise ValueError(f"Phi is undefined for constant error vectors at judge indices {indices}")
    return np.corrcoef(errors)


def effective_sample_sizes(correlation: np.ndarray) -> tuple[float, float, float]:
    correlation = np.asarray(correlation, dtype=float)
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be square")
    k = correlation.shape[0]
    if k < 2:
        raise ValueError("At least two judges are required")
    upper = correlation[np.triu_indices(k, k=1)]
    mean_phi = float(upper.mean())
    denominator = 1 + (k - 1) * mean_phi
    if denominator <= 0:
        raise ValueError("Kish denominator is non-positive; inspect error correlations")
    kish = float(k / denominator)
    largest_eigenvalue = float(np.linalg.eigvalsh(correlation).max())
    eigen = float(k / largest_eigenvalue)
    return mean_phi, kish, eigen


def deterministic_majority(labels: list[str], item_id: str) -> str:
    counts = Counter(labels)
    maximum = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == maximum)
    if len(winners) == 1:
        return winners[0]
    digest = hashlib.sha256(f"{item_id}|{'|'.join(labels)}".encode()).digest()
    return winners[int.from_bytes(digest[:8], "big") % len(winners)]


def records_to_matrices(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Convert complete long-form records into aligned labels and binary errors."""
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        judge = f"{record['model']}#{int(record.get('repeat', 0))}"
        item_id = str(record["item_id"])
        if judge in by_item[item_id]:
            raise ValueError(f"Duplicate judgment for item={item_id}, judge={judge}")
        by_item[item_id][judge] = record

    item_ids = sorted(by_item)
    judges = sorted({judge for item in by_item.values() for judge in item})
    missing = [
        (item_id, judge)
        for item_id in item_ids
        for judge in judges
        if judge not in by_item[item_id]
    ]
    if missing:
        raise ValueError(f"Incomplete panel: {len(missing)} missing judgments; first={missing[0]}")

    predictions = np.empty((len(judges), len(item_ids)), dtype=object)
    gold = np.empty(len(item_ids), dtype=object)
    for item_index, item_id in enumerate(item_ids):
        gold_values = {record["gold_label"] for record in by_item[item_id].values()}
        if len(gold_values) != 1:
            raise ValueError(f"Inconsistent gold labels for item={item_id}: {gold_values}")
        gold[item_index] = gold_values.pop()
        for judge_index, judge in enumerate(judges):
            predictions[judge_index, item_index] = by_item[item_id][judge]["prediction"]
    errors = predictions != gold[np.newaxis, :]
    return item_ids, judges, predictions, gold, errors.astype(int)


def bootstrap_kish_interval(
    errors: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    item_count = errors.shape[1]
    for _ in range(resamples):
        indices = rng.integers(0, item_count, size=item_count)
        sampled = errors[:, indices]
        try:
            _, kish, _ = effective_sample_sizes(phi_matrix(sampled))
        except ValueError:
            continue
        values.append(kish)
    if not values:
        raise ValueError("No valid bootstrap resamples")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _panel_accuracy(
    item_ids: list[str], predictions: np.ndarray, gold: np.ndarray
) -> float:
    panel_predictions = np.array(
        [
            deterministic_majority(predictions[:, index].tolist(), item_id)
            for index, item_id in enumerate(item_ids)
        ]
    )
    return float(np.mean(panel_predictions == gold))


def summarize_panel(
    records: list[dict[str, Any]],
    *,
    bootstrap_resamples: int = 0,
    seed: int = 42,
) -> PanelSummary:
    item_ids, judges, predictions, gold, errors = records_to_matrices(records)
    correlations = phi_matrix(errors)
    mean_phi, kish, eigen = effective_sample_sizes(correlations)
    providers = {
        f"{record['model']}#{int(record.get('repeat', 0))}": record.get("provider")
        for record in records
    }

    def group_phi(first: str, second: str) -> float | None:
        values = [
            correlations[left, right]
            for left in range(len(judges))
            for right in range(left + 1, len(judges))
            if {providers.get(judges[left]), providers.get(judges[right])}
            == {first, second}
            and (
                first != second
                or providers.get(judges[left]) == providers.get(judges[right]) == first
            )
        ]
        return float(np.mean(values)) if values else None

    individual_accuracy = 1 - errors.mean(axis=1)
    panel_accuracy = _panel_accuracy(item_ids, predictions, gold)
    best_accuracy = float(individual_accuracy.max())
    low: float | None = None
    high: float | None = None
    if bootstrap_resamples:
        low, high = bootstrap_kish_interval(
            errors, resamples=bootstrap_resamples, seed=seed
        )
    return PanelSummary(
        items=len(item_ids),
        judges=len(judges),
        mean_phi=mean_phi,
        kish_n_eff=kish,
        eigen_n_eff=eigen,
        panel_accuracy=panel_accuracy,
        best_individual_accuracy=best_accuracy,
        panel_lift=panel_accuracy - best_accuracy,
        within_openai_mean_phi=group_phi("openai", "openai"),
        within_anthropic_mean_phi=group_phi("anthropic", "anthropic"),
        cross_provider_mean_phi=group_phi("openai", "anthropic"),
        kish_ci_low=low,
        kish_ci_high=high,
    )


def compare_conditions(
    baseline_records: list[dict[str, Any]],
    intervention_records: list[dict[str, Any]],
    *,
    bootstrap_resamples: int = 10_000,
    noise_simulations: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired condition comparison plus an independently-noised baseline control."""
    base_ids, base_judges, base_predictions, base_gold, base_errors = records_to_matrices(
        baseline_records
    )
    int_ids, int_judges, int_predictions, int_gold, int_errors = records_to_matrices(
        intervention_records
    )
    if base_ids != int_ids or base_judges != int_judges or not np.array_equal(
        base_gold, int_gold
    ):
        raise ValueError("Conditions must contain the same items, judges, and gold labels")

    base_phi, base_kish, _ = effective_sample_sizes(phi_matrix(base_errors))
    int_phi, int_kish, _ = effective_sample_sizes(phi_matrix(int_errors))
    base_accuracy = _panel_accuracy(base_ids, base_predictions, base_gold)
    int_accuracy = _panel_accuracy(int_ids, int_predictions, int_gold)

    rng = np.random.default_rng(seed)
    delta_kish: list[float] = []
    delta_accuracy: list[float] = []
    for _ in range(bootstrap_resamples):
        indices = rng.integers(0, len(base_ids), size=len(base_ids))
        try:
            _, sampled_base_kish, _ = effective_sample_sizes(
                phi_matrix(base_errors[:, indices])
            )
            _, sampled_int_kish, _ = effective_sample_sizes(
                phi_matrix(int_errors[:, indices])
            )
        except ValueError:
            continue
        sampled_ids = [base_ids[index] for index in indices]
        delta_kish.append(sampled_int_kish - sampled_base_kish)
        delta_accuracy.append(
            _panel_accuracy(sampled_ids, int_predictions[:, indices], int_gold[indices])
            - _panel_accuracy(
                sampled_ids, base_predictions[:, indices], base_gold[indices]
            )
        )
    if not delta_kish:
        raise ValueError("No valid paired bootstrap resamples")

    result: dict[str, Any] = {
        "baseline_mean_phi": base_phi,
        "intervention_mean_phi": int_phi,
        "delta_mean_phi": int_phi - base_phi,
        "baseline_kish_n_eff": base_kish,
        "intervention_kish_n_eff": int_kish,
        "delta_kish_n_eff": int_kish - base_kish,
        "delta_kish_95_ci": np.quantile(delta_kish, [0.025, 0.975]).tolist(),
        "bootstrap_p_delta_kish_le_zero": float(np.mean(np.asarray(delta_kish) <= 0)),
        "baseline_panel_accuracy": base_accuracy,
        "intervention_panel_accuracy": int_accuracy,
        "delta_panel_accuracy": int_accuracy - base_accuracy,
        "delta_accuracy_95_ci": np.quantile(delta_accuracy, [0.025, 0.975]).tolist(),
    }
    if int_phi < base_phi:
        noise_control = _noise_matched_control(
            item_ids=base_ids,
            predictions=base_predictions,
            gold=base_gold,
            target_phi=int_phi,
            simulations=noise_simulations,
            seed=seed + 1,
        )
        result["noise_matched_control"] = noise_control
        result["e2_vs_noise_twin"] = _bootstrap_vs_noise_twin(
            item_ids=base_ids,
            baseline_predictions=base_predictions,
            intervention_predictions=int_predictions,
            gold=base_gold,
            labels=np.asarray(noise_control["labels"], dtype=object),
            flip_probability=float(noise_control["flip_probability"]),
            resamples=bootstrap_resamples,
            seed=seed + 2,
        )
    else:
        result["noise_matched_control"] = None
        result["e2_vs_noise_twin"] = None
    return result


def _label_space(predictions: np.ndarray, gold: np.ndarray) -> np.ndarray:
    labels = sorted({str(value) for value in predictions.flat} | {str(value) for value in gold})
    if len(labels) < 2:
        raise ValueError("Noise matching requires at least two labels")
    return np.asarray(labels, dtype=object)


def _draw_noisy_predictions(
    predictions: np.ndarray,
    *,
    labels: np.ndarray,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    label_to_index = {str(label): index for index, label in enumerate(labels)}
    encoded = np.vectorize(lambda value: label_to_index[str(value)])(predictions)
    changed = encoded.copy()
    mask = rng.random(changed.shape) < probability
    offsets = rng.integers(1, len(labels), size=changed.shape)
    changed[mask] = (changed[mask] + offsets[mask]) % len(labels)
    return labels[changed]


def _noise_matched_control(
    *,
    item_ids: list[str],
    predictions: np.ndarray,
    gold: np.ndarray,
    target_phi: float,
    simulations: int,
    seed: int,
) -> dict[str, float | list[float] | list[str]]:
    rng = np.random.default_rng(seed)
    labels = _label_space(predictions, gold)

    def draw(noise_probability: float) -> tuple[float, float]:
        noisy_predictions = _draw_noisy_predictions(
            predictions,
            labels=labels,
            probability=noise_probability,
            rng=rng,
        )
        noisy_errors = (noisy_predictions != gold[np.newaxis, :]).astype(int)
        mean_phi, _, _ = effective_sample_sizes(phi_matrix(noisy_errors))
        accuracy = _panel_accuracy(item_ids, noisy_predictions, gold)
        return mean_phi, accuracy

    low, high = 0.0, 0.5
    calibration_draws = max(20, min(100, simulations // 10))
    for _ in range(14):
        midpoint = (low + high) / 2
        phis = [draw(midpoint)[0] for _ in range(calibration_draws)]
        if float(np.mean(phis)) > target_phi:
            low = midpoint
        else:
            high = midpoint
    probability = (low + high) / 2
    draws = [draw(probability) for _ in range(simulations)]
    phis = np.array([entry[0] for entry in draws])
    accuracies = np.array([entry[1] for entry in draws])
    return {
        "labels": labels.tolist(),
        "flip_probability": probability,
        "achieved_mean_phi": float(phis.mean()),
        "achieved_mean_phi_95_interval": np.quantile(phis, [0.025, 0.975]).tolist(),
        "panel_accuracy_mean": float(accuracies.mean()),
        "panel_accuracy_95_interval": np.quantile(
            accuracies, [0.025, 0.975]
        ).tolist(),
    }


def _bootstrap_vs_noise_twin(
    *,
    item_ids: list[str],
    baseline_predictions: np.ndarray,
    intervention_predictions: np.ndarray,
    gold: np.ndarray,
    labels: np.ndarray,
    flip_probability: float,
    resamples: int,
    seed: int,
) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        indices = rng.integers(0, len(item_ids), size=len(item_ids))
        sampled_ids = [item_ids[index] for index in indices]
        noisy = _draw_noisy_predictions(
            baseline_predictions[:, indices],
            labels=labels,
            probability=flip_probability,
            rng=rng,
        )
        intervention_accuracy = _panel_accuracy(
            sampled_ids, intervention_predictions[:, indices], gold[indices]
        )
        noise_accuracy = _panel_accuracy(sampled_ids, noisy, gold[indices])
        deltas.append(intervention_accuracy - noise_accuracy)
    values = np.asarray(deltas)
    return {
        "delta_accuracy_mean": float(values.mean()),
        "delta_accuracy_95_ci": np.quantile(values, [0.025, 0.975]).tolist(),
        "one_sided_95_lower_bound": float(np.quantile(values, 0.05)),
        "bootstrap_p_intervention_not_better": float(np.mean(values <= 0)),
    }


def behavioral_invariance(
    baseline_records: list[dict[str, Any]],
    intervention_records: list[dict[str, Any]],
) -> InvarianceSummary:
    base_ids, base_judges, base_predictions, base_gold, base_errors = records_to_matrices(
        baseline_records
    )
    int_ids, int_judges, int_predictions, int_gold, int_errors = records_to_matrices(
        intervention_records
    )
    if base_ids != int_ids or base_judges != int_judges or not np.array_equal(
        base_gold, int_gold
    ):
        raise ValueError("Conditions must contain the same items, judges, and gold labels")

    stable = base_predictions == int_predictions
    unanimous_wrong = np.all(base_errors == 1, axis=0)
    unanimous_correct = np.all(base_errors == 0, axis=0)
    wrong_count = int(unanimous_wrong.sum())
    correct_count = int(unanimous_correct.sum())

    wrong_persistence = (
        float(np.mean(np.all(int_errors[:, unanimous_wrong] == 1, axis=0)))
        if wrong_count
        else None
    )
    correct_any_flip = (
        float(np.mean(np.any(~stable[:, unanimous_correct], axis=0))) if correct_count else None
    )
    if correct_count:
        base_panel = np.asarray(
            [
                deterministic_majority(
                    base_predictions[:, index].tolist(), base_ids[index]
                )
                for index in np.flatnonzero(unanimous_correct)
            ]
        )
        int_panel = np.asarray(
            [
                deterministic_majority(
                    int_predictions[:, index].tolist(), base_ids[index]
                )
                for index in np.flatnonzero(unanimous_correct)
            ]
        )
        correct_panel_flip = float(np.mean(base_panel != int_panel))
    else:
        correct_panel_flip = None

    return InvarianceSummary(
        items=len(base_ids),
        judges=len(base_judges),
        label_stability_rate=float(stable.mean()),
        unanimous_wrong_count=wrong_count,
        unanimous_wrong_persistence=wrong_persistence,
        unanimous_correct_count=correct_count,
        unanimous_correct_any_flip_rate=correct_any_flip,
        unanimous_correct_panel_flip_rate=correct_panel_flip,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in the original order."""
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[original_index]))
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted.tolist()


def evaluate_primary_endpoints(
    comparisons: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Evaluate E1/E2 separately per dataset with one Holm-corrected family."""
    tests: list[tuple[str, str, float]] = []
    for dataset, result in comparisons.items():
        c2 = result["c2_vs_c0"]
        tests.append((dataset, "e1", float(c2["bootstrap_p_delta_kish_le_zero"])))
        e2 = c2.get("e2_vs_noise_twin")
        if e2 is not None:
            tests.append(
                (
                    dataset,
                    "e2",
                    float(e2["bootstrap_p_intervention_not_better"]),
                )
            )

    adjusted = holm_adjust([entry[2] for entry in tests])
    adjusted_lookup = {
        (dataset, endpoint): value
        for (dataset, endpoint, _), value in zip(tests, adjusted, strict=True)
    }
    report: dict[str, Any] = {"holm_family": []}
    for dataset, result in comparisons.items():
        c2 = result["c2_vs_c0"]
        c2b = result["c2b_vs_c0"]
        e1_adjusted = adjusted_lookup[(dataset, "e1")]
        c2_ci = c2["delta_kish_95_ci"]
        c2b_ci = c2b["delta_kish_95_ci"]
        e2 = c2.get("e2_vs_noise_twin")
        e2_adjusted = adjusted_lookup.get((dataset, "e2"))
        report[dataset] = {
            "e1_holm_p": e1_adjusted,
            "e1_c2b_no_material_gain": bool(
                c2b_ci[0] <= 0 <= c2b_ci[1] and abs(c2b["delta_kish_n_eff"]) < 0.5
            ),
            "e1_supports_surface_form": bool(
                e1_adjusted < 0.05
                and c2_ci[0] > 0
                and c2["delta_kish_n_eff"] >= 0.5
                and c2b_ci[0] <= 0 <= c2b_ci[1]
                and abs(c2b["delta_kish_n_eff"]) < 0.5
            ),
            "e2_holm_p": e2_adjusted,
            "e2_supports_useful_diversity": bool(
                e2 is not None
                and e2_adjusted is not None
                and e2_adjusted < 0.05
                and e2["one_sided_95_lower_bound"] > 0
            ),
        }
    report["holm_family"] = [
        {
            "dataset": dataset,
            "endpoint": endpoint,
            "raw_p": raw,
            "adjusted_p": adjusted_lookup[(dataset, endpoint)],
        }
        for dataset, endpoint, raw in tests
    ]
    return report
