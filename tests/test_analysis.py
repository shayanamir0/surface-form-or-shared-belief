import numpy as np
import pytest

from behavioural_test.analysis import (
    behavioral_invariance,
    compare_conditions,
    deterministic_majority,
    effective_sample_sizes,
    evaluate_primary_endpoints,
    holm_adjust,
    phi_matrix,
    summarize_panel,
)


def test_effective_sample_size_independent_identity_matrix() -> None:
    mean_phi, kish, eigen = effective_sample_sizes(np.eye(3))
    assert mean_phi == pytest.approx(0.0)
    assert kish == pytest.approx(3.0)
    assert eigen == pytest.approx(3.0)


def test_effective_sample_size_exchangeable_correlation() -> None:
    correlation = np.full((4, 4), 0.5)
    np.fill_diagonal(correlation, 1.0)
    mean_phi, kish, eigen = effective_sample_sizes(correlation)
    assert mean_phi == pytest.approx(0.5)
    assert kish == pytest.approx(1.6)
    assert eigen == pytest.approx(1.6)


def test_phi_rejects_constant_error_vector() -> None:
    with pytest.raises(ValueError, match="constant error vectors"):
        phi_matrix(np.array([[0, 0, 0], [0, 1, 0]]))


def test_majority_tie_breaking_is_deterministic() -> None:
    labels = ["entailment", "neutral", "contradiction"]
    assert deterministic_majority(labels, "item-1") == deterministic_majority(
        labels, "item-1"
    )


def test_panel_summary() -> None:
    gold = ["entailment", "neutral", "contradiction", "entailment", "neutral", "contradiction"]
    predictions = {
        "model-a": [
            "entailment",
            "neutral",
            "neutral",
            "entailment",
            "contradiction",
            "contradiction",
        ],
        "model-b": ["neutral", "neutral", "contradiction", "entailment", "neutral", "entailment"],
        "model-c": [
            "entailment",
            "neutral",
            "contradiction",
            "neutral",
            "contradiction",
            "contradiction",
        ],
    }
    records = [
        {
            "item_id": f"item-{item_index}",
            "model": model,
            "repeat": 0,
            "condition": "C0",
            "prediction": prediction,
            "gold_label": gold[item_index],
        }
        for model, model_predictions in predictions.items()
        for item_index, prediction in enumerate(model_predictions)
    ]
    summary = summarize_panel(records)
    assert summary.items == 6
    assert summary.judges == 3
    assert 0 <= summary.panel_accuracy <= 1
    assert summary.best_individual_accuracy == pytest.approx(4 / 6)


def test_paired_comparison_builds_noise_matched_control() -> None:
    rng = np.random.default_rng(7)
    item_count = 120
    latent = rng.random(item_count) < 0.3
    baseline_errors = np.vstack(
        [np.logical_xor(latent, rng.random(item_count) < 0.05) for _ in range(4)]
    )
    intervention_errors = rng.random((4, item_count)) < 0.3
    models = ["oa-1", "oa-2", "an-1", "an-2"]

    def records(errors: np.ndarray, condition: str) -> list[dict[str, object]]:
        return [
            {
                "item_id": f"item-{item}",
                "model": model,
                "provider": "openai" if judge < 2 else "anthropic",
                "repeat": 0,
                "condition": condition,
                "prediction": "neutral" if errors[judge, item] else "entailment",
                "gold_label": "entailment",
            }
            for judge, model in enumerate(models)
            for item in range(item_count)
        ]

    result = compare_conditions(
        records(baseline_errors, "C0"),
        records(intervention_errors, "C2"),
        bootstrap_resamples=100,
        noise_simulations=50,
    )
    assert result["delta_kish_n_eff"] > 0
    assert result["noise_matched_control"] is not None
    assert result["noise_matched_control"]["labels"] == ["entailment", "neutral"]
    assert result["e2_vs_noise_twin"] is not None


def test_behavioral_invariance_tracks_shared_failures_and_stability() -> None:
    models = ["a", "b", "c"]
    gold = ["A", "A", "B", "B"]
    baseline = {
        "a": ["B", "A", "B", "B"],
        "b": ["B", "A", "B", "B"],
        "c": ["B", "A", "B", "B"],
    }
    intervention = {
        "a": ["B", "B", "B", "B"],
        "b": ["B", "A", "B", "A"],
        "c": ["B", "A", "B", "B"],
    }

    def records(predictions: dict[str, list[str]], condition: str) -> list[dict[str, object]]:
        return [
            {
                "item_id": f"item-{item}",
                "model": model,
                "repeat": 0,
                "condition": condition,
                "prediction": prediction,
                "gold_label": gold[item],
            }
            for model in models
            for item, prediction in enumerate(predictions[model])
        ]

    result = behavioral_invariance(
        records(baseline, "C0"),
        records(intervention, "C2"),
    )
    assert result.unanimous_wrong_count == 1
    assert result.unanimous_wrong_persistence == 1.0
    assert result.unanimous_correct_count == 3
    assert result.unanimous_correct_any_flip_rate == pytest.approx(2 / 3)
    assert result.label_stability_rate == pytest.approx(10 / 12)


def test_holm_adjustment_and_separate_endpoint_reporting() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    c2 = {
        "bootstrap_p_delta_kish_le_zero": 0.001,
        "delta_kish_n_eff": 0.6,
        "delta_kish_95_ci": [0.2, 0.9],
        "e2_vs_noise_twin": {
            "bootstrap_p_intervention_not_better": 0.001,
            "one_sided_95_lower_bound": 0.01,
        },
    }
    c2b = {"delta_kish_n_eff": 0.0, "delta_kish_95_ci": [-0.1, 0.1]}
    report = evaluate_primary_endpoints(
        {
            "d1": {"c2_vs_c0": c2, "c2b_vs_c0": c2b},
            "d2": {"c2_vs_c0": c2, "c2b_vs_c0": c2b},
        }
    )
    assert report["d1"]["e1_supports_surface_form"]
    assert report["d2"]["e2_supports_useful_diversity"]
    assert {entry["dataset"] for entry in report["holm_family"]} == {"d1", "d2"}
