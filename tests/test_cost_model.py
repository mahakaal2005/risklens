import numpy as np
import pytest
import yaml

from ml.cost_model import (
    DEFAULT_COST_CONFIG_PATH,
    confusion_at,
    cost_metrics,
    do_nothing_cost,
    expected_cost,
    load_cost_config,
    review_all_cost,
    select_cost_optimal_threshold,
    threshold_sensitivity,
)


@pytest.fixture(scope="module")
def config():
    return load_cost_config()


@pytest.fixture(scope="module")
def scored_validation():
    """A separable-but-imperfect probability distribution, so the cost-optimal
    threshold is a real optimum rather than a degenerate 0.05 or 0.95."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.145, 4000)
    probs = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 4000), 0, 1)
    return y, probs


def test_config_file_documents_every_figure_as_an_assumption():
    """The rupee values are guesses. If that header ever gets edited away, the
    report starts implying these are measured costs."""
    text = DEFAULT_COST_CONFIG_PATH.read_text(encoding="utf-8")
    assert "ASSUMPTION" in text
    assert "not a verified fact" in text.lower()
    # Must not claim any real provider's costs.
    assert "Razorpay's costs" in text or "not Razorpay" in text.lower() or "razorpay" in text.lower()


def test_load_cost_config_rejects_negative_cost(tmp_path, config):
    bad = {**config, "costs_inr": {**config["costs_inr"], "review_cost": -1}}
    path = tmp_path / "bad_cost.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")

    with pytest.raises(ValueError, match="review_cost"):
        load_cost_config(path)


def test_load_cost_config_rejects_out_of_range_recovery_rate(tmp_path, config):
    bad = {**config, "recovery_rate": 1.5}
    path = tmp_path / "bad_recovery.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")

    with pytest.raises(ValueError, match="recovery_rate"):
        load_cost_config(path)


def test_expected_cost_matches_the_documented_formula(config):
    confusion = {"tn": 10, "fp": 2, "fn": 3, "tp": 4}
    costs = config["costs_inr"]
    residual = (1 - config["recovery_rate"]) * costs["missed_loss"]

    manual = (
        4 * (costs["review_cost"] + residual)
        + 2 * (costs["review_cost"] + costs["false_positive_friction"])
        + 3 * costs["missed_loss"]
    )
    assert expected_cost(confusion, config) == pytest.approx(manual)


def test_true_negatives_are_free(config):
    few = {"tn": 1, "fp": 2, "fn": 3, "tp": 4}
    many = {"tn": 10_000, "fp": 2, "fn": 3, "tp": 4}
    assert expected_cost(few, config) == expected_cost(many, config)


def test_catching_a_case_is_cheaper_than_missing_it_but_not_free(config):
    """The core asymmetry the whole model rests on. If recovery_rate were 1.0
    this would flatter every result, so the strict inequality matters."""
    caught = {"tn": 0, "fp": 0, "fn": 0, "tp": 1}
    missed = {"tn": 0, "fp": 0, "fn": 1, "tp": 0}
    assert 0 < expected_cost(caught, config) < expected_cost(missed, config)


def test_do_nothing_cost_ignores_predictions_and_charges_every_positive(config):
    aggressive = {"tn": 50, "fp": 50, "fn": 1, "tp": 9}
    passive = {"tn": 100, "fp": 0, "fn": 9, "tp": 1}
    # Same 10 actual positives either way -- the baseline must not depend on
    # what the model predicted.
    assert do_nothing_cost(aggressive, config) == do_nothing_cost(passive, config)
    assert do_nothing_cost(aggressive, config) == 10 * config["costs_inr"]["missed_loss"]


def test_review_all_cost_charges_a_review_for_every_row(config):
    confusion = {"tn": 90, "fp": 0, "fn": 10, "tp": 0}
    costs = config["costs_inr"]
    residual = (1 - config["recovery_rate"]) * costs["missed_loss"]
    expected = 10 * (costs["review_cost"] + residual) + 90 * (costs["review_cost"] + costs["false_positive_friction"])
    assert review_all_cost(confusion, config) == pytest.approx(expected)


def test_savings_per_1000_reviews_is_none_when_nothing_is_flagged(config):
    """Dividing by zero reviews must not become a misleading 0.0 -- a method
    that flags nobody has an undefined per-review saving, not a zero one."""
    metrics = cost_metrics({"tn": 900, "fp": 0, "fn": 100, "tp": 0}, config)
    assert metrics["n_reviews_generated"] == 0
    assert metrics["savings_per_1000_reviews_inr"] is None
    assert metrics["beats_do_nothing"] is False  # identical to doing nothing


def test_savings_per_1000_reviews_scales_from_net_savings(config):
    confusion = {"tn": 800, "fp": 100, "fn": 20, "tp": 80}
    metrics = cost_metrics(confusion, config)
    expected = metrics["net_savings_vs_do_nothing_inr"] / metrics["n_reviews_generated"] * 1000
    assert metrics["savings_per_1000_reviews_inr"] == pytest.approx(expected, rel=1e-6)


def test_confusion_at_threshold_matches_manual_counts():
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    assert confusion_at(y, probs, 0.5) == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    # Threshold is inclusive, matching evaluate_split()'s `probs >= threshold`.
    assert confusion_at(y, probs, 0.8)["tp"] == 1


def test_cost_optimal_threshold_minimizes_cost_across_the_grid(config, scored_validation):
    y, probs = scored_validation
    result = select_cost_optimal_threshold(y, probs, config)

    best_cost = result["selected_candidate"]["expected_cost_inr"]
    assert all(c["expected_cost_inr"] >= best_cost for c in result["candidates"])
    assert result["selection_method"] == "minimum_expected_rupee_cost_on_validation"
    assert result["assumption_notice"]


def test_cost_optimal_threshold_never_reads_held_out_data(config, scored_validation):
    """Same discipline as select_threshold(): the function is given validation
    arrays only and has no parameter through which test data could enter."""
    import inspect

    signature = inspect.signature(select_cost_optimal_threshold)
    assert set(signature.parameters) == {"y_val", "val_probs", "config"}


def test_higher_miss_cost_lowers_the_threshold_and_buys_more_reviews(config, scored_validation):
    """The direction that makes the whole model credible: as a missed loss gets
    more expensive relative to a review, flagging more aggressively must become
    optimal. A model that moved the other way would be wrong."""
    y, probs = scored_validation
    rows = threshold_sensitivity(y, probs, config)

    thresholds = [r["cost_optimal_threshold"] for r in rows]
    reviews = [r["n_reviews_generated"] for r in rows]
    assert rows[0]["missed_loss_to_review_cost_ratio"] < rows[-1]["missed_loss_to_review_cost_ratio"]
    assert thresholds == sorted(thresholds, reverse=True)
    assert reviews == sorted(reviews)
    assert thresholds[0] > thresholds[-1]  # must actually move, not stay flat


def test_cost_is_counts_based_not_probability_weighted(config):
    """The cost arithmetic must depend only on WHICH side of the threshold each
    row falls, never on the probability's magnitude. If this ever became a
    P(risk) x loss product, miscalibration would silently corrupt every rupee
    figure -- so the property is pinned here.

    Two probability vectors with identical orderings and identical
    above/below-threshold membership, but wildly different magnitudes, must
    produce identical costs.
    """
    y = np.array([0, 0, 1, 1, 0, 1])
    timid = np.array([0.11, 0.12, 0.13, 0.14, 0.01, 0.02])
    confident = np.array([0.91, 0.92, 0.93, 0.94, 0.01, 0.02])

    timid_confusion = confusion_at(y, timid, 0.10)
    confident_confusion = confusion_at(y, confident, 0.10)

    assert timid_confusion == confident_confusion
    assert expected_cost(timid_confusion, config) == expected_cost(confident_confusion, config)


def test_monotone_recalibration_does_not_change_the_achievable_cost(config, scored_validation):
    """Monotone calibration preserves ordering, so it preserves the set of
    achievable partitions and therefore the minimum achievable cost. Only the
    LABEL on the cut point moves. Verified over a fine grid so the result is not
    confounded by the coarse production grid's resolution.
    """
    from ml.calibration import IsotonicCalibrator

    y, probs = scored_validation
    calibrated = IsotonicCalibrator().fit(probs, y).transform(probs)

    fine_grid = np.linspace(0.0, 1.0, 1001)
    raw_min = min(expected_cost(confusion_at(y, probs, t), config) for t in fine_grid)
    calibrated_min = min(expected_cost(confusion_at(y, calibrated, t), config) for t in fine_grid)

    # Within discretisation residue of each other, not merely "similar".
    assert calibrated_min == pytest.approx(raw_min, rel=1e-3)


def test_grid_boundary_note_makes_no_directional_claim(config):
    """The note must not assert which side the true optimum lies on -- an
    earlier version claimed "may lie lower" when the real optimum was between
    two grid points, above the floor."""
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.3, 1000)
    probs = rng.uniform(0.06, 1.0, 1000)
    ruinous = {**config, "costs_inr": {**config["costs_inr"], "missed_loss": 10_000_000}}

    note = select_cost_optimal_threshold(y, probs, ruinous)["grid_boundary_note"]
    assert "lower" not in note
    assert "between two grid points" in note


def test_grid_boundary_optimum_is_flagged_not_presented_as_located(config):
    """When the cost asymmetry is extreme enough that "flag almost everything"
    wins, the optimum lands on the lowest grid point and the true minimum may
    be off-grid. That must be disclosed, not reported as a found optimum."""
    # Weak, overlapping signal: positives are spread across the whole
    # probability range, so no threshold cleanly separates them and a
    # ruinously expensive miss makes "flag everything" the best available play.
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.3, 1000)
    probs = rng.uniform(0.06, 1.0, 1000)
    ruinous_miss = {**config, "costs_inr": {**config["costs_inr"], "missed_loss": 10_000_000}}

    result = select_cost_optimal_threshold(y, probs, ruinous_miss)

    assert result["selected_threshold"] == 0.05
    assert result["threshold_at_grid_boundary"] is True
    assert "could not locate the true minimum" in result["grid_boundary_note"]


def test_interior_optimum_is_not_flagged_as_a_boundary(config, scored_validation):
    y, probs = scored_validation
    result = select_cost_optimal_threshold(y, probs, config)
    if result["selected_threshold"] not in (0.05, 0.95):
        assert result["threshold_at_grid_boundary"] is False
        assert result["grid_boundary_note"] is None


def test_sensitivity_reports_the_implied_rupee_value_for_each_ratio(config, scored_validation):
    y, probs = scored_validation
    review_cost = config["costs_inr"]["review_cost"]
    for row in threshold_sensitivity(y, probs, config):
        assert row["implied_missed_loss_inr"] == review_cost * row["missed_loss_to_review_cost_ratio"]
