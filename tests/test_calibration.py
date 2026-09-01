import numpy as np
import pytest

from ml.calibration import (
    DEFAULT_N_BINS,
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    calibrate_and_evaluate,
    calibration_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
)


@pytest.fixture(scope="module")
def overconfident_split():
    """A model that ranks reasonably but states probabilities far above the
    true rate -- the exact failure calibration exists to catch, and the one a
    good PR-AUC would hide."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 4000)
    probs = np.clip(y * 0.35 + rng.normal(0.45, 0.12, 4000), 0.001, 0.999)
    return y[:2000], probs[:2000], y[2000:], probs[2000:]


def test_brier_score_is_zero_for_perfect_confident_predictions():
    y = np.array([0, 1, 1, 0])
    assert brier_score(y, y.astype(float)) == 0.0


def test_brier_score_is_one_for_confidently_wrong_predictions():
    y = np.array([0, 1])
    assert brier_score(y, 1.0 - y.astype(float)) == 1.0


def test_brier_score_punishes_overconfidence_that_pr_auc_would_not():
    """Both arrays rank identically -- same ordering, same PR-AUC -- but one
    states honest probabilities and one is overconfident. Brier must separate
    them; that is the whole reason it is reported."""
    y = np.array([0, 0, 1, 1])
    honest = np.array([0.3, 0.4, 0.6, 0.7])
    overconfident = np.array([0.01, 0.02, 0.98, 0.99])
    # Here the confident model is actually right, so it wins:
    assert brier_score(y, overconfident) < brier_score(y, honest)

    # But when the ordering is right and the confidence is not warranted:
    y_noisy = np.array([0, 1, 0, 1])
    assert brier_score(y_noisy, overconfident) > brier_score(y_noisy, honest)


def test_reliability_curve_covers_the_unit_interval_without_gaps():
    y = np.array([0, 1] * 50)
    probs = np.linspace(0.0, 1.0, 100)
    curve = reliability_curve(y, probs, n_bins=DEFAULT_N_BINS)

    assert len(curve) == DEFAULT_N_BINS
    assert curve[0]["bin_lower"] == 0.0
    assert curve[-1]["bin_upper"] == 1.0
    # Every row lands in exactly one bin -- nothing lost, nothing double-counted.
    assert sum(entry["count"] for entry in curve) == len(y)


def test_probability_of_exactly_one_lands_in_the_final_bin():
    """np.digitize's right-open edges would otherwise push p == 1.0 into an
    out-of-range bin and silently drop the model's most confident prediction."""
    y = np.array([1])
    curve = reliability_curve(y, np.array([1.0]), n_bins=DEFAULT_N_BINS)
    assert curve[-1]["count"] == 1
    assert sum(entry["count"] for entry in curve) == 1


def test_empty_bins_report_null_not_zero():
    """A bin with no predictions has an undefined observed rate. Reporting 0.0
    would read as "the model scored rows here and none were positive"."""
    y = np.array([0, 1])
    curve = reliability_curve(y, np.array([0.05, 0.15]), n_bins=DEFAULT_N_BINS)
    empty = [entry for entry in curve if entry["count"] == 0]
    assert empty
    assert all(entry["observed_positive_rate"] is None for entry in empty)
    assert all(entry["mean_predicted_probability"] is None for entry in empty)


def test_expected_calibration_error_is_zero_for_a_perfectly_calibrated_model():
    # Within each bin the predicted probability equals the observed rate.
    y = np.array([0] * 70 + [1] * 30)
    probs = np.full(100, 0.3)
    assert expected_calibration_error(y, probs) == pytest.approx(0.0, abs=1e-9)


def test_calibration_errors_are_large_for_a_confidently_wrong_model():
    y = np.zeros(100, dtype=int)
    probs = np.full(100, 0.95)
    assert expected_calibration_error(y, probs) == pytest.approx(0.95, abs=1e-9)
    assert maximum_calibration_error(y, probs) == pytest.approx(0.95, abs=1e-9)


def test_maximum_calibration_error_exposes_a_bad_sparse_bin_that_ece_hides():
    """ECE weights by bin count, so a small badly-calibrated high-probability
    bin barely moves it -- but that bin is exactly where escalations come from.
    MCE is reported to make that visible."""
    # 990 well-calibrated rows, plus 10 confidently-wrong high-probability rows.
    y = np.concatenate([np.zeros(990, dtype=int), np.zeros(10, dtype=int)])
    probs = np.concatenate([np.full(990, 0.05), np.full(10, 0.95)])

    ece = expected_calibration_error(y, probs)
    mce = maximum_calibration_error(y, probs)
    assert ece < 0.06  # barely registers
    assert mce > 0.9  # impossible to miss


def test_platt_calibration_corrects_a_systematically_overconfident_model(overconfident_split):
    y_val, val_probs, y_test, test_probs = overconfident_split
    calibrator = PlattCalibrator().fit(val_probs, y_val)
    calibrated = calibrator.transform(test_probs)

    assert brier_score(y_test, calibrated) < brier_score(y_test, test_probs)
    # The mean predicted probability should move toward the true base rate.
    assert abs(calibrated.mean() - y_test.mean()) < abs(test_probs.mean() - y_test.mean())


def test_isotonic_calibration_corrects_a_systematically_overconfident_model(overconfident_split):
    y_val, val_probs, y_test, test_probs = overconfident_split
    calibrated = IsotonicCalibrator().fit(val_probs, y_val).transform(test_probs)

    assert brier_score(y_test, calibrated) < brier_score(y_test, test_probs)
    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()


def test_calibrators_preserve_ranking(overconfident_split):
    """Both calibrators are monotonic by construction, so calibration must not
    change which merchant-week is riskier than which -- it changes only the
    stated number. If ranking moved, PR-AUC and the threshold would silently
    stop meaning what they meant."""
    y_val, val_probs, _, test_probs = overconfident_split
    for calibrator_class in (PlattCalibrator, IsotonicCalibrator):
        calibrated = calibrator_class().fit(val_probs, y_val).transform(test_probs)
        order_before = np.argsort(test_probs, kind="stable")
        # A monotonic map can create ties but must never invert an ordering.
        assert np.all(np.diff(calibrated[order_before]) >= -1e-9)


def test_calibrate_and_evaluate_never_fits_on_test_data(overconfident_split):
    """Fitting on test would make the calibration metrics meaningless. Verified
    behaviourally: calibrating against a corrupted test label vector must not
    change the fitted mapping, because test labels are never used for fitting."""
    y_val, val_probs, y_test, test_probs = overconfident_split

    honest = calibrate_and_evaluate(y_val, val_probs, y_test, test_probs)
    flipped = calibrate_and_evaluate(y_val, val_probs, 1 - y_test, test_probs)

    for variant in ("platt", "isotonic"):
        np.testing.assert_allclose(
            honest["calibrated_test_probabilities"][variant],
            flipped["calibrated_test_probabilities"][variant],
        )


def test_calibrate_and_evaluate_reports_all_three_variants_and_a_note(overconfident_split):
    y_val, val_probs, y_test, test_probs = overconfident_split
    result = calibrate_and_evaluate(y_val, val_probs, y_test, test_probs)

    assert set(result["metrics"]) == {"raw", "platt", "isotonic"}
    assert result["best_by_brier_score"] in {"raw", "platt", "isotonic"}
    assert result["fit_on"] == "validation_split"
    # The double-use-of-validation caveat must travel with the numbers.
    assert "double duty" in result["methodology_note"]


def test_best_by_brier_score_is_chosen_strictly_by_brier(overconfident_split):
    y_val, val_probs, y_test, test_probs = overconfident_split
    result = calibrate_and_evaluate(y_val, val_probs, y_test, test_probs)

    best = result["best_by_brier_score"]
    best_score = result["metrics"][best]["brier_score"]
    assert all(m["brier_score"] >= best_score for m in result["metrics"].values())


def test_calibration_metrics_reports_base_rate_alongside_mean_prediction():
    """The single most readable calibration check is "what does it say on
    average" vs "what actually happens" -- both must be present."""
    y = np.array([0] * 80 + [1] * 20)
    metrics = calibration_metrics(y, np.full(100, 0.5))
    assert metrics["observed_positive_rate"] == pytest.approx(0.2)
    assert metrics["mean_predicted_probability"] == pytest.approx(0.5)
    assert len(metrics["reliability_curve"]) == DEFAULT_N_BINS
