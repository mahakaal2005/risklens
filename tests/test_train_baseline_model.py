import numpy as np
import pandas as pd

from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN
from ml.model_utils import ML_FEATURE_COLUMNS
from ml.split_data import split_dataset
from ml.train_baseline_model import _design_matrix, select_threshold, train

EXCLUDED_FROM_FEATURES = {"merchant_id", "week_start", LABEL_COLUMN, LATENT_STATE_COLUMN}


def test_design_matrix_excludes_target_latent_id_and_date_fields(tmp_path):
    from ml.generate_synthetic_data import generate_dataset

    df = generate_dataset(seed=42, n_merchants=60, n_weeks=30)
    X = _design_matrix(df)
    assert EXCLUDED_FROM_FEATURES.isdisjoint(set(X.columns))
    assert set(X.columns) == set(ML_FEATURE_COLUMNS)


def test_saved_feature_list_contains_only_approved_fields(tmp_path):
    from ml.generate_synthetic_data import generate_dataset

    df = generate_dataset(seed=42, n_merchants=220, n_weeks=52)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    artifact_dir = tmp_path / "artifacts"

    result = train(csv_path=csv_path, seed=42, artifact_dir=artifact_dir)
    feature_list = result["metadata"]["feature_columns"]

    assert set(feature_list) == set(ML_FEATURE_COLUMNS)
    assert EXCLUDED_FROM_FEATURES.isdisjoint(set(feature_list))


def test_model_artifact_metadata_contains_required_fields(tmp_path):
    from ml.generate_synthetic_data import generate_dataset

    df = generate_dataset(seed=42, n_merchants=220, n_weeks=52)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    artifact_dir = tmp_path / "artifacts"

    result = train(csv_path=csv_path, seed=42, artifact_dir=artifact_dir)
    metadata = result["metadata"]

    for key in ["seed", "feature_columns", "training_row_count", "selected_threshold", "split_info"]:
        assert key in metadata
    assert metadata["split_info"]["train"]["week_start_max"] < metadata["split_info"]["validation"]["week_start_min"]
    assert metadata["split_info"]["validation"]["week_start_max"] < metadata["split_info"]["test"]["week_start_min"]


def test_preprocessing_is_fit_only_on_training_data(tmp_path):
    from ml.generate_synthetic_data import generate_dataset

    df = generate_dataset(seed=42, n_merchants=220, n_weeks=52)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    artifact_dir = tmp_path / "artifacts"

    result = train(csv_path=csv_path, seed=42, artifact_dir=artifact_dir)
    pipeline = result["pipeline"]
    train_df = result["split"]["train"]

    scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    train_features = _design_matrix(train_df)

    numeric_cols = [c for c in train_features.columns if c not in ("merchant_category", "previous_review_outcome")]
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    imputed = imputer.transform(train_features[numeric_cols])

    np.testing.assert_allclose(scaler.mean_, np.nanmean(imputed, axis=0), rtol=1e-2, atol=1e-2)


def test_threshold_selection_uses_validation_data_only():
    y_val = np.array([0] * 90 + [1] * 10)
    rng = np.random.default_rng(0)
    val_probs = np.concatenate([rng.uniform(0, 0.3, 90), rng.uniform(0.4, 0.9, 10)])

    result = select_threshold(y_val, val_probs)
    assert 0.05 <= result["selected_threshold"] <= 0.95
    assert "candidates" in result
    assert all("threshold" in c for c in result["candidates"])


def test_held_out_data_cannot_alter_selected_threshold(tmp_path):
    from ml.generate_synthetic_data import generate_dataset

    df = generate_dataset(seed=42, n_merchants=220, n_weeks=52)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    artifact_dir_a = tmp_path / "artifacts_a"
    result_a = train(csv_path=csv_path, seed=42, artifact_dir=artifact_dir_a)

    # Corrupt the test split's labels in a copy of the CSV; the selected
    # threshold must be identical, since select_threshold() never receives
    # the test split at all.
    corrupted = df.copy()
    split = split_dataset(corrupted)
    test_weeks = set(split["test"]["week_start"].unique())
    corrupted.loc[corrupted["week_start"].isin(test_weeks), LABEL_COLUMN] = 1
    corrupted_csv = tmp_path / "corrupted.csv"
    corrupted.to_csv(corrupted_csv, index=False)

    artifact_dir_b = tmp_path / "artifacts_b"
    result_b = train(csv_path=corrupted_csv, seed=42, artifact_dir=artifact_dir_b)

    assert result_a["threshold_result"]["selected_threshold"] == result_b["threshold_result"]["selected_threshold"]
