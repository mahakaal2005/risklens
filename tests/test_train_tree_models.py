import pytest

from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN
from ml.generate_synthetic_data import generate_dataset
from ml.model_utils import ML_FEATURE_COLUMNS
from ml.train_tree_models import TREE_MODELS, _design_matrix, train_all, train_one

EXCLUDED_FROM_FEATURES = {"merchant_id", "week_start", LABEL_COLUMN, LATENT_STATE_COLUMN}


@pytest.fixture(scope="module")
def small_csv(tmp_path_factory):
    df = generate_dataset(seed=42, n_merchants=220, n_weeks=52)
    csv_path = tmp_path_factory.mktemp("data") / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.parametrize("model_key", list(TREE_MODELS.keys()))
def test_each_tree_model_trains_and_saves_valid_metadata(small_csv, tmp_path, model_key):
    artifact_dir = tmp_path / f"artifacts_{model_key}"
    result = train_one(model_key, csv_path=small_csv, seed=42, artifact_dir=artifact_dir)
    metadata = result["metadata"]

    assert set(metadata["feature_columns"]) == set(ML_FEATURE_COLUMNS)
    assert EXCLUDED_FROM_FEATURES.isdisjoint(set(metadata["feature_columns"]))
    assert metadata["role"] == "evaluation_comparison_only"
    for key in ["seed", "training_row_count", "selected_threshold", "split_info"]:
        assert key in metadata

    artifact_path = artifact_dir / f"{TREE_MODELS[model_key]['artifact_stem']}_v{TREE_MODELS[model_key]['model_version']}.joblib"
    assert artifact_path.exists()


def test_design_matrix_excludes_target_latent_id_and_date_fields(small_csv):
    import pandas as pd

    df = pd.read_csv(small_csv)
    X = _design_matrix(df)
    assert EXCLUDED_FROM_FEATURES.isdisjoint(set(X.columns))
    assert set(X.columns) == set(ML_FEATURE_COLUMNS)


def test_train_all_produces_both_models(small_csv, tmp_path):
    artifact_dir = tmp_path / "artifacts_all"
    results = train_all(csv_path=small_csv, seed=42, artifact_dir=artifact_dir)
    assert set(results.keys()) == set(TREE_MODELS.keys())
    for model_key, result in results.items():
        assert result["pipeline"] is not None
        assert 0.05 <= result["threshold_result"]["selected_threshold"] <= 0.95
