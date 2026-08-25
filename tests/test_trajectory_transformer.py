import numpy as np
import pandas as pd
import pytest

from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN
from ml.generate_synthetic_data import generate_dataset
from ml.split_data import split_dataset
from ml.trajectory_transformer import DEFAULT_WINDOW, TrajectoryModel, build_sequences

SMALL_N_MERCHANTS = 40
SMALL_N_WEEKS = 20
EXCLUDED_FROM_FEATURES = {"merchant_id", "week_start", LABEL_COLUMN, LATENT_STATE_COLUMN}


@pytest.fixture(scope="module")
def small_split():
    df = generate_dataset(seed=42, n_merchants=SMALL_N_MERCHANTS, n_weeks=SMALL_N_WEEKS)
    return split_dataset(df)


@pytest.fixture(scope="module")
def fitted_model(small_split):
    model = TrajectoryModel(seed=42)
    model.fit(small_split["train"], epochs=2, verbose=False)
    return model


def test_build_sequences_shapes_and_current_week_is_last_position(small_split):
    df = small_split["train"]
    week_vectors = np.arange(len(df) * 3, dtype=np.float32).reshape(len(df), 3)

    sequences, padding_mask = build_sequences(df, week_vectors, window=DEFAULT_WINDOW)

    assert sequences.shape == (len(df), DEFAULT_WINDOW, 3)
    assert padding_mask.shape == (len(df), DEFAULT_WINDOW)
    # The final position of every row's window is that row's own week.
    np.testing.assert_allclose(sequences[:, -1, :], week_vectors)
    # The current week is never padding.
    assert not padding_mask[:, -1].any()


def test_first_week_of_a_merchant_is_padded_but_not_dropped(small_split):
    df = small_split["train"].reset_index(drop=True)
    week_vectors = np.zeros((len(df), 2), dtype=np.float32)

    _, padding_mask = build_sequences(df, week_vectors, window=DEFAULT_WINDOW)

    first_rows = df.sort_values(["merchant_id", "week_start"]).groupby("merchant_id").head(1).index
    for row in first_rows:
        # Only the current-week position is real; everything earlier is padding.
        assert padding_mask[row, :-1].all()
        assert not padding_mask[row, -1]

    assert len(padding_mask) == len(df)  # no row dropped for lacking history


def test_context_supplies_history_for_split_boundary_rows(small_split):
    """The rows at the start of a split used to be padded even though their
    real prior weeks exist in the split immediately before."""
    test_df = small_split["test"].reset_index(drop=True)
    full = pd.concat([small_split["train"], small_split["validation"], small_split["test"]], ignore_index=True)

    vectors = np.arange(len(test_df) * 2, dtype=np.float32).reshape(len(test_df), 2)
    context_vectors = np.zeros((len(full), 2), dtype=np.float32)

    _, truncated_mask = build_sequences(test_df, vectors, window=DEFAULT_WINDOW)
    _, contextual_mask = build_sequences(
        test_df, vectors, window=DEFAULT_WINDOW, context_df=full, context_vectors=context_vectors
    )

    boundary_rows = test_df.sort_values(["merchant_id", "week_start"]).groupby("merchant_id").head(DEFAULT_WINDOW - 1).index

    assert truncated_mask[boundary_rows].any(), "fixture must actually contain truncated boundary rows"
    # Every merchant here has a full train+validation history behind it, so no
    # boundary row should need padding any more.
    assert not contextual_mask[boundary_rows].any()
    # Rows that already had a full window are unaffected.
    np.testing.assert_array_equal(
        truncated_mask[~np.isin(np.arange(len(test_df)), boundary_rows)],
        contextual_mask[~np.isin(np.arange(len(test_df)), boundary_rows)],
    )


def test_context_never_pulls_in_a_week_at_or_after_the_target_week(small_split):
    """Temporal discipline: context may only contribute strictly earlier weeks.
    A context row sharing the target's week (or later) must never be read."""
    test_df = small_split["test"].reset_index(drop=True)
    target_vectors = np.ones((len(test_df), 1), dtype=np.float32)

    # Poison every context row with a distinctive value. Any context row at or
    # after a target's week that leaked in would show up as a 9.0.
    context_df = test_df.copy()
    context_vectors = np.full((len(context_df), 1), 9.0, dtype=np.float32)

    sequences, padding_mask = build_sequences(
        test_df, target_vectors, window=DEFAULT_WINDOW,
        context_df=context_df, context_vectors=context_vectors,
    )

    # context_df duplicates test_df exactly, so every context week is a target
    # week: none of them is strictly earlier than itself, and the de-duplication
    # must hand the target row the win.
    assert not (sequences == 9.0).any()
    # Non-padding positions all came from the target frame.
    assert (sequences[~padding_mask] == 1.0).all()


def test_boundary_row_attention_is_not_forced_onto_the_current_week(fitted_model, small_split):
    """Input-side assertion only: the model now has real prior weeks available
    to attend over. It may still choose to weight the current week heavily --
    that is a learned outcome and is not asserted here."""
    test_df = small_split["test"].reset_index(drop=True)
    full = pd.concat([small_split["train"], small_split["validation"], small_split["test"]], ignore_index=True)

    boundary_rows = test_df.sort_values(["merchant_id", "week_start"]).groupby("merchant_id").head(1).index

    truncated = fitted_model.attention_by_week(test_df)
    contextual = fitted_model.attention_by_week(test_df, history_df=full)

    # Truncated: the very first test week of each merchant had nothing else to
    # attend to, so all attention lands on the current week by construction.
    np.testing.assert_allclose(truncated[boundary_rows, -1], 1.0, atol=1e-4)
    # With context, attention is spread over real weeks rather than pinned.
    assert (contextual[boundary_rows, -1] < 1.0 - 1e-4).all()
    assert (contextual[boundary_rows, :-1] > 0).any()


def test_predict_returns_probability_per_row(fitted_model, small_split):
    probabilities = fitted_model.predict(small_split["validation"])
    assert probabilities.shape == (len(small_split["validation"]),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_attention_weights_are_per_week_and_ignore_padding(fitted_model, small_split):
    df = small_split["validation"].reset_index(drop=True)
    attention = fitted_model.attention_by_week(df)

    assert attention.shape == (len(df), DEFAULT_WINDOW)
    assert (attention >= 0).all()

    # Rows with a full window of history should have attention summing to ~1;
    # padded rows have the padded positions zeroed out, so they sum to <= 1.
    row_sums = attention.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-4).all()
    assert (row_sums > 0).all()


def test_design_matrix_never_sees_label_or_latent_state(fitted_model, small_split):
    from ml.model_utils import design_matrix

    features = design_matrix(small_split["train"])
    assert EXCLUDED_FROM_FEATURES.isdisjoint(set(features.columns))


def test_predictions_are_deterministic_for_a_fitted_model(fitted_model, small_split):
    first = fitted_model.predict(small_split["validation"])
    second = fitted_model.predict(small_split["validation"])
    np.testing.assert_allclose(first, second)
