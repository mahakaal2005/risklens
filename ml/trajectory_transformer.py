"""Merchant risk trajectory model: a small Transformer encoder that reads the
last N weeks of a merchant's features instead of a single merchant-week row.

Motivation: a merchant whose refund rate crept up for six straight weeks is a
different risk than one with an identical current refund rate that spiked once.
Every other model in this project (Logistic Regression, Random Forest,
Gradient Boosting) sees one row at a time and cannot express that difference.

Design decisions, consistent with the rest of the project:

- **Comparison baseline only.** Like Random Forest and Gradient Boosting, this
  model is evaluated on the same held-out test set but is never used for live
  case scoring -- ml/case_packet.py and app/services/case_service.py keep using
  Logistic Regression, per the preference for interpretable models in anything
  merchant- or reviewer-facing.
- **Same preprocessing as every other model.** Each week's "token" is the exact
  numeric+one-hot vector produced by ml/model_utils.py's shared preprocessing
  pipeline, fit on the training split only -- not a second, divergent feature
  path.
- **Attention weights are exposed per week.** The point of a sequence model
  here is not only accuracy; it is a second, independent explanation modality
  ("weeks 5 and 6 drove this flag") alongside the existing feature-level
  before/after explanation. That requires reading attention weights out, which
  nn.TransformerEncoder does not cleanly support, so the encoder layer below is
  written explicitly rather than assembled from nn.TransformerEncoderLayer.

Trained on synthetic data for demonstration only. Not a real-world chargeback
prediction model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ml.features import LABEL_COLUMN
from ml.model_utils import build_preprocessing_pipeline, design_matrix

DEFAULT_WINDOW = 8
DEFAULT_D_MODEL = 32
DEFAULT_N_HEADS = 4
DEFAULT_FEEDFORWARD = 64
DEFAULT_DROPOUT = 0.1
DEFAULT_LAYERS = 2

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-3


def build_sequences(
    df: pd.DataFrame,
    week_vectors: np.ndarray,
    window: int = DEFAULT_WINDOW,
    context_df: pd.DataFrame | None = None,
    context_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Turns per-row preprocessed feature vectors into per-row trailing-window
    sequences, grouped by merchant and ordered by week.

    week_vectors[i] must be the preprocessed feature vector for df row i.

    Returns (sequences, padding_mask):
      sequences[i]     -- (window, n_features); position window-1 is row i's own
                          week, position window-1-k is k weeks earlier.
      padding_mask[i]  -- (window,) bool; True where the position is padding
                          (that merchant had no week that far back yet).

    context_df/context_vectors supply read-only prior weeks that are not
    themselves being scored. Without them, a df covering only one split has its
    trailing windows truncated at the split boundary: at evaluation time every
    merchant's first `window - 1` test weeks would be padded even though the
    real prior weeks exist in the validation split immediately before. That
    does not leak labels (only feature vectors are read, and the truncation
    biases recall downward), but it measures "does trajectory help when 7% of
    rows have their history cut" rather than the intended question.

    Temporal discipline: only context rows *strictly before* a target row's
    week_start are ever used. Where a context row and a target row share a
    (merchant_id, week_start), the target row wins and the context copy is
    dropped, so a row can never serve as its own history.

    A merchant whose real history is genuinely shorter than the window still
    gets zero-padding plus a mask, rather than being dropped -- silently
    discarding those rows would change the evaluated population and make this
    model's metrics incomparable to the other models'.
    """
    n_rows, n_features = week_vectors.shape
    sequences = np.zeros((n_rows, window, n_features), dtype=np.float32)
    padding_mask = np.ones((n_rows, window), dtype=bool)

    # Positional (not label-based) indexing throughout: df may carry any index,
    # and week_vectors is aligned to row position, not index label.
    if context_df is None or context_vectors is None or len(context_df) == 0:
        merchants = df["merchant_id"].to_numpy()
        weeks = df["week_start"].to_numpy()
        vectors = week_vectors
        # Target row index for each entry; -1 marks a context-only entry.
        target_of = np.arange(n_rows)
    else:
        merchants = np.concatenate([context_df["merchant_id"].to_numpy(), df["merchant_id"].to_numpy()])
        weeks = np.concatenate([context_df["week_start"].to_numpy(), df["week_start"].to_numpy()])
        vectors = np.concatenate([context_vectors, week_vectors])
        target_of = np.concatenate([np.full(len(context_df), -1), np.arange(n_rows)])

    # Sort by (merchant, week, is_target) so that within a duplicated week the
    # target row sorts last and wins the de-duplication below.
    is_target = (target_of >= 0).astype(np.int8)
    ordering = np.lexsort((is_target, weeks, merchants))
    merchants_sorted = merchants[ordering]
    weeks_sorted = weeks[ordering]

    # Keep the last entry of each (merchant, week) run -- the target row when
    # one exists, otherwise the context copy.
    is_last_of_week = np.r_[
        (merchants_sorted[:-1] != merchants_sorted[1:]) | (weeks_sorted[:-1] != weeks_sorted[1:]),
        True,
    ]
    ordering = ordering[is_last_of_week]
    merchants_sorted = merchants[ordering]

    # After de-duplication each merchant's weeks are strictly increasing, so
    # "the entries preceding this one" are exactly "weeks strictly earlier".
    group_starts = np.flatnonzero(np.r_[True, merchants_sorted[1:] != merchants_sorted[:-1]])
    group_bounds = np.r_[group_starts, len(ordering)]

    for start, end in zip(group_bounds[:-1], group_bounds[1:]):
        merchant_entries = ordering[start:end]
        for offset, entry in enumerate(merchant_entries):
            target_row = target_of[entry]
            if target_row < 0:
                continue  # context-only week: supplies history, is never scored
            available = min(window, offset + 1)
            history_entries = merchant_entries[offset + 1 - available: offset + 1]
            sequences[target_row, window - available:] = vectors[history_entries]
            padding_mask[target_row, window - available:] = False

    return sequences, padding_mask


class _AttentionEncoderLayer(nn.Module):
    """One pre-norm self-attention + feed-forward block that returns its own
    attention weights, so a prediction can be explained by which weeks the model
    actually attended to."""

    def __init__(self, d_model: int, n_heads: int, feedforward: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm_attention(x)
        attended, attention_weights = self.attention(
            normed, normed, normed,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feedforward(self.norm_feedforward(x)))
        return x, attention_weights


class TrajectoryTransformer(nn.Module):
    """Small (~50k parameter) encoder over a merchant's trailing-week sequence.

    Deliberately tiny: 900 merchants of synthetic data cannot support a large
    model, and pretending otherwise would produce an impressive-sounding
    architecture with a worse honest result.
    """

    def __init__(
        self,
        n_features: int,
        window: int = DEFAULT_WINDOW,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        feedforward: int = DEFAULT_FEEDFORWARD,
        dropout: float = DEFAULT_DROPOUT,
        n_layers: int = DEFAULT_LAYERS,
    ):
        super().__init__()
        self.window = window
        self.input_projection = nn.Linear(n_features, d_model)
        self.position_embedding = nn.Embedding(window, d_model)
        self.layers = nn.ModuleList(
            [_AttentionEncoderLayer(d_model, n_heads, feedforward, dropout) for _ in range(n_layers)]
        )
        self.norm_output = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, sequences: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, last_layer_attention_weights)."""
        positions = torch.arange(self.window, device=sequences.device)
        x = self.input_projection(sequences) + self.position_embedding(positions)

        attention_weights = None
        for layer in self.layers:
            x, attention_weights = layer(x, padding_mask)

        x = self.norm_output(x)

        # Mean-pool over real (non-padding) positions only, so a merchant with a
        # short history is not diluted by zero-padding it never had.
        keep = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

        return self.head(pooled).squeeze(-1), attention_weights


class TrajectoryModel:
    """Wrapper holding the fitted preprocessing pipeline plus the trained
    network, with a plain (DataFrame in, probability array out) interface so
    ml/evaluate_model.py can score it exactly like the sklearn models without
    knowing anything about tensors."""

    def __init__(self, window: int = DEFAULT_WINDOW, seed: int = 42):
        self.window = window
        self.seed = seed
        self.preprocessor = None
        self.network: TrajectoryTransformer | None = None

    def _week_vectors(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        features = design_matrix(df)
        if fit:
            transformed = self.preprocessor.fit_transform(features)
        else:
            transformed = self.preprocessor.transform(features)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=np.float32)

    def _tensors(
        self,
        df: pd.DataFrame,
        fit: bool = False,
        history_df: pd.DataFrame | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        week_vectors = self._week_vectors(df, fit=fit)
        context_vectors = None
        if history_df is not None and len(history_df) > 0:
            # Transform only -- never fit. The preprocessing pipeline stays
            # fit on the training split alone.
            context_vectors = self._week_vectors(history_df, fit=False)
        sequences, padding_mask = build_sequences(
            df, week_vectors, self.window, context_df=history_df, context_vectors=context_vectors
        )
        return torch.from_numpy(sequences), torch.from_numpy(padding_mask)

    def fit(
        self,
        train_df: pd.DataFrame,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        verbose: bool = True,
    ) -> "TrajectoryModel":
        torch.manual_seed(self.seed)
        self.preprocessor = build_preprocessing_pipeline()

        sequences, padding_mask = self._tensors(train_df, fit=True)
        labels = torch.from_numpy(train_df[LABEL_COLUMN].to_numpy().astype(np.float32))

        self.network = TrajectoryTransformer(n_features=sequences.shape[-1], window=self.window)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        loss_function = nn.BCEWithLogitsLoss()

        n_rows = sequences.shape[0]
        self.network.train()
        for epoch in range(epochs):
            permutation = torch.randperm(n_rows)
            epoch_loss = 0.0
            for start in range(0, n_rows, batch_size):
                batch = permutation[start: start + batch_size]
                optimizer.zero_grad()
                logits, _ = self.network(sequences[batch], padding_mask[batch])
                loss = loss_function(logits, labels[batch])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item() * len(batch)
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs} train_loss={epoch_loss / n_rows:.4f}")

        return self

    def predict(self, df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> np.ndarray:
        """Probability of label_high_loss_next_30d, aligned to df's row order.

        history_df is an optional read-only feature frame (typically the full
        train+validation+test frame) used to supply trailing-window context for
        rows near df's earliest week. Only its feature columns are read, only
        weeks strictly before each target week are used, and it never
        participates in fitting or gradient updates.
        """
        sequences, padding_mask = self._tensors(df, fit=False, history_df=history_df)
        self.network.eval()
        with torch.no_grad():
            logits, _ = self.network(sequences, padding_mask)
            return torch.sigmoid(logits).numpy()

    def attention_by_week(self, df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> np.ndarray:
        """Per-row attention over the trailing window, as (n_rows, window).

        Column index window-1 is the current week, window-1-k is k weeks back.
        Padding positions are zeroed so a short-history merchant's explanation
        never claims attention on a week that does not exist.

        history_df has the same meaning and the same read-only, strictly-prior
        contract as in predict().

        Caveat: attention weight is a heuristic proxy for temporal importance,
        not a causal explanation of the prediction.
        """
        sequences, padding_mask = self._tensors(df, fit=False, history_df=history_df)
        self.network.eval()
        with torch.no_grad():
            _, attention_weights = self.network(sequences, padding_mask)
            # Average over query positions: "how much did the sequence as a
            # whole attend to each week", which is the reviewer-facing question.
            per_week = attention_weights.mean(dim=1)
            per_week = per_week.masked_fill(padding_mask, 0.0)
            return per_week.numpy()
