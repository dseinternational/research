# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compute joint feature-block permutation scores within each held-out fold.

Use ``ml.permutation.pooled_oof_permutation_deltas`` when the score must be
computed once over pooled held-out predictions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def grouped_permutation_importance(
    estimators: Iterable[Any],
    X: pd.DataFrame,
    y: np.ndarray | Sequence[float],
    test_indices: Iterable[np.ndarray],
    cluster_cols: Mapping[int, Sequence[int]],
    *,
    n_repeats: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Joint (grouped) out-of-fold permutation deltas, one block per cluster.

    Each repeat shuffles a cluster's columns together within one held-out fold.
    This preserves their within-row relationships while changing which row
    supplies them. The random generator resets to ``seed`` for each fold.
    Results concatenate fold-specific score changes; they are not changes
    in a score computed over pooled predictions.

    Parameters
    ----------
    estimators
        Per-fold fitted estimators (e.g. ``cross_validate(..., return_estimator=True)``).
    X : pandas.DataFrame
        The full design matrix; folds are selected positionally via ``.iloc``.
    y : array-like
        The full target vector.
    test_indices
        Per-fold held-out row positions (aligned with ``estimators``).
    cluster_cols : dict
        Maps cluster id -> list of column *positions* in ``X``.
    n_repeats : int
        Permutation repeats per cluster per fold.
    seed : int
        RNG seed, reset at the start of each fold.

    Returns
    -------
    dict
        Cluster id -> array of deltas (held-out RMSE rise when the block is
        permuted). Positive values mean that this shuffle worsened prediction
        error on that fold; they do not establish a causal effect.
    """
    y = np.asarray(y, dtype=float)
    deltas: dict[int, list[float]] = {c: [] for c in cluster_cols}
    for est, val_idx in zip(estimators, test_indices, strict=True):
        X_val = X.iloc[val_idx]
        y_val = y[val_idx]
        base_rmse = root_mean_squared_error(y_val, est.predict(X_val))
        n = len(X_val)
        rng = np.random.default_rng(seed)  # reset per fold (matches the per-feature loop)
        for c, cols in cluster_cols.items():
            for _ in range(n_repeats):
                perm = rng.permutation(n)
                Xp = X_val.copy()
                Xp.iloc[:, cols] = X_val.iloc[perm, cols].to_numpy()  # joint block shuffle
                deltas[c].append(root_mean_squared_error(y_val, est.predict(Xp)) - base_rmse)
    return {c: np.asarray(v) for c, v in deltas.items()}
