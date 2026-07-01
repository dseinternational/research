# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Grouped (block) out-of-fold permutation importance.

A pure-numeric routine, estimator-agnostic and independent of any pipeline, so it
is importable and unit-testable on its own.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import root_mean_squared_error


def grouped_permutation_importance(
    estimators,
    X,
    y,
    test_indices,
    cluster_cols,
    *,
    n_repeats,
    seed,
):
    """Joint (grouped) out-of-fold permutation deltas, one block per cluster.

    For each held-out fold it permutes ALL of a cluster's columns together (one
    row-permutation per repeat applied to the whole block), which removes the
    within-cluster substitution dilution that deflates per-feature scores. The RNG
    is reset to ``seed`` at the start of each fold, mirroring a per-feature
    permutation loop that passes a fixed ``random_state`` per fold.

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
        permuted; positive = the cluster was useful).
    """
    y = np.asarray(y, dtype=float)
    deltas: dict[int, list[float]] = {c: [] for c in cluster_cols}
    for est, val_idx in zip(estimators, test_indices, strict=False):
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
