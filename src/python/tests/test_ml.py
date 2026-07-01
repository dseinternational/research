# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the shared ``dse_research_utils.ml`` package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from dse_research_utils.ml.cross_validation import cross_validation_score_rows
from dse_research_utils.ml.feature_dependence import (
    distance_corr_matrix,
    mutual_info_dissimilarity,
    spearman_distance_matrix,
)
from dse_research_utils.ml.importance import grouped_permutation_importance


def test_cross_validation_score_rows_sign_logic():
    scores = {
        "test_neg_mean_absolute_error": np.array([-1.0, -3.0]),  # stored negative
        "test_r2": np.array([-0.2, 0.4]),  # signed, kept as-is
        "fit_time": np.array([0.1, 0.2]),  # ignored (not test_*)
    }
    rows = {r["metric"]: r for r in cross_validation_score_rows(scores)}
    assert set(rows) == {"neg_mean_absolute_error", "r2"}
    assert rows["neg_mean_absolute_error"]["mean"] == pytest.approx(2.0)  # flipped positive
    assert rows["r2"]["mean"] == pytest.approx(0.1)  # true (possibly negative) sign kept


def test_spearman_distance_matrix_basic():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    df = pd.DataFrame({"a": a, "b": 2 * a + 1.0, "c": rng.normal(size=200)})
    dist, _corr = spearman_distance_matrix(df)
    assert dist.shape == (3, 3)
    assert np.allclose(np.diag(dist), 0.0)
    assert np.allclose(dist, dist.T)
    # a and b are monotonic transforms -> Spearman |corr| = 1 -> distance 0.
    assert dist[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_distance_corr_matrix_basic():
    pytest.importorskip("dcor")
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = np.column_stack([a, 2 * a + 1.0, rng.normal(size=200)])
    M = distance_corr_matrix(X)
    assert M.shape == (3, 3)
    assert np.allclose(np.diag(M), 1.0)
    assert np.allclose(M, M.T)
    # a and (2a+1) are a deterministic linear transform of each other -> dcor ~ 1.
    assert M[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_distance_corr_matrix_missing_dcor_raises_clear_error(monkeypatch):
    # Simulate the optional 'dcor' dependency being absent: the lazy import
    # inside distance_corr_matrix must raise a ModuleNotFoundError naming the
    # 'dependence' extra, not a bare import traceback.
    import sys

    monkeypatch.setitem(sys.modules, "dcor", None)
    X = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.raises(ModuleNotFoundError, match=r"dependence.*extra"):
        distance_corr_matrix(X)


def test_mutual_info_dissimilarity_diagonal_zero():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 4))
    d = mutual_info_dissimilarity(X, random_state=0)
    assert d.shape == (4, 4)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)


def test_grouped_permutation_importance_flags_predictive_block():
    rng = np.random.default_rng(2)
    n = 200
    x_signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "s0": x_signal,
            "s1": x_signal + rng.normal(scale=0.01, size=n),  # same cluster as s0
            "noise": rng.normal(size=n),
        }
    )
    y = 3.0 * x_signal + rng.normal(scale=0.1, size=n)

    estimators, test_indices = [], []
    for tr, te in KFold(n_splits=4, shuffle=True, random_state=0).split(X):
        est = clone(LinearRegression()).fit(X.iloc[tr], y[tr])
        estimators.append(est)
        test_indices.append(te)

    cluster_cols = {0: [0, 1], 1: [2]}  # signal block vs noise
    deltas = grouped_permutation_importance(
        estimators, X, y, test_indices, cluster_cols, n_repeats=5, seed=0
    )
    # Permuting the signal block should raise held-out RMSE; the noise block barely.
    assert deltas[0].mean() > 0.5
    assert abs(deltas[1].mean()) < deltas[0].mean()
