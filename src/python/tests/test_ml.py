# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the shared ``dse_research_utils.ml`` package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import dse_research_utils.ml.feature_dependence as feature_dependence
from dse_research_utils.ml.cross_validation import DEFAULT_REGRESSION_SCORERS, cross_validation_score_rows
from dse_research_utils.ml.feature_dependence import (
    distance_corr_matrix,
    mutual_info_dissimilarity,
    spearman_distance_matrix,
)
from dse_research_utils.ml.importance import grouped_permutation_importance
from dse_research_utils.ml.search import hyperparam_search_randomized


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


def test_cross_validation_score_rows_flips_default_display_labels():
    scores = {
        f"test_{label}": np.array([-1.0, -3.0])
        for label, scorer in DEFAULT_REGRESSION_SCORERS.items()
        if scorer.startswith("neg_")
    }
    rows = {r["metric"]: r for r in cross_validation_score_rows(scores)}
    assert rows["Mean Absolute Error (MAE)"]["mean"] == pytest.approx(2.0)
    assert rows["Root Mean Squared Error (RMSE)"]["mean"] == pytest.approx(2.0)
    assert rows["Median Absolute Error (MedAE)"]["mean"] == pytest.approx(2.0)


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


def test_distance_corr_dissimilarity_linkage_uses_average_linkage(monkeypatch):
    dissim = np.array(
        [
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.7],
            [0.8, 0.7, 0.0],
        ]
    )
    expected_linkage = np.array([[0.0, 1.0, 0.2, 2.0], [2.0, 3.0, 0.75, 3.0]])

    monkeypatch.setattr(
        feature_dependence,
        "distance_corr_dissimilarity",
        lambda X: (dissim.copy(), 1.0 - dissim),
    )
    monkeypatch.setattr(feature_dependence.hierarchy, "average", lambda condensed: expected_linkage)
    monkeypatch.setattr(feature_dependence.hierarchy, "ward", lambda condensed: pytest.fail("ward should not be used"))

    out_dissim, condensed, linkage = feature_dependence.distance_corr_dissimilarity_linkage(np.ones((5, 3)))
    np.testing.assert_allclose(out_dissim, dissim)
    np.testing.assert_allclose(condensed, np.array([0.2, 0.8, 0.7]))
    np.testing.assert_allclose(linkage, expected_linkage)


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
    deltas = grouped_permutation_importance(estimators, X, y, test_indices, cluster_cols, n_repeats=5, seed=0)
    # Permuting the signal block should raise held-out RMSE; the noise block barely.
    assert deltas[0].mean() > 0.5
    assert abs(deltas[1].mean()) < deltas[0].mean()


def test_hyperparam_search_randomized_multimetric_requires_refit_metric():
    X = np.arange(20.0).reshape(-1, 1)
    y = np.arange(20.0)
    scoring = {"mae": "neg_mean_absolute_error", "r2": "r2"}

    with pytest.raises(ValueError, match="refit"):
        hyperparam_search_randomized(
            X,
            y,
            None,
            DummyRegressor(),
            {"strategy": ["mean", "median"]},
            n_iter=2,
            scoring=scoring,
            cv=KFold(2),
            random_state=0,
        )


def test_hyperparam_search_randomized_multimetric_refit_key_returns_best_params():
    X = np.arange(20.0).reshape(-1, 1)
    y = np.arange(20.0)
    search, results, best_params = hyperparam_search_randomized(
        X,
        y,
        None,
        DummyRegressor(),
        {"strategy": ["mean", "median"]},
        n_iter=2,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        refit="mae",
        cv=KFold(2),
        random_state=0,
    )

    assert search.refit == "mae"
    assert "mean_test_mae" in results.columns
    assert best_params["strategy"] in {"mean", "median"}


def test_grouped_permutation_importance_rejects_mismatched_fold_inputs():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="zip\\(\\) argument"):
        grouped_permutation_importance([], X, y, [np.array([0, 1])], {0: [0]}, n_repeats=1, seed=0)
