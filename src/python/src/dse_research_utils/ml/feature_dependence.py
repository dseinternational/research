# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pairwise feature-dependence and clustering primitives.

Spearman / distance-correlation / mutual-information dissimilarity matrices over a
``(n_samples, n_features)`` design, plus an average-linkage helper, for feature
clustering and redundancy analysis. Distance correlation uses the optional ``dcor``
dependency (install the ``dependence`` extra: ``pip install
dse-research-utils[dependence]``), imported lazily so it is only required when
that path is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression

from dse_research_utils.ml.feature_groups import linkage_from_dissimilarity


def spearman_distance_matrix(X: pd.DataFrame | np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature-wise Spearman correlations and ``1 - abs(correlation)``.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Rows are observations and columns are features.

    Returns
    -------
    distance : ndarray of shape (n_features, n_features)
        Symmetric dissimilarity matrix with a zero diagonal.
    corr : ndarray of shape (n_features, n_features)
        Symmetric correlation matrix with a unit diagonal.

    Notes
    -----
    Each pair uses observations available for both features, with at least two
    required. Unavailable correlations become zero before the diagonal is set
    to one. A zero returned off-diagonal correlation can therefore mean missing
    information, rather than an estimated absence of association.
    """
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    corr = df.corr(method="spearman", min_periods=2).to_numpy()

    # Replace NaNs and infinities
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)

    # Enforce exact symmetry and unit diagonal
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    # Convert to distance matrix: in [0,1], 0 on diagonal
    distance = 1.0 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)

    return distance, corr


def distance_corr_matrix(X: pd.DataFrame | np.ndarray | list[float]) -> np.ndarray:
    """Estimate pairwise distance correlations from finite observation pairs.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input values, converted to float64. Requires the ``dependence`` extra.

    Returns
    -------
    M : ndarray of shape (n_features, n_features)
        Symmetric matrix clipped to [0, 1], with a unit diagonal. Each pair uses
        only rows where both features are finite. Fewer than two such rows or
        a non-finite estimate produces zero for that pair.

    Notes
    -----
    The dcor estimator can measure nonlinear association. A zero sample value,
    including the fallback for missing information, does not prove population
    independence. The cost depends on both the feature count and observations
    per pair. See https://dcor.readthedocs.io/en/latest/functions/dcor.distance_correlation.html.
    """
    try:
        import dcor  # lazy: only this function needs the optional dependency
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "distance_corr_matrix requires dcor; install the 'dependence' extra "
            "(pip install dse-research-utils[dependence])."
        ) from exc

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[1]
    M = np.ones((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            xi = X[:, i]
            xj = X[:, j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            # Not enough data points
            if mask.sum() < 2:
                val = 0.0
            else:
                val = dcor.distance_correlation(xi[mask], xj[mask])
                # Calculation failed (e.g., constant values)
                if not np.isfinite(val):
                    val = 0.0

            M[i, j] = val
            M[j, i] = val

    np.fill_diagonal(M, 1.0)
    np.clip(M, 0.0, 1.0, out=M)
    return M


def distance_corr_dissimilarity(X: pd.DataFrame | np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``1 - distance_correlation`` and the underlying correlation matrix.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data passed to :func:`distance_corr_matrix`.

    Returns
    -------
    dissim : ndarray of shape (n_features, n_features)
        Symmetric dissimilarities clipped to [0, 1], with a zero diagonal.
        Missing-information fallbacks become dissimilarities of one.
    corr_matrix : ndarray of shape (n_features, n_features)
        The underlying pairwise correlation estimates.

    Notes
    -----
    These dissimilarities do not establish Euclidean distance or population
    independence. See :func:`distance_corr_matrix` for finite-row handling.
    """
    corr_matrix = distance_corr_matrix(X)
    dissim = 1.0 - corr_matrix

    np.clip(dissim, 0.0, 1.0, out=dissim)
    np.fill_diagonal(dissim, 0.0)

    return dissim, corr_matrix


def distance_corr_dissimilarity_linkage(
    X: pd.DataFrame | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an average-linkage tree from distance-correlation dissimilarities.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data passed to :func:`distance_corr_dissimilarity`.

    Returns
    -------
    dissim : ndarray of shape (n_features, n_features)
        Square dissimilarity matrix, including its missing-information fallbacks.
    condensed : ndarray of shape (n_features * (n_features - 1) / 2,)
        Upper-triangle values in SciPy's condensed ordering.
    linkage : ndarray
        Average-linkage tree. Zero or one feature returns shape ``(0, 4)``.

    Notes
    -----
    Average linkage accepts these precomputed dissimilarities. Ward linkage
    requires Euclidean distances, which distance-correlation dissimilarities
    do not in general establish.
    """
    dissim, _corr_matrix = distance_corr_dissimilarity(X)
    condensed = squareform(dissim)
    linkage = linkage_from_dissimilarity(dissim, method="average")
    return dissim, condensed, linkage


def mutual_info_dissimilarity(
    X: pd.DataFrame | np.ndarray,
    discrete_features: str | bool | list[bool] | np.ndarray = "auto",
    n_neighbors: int = 3,
    copy: bool = True,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Build symmetric dissimilarities from estimated mutual information.

    Each feature acts in turn as the continuous target of
    ``mutual_info_regression``. Its scores are divided by the largest score in
    that row, then subtracted from one. The result is averaged with its transpose
    and its diagonal is set to zero.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Input feature matrix. Every target column is treated as continuous.
    discrete_features : {'auto', bool, array-like}, default 'auto'
        Predictor-feature flags passed to scikit-learn. ``'auto'`` treats dense
        inputs as continuous; it does not infer discrete features from dtype.
    n_neighbors : int, default 3
        Neighbour count for estimating mutual information with continuous values.
    copy : bool, default True
        Passed to scikit-learn. With False, its estimator can overwrite input data.
    random_state : int or None, default None
        Seed for noise used to break ties in continuous values.
    n_jobs : int or None, default None
        Parallel jobs for the estimator. -1 requests all processors; None uses
        one job unless a joblib backend context changes it.

    Returns
    -------
    dissim : ndarray of shape (n_features, n_features)
        Symmetric dissimilarities with an exactly zero diagonal. An all-zero
        score row becomes ones before symmetrisation and diagonal replacement.
        A distance of one does not prove population independence.

    Notes
    -----
    The row-wise normalisation is a relative comparison of estimated scores,
    not a test of independence or a guaranteed scale-invariant measure.
    See https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html.
    """
    # Accept DataFrames or ndarrays uniformly.
    if hasattr(X, "iloc"):
        n_features = X.shape[1]
        target_column = lambda i: X.iloc[:, i]  # noqa: E731
    else:
        X_arr = np.asarray(X)
        n_features = X_arr.shape[1]
        target_column = lambda i: X_arr[:, i]  # noqa: E731

    dissim = np.zeros((n_features, n_features))

    for i in range(n_features):
        mi_scores = mutual_info_regression(
            X,
            target_column(i),
            discrete_features=discrete_features,
            n_neighbors=n_neighbors,
            copy=copy,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        # When every feature has zero MI against the target column (e.g.
        # a constant column), the max is zero; treat the row as fully
        # dissimilar (1.0) rather than dividing by zero.
        max_mi = float(mi_scores.max())
        if max_mi > 0:
            dissim[i, :] = 1.0 - mi_scores / max_mi
        else:
            dissim[i, :] = 1.0

    dissim = (dissim + dissim.T) / 2  # Symmetrize
    # Set the diagonal to zero by convention, including all-zero score rows.
    np.fill_diagonal(dissim, 0.0)
    return dissim
