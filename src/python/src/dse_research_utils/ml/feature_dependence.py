# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pairwise feature-dependence and clustering primitives.

Spearman / distance-correlation / mutual-information dissimilarity matrices over a
``(n_samples, n_features)`` design, plus a Ward-linkage helper, for feature
clustering and redundancy analysis. Distance correlation uses the optional ``dcor``
dependency (install the ``dependence`` extra: ``pip install
dse-research-utils[dependence]``), imported lazily so it is only required when
that path is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression


def spearman_distance_matrix(X: pd.DataFrame | np.ndarray | list[float]):
    """
    Compute Spearman rank correlation between all pairs of features in X (DataFrame or ndarray).
    Returns distance matrix (1 - |correlation|) and correlation matrix.

    Parameters
    ----------
    X: DataFrame or ndarray of shape (n_samples, n_features)

    Returns
    -------
    distance: distance matrix (1 - |correlation|) - ndarray of shape (n_features, n_features)
    corr: correlation matrix - ndarray of shape (n_features, n_features)

    Notes
    -----
    Missing values are handled by pairwise-complete deletion (pandas
    ``min_periods=2``). Both DataFrame and ndarray inputs route through the
    same code path so the result does not depend on the input container — an
    earlier version computed the ndarray case with
    ``scipy.stats.spearmanr(nan_policy="propagate")``, which (a) returned a
    bare scalar for exactly two columns and (b) zeroed out whole rows/columns
    whenever a single value was NaN, silently disagreeing with the DataFrame
    path.
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


def distance_corr_matrix(X: pd.DataFrame | np.ndarray | list[float]):
    """
    Compute pairwise distance correlation matrix for all feature pairs.

    Distance correlation measures both linear and non-linear dependence between variables,
    returning values in [0, 1] where 0 = independence and 1 = complete dependence.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.

    Returns
    -------
    M : ndarray of shape (n_features, n_features)
        Symmetric matrix where M[i,j] is the distance correlation between features i and j.
        Diagonal elements equal 1.0 (self-correlation).

    Notes
    -----
    - Unlike Pearson/Spearman correlation, distance correlation detects non-linear relationships
    - More computationally expensive than standard correlation (O(n²) for n features)
    - Values range from 0 (independent) to 1 (perfectly dependent)
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


def distance_corr_dissimilarity(X: pd.DataFrame | np.ndarray | list[float]):
    """
    Compute pairwise distance correlation dissimilarity matrix for all feature pairs.

    Converts distance correlation (similarity) into dissimilarity metric by computing 1 - dcor.
    This dissimilarity matrix is suitable for clustering algorithms that expect distance-like measures.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.

    Returns
    -------
    dissim : ndarray of shape (n_features, n_features)
        Symmetric dissimilarity matrix where dissim[i,j] = 1 - distance_correlation(i,j).
        Values range from 0 (perfectly dependent) to 1 (independent).
        Diagonal elements equal 0 (zero distance to self).

    Notes
    -----
    - Distance correlation in [0,1]: 0 = independent, 1 = dependent
    - Dissimilarity in [0,1]: 0 = similar/dependent, 1 = dissimilar/independent
    """
    corr_matrix = distance_corr_matrix(X)
    dissim = 1.0 - corr_matrix

    np.clip(dissim, 0.0, 1.0, out=dissim)
    np.fill_diagonal(dissim, 0.0)

    return dissim, corr_matrix


def distance_corr_dissimilarity_linkage(X):
    """
    Compute condensed distance correlation dissimilarity matrix for linkage clustering.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.

    Returns
    -------
    dissim : ndarray of shape (n_features, n_features)
        Symmetric dissimilarity matrix where dissim[i,j] = 1 - distance_correlation(i,j).
        Values range from 0 (perfectly dependent) to 1 (independent).
        Diagonal elements equal 0 (zero distance to self).

    condensed : ndarray of shape (n_features * (n_features - 1) / 2,)
        Condensed dissimilarity matrix suitable for linkage clustering algorithms.

    linkage : ndarray
        Linkage matrix resulting from hierarchical clustering using Ward's method.

    Notes
    -----
    - Uses distance correlation dissimilarity metric
    - Returns condensed form for compatibility with scipy linkage functions
    """
    dissim, _corr_matrix = distance_corr_dissimilarity(X)
    condensed = squareform(dissim)
    linkage = hierarchy.ward(condensed)
    return dissim, condensed, linkage


def mutual_info_dissimilarity(
    X,
    discrete_features="auto",
    n_neighbors=3,
    copy=True,
    random_state=None,
    n_jobs=None,
):
    """
    Compute pairwise mutual information dissimilarity matrix for all feature pairs.

    Uses mutual information regression to measure dependence between features, then converts
    to dissimilarity by computing 1 - normalized MI. Mutual information captures both linear
    and non-linear relationships.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.
    discrete_features : {'auto', bool, array-like}, default='auto'
        Indicates which features are discrete. 'auto' infers from data type.
    n_neighbors : int, default=3
        Number of neighbors for k-NN based MI estimation.
    copy : bool, default=True
        Whether to make a copy of X.
    random_state : int, RandomState instance or None, default=None
        Random seed for reproducibility.
    n_jobs : int or None, default=None
        Number of parallel jobs. None means 1, -1 uses all processors.

    Returns
    -------
    dissim : ndarray of shape (n_features, n_features)
        Symmetric dissimilarity matrix where dissim[i,j] = 1 - normalized_MI(i,j).
        Values range from 0 (high mutual information) to 1 (independent).
        Diagonal elements near 0 (features maximally informative about themselves).

    Notes
    -----
    - Mutual information quantifies information shared between variables
    - Normalized by max MI score per feature for scale invariance
    - Symmetrized by averaging with transpose: (dissim + dissim.T) / 2
    - Suitable for feature clustering and selection
    - More robust to non-linear relationships than correlation-based methods
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
    # Each feature is maximally informative about itself; force the
    # diagonal so the returned matrix is a proper dissimilarity.
    np.fill_diagonal(dissim, 0.0)
    return dissim
