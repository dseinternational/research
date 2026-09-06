# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cluster precomputed feature dissimilarities without choosing study policy."""

from collections.abc import Sequence
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def _real_matrix(values: ArrayLike, name: str) -> np.ndarray:
    if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
        raise ValueError(f"{name} must not contain masked entries")
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_float64(array: np.ndarray, name: str) -> np.ndarray:
    try:
        with np.errstate(over="raise", invalid="raise"):
            array = np.array(array, dtype=np.float64, copy=True)
    except FloatingPointError as exc:
        raise ValueError(f"{name} entries must be representable as finite float64 values") from exc
    return array


def linkage_from_dissimilarity(
    matrix: ArrayLike,
    *,
    method: Literal["single", "complete", "average", "weighted"],
) -> np.ndarray:
    """Construct a linkage tree from a validated square dissimilarity matrix.

    Parameters
    ----------
    matrix
        Square real matrix in feature order, with finite nonnegative entries,
        exact symmetry and an exactly zero diagonal. Values need not be at most
        one or satisfy the triangle inequality. Labels on array-like inputs do
        not define ordering here; callers retain the feature order separately.
        Missing values, asymmetry and rounding errors are not repaired.
    method : {"single", "complete", "average", "weighted"}
        SciPy linkage method. Ward, centroid and median linkage are excluded
        because an arbitrary dissimilarity matrix need not be Euclidean.

    Returns
    -------
    numpy.ndarray
        SciPy linkage matrix, shape ``(n_features - 1, 4)`` for two or more
        features. Zero or one feature returns an empty float64 ``(0, 4)``
        matrix. Inputs are not modified.

    Notes
    -----
    The matrix is converted to float64 before clustering. Input order and
    SciPy's handling of tied distances determine the leaf references. No
    imputation, dependence calculation or feature selection is performed.
    """
    if not isinstance(method, str) or method not in ("single", "complete", "average", "weighted"):
        raise ValueError(
            "method must be single, complete, average or weighted; Euclidean-only methods are not supported"
        )
    dissimilarity = _real_matrix(matrix, "matrix")
    if dissimilarity.shape[0] != dissimilarity.shape[1]:
        raise ValueError("matrix must be square")
    if np.any(dissimilarity < 0):
        raise ValueError("matrix must contain nonnegative dissimilarities")
    if np.any(np.diag(dissimilarity) != 0):
        raise ValueError("matrix must have an exactly zero diagonal")
    if not np.array_equal(dissimilarity, dissimilarity.T):
        raise ValueError("matrix must be exactly symmetric")
    if len(dissimilarity) < 2:
        return np.empty((0, 4), dtype=np.float64)
    dissimilarity = _as_float64(dissimilarity, "matrix")
    linkage = hierarchy.linkage(squareform(dissimilarity, checks=False), method=method)
    if not np.isfinite(linkage).all():
        raise ValueError("Clustering produced nonfinite linkage values")
    return linkage


def _validated_linkage(linkage: ArrayLike, n_features: int) -> np.ndarray:
    matrix = _real_matrix(linkage, "linkage")
    if matrix.shape != (max(n_features - 1, 0), 4):
        raise ValueError(
            "linkage must have four columns and exactly one fewer row than names, or zero rows for no names"
        )
    if n_features < 2:
        return _as_float64(matrix, "linkage")
    if np.any(matrix[:, 2] < 0) or np.any(matrix[1:, 2] < matrix[:-1, 2]):
        raise ValueError("linkage merge heights must be nonnegative and nondecreasing")
    sizes = np.ones(2 * n_features - 1, dtype=np.intp)
    consumed = np.zeros(2 * n_features - 1, dtype=bool)
    for row, (left, right, _height, count) in enumerate(matrix):
        parent = n_features + row
        if left < 0 or right < 0 or left >= parent or right >= parent:
            raise ValueError("linkage children must reference existing leaves or earlier clusters")
        if left != np.floor(left) or right != np.floor(right):
            raise ValueError("linkage child references must be integers")
        left_id, right_id = int(left), int(right)
        if left_id == right_id or consumed[left_id] or consumed[right_id]:
            raise ValueError("linkage must not reuse a leaf or cluster")
        sizes[parent] = sizes[left_id] + sizes[right_id]
        if count != sizes[parent]:
            raise ValueError("linkage member counts must equal the number of descendant features")
        consumed[left_id] = consumed[right_id] = True
    return _as_float64(matrix, "linkage")


def feature_groups_from_linkage(
    names: Sequence[str],
    linkage: ArrayLike,
    *,
    threshold: float,
) -> dict[int, list[str]]:
    """Cut a linkage tree while retaining feature order and SciPy cluster labels.

    Parameters
    ----------
    names
        Unique, nonempty feature names, in the original matrix's feature order.
        The tree contains positions, so it cannot verify the meaning of names
        supplied in a different order.
    linkage
        A finite, structurally valid SciPy linkage matrix with nonnegative,
        nondecreasing merge heights. Its leaf references and member counts must
        agree with ``names``. Empty inputs use shape ``(0, 4)``.
    threshold
        Finite, nonnegative cut height, passed to SciPy's ``distance`` criterion.
        Equality at the threshold includes the merge. The caller chooses the
        cutoff and the meaning of the original dissimilarities.

    Returns
    -------
    dict[int, list[str]]
        Groups keyed by SciPy's original integer cut labels. Groups are ordered
        by their first feature in ``names``; members retain their input order.
        Empty names return ``{}``; a singleton returns ``{1: [name]}``.
        Callers can assign their own identifiers by enumerating the groups,
        or keep the numeric labels for existing table joins.

    Notes
    -----
    Cluster labels are local to this tree and cut, not persistent identifiers
    across changed feature orders, distance matrices or thresholds.

    Average linkage uses the average distance between merging clusters. Its
    cut height does not bound every pairwise distance within a group and does
    not guarantee a minimum pairwise correlation.
    """
    if isinstance(names, str):
        raise TypeError("names must be a sequence of feature names, not a string")
    feature_names = list(names)
    if any(not isinstance(name, str) or not name for name in feature_names):
        raise ValueError("names must contain nonempty strings")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("names must be unique")
    if isinstance(threshold, bool | np.bool_) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real number")
    try:
        cut = float(threshold)
    except OverflowError as exc:
        raise ValueError("threshold must be finite and nonnegative") from exc
    if not np.isfinite(cut) or cut < 0:
        raise ValueError("threshold must be finite and nonnegative")
    matrix = _validated_linkage(linkage, len(feature_names))
    if not feature_names:
        return {}
    if len(feature_names) == 1:
        return {1: feature_names}
    labels = hierarchy.fcluster(matrix, t=cut, criterion="distance")
    groups: dict[int, list[str]] = {}
    for name, label in zip(feature_names, labels, strict=True):
        groups.setdefault(int(label), []).append(name)
    return groups
