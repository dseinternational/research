# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Feature grouping contracts, including the downstream table adapters."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from dse_research_utils.ml.feature_dependence import distance_corr_dissimilarity_linkage
from dse_research_utils.ml.feature_groups import feature_groups_from_linkage, linkage_from_dissimilarity


def _dissimilarity():
    matrix = np.full((5, 5), 0.9)
    np.fill_diagonal(matrix, 0)
    matrix[0, 2] = matrix[2, 0] = 0.2
    matrix[1, 3] = matrix[3, 1] = 0.1
    return matrix


@pytest.mark.parametrize("method", ["single", "complete", "average", "weighted"])
def test_linkage_matches_scipy_and_preserves_input(method):
    matrix = _dissimilarity()
    before = matrix.copy()
    actual = linkage_from_dissimilarity(matrix, method=method)
    expected = hierarchy.linkage(squareform(matrix), method=method)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(matrix, before)


def test_average_linkage_known_small_tree():
    actual = linkage_from_dissimilarity([[0, 0.2, 0.8], [0.2, 0, 0.7], [0.8, 0.7, 0]], method="average")
    np.testing.assert_array_equal(actual, [[0, 1, 0.2, 2], [2, 3, 0.75, 3]])


@pytest.mark.parametrize("n_features", [0, 1])
def test_zero_and_one_feature_outputs(n_features):
    linkage = linkage_from_dissimilarity(np.zeros((n_features, n_features)), method="average")
    assert linkage.shape == (0, 4)
    assert linkage.dtype == np.float64
    names = ["only"] if n_features else []
    assert feature_groups_from_linkage(names, linkage, threshold=0.3) == ({1: ["only"]} if n_features else {})


@pytest.mark.parametrize("n_features", [0, 1])
def test_existing_raw_data_helper_handles_zero_and_one_feature(n_features):
    pytest.importorskip("dcor")
    dissimilarity, condensed, linkage = distance_corr_dissimilarity_linkage(np.ones((10, n_features)))
    assert dissimilarity.shape == (n_features, n_features)
    assert condensed.shape == (0,)
    assert linkage.shape == (0, 4)


def test_existing_raw_data_helper_agrees_with_reusing_its_distance_matrix():
    pytest.importorskip("dcor")
    rng = np.random.default_rng(49)
    x = rng.normal(size=(40, 3))
    dissimilarity, condensed, linkage = distance_corr_dissimilarity_linkage(x)
    np.testing.assert_array_equal(condensed, squareform(dissimilarity))
    np.testing.assert_array_equal(linkage, hierarchy.average(condensed))
    np.testing.assert_array_equal(linkage, linkage_from_dissimilarity(dissimilarity, method="average"))


def test_feature_order_is_retained_without_renumbering_scipy_labels():
    names = ["z", "a", "q", "b", "m"]
    linkage = linkage_from_dissimilarity(_dissimilarity(), method="average")
    before = linkage.copy()
    groups = feature_groups_from_linkage(names, linkage, threshold=0.3)
    assert list(groups.items()) == [(2, ["z", "q"]), (1, ["a", "b"]), (3, ["m"])]
    np.testing.assert_array_equal(linkage, before)
    assert names == ["z", "a", "q", "b", "m"]


def test_reordering_features_and_the_matrix_keeps_membership_with_new_input_order():
    names = np.array(["z", "a", "q", "b", "m"])
    order = np.array([4, 2, 0, 3, 1])
    matrix = _dissimilarity()[np.ix_(order, order)]
    groups = feature_groups_from_linkage(
        names[order], linkage_from_dissimilarity(matrix, method="average"), threshold=0.3
    )
    assert list(groups.values()) == [["m"], ["q", "z"], ["b", "a"]]


def test_usbc_adapter_retains_first_member_group_ids_and_member_order():
    groups = feature_groups_from_linkage(
        ["z", "a", "q", "b", "m"], linkage_from_dissimilarity(_dissimilarity(), method="average"), threshold=0.3
    )
    usbc_groups = {f"cluster_{number:02d}": members for number, members in enumerate(groups.values(), start=1)}
    assert usbc_groups == {"cluster_01": ["z", "q"], "cluster_02": ["a", "b"], "cluster_03": ["m"]}


def test_lrp_adapter_preserves_numeric_ids_for_feature_and_cluster_importance_joins():
    names = ["z", "a", "q", "b", "m"]
    linkage = linkage_from_dissimilarity(_dissimilarity(), method="average")
    groups = feature_groups_from_linkage(names, linkage, threshold=0.4)
    cluster_table = (
        pd.DataFrame(
            [(name, label) for label, members in groups.items() for name in members], columns=["feature", "cluster_id"]
        )
        .astype({"cluster_id": np.int32})
        .sort_values(["cluster_id", "feature"], ignore_index=True)
    )
    expected = pd.DataFrame(
        {"feature": ["a", "b", "q", "z", "m"], "cluster_id": np.array([1, 1, 2, 2, 3], dtype=np.int32)}
    )
    pd.testing.assert_frame_equal(cluster_table, expected)
    importance = pd.DataFrame({"feature": ["m", "q", "b", "a", "z"], "importance": [5, 4, 3, 2, 1]})
    cluster_importance = pd.DataFrame({"cluster_id": [3, 1, 2], "group_importance": [10, 20, 30]})
    joined = cluster_table.merge(importance, on="feature").merge(cluster_importance, on="cluster_id")
    assert joined.set_index("feature")["group_importance"].to_dict() == {"a": 20, "b": 20, "q": 30, "z": 30, "m": 10}
    assert joined.set_index("feature")["importance"].to_dict() == dict(
        zip(importance.feature, importance.importance, strict=True)
    )


def test_average_linkage_cut_does_not_bound_all_pairwise_distances():
    matrix = np.array([[0, 0.1, 0.2], [0.1, 0, 0.8], [0.2, 0.8, 0]])
    linkage = linkage_from_dissimilarity(matrix, method="average")
    # The final merge averages 0.2 and 0.8, despite the 0.8 pair inside the group.
    assert feature_groups_from_linkage(["a", "b", "c"], linkage, threshold=0.5) == {1: ["a", "b", "c"]}
    assert max(matrix.ravel()) > 0.5
    assert len(feature_groups_from_linkage(["a", "b", "c"], linkage, threshold=np.nextafter(0.5, 0))) == 2


def test_zero_cut_height_includes_zero_distance_merges():
    linkage = linkage_from_dissimilarity(np.zeros((3, 3)), method="average")
    assert feature_groups_from_linkage(["a", "b", "c"], linkage, threshold=0) == {1: ["a", "b", "c"]}


def test_general_dissimilarities_may_exceed_one_and_violate_triangle_inequality():
    matrix = np.array([[0, 2, 10], [2, 0, 3], [10, 3, 0]])
    actual = linkage_from_dissimilarity(matrix, method="average")
    np.testing.assert_array_equal(actual, [[0, 1, 2, 2], [2, 3, 6.5, 3]])


@pytest.mark.parametrize("method", ["ward", "centroid", "median", "unknown", "Average", None])
def test_unsupported_methods_are_rejected_even_for_singletons(method):
    with pytest.raises(ValueError, match="Euclidean-only"):
        linkage_from_dissimilarity([[0]], method=method)


@pytest.mark.parametrize(
    ("matrix", "error"),
    [
        ([], "two-dimensional"),
        ([[0, 1]], "square"),
        ([[1]], "zero diagonal"),
        ([[0, -1], [-1, 0]], "nonnegative"),
        ([[0, 1], [2, 0]], "symmetric"),
        ([[0, np.nan], [np.nan, 0]], "finite"),
        ([[0, np.inf], [np.inf, 0]], "finite"),
        ([[0, -np.inf], [-np.inf, 0]], "finite"),
        (np.ma.array([[0, 1], [1, 0]], mask=[[False, True], [True, False]]), "masked"),
    ],
)
def test_invalid_dissimilarities_are_rejected_without_cleanup(matrix, error):
    with pytest.raises(ValueError, match=error):
        linkage_from_dissimilarity(matrix, method="average")


def test_symmetry_is_validated_before_float64_conversion_can_hide_a_difference():
    matrix = np.array([[0, 2**60], [2**60 + 1, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="symmetric"):
        linkage_from_dissimilarity(matrix, method="average")


@pytest.mark.parametrize("matrix", [[[False, True], [True, False]], [["0"]], [[0j]]])
def test_nonnumeric_or_boolean_matrices_are_rejected(matrix):
    with pytest.raises(TypeError, match="real numeric"):
        linkage_from_dissimilarity(matrix, method="average")


@pytest.mark.parametrize(
    ("matrix", "error"),
    [
        ([[0, 1, 0.1]], "four columns"),
        (np.empty((0, 4)), "fewer row"),
        ([[0, 1, -0.1, 2], [2, 3, 0.5, 3]], "heights"),
        ([[0, 1, 0.5, 2], [2, 3, 0.1, 3]], "heights"),
        ([[0, 1, np.nan, 2], [2, 3, 0.5, 3]], "finite"),
        ([[0, 1, np.inf, 2], [2, 3, 0.5, 3]], "finite"),
        ([[-1, 1, 0.1, 2], [2, 3, 0.5, 3]], "existing leaves"),
        ([[0, 3, 0.1, 2], [2, 3, 0.5, 3]], "existing leaves"),
        ([[0, 4, 0.1, 2], [2, 3, 0.5, 3]], "existing leaves"),
        ([[0.5, 1, 0.1, 2], [2, 3, 0.5, 3]], "integers"),
        ([[0, 0, 0.1, 2], [2, 3, 0.5, 3]], "reuse"),
        ([[0, 1, 0.1, 2], [0, 3, 0.5, 3]], "reuse"),
        ([[0, 1, 0.1, 3], [2, 3, 0.5, 3]], "member counts"),
        ([[0, 1, 0.1, 2], [2, 3, 0.5, 2.5]], "member counts"),
        ([[0, 1, 0.1, 2], [2, 3, 0.5, 2]], "member counts"),
    ],
)
def test_malformed_linkage_is_rejected_before_tree_cutting(matrix, error):
    with pytest.raises(ValueError, match=error):
        feature_groups_from_linkage(["a", "b", "c"], matrix, threshold=0.3)


def test_reusing_an_already_merged_cluster_is_rejected():
    matrix = [[0, 1, 0.1, 2], [2, 4, 0.2, 3], [3, 4, 0.3, 3]]
    with pytest.raises(ValueError, match="reuse"):
        feature_groups_from_linkage(["a", "b", "c", "d"], matrix, threshold=0.3)


@pytest.mark.parametrize("names", [["a", "a"], [""], [None], [1]])
def test_invalid_names_are_rejected(names):
    with pytest.raises(ValueError, match="names"):
        feature_groups_from_linkage(names, np.empty((0, 4)), threshold=0.3)


@pytest.mark.parametrize("names", [[], ["only"]])
def test_degenerate_names_still_require_an_empty_linkage(names):
    with pytest.raises(ValueError, match="four columns"):
        feature_groups_from_linkage(names, [[0, 1, 0.1, 2]], threshold=0.3)


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, -np.inf, 10**400])
def test_invalid_cut_heights_are_rejected_even_for_empty_inputs(threshold):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        feature_groups_from_linkage([], np.empty((0, 4)), threshold=threshold)


@pytest.mark.parametrize("threshold", [True, np.bool_(False), "0.3", None, 1j])
def test_cut_heights_must_be_real_numbers(threshold):
    with pytest.raises(TypeError, match="real number"):
        feature_groups_from_linkage([], np.empty((0, 4)), threshold=threshold)


def test_clustering_precomputed_distances_does_not_need_optional_dependence_imports():
    code = """
import sys
sys.modules['dcor'] = None
from dse_research_utils.ml.feature_groups import feature_groups_from_linkage, linkage_from_dissimilarity
tree = linkage_from_dissimilarity([[0, .2], [.2, 0]], method='average')
assert feature_groups_from_linkage(['a', 'b'], tree, threshold=.3) == {1: ['a', 'b']}
assert sys.modules['dcor'] is None
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
