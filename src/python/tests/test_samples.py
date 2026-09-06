# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dse_research_utils.statistics.samples import sample_matrix


def test_multidimensional_values_follow_declared_coordinate_order():
    source = xr.DataArray(
        np.arange(48).reshape(2, 3, 2, 4),
        dims=("visit", "draw", "child", "chain"),
        coords={"visit": [5, 1], "draw": [10, 2, 7], "child": ["b", "a"], "chain": [4, 3, 2, 1]},
    )
    original = source.copy(deep=True)
    result = sample_matrix(source, sample_dims=("chain", "draw"), observation_dims=("child", "visit"))

    assert result.sample_index.tolist() == [(c, d) for c in [4, 3, 2, 1] for d in [10, 2, 7]]
    assert result.observation_index.tolist() == [(c, v) for c in ["b", "a"] for v in [5, 1]]
    for row, (child, visit) in enumerate(result.observation_index):
        for column, (chain, draw) in enumerate(result.sample_index):
            assert result.values[row, column] == source.sel(child=child, visit=visit, chain=chain, draw=draw).item()

    observed = xr.DataArray(
        [[4.0, np.nan], [6.0, 7.0]],
        dims=("visit", "child"),
        coords={"visit": source.visit, "child": source.child},
    )
    np.testing.assert_equal(result.observed_values(observed), [4.0, 6.0, np.nan, 7.0])
    xr.testing.assert_identical(source, original)


def test_draw_only_and_names_that_would_collide_with_stack_output():
    source = xr.DataArray(
        [[7, 8, 9], [10, 11, 12]],
        dims=("sample", "observation"),
        coords={"sample": [9, 1], "observation": ["c", "a", "b"], "__sample__": 42},
    )
    result = sample_matrix(source, sample_dims=("sample",), observation_dims=("observation",))
    np.testing.assert_equal(result.values, [[7, 10], [8, 11], [9, 12]])
    pd.testing.assert_index_equal(result.sample_index, source.indexes["sample"])
    pd.testing.assert_index_equal(result.observation_index, source.indexes["observation"])
    assert np.shares_memory(result.values, source.values)


def test_previously_stacked_samples_preserve_their_original_multiindex():
    source = xr.DataArray(
        np.arange(12).reshape(2, 3, 2),
        dims=("chain", "draw", "observation"),
        coords={"chain": [3, 0], "draw": [7, 1, 5], "observation": ["b", "a"]},
    ).stack(sample=("draw", "chain"))
    result = sample_matrix(source, sample_dims=("sample",), observation_dims=("observation",))
    pd.testing.assert_index_equal(result.sample_index, source.indexes["sample"])
    np.testing.assert_equal(result.values, source.transpose("observation", "sample").values)


def test_multidimensional_index_retains_categorical_and_datetime_metadata():
    visits = pd.DatetimeIndex(["2026-09-06", "2026-08-01"], tz="Europe/London", name="visit")
    children = pd.CategoricalIndex(["b", "a"], categories=["c", "b", "a"], ordered=True, name="child")
    source = xr.DataArray(
        np.arange(8).reshape(2, 2, 2),
        dims=("draw", "visit", "child"),
        coords={"draw": [1, 0], "visit": visits, "child": children},
    )
    result = sample_matrix(source, sample_dims=("draw",), observation_dims=("visit", "child"))
    pd.testing.assert_index_equal(result.observation_index, pd.MultiIndex.from_product([visits, children]))
    np.testing.assert_equal(result.observed_values(source.isel(draw=0, drop=True)), [0, 1, 2, 3])


def test_previously_stacked_dimension_can_be_combined_with_another_sample_dimension():
    source = xr.DataArray(
        np.arange(12).reshape(2, 3, 2),
        dims=("chain", "draw", "replicate"),
        coords={"chain": [3, 0], "draw": [7, 1, 5], "replicate": ["b", "a"]},
    ).stack(sample=("draw", "chain"))
    result = sample_matrix(source, sample_dims=("sample", "replicate"), observation_dims=())
    assert result.sample_index.tolist() == [
        ((draw, chain), replicate) for draw in [7, 1, 5] for chain in [3, 0] for replicate in ["b", "a"]
    ]
    for column, ((draw, chain), replicate) in enumerate(result.sample_index):
        assert result.values[0, column] == source.sel(draw=draw, chain=chain, replicate=replicate).item()


def test_scalar_posterior_has_one_observation_and_accepts_scalar_observed_value():
    source = xr.DataArray([np.nan, np.inf, -np.inf, 3], dims="draw", coords={"draw": [1, 3, 7, 9]})
    result = sample_matrix(source, sample_dims=("draw",), observation_dims=())
    np.testing.assert_equal(result.values, [[np.nan, np.inf, -np.inf, 3]])
    np.testing.assert_equal(result.observed_values(xr.DataArray(np.nan)), [np.nan])
    assert result.observation_dims == ()
    assert result.observation_index.equals(pd.RangeIndex(1))


@pytest.mark.parametrize(
    ("sample_dims", "observation_dims", "message"),
    [
        ((), ("row", "draw"), "at least one"),
        (("draw", "draw"), ("row",), "repeat"),
        (("draw",), ("row", "row"), "repeat"),
        (("draw", "row"), ("row",), "overlap"),
        (("draw",), (), "declare every"),
        (("draw",), ("row", "event"), "declare every"),
        (("draw",), ("",), "non-empty string"),
        (("draw",), (3,), "non-empty string"),
    ],
)
def test_invalid_dimension_declarations_are_rejected(sample_dims, observation_dims, message):
    source = xr.DataArray([[1, 2]], dims=("row", "draw"), coords={"row": ["a"], "draw": [1, 2]})
    with pytest.raises(ValueError, match=message):
        sample_matrix(source, sample_dims=sample_dims, observation_dims=observation_dims)


def test_bare_string_dimension_sequence_is_rejected():
    source = xr.DataArray([1], dims="draw", coords={"draw": [1]})
    with pytest.raises(TypeError, match="sequence"):
        sample_matrix(source, sample_dims="draw", observation_dims=())


def test_undeclared_likelihood_event_dimension_is_not_flattened():
    source = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("draw", "row", "event"),
        coords={"draw": [0, 1], "row": ["c", "a", "b"], "event": [1, 2, 3, 4]},
    )
    with pytest.raises(ValueError, match="declare every"):
        sample_matrix(source, sample_dims=("draw",), observation_dims=("row",))


@pytest.mark.parametrize("dimension", ["row", "draw"])
@pytest.mark.parametrize("labels", [[], [1, 1], [1, np.nan]])
def test_empty_duplicate_and_missing_coordinates_are_rejected(dimension, labels):
    sizes = {"draw": [0, 1], "row": ["b", "a"]}
    sizes[dimension] = labels
    source = xr.DataArray(np.zeros((len(sizes["row"]), len(sizes["draw"]))), dims=("row", "draw"), coords=sizes)
    with pytest.raises(ValueError, match=r"empty|unique|missing"):
        sample_matrix(source, sample_dims=("draw",), observation_dims=("row",))


@pytest.mark.parametrize("dimension", ["row", "draw"])
def test_implicit_positional_coordinates_are_rejected(dimension):
    source = xr.DataArray([[1, 2]], dims=("row", "draw"), coords={"row": [0], "draw": [0, 1]})
    source = source.drop_vars(dimension)
    with pytest.raises(ValueError, match="explicit"):
        sample_matrix(source, sample_dims=("draw",), observation_dims=("row",))


def test_equal_shape_does_not_establish_observed_alignment():
    source = xr.DataArray([[1, 2], [3, 4]], dims=("row", "draw"), coords={"row": [0, 1], "draw": [0, 1]})
    result = sample_matrix(source, sample_dims=("draw",), observation_dims=("row",))
    for observed in (
        xr.DataArray([3, 2], dims="row", coords={"row": [1, 0]}),
        xr.DataArray([3, 2], dims="row", coords={"row": [0, 2]}),
        xr.DataArray([3, 2], dims="row", coords={"row": [0.0, 1.0]}),
    ):
        with pytest.raises(ValueError, match="does not match"):
            result.observed_values(observed)
    with pytest.raises(ValueError, match="explicit"):
        result.observed_values(xr.DataArray([3, 2], dims="row"))
    with pytest.raises(ValueError, match="exactly"):
        result.observed_values(xr.DataArray([3, 2], dims="other"))


@pytest.mark.parametrize("labels", [[0, 0], [0, np.nan]])
def test_ambiguous_observed_coordinate_labels_are_rejected(labels):
    source = xr.DataArray([[1, 2], [3, 4]], dims=("row", "draw"), coords={"row": [0, 1], "draw": [0, 1]})
    result = sample_matrix(source, sample_dims=("draw",), observation_dims=("row",))
    with pytest.raises(ValueError, match=r"unique|missing"):
        result.observed_values(xr.DataArray([3, 2], dims="row", coords={"row": labels}))


def test_already_stacked_observation_identity_is_checked():
    source = xr.DataArray(
        np.arange(8).reshape(2, 2, 2),
        dims=("draw", "child", "visit"),
        coords={"draw": [2, 0], "child": ["b", "a"], "visit": [3, 1]},
    ).stack(row=("visit", "child"))
    result = sample_matrix(source, sample_dims=("draw",), observation_dims=("row",))
    observed = source.isel(draw=0, drop=True)
    np.testing.assert_equal(result.observed_values(observed), result.values[:, 0])
    with pytest.raises(ValueError, match="does not match"):
        result.observed_values(observed.isel(row=[1, 0, 2, 3]))


def test_missing_label_in_stacked_index_is_rejected():
    source = xr.DataArray(np.ones((2, 2)), dims=("chain", "draw"), coords={"chain": [1, np.nan], "draw": [0, 1]}).stack(
        sample=("chain", "draw")
    )
    with pytest.raises(ValueError, match="missing"):
        sample_matrix(source, sample_dims=("sample",), observation_dims=())


def test_coordinate_metadata_is_detached_from_later_source_changes():
    source = xr.DataArray([[1, 2]], dims=("row", "draw"), coords={"row": [0], "draw": [0, 1]})
    result = sample_matrix(source, sample_dims=("draw",), observation_dims=("row",))
    source.coords["row"] = ("row", [7])
    assert result.observation_index.tolist() == [0]
    np.testing.assert_equal(result.observed_values(xr.DataArray([4], dims="row", coords={"row": [0]})), [4])
    with pytest.raises(ValueError, match="does not match"):
        result.observed_values(xr.DataArray([4], dims="row", coords={"row": [7]}))
