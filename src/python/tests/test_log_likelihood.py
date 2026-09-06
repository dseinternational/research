# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contracts for likelihood aggregation independent of study and trace policy."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dse_research_utils.statistics.log_likelihood import LogLikelihoodFactor, aggregate_log_likelihood


def _array(values, *, row_dim="row", row_ids=None):
    values = np.asarray(values, dtype=float)
    coords = {"chain": [2, 5][: values.shape[0]], "draw": [10, 30, 70][: values.shape[1]]}
    if row_ids is not None:
        coords[row_dim] = row_ids
    return xr.DataArray(values, dims=("chain", "draw", row_dim), coords=coords)


def _factor(values, units, **kwargs):
    return LogLikelihoodFactor(_array(values, **kwargs), kwargs.get("row_dim", "row"), units)


def test_multiple_factors_repeated_rows_and_unit_order_conserve_draw_totals():
    first = _factor(-np.arange(1, 25).reshape(2, 3, 4), ["b", "a", "b", "c"])
    second = _factor(-np.arange(1, 13).reshape(2, 3, 2), ["a", "c"], row_dim="other_row")
    original = first.values.copy(deep=True)

    actual = aggregate_log_likelihood([first, second], unit_ids=["c", "a", "b"], unit_dim="child")

    expected = np.stack(
        [
            first.values.values[..., 3] + second.values.values[..., 1],
            first.values.values[..., 1] + second.values.values[..., 0],
            first.values.values[..., 0] + first.values.values[..., 2],
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(actual.values, expected)
    np.testing.assert_array_equal(actual.sum("child"), first.values.sum("row") + second.values.sum("other_row"))
    assert actual.dims == ("chain", "draw", "child")
    assert actual.child.values.tolist() == ["c", "a", "b"]
    xr.testing.assert_identical(actual.chain, first.values.chain)
    xr.testing.assert_identical(actual.draw, first.values.draw)
    xr.testing.assert_identical(first.values, original)


def test_masked_factors_keep_administrations_separate_from_child_units():
    administration_ids = np.array(["visit-a", "visit-b", "visit-c", "visit-d"])
    child_ids = np.array(["child-a", "child-a", "child-b", "child-c"])
    understood_mask = np.array([True, True, False, True])
    spoken_mask = np.array([False, True, True, False])
    understood = _array([[[-1, -2, -4]]])
    spoken = _array([[[-20, -30]]], row_dim="spoken_row")
    administration = aggregate_log_likelihood(
        [
            LogLikelihoodFactor(understood, "row", administration_ids[understood_mask]),
            LogLikelihoodFactor(spoken, "spoken_row", administration_ids[spoken_mask]),
        ],
        unit_ids=administration_ids,
        unit_dim="administration",
    )
    child = aggregate_log_likelihood(
        [
            LogLikelihoodFactor(understood, "row", child_ids[understood_mask]),
            LogLikelihoodFactor(spoken, "spoken_row", child_ids[spoken_mask]),
        ],
        unit_ids=["child-a", "child-b", "child-c"],
        unit_dim="child",
    )

    np.testing.assert_array_equal(administration.values, [[[-1, -22, -30, -4]]])
    np.testing.assert_array_equal(child.values, [[[-23, -30, -4]]])
    np.testing.assert_array_equal(administration.sum("administration"), child.sum("child"))


def test_only_declared_event_dimensions_are_summed_independent_of_axis_order():
    values = -np.arange(1, 97).reshape(2, 3, 4, 2, 2)
    array = xr.DataArray(
        values,
        dims=("chain", "draw", "row", "cell", "response"),
        coords={"chain": [5, 8], "draw": [2, 4, 8]},
    ).transpose("response", "row", "draw", "cell", "chain")
    factor = LogLikelihoodFactor(array, "row", [0, 1, 0, 1], event_dims=("response", "cell"))

    actual = aggregate_log_likelihood([factor], unit_ids=[0, 1], unit_dim="unit")

    np.testing.assert_array_equal(actual.values[..., 0], values[:, :, [0, 2]].sum(axis=(2, 3, 4)))
    np.testing.assert_array_equal(actual.values[..., 1], values[:, :, [1, 3]].sum(axis=(2, 3, 4)))
    np.testing.assert_array_equal(actual.sum("unit"), values.sum(axis=(2, 3, 4)))


def test_labelled_row_map_requires_exact_row_coordinates():
    array = _array([[[-1, -2, -3]]], row_ids=[50, 10, 30])
    row_map = xr.DataArray(["a", "b", "a"], dims="row", coords={"row": [50, 10, 30]})
    actual = aggregate_log_likelihood(
        [LogLikelihoodFactor(array, "row", row_map)], unit_ids=["a", "b"], unit_dim="unit"
    )
    np.testing.assert_array_equal(actual.values, [[[-4, -2]]])

    shuffled_map = row_map.isel(row=[1, 0, 2])
    with pytest.raises(ValueError, match="row map coordinate"):
        aggregate_log_likelihood(
            [LogLikelihoodFactor(array, "row", shuffled_map)], unit_ids=["a", "b"], unit_dim="unit"
        )


def test_plain_row_map_is_positional_even_when_row_coordinate_is_not_sorted():
    factor = _factor([[[-1, -2, -3]]], ["a", "b", "a"], row_ids=[50, 10, 30])
    actual = aggregate_log_likelihood([factor], unit_ids=["b", "a"], unit_dim="unit")
    np.testing.assert_array_equal(actual.values, [[[-2, -4]]])


def test_singleton_row_and_nonstandard_sample_dimension_are_preserved():
    values = xr.DataArray([[-2], [-3]], dims=("sample", "row"), coords={"sample": [100, 500]})
    actual = aggregate_log_likelihood(
        [LogLikelihoodFactor(values, "row", [9])], unit_ids=[9], unit_dim="unit", sample_dims=("sample",)
    )
    assert actual.dims == ("sample", "unit")
    assert actual.shape == (2, 1)
    np.testing.assert_array_equal(actual.values, [[-2], [-3]])
    xr.testing.assert_identical(actual["sample"], values["sample"])


def test_empty_optional_factor_does_not_create_phantom_units():
    empty = _factor(np.empty((1, 1, 0)), [])
    observed = _factor([[[-2, -3]]], ["a", "b"])
    actual = aggregate_log_likelihood([empty, observed], unit_ids=["a", "b"], unit_dim="unit")
    np.testing.assert_array_equal(actual.values, [[[-2, -3]]])


def test_sample_auxiliary_coordinates_are_retained_with_attributes():
    first = _factor([[[-1], [-2]]], ["a"])
    second = _factor([[[-3], [-4]]], ["a"])
    for factor in (first, second):
        factor.values.coords["iteration"] = ("draw", [101, 201])
        factor.values.coords["iteration"].attrs["description"] = "Original iteration number"
        factor.values.coords["draw"].attrs["thinning"] = 100
    actual = aggregate_log_likelihood([first, second], unit_ids=["a"], unit_dim="unit")
    xr.testing.assert_identical(actual.iteration, first.values.iteration)
    xr.testing.assert_identical(actual.draw, first.values.draw)


def test_sample_multiindex_metadata_is_retained():
    index = pd.MultiIndex.from_tuples([(2, 10), (2, 20), (5, 10)], names=["original_chain", "iteration"])
    array = xr.DataArray(
        [[-1], [-2], [-3]], dims=("sample", "row"), coords=xr.Coordinates.from_pandas_multiindex(index, "sample")
    )
    actual = aggregate_log_likelihood(
        [LogLikelihoodFactor(array, "row", [0])], unit_ids=[0], unit_dim="unit", sample_dims=("sample",)
    )
    assert actual.indexes["sample"].identical(index)
    xr.testing.assert_identical(actual["sample"], array["sample"])


def test_sample_multiindex_with_missing_level_is_rejected():
    index = pd.MultiIndex.from_tuples([(2, 10), (2, np.nan)], names=["original_chain", "iteration"])
    array = xr.DataArray(
        [[-1], [-2]], dims=("sample", "row"), coords=xr.Coordinates.from_pandas_multiindex(index, "sample")
    )
    with pytest.raises(ValueError, match="missing labels"):
        aggregate_log_likelihood(
            [LogLikelihoodFactor(array, "row", [0])], unit_ids=[0], unit_dim="unit", sample_dims=("sample",)
        )


def test_output_unit_dimension_does_not_replace_a_sample_auxiliary_coordinate():
    factor = _factor([[[-1], [-2]]], [0])
    factor.values.coords["unit"] = ("draw", [101, 201])
    with pytest.raises(ValueError, match="retained sample coordinate"):
        aggregate_log_likelihood([factor], unit_ids=[0], unit_dim="unit")


def test_output_unit_multiindex_level_does_not_replace_a_sample_auxiliary_coordinate():
    factor = _factor([[[-1, -2], [-3, -4]]], [(1, "a"), (2, "b")])
    factor.values.coords["iteration"] = ("draw", [101, 201])
    units = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")], names=["iteration", "visit"])
    with pytest.raises(ValueError, match="retained sample coordinate"):
        aggregate_log_likelihood([factor], unit_ids=units, unit_dim="unit")


def test_output_unit_multiindex_is_preserved_without_implicit_promotion():
    factor = _factor([[[-1, -2], [-3, -4]]], [(1, "a"), (2, "b")])
    factor.values.coords["iteration"] = ("draw", [101, 201])
    units = pd.MultiIndex.from_tuples([(2, "b"), (1, "a")], names=["child", "visit"])
    actual = aggregate_log_likelihood([factor], unit_ids=units, unit_dim="unit")
    assert actual.indexes["unit"].identical(units)
    xr.testing.assert_identical(actual.iteration, factor.values.iteration)
    np.testing.assert_array_equal(actual.values, [[[-2, -1], [-4, -3]]])


@pytest.mark.parametrize("kind", ["dtype", "attributes"])
def test_sample_coordinate_types_and_metadata_must_agree(kind):
    first = _factor([[[-1], [-2]]], [0])
    second = _factor([[[-3], [-4]]], [0])
    if kind == "dtype":
        second.values.coords["draw"] = second.values.coords["draw"].astype(float)
    else:
        second.values.coords["draw"].attrs["meaning"] = "Different sampling origin"
    with pytest.raises(ValueError, match="sample coordinates do not match"):
        aggregate_log_likelihood([first, second], unit_ids=[0], unit_dim="unit")


def test_row_map_coordinate_types_must_agree():
    array = _array([[[-1, -2]]], row_ids=[10, 20])
    row_map = xr.DataArray([0, 0], dims="row", coords={"row": [10.0, 20.0]})
    with pytest.raises(ValueError, match="row map coordinate"):
        aggregate_log_likelihood([LogLikelihoodFactor(array, "row", row_map)], unit_ids=[0], unit_dim="unit")


@pytest.mark.parametrize("axis", ["draw", "chain"])
def test_reordered_sample_coordinates_are_rejected_instead_of_realigned(axis):
    first = _factor(-np.ones((2, 3, 1)), [0])
    other = first.values.isel({axis: slice(None, None, -1)}).copy()
    with pytest.raises(ValueError, match="sample coordinates do not match"):
        aggregate_log_likelihood([first, LogLikelihoodFactor(other, "row", [0])], unit_ids=[0], unit_dim="unit")


@pytest.mark.parametrize("change", ["different_labels", "shorter", "auxiliary", "missing_auxiliary"])
def test_mismatched_sample_coordinates_are_rejected(change):
    first = _factor([[[-1], [-2]]], [0])
    second = _factor([[[-3], [-4]]], [0])
    if change == "different_labels":
        second.values.coords["draw"] = [10, 40]
    elif change == "shorter":
        second = LogLikelihoodFactor(second.values.isel(draw=[0]), "row", [0])
    else:
        first.values.coords["iteration"] = ("draw", [101, 201])
        if change == "auxiliary":
            second.values.coords["iteration"] = ("draw", [101, 202])
    with pytest.raises(ValueError, match="sample coordinates do not match"):
        aggregate_log_likelihood([first, second], unit_ids=[0], unit_dim="unit")


@pytest.mark.parametrize("coordinate", [[10, 10], [10, np.nan], [10, None]])
def test_invalid_sample_coordinate_labels_are_rejected(coordinate):
    array = _array([[[-1], [-2]]]).assign_coords(draw=coordinate)
    with pytest.raises(ValueError, match="Sample coordinate"):
        aggregate_log_likelihood([LogLikelihoodFactor(array, "row", [0])], unit_ids=[0], unit_dim="unit")


def test_missing_sample_coordinates_are_rejected():
    array = xr.DataArray([[[-1]]], dims=("chain", "draw", "row"))
    with pytest.raises(ValueError, match="explicit one-dimensional coordinate"):
        aggregate_log_likelihood([LogLikelihoodFactor(array, "row", [0])], unit_ids=[0], unit_dim="unit")


@pytest.mark.parametrize(
    ("units", "row_units", "error"),
    [
        ([], [], "unit_ids must not be empty"),
        ([1, 1], [1], "duplicate labels"),
        ([1, None], [1], "missing labels"),
        ([1, np.nan], [1], "missing labels"),
        ([1, pd.NA], [1], "missing labels"),
        ([1], [None], "missing labels"),
        ([1], [np.nan], "missing labels"),
        ([1], [2], "absent from unit_ids"),
        ([1, 2], [1], "at least one factor row"),
        ([0, 1], [False, True], "boolean masks"),
        ([False, True], [0, 1], "boolean masks"),
        ([[1]], [1], "one-dimensional"),
    ],
)
def test_invalid_unit_maps_are_rejected(units, row_units, error):
    factor = _factor(np.zeros((1, 1, len(row_units))), row_units)
    with pytest.raises((ValueError, TypeError), match=error):
        aggregate_log_likelihood([factor], unit_ids=units, unit_dim="unit")


def test_different_scalar_label_types_are_not_coerced_to_strings():
    factor = _factor([[[-1, -2]]], [1, "1"])
    actual = aggregate_log_likelihood([factor], unit_ids=["1", 1], unit_dim="unit")
    assert actual.unit.values.tolist() == ["1", 1]
    np.testing.assert_array_equal(actual.values, [[[-2, -1]]])


@pytest.mark.parametrize("field", ["unit_ids", "row_unit_ids"])
def test_masked_string_labels_are_rejected_before_pandas_can_discard_the_mask(field):
    labels = np.ma.array(["a", "b"], mask=[True, False])
    factor = _factor([[[-1, -2]]], labels if field == "row_unit_ids" else ["a", "b"])
    with pytest.raises(ValueError, match="masked labels"):
        aggregate_log_likelihood([factor], unit_ids=labels if field == "unit_ids" else ["a", "b"], unit_dim="unit")


def test_row_map_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="length does not match"):
        aggregate_log_likelihood([_factor([[[-1, -2]]], [0])], unit_ids=[0], unit_dim="unit")


@pytest.mark.parametrize("labels", [np.zeros((1, 2)), xr.DataArray([[0, 0]], dims=("a", "b"))])
def test_multidimensional_row_maps_are_rejected(labels):
    with pytest.raises(ValueError, match="one-dimensional"):
        aggregate_log_likelihood([_factor([[[-1, -2]]], labels)], unit_ids=[0], unit_dim="unit")


@pytest.mark.parametrize("kind", ["wrong_dimension", "missing_map_coordinate", "missing_factor_coordinate"])
def test_labelled_row_map_cannot_fall_back_to_position(kind):
    array = _array([[[-1, -2]]], row_ids=[10, 20])
    row_map = xr.DataArray([0, 0], dims="row", coords={"row": [10, 20]})
    if kind == "wrong_dimension":
        row_map = row_map.rename(row="other")
    elif kind == "missing_map_coordinate":
        row_map = row_map.drop_vars("row")
    else:
        array = array.drop_vars("row")
    with pytest.raises(ValueError, match=r"row dimension|explicit row coordinate"):
        aggregate_log_likelihood([LogLikelihoodFactor(array, "row", row_map)], unit_ids=[0], unit_dim="unit")


@pytest.mark.parametrize("row_ids", [[10, 10], [10, np.nan]])
def test_labelled_row_map_requires_unique_nonmissing_row_coordinates(row_ids):
    array = _array([[[-1, -2]]], row_ids=row_ids)
    row_map = xr.DataArray([0, 0], dims="row", coords={"row": row_ids})
    with pytest.raises(ValueError, match="Row coordinate"):
        aggregate_log_likelihood([LogLikelihoodFactor(array, "row", row_map)], unit_ids=[0], unit_dim="unit")


def test_missing_factors_and_duplicate_array_objects_are_rejected():
    with pytest.raises(ValueError, match="At least one"):
        aggregate_log_likelihood([], unit_ids=[0], unit_dim="unit")
    factor = _factor([[[-1]]], [0])
    same_array = LogLikelihoodFactor(factor.values, "row", [0])
    for factors in ([factor, factor], [factor, same_array]):
        with pytest.raises(ValueError, match="same factor array"):
            aggregate_log_likelihood(factors, unit_ids=[0], unit_dim="unit")


def test_distinct_same_named_factor_arrays_are_valid():
    first = _factor([[[-1]]], [0])
    second = _factor([[[-2]]], [0])
    first.values.name = second.values.name = "y_obs"
    actual = aggregate_log_likelihood([first, second], unit_ids=[0], unit_dim="unit")
    np.testing.assert_array_equal(actual.values, [[[-3]]])


def test_scalar_factor_needs_an_explicit_row_dimension():
    with pytest.raises(ValueError, match=r"Declared dimension.*absent"):
        LogLikelihoodFactor(xr.DataArray(-2.0), "row", [0])


def test_undeclared_singleton_event_dimension_is_not_silently_squeezed():
    array = _array([[[-1]]]).expand_dims(cell=[0])
    with pytest.raises(ValueError, match="Every factor dimension must be declared"):
        aggregate_log_likelihood([LogLikelihoodFactor(array, "row", [0])], unit_ids=[0], unit_dim="unit")


@pytest.mark.parametrize("event_dims", [("cell", "cell"), ("row",), ("absent",)])
def test_invalid_event_dimension_declarations_are_rejected(event_dims):
    array = _array([[[-1]]]).expand_dims(cell=[0])
    with pytest.raises(ValueError, match=r"duplicate|also be an event|absent"):
        LogLikelihoodFactor(array, "row", [0], event_dims=event_dims)


def test_empty_event_dimension_is_rejected():
    array = _array([[[-1]]]).expand_dims(cell=[])
    with pytest.raises(ValueError, match="Event dimensions must not be empty"):
        LogLikelihoodFactor(array, "row", [0], event_dims=("cell",))


def test_empty_sample_dimension_is_rejected():
    factor = _factor(np.empty((1, 0, 1)), [0])
    with pytest.raises(ValueError, match="Sample dimensions must not be empty"):
        aggregate_log_likelihood([factor], unit_ids=[0], unit_dim="unit")


@pytest.mark.parametrize("sample_dims", [(), ("draw", "draw"), ("chain", "draw", "row")])
def test_invalid_sample_dimension_declarations_are_rejected(sample_dims):
    with pytest.raises(ValueError, match=r"empty|duplicate|distinct roles"):
        aggregate_log_likelihood([_factor([[[-1]]], [0])], unit_ids=[0], unit_dim="unit", sample_dims=sample_dims)


def test_output_unit_dimension_must_not_replace_a_sample_dimension():
    with pytest.raises(ValueError, match="must differ"):
        aggregate_log_likelihood([_factor([[[-1]]], [0])], unit_ids=[0], unit_dim="draw")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_invalid_log_likelihood_values_are_not_skipped(bad):
    array = _array([[[-1, bad]]]).expand_dims(cell=[0])
    factor = LogLikelihoodFactor(array, "row", [0, 0], event_dims=("cell",))
    with pytest.raises(ValueError, match="NaN or positive infinity"):
        aggregate_log_likelihood([factor], unit_ids=[0], unit_dim="unit")


def test_impossible_observation_remains_negative_infinity_after_event_and_factor_sums():
    array = xr.DataArray(
        [[[[-1, -np.inf], [-2, -3]]]],
        dims=("chain", "draw", "row", "cell"),
        coords={"chain": [2], "draw": [10]},
    )
    factor = LogLikelihoodFactor(array, "row", [0, 1], event_dims=("cell",))
    other = _factor([[[-5, -6]]], [0, 1])
    actual = aggregate_log_likelihood([factor, other], unit_ids=[0, 1], unit_dim="unit")
    np.testing.assert_array_equal(actual.values, [[[-np.inf, -11]]])


def test_float32_values_are_widened_before_event_reduction():
    array = xr.DataArray(
        np.array([[[[-1e8, -1, -1]]]], dtype=np.float32),
        dims=("chain", "draw", "row", "cell"),
        coords={"chain": [2], "draw": [10]},
    )
    factor = LogLikelihoodFactor(array, "row", [0], event_dims=("cell",))
    actual = aggregate_log_likelihood([factor], unit_ids=[0], unit_dim="unit")
    assert actual.dtype == np.float64
    assert actual.item() == -100000002.0
    assert array.values.sum(axis=-1).item() == -100000000.0


@pytest.mark.parametrize("stage", ["events", "rows", "factors"])
@pytest.mark.parametrize("value", [-1e308, 1e308])
def test_floating_point_overflow_is_rejected(stage, value):
    if stage == "events":
        array = _array([[[value]]]).expand_dims(cell=[0, 1])
        factors = [LogLikelihoodFactor(array, "row", [0], event_dims=("cell",))]
    elif stage == "rows":
        factors = [_factor([[[value, value]]], [0, 0])]
    else:
        factors = [_factor([[[value]]], [0]), _factor([[[value]]], [0])]
    with pytest.raises(ValueError, match="overflowed"):
        aggregate_log_likelihood(factors, unit_ids=[0], unit_dim="unit")


@pytest.mark.parametrize("values", [[[["-1"]]], [[[-1 + 2j]]], [[[True]]]])
def test_non_real_numeric_values_are_rejected(values):
    array = xr.DataArray(values, dims=("chain", "draw", "row"), coords={"chain": [0], "draw": [0]})
    with pytest.raises(TypeError, match="real numbers"):
        aggregate_log_likelihood([LogLikelihoodFactor(array, "row", [0])], unit_ids=[0], unit_dim="unit")
