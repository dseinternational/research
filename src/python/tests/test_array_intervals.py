# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Axis, precision and missing-value contracts for array equal-tailed intervals."""

import warnings

import numpy as np
import pytest

from dse_research_utils.statistics.array_intervals import equal_tail_interval
from dse_research_utils.statistics.intervals import bands, eti_1d, summarise_bands


def test_known_linear_quantiles_return_zero_dimensional_float_arrays():
    lower, upper = equal_tail_interval([0, 1, 2, 3, 4], prob=0.5, axis=None, nonfinite="raise")
    assert lower.shape == upper.shape == ()
    assert lower.dtype == upper.dtype == np.float64
    assert lower == 1
    assert upper == 3


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3, (0, 2), (2, 0), (-1, 0), None])
@pytest.mark.parametrize("keepdims", [False, True])
def test_arbitrary_axes_match_known_numpy_quantiles_and_retain_shape(axis, keepdims):
    samples = np.arange(60.0).reshape(3, 4, 5)
    before = samples.copy()
    lower, upper = equal_tail_interval(samples, prob=0.8, axis=axis, nonfinite="raise", keepdims=keepdims)
    tail = (1 - 0.8) / 2
    np.testing.assert_array_equal(lower, np.quantile(samples, tail, axis=axis, keepdims=keepdims))
    np.testing.assert_array_equal(upper, np.quantile(samples, 1 - tail, axis=axis, keepdims=keepdims))
    np.testing.assert_array_equal(samples, before)


def test_chain_and_draw_reduction_does_not_merge_observation_distributions():
    samples = np.array([[[1, 101], [3, 103]], [[5, 105], [7, 107]]])
    lower, upper = equal_tail_interval(samples, prob=0.5, axis=(0, 1), nonfinite="raise")
    np.testing.assert_array_equal(lower, [2.5, 102.5])
    np.testing.assert_array_equal(upper, [5.5, 105.5])


def test_noncontiguous_axis_permutation_preserves_intervals():
    samples = np.arange(120.0).reshape(2, 3, 4, 5)
    expected = equal_tail_interval(samples, prob=0.89, axis=(0, 3), nonfinite="raise")
    moved = samples.transpose(3, 1, 0, 2)
    actual = equal_tail_interval(moved, prob=0.89, axis=(2, 0), nonfinite="raise")
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)


def test_keepdims_bounds_broadcast_against_input():
    samples = np.arange(120.0).reshape(2, 3, 4, 5)
    lower, upper = equal_tail_interval(samples, prob=0.89, axis=(3, 0), nonfinite="raise", keepdims=True)
    assert lower.shape == upper.shape == (1, 3, 4, 1)
    assert ((samples >= lower) & (samples <= upper)).shape == samples.shape


@pytest.mark.parametrize("policy", ["propagate", "omit_nan", "omit_nonfinite", "raise"])
def test_scalar_input_is_one_sample(policy):
    lower, upper = equal_tail_interval(4.0, prob=0.89, axis=None, nonfinite=policy)
    assert lower.shape == upper.shape == ()
    assert lower == upper == 4


@pytest.mark.parametrize("policy", ["propagate", "omit_nan", "omit_nonfinite"])
def test_all_missing_slices_return_nan_without_warnings(policy):
    samples = np.array([[np.nan, np.nan, np.nan], [1, 2, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lower, upper = equal_tail_interval(samples, prob=0.5, axis=1, nonfinite=policy)
    assert np.isnan(lower[0]) and np.isnan(upper[0])
    if policy == "propagate":
        assert np.isnan(lower[1]) and np.isnan(upper[1])
    else:
        assert lower[1] == 1.25
        assert upper[1] == 1.75


def test_nan_omission_and_all_nonfinite_omission_are_distinct():
    samples = [-np.inf, 1, 2, 3, np.inf, np.nan]
    propagated = equal_tail_interval(samples, prob=0.5, axis=None, nonfinite="propagate")
    omitted_nan = equal_tail_interval(samples, prob=0.5, axis=None, nonfinite="omit_nan")
    finite_only = equal_tail_interval(samples, prob=0.5, axis=None, nonfinite="omit_nonfinite")
    assert all(np.isnan(bound) for bound in propagated)
    assert omitted_nan[0] == 1
    assert np.isnan(omitted_nan[1])
    assert finite_only == (1.5, 2.5)


@pytest.mark.parametrize("samples", [[np.inf], [-np.inf], [0, np.inf], [-np.inf, 0], [-np.inf, np.inf], [1, 2, np.inf]])
@pytest.mark.parametrize("prob", [0.5, 0.89, 1.0])
@pytest.mark.parametrize("policy", ["propagate", "omit_nan"])
def test_retained_infinities_follow_numpy_interpolation_without_warnings(samples, prob, policy):
    tail = (1 - prob) / 2
    with np.errstate(invalid="ignore"):
        expected = np.quantile(np.asarray(samples, dtype=float), [tail, 1 - tail])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = equal_tail_interval(samples, prob=prob, axis=None, nonfinite=policy)
    np.testing.assert_equal(actual, expected)


def test_all_nonfinite_omission_leaves_no_artificial_finite_bound():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lower, upper = equal_tail_interval([np.nan, np.inf, -np.inf], prob=1, axis=None, nonfinite="omit_nonfinite")
    assert np.isnan(lower) and np.isnan(upper)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_raise_policy_rejects_any_nonfinite_value(value):
    with pytest.raises(ValueError, match="must be finite"):
        equal_tail_interval([[1, 2], [3, value]], prob=0.89, axis=1, nonfinite="raise")


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 3)])
@pytest.mark.parametrize("policy", ["propagate", "omit_nan", "omit_nonfinite", "raise"])
def test_empty_sample_axes_return_nan_for_each_remaining_position(shape, policy):
    samples = np.empty(shape)
    axis = shape.index(0)
    lower, upper = equal_tail_interval(samples, prob=0.89, axis=axis, nonfinite=policy)
    assert lower.shape == upper.shape == tuple(length for dim, length in enumerate(shape) if dim != axis)
    assert np.isnan(lower).all() and np.isnan(upper).all()


def test_no_remaining_positions_returns_empty_bounds_with_keepdims():
    lower, upper = equal_tail_interval(np.empty((0, 3)), prob=1, axis=1, nonfinite="raise", keepdims=True)
    assert lower.shape == upper.shape == (0, 1)


def test_empty_all_axis_reduction_returns_scalar_nan():
    lower, upper = equal_tail_interval(np.empty((2, 0, 3)), prob=1, axis=None, nonfinite="raise")
    assert lower.shape == upper.shape == ()
    assert np.isnan(lower) and np.isnan(upper)


def test_full_probability_returns_finite_range():
    actual = equal_tail_interval([9, 3, 6, 2], prob=1, axis=None, nonfinite="raise")
    assert actual == (2, 9)


def test_small_positive_probability_is_valid():
    actual = equal_tail_interval([0, 10], prob=np.nextafter(0.0, 1.0), axis=None, nonfinite="raise")
    assert actual == (5, 5)


@pytest.mark.parametrize("prob", [0, -0.1, 1.01, np.nan, np.inf, -np.inf, 10**400])
def test_invalid_probabilities_are_rejected_even_for_empty_inputs(prob):
    with pytest.raises(ValueError, match="prob must"):
        equal_tail_interval([], prob=prob, axis=None, nonfinite="omit_nonfinite")


@pytest.mark.parametrize("prob", [True, "0.89", None, 0.5j])
def test_probability_must_be_a_real_number(prob):
    with pytest.raises(TypeError, match="real number"):
        equal_tail_interval([], prob=prob, axis=None, nonfinite="omit_nonfinite")


@pytest.mark.parametrize("axis", [(), (0, 0), (1, -1), 2, -3])
def test_invalid_axes_are_rejected_even_for_empty_arrays(axis):
    with pytest.raises(ValueError, match="axis"):
        equal_tail_interval(np.empty((0, 3)), prob=0.89, axis=axis, nonfinite="propagate")


@pytest.mark.parametrize("axis", [True, 0.0, [0], (False,), (1.0,)])
def test_axis_requires_integers(axis):
    with pytest.raises(TypeError, match="axis"):
        equal_tail_interval(np.empty((0, 3)), prob=0.89, axis=axis, nonfinite="propagate")


def test_scalar_axis_must_be_none():
    with pytest.raises(ValueError, match="out of range"):
        equal_tail_interval(1.0, prob=0.89, axis=0, nonfinite="raise")


@pytest.mark.parametrize("policy", ["omit", "ignore", "finite", None, []])
def test_unknown_nonfinite_policy_is_rejected_even_for_empty_arrays(policy):
    with pytest.raises(ValueError, match="nonfinite must"):
        equal_tail_interval([], prob=0.89, axis=None, nonfinite=policy)


@pytest.mark.parametrize("keepdims", [0, 1, "true", None])
def test_keepdims_requires_boolean(keepdims):
    with pytest.raises(TypeError, match="keepdims"):
        equal_tail_interval([], prob=0.89, axis=None, nonfinite="raise", keepdims=keepdims)


@pytest.mark.parametrize("values", [[True, False], [1 + 2j], ["1", "2"], np.array([1], dtype=object)])
def test_values_must_have_real_numeric_dtype(values):
    with pytest.raises(TypeError, match="real numeric"):
        equal_tail_interval(values, prob=0.89, axis=None, nonfinite="raise")


@pytest.mark.parametrize("mask", [[False, True], [False, False]])
def test_masked_arrays_require_explicit_conversion(mask):
    with pytest.raises(ValueError, match="masked array"):
        equal_tail_interval(np.ma.array([1.0, 2.0], mask=mask), prob=0.89, axis=None, nonfinite="omit_nan")


def test_float32_is_converted_before_quantile_calculation():
    samples = np.array([0, 1, 3, 5], dtype=np.float32)
    lower, upper = equal_tail_interval(samples, prob=0.89, axis=None, nonfinite="raise")
    tail = (1 - 0.89) / 2
    assert lower.dtype == upper.dtype == np.float64
    assert lower == np.quantile(samples.astype(float), tail)
    assert upper == np.quantile(samples.astype(float), 1 - tail)
    assert lower != np.quantile(samples, tail)


def test_large_signed_integer_difference_does_not_overflow_integer_arithmetic():
    samples = np.array([-(2**63), 2**63 - 1], dtype=np.int64)
    lower, upper = equal_tail_interval(samples, prob=0.5, axis=None, nonfinite="raise")
    assert lower == -(2**62)
    assert upper == 2**62


@pytest.mark.parametrize("policy", ["propagate", "omit_nan", "omit_nonfinite", "raise"])
def test_finite_quantile_overflow_raises_instead_of_returning_reversed_bounds(policy):
    with pytest.raises(ValueError, match="Quantile arithmetic overflowed"):
        equal_tail_interval([-1e308, 1e308], prob=0.89, axis=None, nonfinite=policy)


def test_interval_omission_does_not_choose_a_mean_or_median_summary():
    draws = np.array([0, 0, 0, 12.0])
    lower, upper = equal_tail_interval(draws, prob=0.5, axis=None, nonfinite="raise")
    mean_summary = {"mean": float(draws.mean()), "lo": float(lower), "hi": float(upper)}
    assert mean_summary == {"mean": 3, "lo": 0, "hi": 3}
    median_summary = summarise_bands(draws, np.array([1]), outer=0.5)
    assert median_summary.loc[0, "median"] == 0
    assert median_summary.loc[0, "ci_lo"] == lower
    assert median_summary.loc[0, "ci_hi"] == upper


def test_existing_intervals_keep_finite_filtering_full_coverage_and_median_contract():
    draws = np.array([[0, 0, 0, 12, np.nan, np.inf]])
    assert eti_1d(draws[0], eti_prob=1) == (0, 12)
    np.testing.assert_array_equal(bands(draws, prob=1), [[0, 12]])
    summary = summarise_bands(draws, np.array([1]), outer=1)
    assert summary.loc[0, "median"] == 0
    assert summary.loc[0, "ci_lo"] == 0
    assert summary.loc[0, "ci_hi"] == 12
