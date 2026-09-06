# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import fields

import numpy as np
import pytest

from dse_research_utils.statistics.predictive import predictive_observation_checks


def test_point_mass_and_closed_interval_edges():
    observed = np.array([3, 5, 5, 15])
    replicated = np.array([[3] * 5, [3] * 5, [0, 5, 10, 15, 20], [0, 5, 10, 15, 20]])
    result = predictive_observation_checks(observed, replicated, interval_probs=(0.9, 0.5))
    assert result.interval_probs == (0.9, 0.5)
    np.testing.assert_array_equal(result.predictive_mean, [3, 3, 10, 10])
    np.testing.assert_array_equal(result.predictive_median, [3, 3, 10, 10])
    np.testing.assert_array_equal(result.lower[:, 1], [3, 3, 5, 5])
    np.testing.assert_array_equal(result.upper[:, 1], [3, 3, 15, 15])
    np.testing.assert_array_equal(result.inside, [[True, True], [False, False], [True, True], [True, True]])
    np.testing.assert_array_equal(result.predictive_mass[:, 1], [1, 1, 0.6, 0.6])
    assert result.midpoint_pit is result.expected_midpoint_pit_variance is None


def test_bernoulli_reference_is_not_the_continuous_uniform_variance():
    result = predictive_observation_checks(
        [0, 1], [[0, 0, 1, 1], [0, 0, 1, 1]], interval_probs=(0.5, 0.9), pit_method="midpoint"
    )
    np.testing.assert_array_equal(result.midpoint_pit, [0.25, 0.75])
    np.testing.assert_array_equal(result.expected_midpoint_pit_variance, [0.0625, 0.0625])
    np.testing.assert_array_equal(result.predictive_mass, np.ones((2, 2)))
    np.testing.assert_array_equal(result.inside, np.ones((2, 2), dtype=bool))


def test_interpolated_interval_mass_can_be_below_nominal():
    result = predictive_observation_checks([0], [[0, 1]], interval_probs=(0.9,), pit_method="midpoint")
    np.testing.assert_allclose(result.lower, [[0.05]])
    np.testing.assert_allclose(result.upper, [[0.95]])
    assert result.predictive_mass.item() == 0
    assert not result.inside.item()
    assert result.midpoint_pit.item() == 0.25


def test_single_sample_has_zero_midpoint_reference_variance():
    result = predictive_observation_checks([2, 9], [[2], [3]], interval_probs=(0.89,), pit_method="midpoint")
    np.testing.assert_array_equal(result.lower, [[2], [3]])
    np.testing.assert_array_equal(result.upper, result.lower)
    np.testing.assert_array_equal(result.midpoint_pit, [0.5, 1.0])
    np.testing.assert_array_equal(result.expected_midpoint_pit_variance, [0, 0])


@pytest.mark.parametrize("sample_axis", [0, 1])
@pytest.mark.parametrize("chunk_size", [1, 3, 32])
def test_axis_and_chunking_preserve_every_output_and_inputs(sample_axis, chunk_size):
    rng = np.random.default_rng(2)
    observations = rng.integers(0, 9, 11)
    replicated = rng.integers(0, 10, (11, 25), dtype=np.int16)
    original_observations = observations.copy()
    original_replicated = replicated.copy()
    expected = predictive_observation_checks(
        observations, replicated, interval_probs=(0.5, 0.8, 0.89, 0.95), pit_method="midpoint"
    )
    result = predictive_observation_checks(
        observations,
        replicated if sample_axis == 1 else replicated.T,
        interval_probs=expected.interval_probs,
        sample_axis=sample_axis,
        observation_chunk_size=chunk_size,
        pit_method="midpoint",
    )
    for field in fields(result):
        np.testing.assert_array_equal(getattr(result, field.name), getattr(expected, field.name))
    np.testing.assert_array_equal(observations, original_observations)
    np.testing.assert_array_equal(replicated, original_replicated)


def test_float32_reductions_and_midpoint_operation_order_match_existing_calibration():
    draws = np.array([[1e8, 1, -1e8]], dtype=np.float32)
    result = predictive_observation_checks([1], draws, interval_probs=(0.5,))
    assert result.predictive_mean.item() == draws.mean(axis=1).item() == 0.0
    assert result.predictive_mean.item() != draws.astype(float).mean(axis=1).item()
    result = predictive_observation_checks([1], [np.arange(5)], interval_probs=(0.5,), pit_method="midpoint")
    assert result.midpoint_pit.item() == 0.30000000000000004


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int16])
def test_quantile_and_median_dtypes_match_scalar_consumer_reductions(dtype):
    draws = np.array([[0, 0, 1], [1, 2, 3]], dtype=dtype)
    result = predictive_observation_checks([0, 2], draws, interval_probs=(0.8,))
    tail = (1 - 0.8) / 2
    expected_lower = np.quantile(draws, tail, axis=1)
    expected_upper = np.quantile(draws, 1 - tail, axis=1)
    assert result.lower.dtype == expected_lower.dtype
    assert result.upper.dtype == expected_upper.dtype
    assert result.predictive_median.dtype == np.median(draws, axis=1).dtype
    np.testing.assert_array_equal(result.lower[:, 0], expected_lower)
    np.testing.assert_array_equal(result.upper[:, 0], expected_upper)
    np.testing.assert_array_equal(
        (result.upper[:, 0] - result.lower[:, 0]).astype(float), (expected_upper - expected_lower).astype(float)
    )


def test_float32_width_is_subtracted_before_widening_to_group_summary_dtype():
    draws = np.array([[1e8, 1, -1e8]], dtype=np.float32)
    result = predictive_observation_checks([1], draws, interval_probs=(0.5,))
    assert (result.upper - result.lower).astype(float).item() == 100_000_000.0


def test_finite_draws_that_overflow_do_not_return_infinite_summaries():
    extreme = np.finfo(np.float32).max
    with pytest.raises(ValueError, match="overflowed"):
        predictive_observation_checks([1], np.array([[extreme, extreme]], dtype=np.float32), interval_probs=(0.9,))


@pytest.mark.parametrize("masked_input", ["observed", "replicated"])
def test_masked_values_are_not_silently_included(masked_input):
    observed = np.array([1, 9])
    replicated = np.array([[0, 1, 2], [7, 8, 9]])
    if masked_input == "observed":
        observed = np.ma.array(observed, mask=[False, True])
    else:
        replicated = np.ma.array(replicated, mask=[[False, False, False], [False, False, True]])
    with pytest.raises(ValueError, match="masked arrays"):
        predictive_observation_checks(observed, replicated, interval_probs=(0.9,))


def test_upper_quantile_uses_complement_of_lower_tail_at_rounding_boundary():
    observed = [0.5000005000000001]
    result = predictive_observation_checks(observed, [[0.0, 1.0]], interval_probs=(1e-6,))
    assert result.upper.item() == observed[0]
    assert result.inside.item()


@pytest.mark.parametrize(
    ("observed", "replicated", "message"),
    [
        ([[1]], [[1, 2]], "1-D"),
        ([1], [1, 2], "2-D"),
        ([1, 2], [[1, 2]], "row-aligned"),
        ([], np.empty((0, 2)), "require observations"),
        ([1], np.empty((1, 0)), "require observations"),
        ([np.nan], [[1, 2]], "observed must be finite"),
        ([np.inf], [[1, 2]], "observed must be finite"),
        ([1], [[1, np.nan]], "finite predictive"),
        ([1], [[1, -np.inf]], "finite predictive"),
    ],
)
def test_invalid_or_nonfinite_data_is_not_silently_dropped(observed, replicated, message):
    with pytest.raises(ValueError, match=message):
        predictive_observation_checks(observed, replicated, interval_probs=(0.9,))


@pytest.mark.parametrize(
    "observed,replicated", [(["1"], [[1, 2]]), ([1], [[1 + 1j, 2]]), ([1], [["1", "2"]]), ([1], [[True, False]])]
)
def test_non_real_numeric_data_requires_explicit_caller_conversion(observed, replicated):
    with pytest.raises(TypeError, match="real numeric"):
        predictive_observation_checks(observed, replicated, interval_probs=(0.9,))


@pytest.mark.parametrize("probs", [(), (0,), (1,), (np.nan,), (np.inf,), (-0.1,), (0.5, 0.5)])
def test_probabilities_are_explicit_valid_and_unique(probs):
    with pytest.raises(ValueError, match="interval_probs"):
        predictive_observation_checks([1], [[0, 1, 2]], interval_probs=probs)


@pytest.mark.parametrize("sample_axis", [-1, 2, True])
def test_invalid_sample_axis(sample_axis):
    with pytest.raises(ValueError, match="sample_axis"):
        predictive_observation_checks([1], [[0, 1]], interval_probs=(0.9,), sample_axis=sample_axis)


@pytest.mark.parametrize("chunk_size,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)])
def test_invalid_chunk_size(chunk_size, error):
    with pytest.raises(error):
        predictive_observation_checks([1], [[0, 1]], interval_probs=(0.9,), observation_chunk_size=chunk_size)


def test_midpoint_pit_requires_an_explicit_supported_method():
    with pytest.raises(ValueError, match="pit_method"):
        predictive_observation_checks([1], [[0, 1]], interval_probs=(0.9,), pit_method="randomised")


def test_missing_observation_filter_is_explicit_and_retains_row_identity():
    row_ids = np.array(["one", "two", "three"])
    observed = np.array([1, np.nan, 3])
    draws = np.array([[0, 1, 2], [4, 5, 6], [2, 3, 4]])
    keep = np.isfinite(observed)
    result = predictive_observation_checks(observed[keep], draws[keep], interval_probs=(0.5,))
    np.testing.assert_array_equal(row_ids[keep], ["one", "three"])
    np.testing.assert_array_equal(result.predictive_median, [1, 3])
    np.testing.assert_array_equal(result.inside, [[True], [True]])
