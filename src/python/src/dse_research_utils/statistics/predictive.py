# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Per-observation predictive summaries with explicit empirical references."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class PredictiveObservationChecks:
    """Predictive summaries in input observation and interval-probability order.

    The returned arrays are newly allocated. The frozen record prevents field
    reassignment, but does not make its NumPy arrays immutable.
    """

    interval_probs: tuple[float, ...]
    """Requested nominal probabilities, in their original order."""

    predictive_mean: np.ndarray
    """Mean per observation, shape ``(observation,)``."""

    predictive_median: np.ndarray
    """Median per observation, shape ``(observation,)``."""

    lower: np.ndarray
    """Lower bounds, shape ``(observation, interval)``."""

    upper: np.ndarray
    """Upper bounds, shape ``(observation, interval)``."""

    inside: np.ndarray
    """Whether each observation lies in each closed interval, same shape as bounds."""

    predictive_mass: np.ndarray
    """Fraction of that observation's predictive draws within each closed interval."""

    midpoint_pit: np.ndarray | None
    """Probability below the observed value plus half the tied mass, if requested."""

    expected_midpoint_pit_variance: np.ndarray | None
    """Midpoint-PIT reference variance of each empirical predictive distribution."""


def predictive_observation_checks(
    observed: ArrayLike,
    replicated: ArrayLike,
    *,
    interval_probs: Sequence[float],
    sample_axis: int = 1,
    pit_method: Literal["midpoint"] | None = None,
    observation_chunk_size: int = 256,
) -> PredictiveObservationChecks:
    """Compare observations with intervals and ranks in predictive draws.

    Parameters
    ----------
    observed
        One-dimensional, nonempty real numeric observations. Every value must
        be finite. The caller chooses any missing-row mask before calling.
    replicated
        Two-dimensional, nonempty real numeric predictive draws. Boolean draws
        must first be converted to numeric zero/one values. The axis other
        than ``sample_axis`` must match ``observed`` in both length and order.
        Every draw must be finite; this function never drops samples or rows.
        Masked arrays are rejected for both inputs; filter or fill them explicitly.
        Use ``statistics.samples.sample_matrix`` first for labelled arrays.
    interval_probs
        Distinct nominal probabilities strictly between zero and one. Output
        columns retain this order. There is no project-specific default width.
    sample_axis : int, default 1
        Axis containing samples, either 0 or 1.
    pit_method : {"midpoint", None}, default None
        Request the deterministic midpoint probability integral transform (PIT)
        and its empirical predictive reference variance, or omit both. No
        randomisation is performed.
    observation_chunk_size : int, default 256
        Maximum observations processed together. Apart from the input and
        output arrays, temporary draw-sized arrays are limited to one chunk.

    Returns
    -------
    PredictiveObservationChecks
        Per-row means, medians, intervals, inclusion flags and empirical mass,
        plus optional midpoint-PIT quantities. No grouping or file writing.

    Raises
    ------
    ValueError
        For invalid shapes, axes, probabilities, non-finite values, masked arrays,
        overflowing summaries or invalid options.
    TypeError
        For non-real numeric inputs or a non-integer chunk size.

    Notes
    -----
    Bounds use NumPy's linear quantiles at ``tail = (1 - p) / 2`` and
    ``1 - tail``. Inclusion uses both endpoints. The empirical mass inside
    these interpolated bounds need not equal or exceed the nominal probability,
    even for discrete draws. It is measured directly and returned separately.

    Midpoint PIT is ``mean(draw < observed) + 0.5 * mean(draw == observed)``.
    For empirical mass ``p_k`` on distinct values, its reference variance is
    ``(1 - sum(p_k**3)) / 12``. This is a finite-draw estimate of the predictive
    reference, not the continuous uniform variance of 1/12. Checks on data used
    to fit a model do not establish calibration for new observations.

    Scalar Python quantile probabilities preserve NumPy's native floating draw
    dtype in bounds and medians; integer draws give float64 bounds and medians.
    The mean is reduced in the input dtype, then stored as float64. These rules
    preserve the existing consumers' arithmetic. A grouped mean of interval
    widths may need ``(upper - lower).astype(float)`` before aggregation to retain
    a consumer's intermediate storage convention.
    The caller retains scientific grouping, CSV schemas and missingness policy.
    """
    if np.ma.isMaskedArray(observed) or np.ma.isMaskedArray(replicated):
        raise ValueError("masked arrays require explicit filtering or filling before calling")
    y = np.asarray(observed)
    draws = np.asarray(replicated)
    if y.dtype.kind not in "biuf" or draws.dtype.kind not in "iuf":
        raise TypeError("observed and replicated must contain real numeric values")
    if isinstance(sample_axis, bool | np.bool_) or sample_axis not in (0, 1):
        raise ValueError("sample_axis must be 0 or 1")
    if draws.ndim != 2 or y.ndim != 1:
        raise ValueError("observed must be 1-D and replicated must be 2-D")
    if sample_axis == 0:
        draws = draws.T
    if y.shape != (draws.shape[0],):
        raise ValueError("observed and replicated are not row-aligned")
    if not y.size or not draws.shape[1]:
        raise ValueError("checks require observations and predictive samples")
    if not np.isfinite(y).all():
        raise ValueError("observed must be finite; filter missing rows explicitly before calling")
    probs = tuple(float(prob) for prob in interval_probs)
    if not probs or any(not 0 < prob < 1 for prob in probs) or len(set(probs)) != len(probs):
        raise ValueError("interval_probs must be distinct probabilities strictly between 0 and 1")
    if pit_method not in (None, "midpoint"):
        raise ValueError("pit_method must be None or 'midpoint'")
    if isinstance(observation_chunk_size, bool | np.bool_):
        raise TypeError("observation_chunk_size must be an integer")
    chunk_size = operator.index(observation_chunk_size)
    if chunk_size <= 0:
        raise ValueError("observation_chunk_size must be positive")

    n_obs = len(y)
    shape = (n_obs, len(probs))
    reduction_dtype = draws.dtype if draws.dtype.kind == "f" else np.dtype(float)
    mean = np.empty(n_obs, dtype=float)
    median = np.empty(n_obs, dtype=reduction_dtype)
    lower = np.empty(shape, dtype=reduction_dtype)
    upper = np.empty(shape, dtype=reduction_dtype)
    inside = np.empty(shape, dtype=bool)
    mass = np.empty(shape, dtype=float)
    pit = np.empty(n_obs, dtype=float) if pit_method is not None else None
    pit_variance = np.empty(n_obs, dtype=float) if pit_method is not None else None

    for start in range(0, n_obs, chunk_size):
        stop = min(start + chunk_size, n_obs)
        chunk = draws[start:stop]
        if not np.isfinite(chunk).all():
            raise ValueError("replicated must contain only finite predictive draws")
        try:
            with np.errstate(over="raise", invalid="raise"):
                mean[start:stop] = chunk.mean(axis=1)
                median[start:stop] = np.median(chunk, axis=1)
                for column, prob in enumerate(probs):
                    # Python scalar q preserves float32 quantile arithmetic.
                    # A float64 vector of q values can change even inclusion at
                    # an endpoint, not only the final displayed precision.
                    tail = (1 - prob) / 2
                    lower[start:stop, column] = np.quantile(chunk, tail, axis=1, method="linear")
                    upper[start:stop, column] = np.quantile(chunk, 1 - tail, axis=1, method="linear")
        except FloatingPointError as exc:
            raise ValueError("predictive summaries overflowed or produced an invalid value") from exc
        inside[start:stop] = (y[start:stop, None] >= lower[start:stop]) & (y[start:stop, None] <= upper[start:stop])
        for column in range(len(probs)):
            mass[start:stop, column] = np.mean(
                (chunk >= lower[start:stop, column, None]) & (chunk <= upper[start:stop, column, None]), axis=1
            )
        if pit is not None and pit_variance is not None:
            pit[start:stop] = np.mean(chunk < y[start:stop, None], axis=1) + 0.5 * np.mean(
                chunk == y[start:stop, None], axis=1
            )
            for offset, row in enumerate(chunk):
                _, counts = np.unique(row, return_counts=True)
                probabilities = counts / row.size
                pit_variance[start + offset] = (1 - float(np.sum(probabilities**3))) / 12

    return PredictiveObservationChecks(probs, mean, median, lower, upper, inside, mass, pit, pit_variance)
