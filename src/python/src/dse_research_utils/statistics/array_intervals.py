# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Equal-tailed array intervals with explicit reduction and missing-value rules."""

import operator
from math import prod
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

NonfinitePolicy = Literal["propagate", "omit_nan", "omit_nonfinite", "raise"]
"""How array interval reductions handle NaN and infinite samples."""


def _reduction_axes(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, tuple):
        if not axis:
            raise ValueError("axis must not be an empty tuple")
        axes = axis
    else:
        axes = (axis,)
    normalized = []
    for item in axes:
        if isinstance(item, bool | np.bool_):
            raise TypeError("axis must contain integers, not booleans")
        try:
            value = operator.index(item)
        except TypeError as exc:
            raise TypeError("axis must be an integer, a tuple of integers or None") from exc
        if value < -ndim or value >= ndim:
            raise ValueError(f"axis {value} is out of range for an array with {ndim} dimensions")
        normalized.append(value % ndim)
    if len(set(normalized)) != len(normalized):
        raise ValueError("axis must not repeat a dimension")
    return tuple(normalized)


def equal_tail_interval(
    samples: ArrayLike,
    *,
    prob: float,
    axis: int | tuple[int, ...] | None,
    nonfinite: NonfinitePolicy,
    keepdims: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce sample axes to equal-tailed interval bounds.

    Parameters
    ----------
    samples
        Real numeric array-like input. Values are converted to float64 before
        calculation; masked arrays must first be converted explicitly, such as
        by filling their masked entries with NaN. The input is not modified.
    prob
        Interval probability in ``(0, 1]``. There is no implicit coverage.
    axis
        Sample axis, a nonempty tuple of sample axes, or ``None`` to reduce all
        dimensions. Negative axes are accepted; repeated axes are rejected.
        A tuple can combine chain and draw dimensions without folding other
        dimensions into the sampled distribution.
    nonfinite : {"propagate", "omit_nan", "omit_nonfinite", "raise"}
        ``propagate`` retains all values: any NaN makes both bounds NaN for its
        slice. ``omit_nan`` drops NaNs but retains infinities. Both use NumPy's
        interpolation behavior for infinite values, which can produce infinite
        or NaN bounds, including when ``prob=1``. ``omit_nonfinite`` removes both
        NaN and infinite values. ``raise`` rejects any nonfinite input value.
        Expected nonfinite interpolation does not emit runtime warnings.
    keepdims
        Retain reduced axes with length one, suitable for broadcasting bounds
        against the input. The default removes reduced axes.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Lower and upper float64 arrays. Without ``keepdims``, unreduced axes
        retain their input order. Reducing every axis returns two 0-D arrays.
        Empty or entirely omitted slices return NaN bounds without warnings.
        No output positions returns empty arrays with the corresponding shape.

    Raises
    ------
    TypeError
        Inputs are nonnumeric or complex, axes are not integers, probability is
        not a real number, or ``keepdims`` is not boolean.
    ValueError
        Coverage, axes or policy are invalid; a masked array is supplied;
        ``raise`` finds nonfinite data; or finite arithmetic overflows.

    Notes
    -----
    Bounds use NumPy's ``linear`` quantiles at ``(1 - prob) / 2`` and
    ``1 - (1 - prob) / 2``. Float64 conversion is an explicit numerical
    contract: it can differ from quantiles computed directly in float32.
    Finite inputs that overflow during conversion or quantile calculation
    raise instead of returning bounds corrupted by arithmetic overflow.

    No mean, median, reporting label or interval-column naming is selected.
    The existing ``statistics.intervals`` helpers retain their own defaults,
    finite-sample filtering and percentile calculations.
    """
    if isinstance(prob, bool | np.bool_) or not isinstance(prob, Real):
        raise TypeError("prob must be a real number")
    if not 0 < prob <= 1:
        raise ValueError("prob must be in (0, 1]")
    try:
        probability = float(prob)
    except OverflowError as exc:
        raise ValueError("prob must be in (0, 1]") from exc
    if not 0 < probability <= 1:
        raise ValueError("prob must be in (0, 1]")
    if not isinstance(nonfinite, str) or nonfinite not in ("propagate", "omit_nan", "omit_nonfinite", "raise"):
        raise ValueError("nonfinite must be propagate, omit_nan, omit_nonfinite or raise")
    if not isinstance(keepdims, bool | np.bool_):
        raise TypeError("keepdims must be boolean")
    if np.ma.isMaskedArray(samples):
        raise ValueError("samples must not be a masked array; convert its missing values explicitly")
    values = np.asarray(samples)
    if values.dtype.kind not in "iuf":
        raise TypeError("samples must contain real numeric values")
    axes = _reduction_axes(axis, values.ndim)
    try:
        with np.errstate(over="raise", invalid="raise"):
            values = np.asarray(values, dtype=np.float64)
    except FloatingPointError as exc:
        raise ValueError("samples overflowed during float64 conversion") from exc
    if nonfinite == "raise" and not np.isfinite(values).all():
        raise ValueError("samples must be finite when nonfinite='raise'")

    retained_axes = tuple(dim for dim in range(values.ndim) if dim not in axes)
    output_shape = tuple(values.shape[dim] for dim in retained_axes)
    n_outputs = prod(output_shape)
    n_samples = prod(values.shape[dim] for dim in axes)
    rows = values.transpose(*retained_axes, *axes).reshape(n_outputs, n_samples)
    lower = np.full(n_outputs, np.nan, dtype=np.float64)
    upper = np.full(n_outputs, np.nan, dtype=np.float64)
    tail = (1 - probability) / 2
    quantiles = [tail, 1 - tail]
    for position, row in enumerate(rows):
        if nonfinite == "omit_nan":
            row = row[~np.isnan(row)]
        elif nonfinite == "omit_nonfinite":
            row = row[np.isfinite(row)]
        elif np.isnan(row).any():
            continue
        if not row.size:
            continue
        invalid = "ignore" if np.isinf(row).any() else "raise"
        try:
            with np.errstate(over="raise", invalid=invalid):
                lower[position], upper[position] = np.quantile(row, quantiles, method="linear")
        except FloatingPointError as exc:
            raise ValueError("Quantile arithmetic overflowed or produced an invalid result") from exc
    if keepdims:
        output_shape = tuple(1 if dim in axes else length for dim, length in enumerate(values.shape))
    return lower.reshape(output_shape), upper.reshape(output_shape)
