# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sum likelihood contributions onto explicitly chosen evaluation units.

This module does not choose the unit held out by cross-validation or integrate
latent effects. For example, mapping repeated administrations to child labels
changes the unit being scored; summing them does not make a conditional
likelihood into a likelihood for a new child.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

type _UnitIds = Sequence[Hashable] | np.ndarray | pd.Index | xr.DataArray


@dataclass(frozen=True)
class LogLikelihoodFactor:
    """One likelihood factor and an explicit map from its rows to output units.

    A plain sequence of ``row_unit_ids`` maps by position. A DataArray map must
    use ``row_dim`` and carry exactly the same row coordinate as ``values``.
    Repeated unit labels are valid: their row contributions will be summed.
    """

    values: xr.DataArray
    """Log likelihood, with named sample, row and optional event dimensions."""

    row_dim: str
    """Dimension whose entries are mapped to the requested output units."""

    row_unit_ids: _UnitIds
    """One nonmissing unit label per row, in the factor's row order."""

    event_dims: tuple[str, ...] = ()
    """Dimensions to sum within each row; no other dimension is reduced."""

    def __post_init__(self) -> None:
        if not isinstance(self.values, xr.DataArray):
            raise TypeError("values must be an xarray.DataArray")
        _dimension_names((self.row_dim,), "row_dim")
        _dimension_names(self.event_dims, "event_dims", allow_empty=True)
        if self.row_dim in self.event_dims:
            raise ValueError("row_dim must not also be an event dimension")
        for dim in (self.row_dim, *self.event_dims):
            if dim not in self.values.dims:
                raise ValueError(f"Declared dimension {dim!r} is absent from the factor")
        if any(self.values.sizes[dim] == 0 for dim in self.event_dims):
            raise ValueError("Event dimensions must not be empty")


def _dimension_names(names: Sequence[str], label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(names, str):
        raise TypeError(f"{label} must be a sequence of dimension names")
    result = tuple(names)
    if not allow_empty and not result:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(name, str) or not name for name in result):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate dimension names")
    return result


def _label_index(values: _UnitIds, label: str, *, unique: bool = False) -> pd.Index:
    if isinstance(values, np.ma.MaskedArray) and np.ma.getmaskarray(values).any():
        raise ValueError(f"{label} contains masked labels")
    if isinstance(values, xr.DataArray):
        if values.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        values = values.values
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    try:
        index = values if isinstance(values, pd.Index) else pd.Index(values, tupleize_cols=False)
        for value in index:
            hash(value)
        missing = (
            bool(index.to_frame(index=False).isna().any().any())
            if isinstance(index, pd.MultiIndex)
            else bool(index.isna().any())
        )
        duplicates = unique and not index.is_unique
        # Boolean masks must be applied to a sequence of unit labels first.
        # Otherwise False and True can silently match integer unit IDs 0 and 1.
        boolean_labels = any(isinstance(value, (bool, np.bool_)) for value in index)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError(f"{label} must be a one-dimensional sequence of scalar labels") from exc
    if missing:
        raise ValueError(f"{label} contains missing labels")
    if boolean_labels:
        raise ValueError(f"{label} must contain unit labels, not boolean masks")
    if duplicates:
        raise ValueError(f"{label} contains duplicate labels")
    return index


def _row_positions(factor: LogLikelihoodFactor, units: pd.Index) -> np.ndarray:
    labels = _label_index(factor.row_unit_ids, "row_unit_ids")
    if len(labels) != factor.values.sizes[factor.row_dim]:
        raise ValueError("row_unit_ids length does not match the factor's row dimension")
    if isinstance(factor.row_unit_ids, xr.DataArray):
        row_map = factor.row_unit_ids
        if row_map.dims != (factor.row_dim,):
            raise ValueError("The row map must use the factor's row dimension")
        if factor.row_dim not in row_map.indexes or factor.row_dim not in factor.values.indexes:
            raise ValueError("A DataArray row map and its factor must both have an explicit row coordinate index")
        _label_index(row_map.indexes[factor.row_dim], "Row coordinate", unique=True)
        if not _same_coordinate(row_map.coords, factor.values.coords, factor.row_dim):
            raise ValueError("The row map coordinate does not exactly match the factor's row coordinate")
    positions = units.get_indexer(labels)
    if np.any(positions < 0):
        raise ValueError("row_unit_ids contains labels absent from unit_ids")
    return positions


def _same_coordinate(left: xr.Coordinates, right: xr.Coordinates, name: Hashable) -> bool:
    if left[name].dtype != right[name].dtype or not left[name].variable.identical(right[name].variable):
        return False
    if (name in left.indexes) != (name in right.indexes):
        return False
    return name not in left.indexes or left.indexes[name].identical(right.indexes[name])


def _sample_coordinates(array: xr.DataArray, sample_dims: tuple[str, ...]) -> xr.Coordinates:
    for dim in sample_dims:
        if dim not in array.dims:
            raise ValueError(f"Sample dimension {dim!r} is absent from a factor")
        if array.sizes[dim] == 0:
            raise ValueError("Sample dimensions must not be empty")
        if dim not in array.indexes or array.coords[dim].dims != (dim,):
            raise ValueError(f"Sample dimension {dim!r} must have an explicit one-dimensional coordinate index")
        _label_index(array.indexes[dim], f"Sample coordinate {dim!r}", unique=True)
    coordinates = {
        name: coord.variable
        for name, coord in array.coords.items()
        if coord.dims and set(coord.dims) <= set(sample_dims)
    }
    return xr.Coordinates(
        coordinates, indexes={name: index for name, index in array.xindexes.items() if name in coordinates}
    )


def aggregate_log_likelihood(
    factors: Sequence[LogLikelihoodFactor],
    *,
    unit_ids: _UnitIds,
    unit_dim: str,
    sample_dims: tuple[str, ...] = ("chain", "draw"),
) -> xr.DataArray:
    """Sum factor rows and declared event dimensions onto explicit units.

    Parameters
    ----------
    factors
        One or more factors. Each input array must be a distinct object; passing
        the same array twice is rejected to catch accidental double counting.
        Callers remain responsible for selecting nonoverlapping factors.
    unit_ids
        Unique, nonmissing output labels in the desired order. Every label must
        receive at least one row across the factors. Boolean masks are not unit
        labels: pass, for example, ``administration_ids[mask]`` as a row map.
    unit_dim
        Name of the output unit dimension. Must differ from the sample names.
    sample_dims
        Names and output order of the sample dimensions. Each must have explicit,
        unique, nonmissing coordinate labels. Factors must have matching sample
        coordinate indexes, including their order, data types, index metadata,
        attributes and any sample-only auxiliary coordinates. No factor is
        silently reordered or broadcast.

    Returns
    -------
    xarray.DataArray
        A new array named ``log_likelihood`` with dimensions
        ``(*sample_dims, unit_dim)``. Sample coordinates are retained. Input
        arrays and trace groups are not changed.

    Notes
    -----
    NaN and positive infinity are rejected. Negative infinity is retained
    because an impossible observation has log likelihood ``-inf``; a sum
    containing it is also ``-inf``. Floating-point overflow raises rather than
    introducing a new infinite value. No sum skips missing values. Values are
    converted to float64 before either event or row reductions. This may differ
    from an existing float32 event sum; callers must review that numerical
    change when migrating lower-precision likelihood arrays.

    A factor may have zero rows if other factors cover all output units. Empty
    sample or event dimensions, no factors, and no output units are rejected.
    A scalar likelihood needs an explicit singleton row dimension. Event
    dimensions must be declared even when their size is one.

    This operation only sums supplied contributions. It does not perform
    leave-one-out estimation or integrate subject-specific latent effects.
    """
    sample_dims = _dimension_names(sample_dims, "sample_dims")
    _dimension_names((unit_dim,), "unit_dim")
    if unit_dim in sample_dims:
        raise ValueError("unit_dim must differ from the sample dimension names")
    factors = tuple(factors)
    if not factors:
        raise ValueError("At least one likelihood factor is required")
    if any(not isinstance(factor, LogLikelihoodFactor) for factor in factors):
        raise TypeError("factors must contain LogLikelihoodFactor objects")
    if len({id(factor.values) for factor in factors}) != len(factors):
        raise ValueError("The same factor array was supplied more than once")
    units = _label_index(unit_ids, "unit_ids", unique=True)
    if not len(units):
        raise ValueError("unit_ids must not be empty")

    for factor in factors:
        # A frozen dataclass does not stop a caller changing its DataArray.
        factor.__post_init__()
        roles = (*sample_dims, factor.row_dim, *factor.event_dims)
        if len(set(roles)) != len(roles):
            raise ValueError("Sample, row and event dimensions must have distinct roles")
        if len(factor.values.dims) != len(roles) or set(factor.values.dims) != set(roles):
            raise ValueError("Every factor dimension must be declared as a sample, row or event dimension")

    coordinates = _sample_coordinates(factors[0].values, sample_dims)
    unit_coordinates = (
        xr.Coordinates.from_pandas_multiindex(units, unit_dim)
        if isinstance(units, pd.MultiIndex)
        else xr.Coordinates({unit_dim: units})
    )
    if coordinates.keys() & unit_coordinates.keys():
        raise ValueError("Output unit coordinates must not replace a retained sample coordinate")
    sample_shape = tuple(factors[0].values.sizes[dim] for dim in sample_dims)
    combined = np.zeros((*sample_shape, len(units)), dtype=float)
    covered = np.zeros(len(units), dtype=bool)
    for factor in factors:
        factor_coordinates = _sample_coordinates(factor.values, sample_dims)
        if coordinates.keys() != factor_coordinates.keys() or any(
            not _same_coordinate(coordinates, factor_coordinates, name) for name in coordinates
        ):
            raise ValueError("Factor sample coordinates do not match exactly")
        positions = _row_positions(factor, units)
        covered[positions] = True
        ordered = factor.values.transpose(*sample_dims, factor.row_dim, *factor.event_dims)
        if not np.issubdtype(ordered.dtype, np.number) or np.issubdtype(ordered.dtype, np.complexfloating):
            raise TypeError("Log likelihood values must be real numbers")
        try:
            with np.errstate(over="raise", invalid="raise"):
                values = np.asarray(ordered.values, dtype=float)
                if np.isnan(values).any() or np.isposinf(values).any():
                    raise ValueError("Log likelihood values must not contain NaN or positive infinity")
                if factor.event_dims:
                    event_axes = tuple(range(len(sample_dims) + 1, values.ndim))
                    values = values.sum(axis=event_axes)
                np.add.at(combined, (*((slice(None),) * len(sample_dims)), positions), values)
        except FloatingPointError as exc:
            raise ValueError("Log likelihood aggregation overflowed or produced an invalid sum") from exc
    if not covered.all():
        raise ValueError("Every unit_ids label must receive at least one factor row")
    return xr.DataArray(
        combined,
        dims=(*sample_dims, unit_dim),
        coords=coordinates.assign(unit_coordinates),
        name="log_likelihood",
    )
