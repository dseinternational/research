# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Convert explicitly labelled sample arrays without losing observation identity."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import prod

import numpy as np
import pandas as pd
import xarray as xr


def _dimension_names(dimensions: Sequence[str], argument: str, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(dimensions, str):
        raise TypeError(f"{argument} must be a sequence of dimension names, not a string")
    names = tuple(dimensions)
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"{argument} must contain non-empty string names")
    if not allow_empty and not names:
        raise ValueError(f"{argument} must contain at least one dimension")
    if len(set(names)) != len(names):
        raise ValueError(f"{argument} must not repeat dimension names")
    return names


def _dimension_indexes(array: xr.DataArray, dimensions: tuple[str, ...]) -> tuple[pd.Index, ...]:
    indexes: list[pd.Index] = []
    for dimension in dimensions:
        if array.sizes[dimension] == 0:
            raise ValueError(f"dimension {dimension!r} must not be empty")
        if dimension not in array.indexes:
            raise ValueError(f"dimension {dimension!r} must have an explicit coordinate index")
        index = array.indexes[dimension]
        if not index.is_unique:
            raise ValueError(f"dimension {dimension!r} must have unique coordinate labels")
        if isinstance(index, pd.MultiIndex):
            if any(name is None for name in index.names) or len(set(index.names)) != index.nlevels:
                raise ValueError(f"dimension {dimension!r} must have distinct, named coordinate levels")
            missing = bool(index.to_frame(index=False).isna().any().any())
        else:
            missing = bool(index.isna().any())
        if missing:
            raise ValueError(f"dimension {dimension!r} must not have missing coordinate labels")
        indexes.append(index.copy(deep=True))
    return tuple(indexes)


def _axis_index(indexes: tuple[pd.Index, ...], dimensions: tuple[str, ...]) -> pd.Index:
    if not indexes:
        return pd.RangeIndex(1)
    if len(indexes) == 1:
        return indexes[0].copy(deep=True)
    # A previously stacked dimension contributes tuple labels as one factor.
    # Avoid xarray.stack: an output name could collide with an existing label,
    # and stack cannot combine a MultiIndex dimension with another dimension.
    factors = [
        pd.Index(index.to_numpy(), tupleize_cols=False) if isinstance(index, pd.MultiIndex) else index
        for index in indexes
    ]
    return pd.MultiIndex.from_product(factors, names=dimensions)


@dataclass(frozen=True)
class SampleMatrix:
    """A matrix of labelled observations by labelled samples.

    Create instances with :func:`sample_matrix`. The arrays may share memory
    with their inputs. Copy ``values`` before modifying it if the source array
    must remain unchanged. Index metadata is copied so later coordinate edits
    cannot change the recorded alignment contract.
    """

    values: np.ndarray
    """Sample values with shape ``(n_observations, n_samples)``."""

    sample_index: pd.Index
    """Sample labels in column order; multiple dimensions form a MultiIndex."""

    observation_index: pd.Index
    """Observation labels in row order; scalar nodes have a one-row RangeIndex."""

    sample_dims: tuple[str, ...]
    """Sample dimensions in the caller's declared order."""

    observation_dims: tuple[str, ...]
    """Observation dimensions in the caller's declared order."""

    _observation_indexes: tuple[pd.Index, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape != (len(self.observation_index), len(self.sample_index)):
            raise ValueError("values must have one row per observation and one column per sample")
        if len(self._observation_indexes) != len(self.observation_dims):
            raise ValueError("observation indexes must describe every observation dimension")

    def observed_values(self, observed: xr.DataArray) -> np.ndarray:
        """Return observed values in matrix row order after checking labels.

        Parameters
        ----------
        observed : xarray.DataArray
            Values with exactly the declared observation dimensions. Dimension
            order may differ, but each explicit index must be identical to the
            corresponding source index, including label order, names and dtype.
            Auxiliary coordinates do not define row identity and are ignored.

        Returns
        -------
        numpy.ndarray
            One-dimensional observed values. Missing values and the input dtype
            are retained. The result may share memory with ``observed``.

        Raises
        ------
        ValueError
            Dimensions or coordinate indexes differ. Equal shapes alone do not
            establish alignment; this method never reindexes or drops rows.
        """
        if not isinstance(observed, xr.DataArray):
            raise TypeError("observed must be an xarray.DataArray")
        if len(set(observed.dims)) != observed.ndim or set(observed.dims) != set(self.observation_dims):
            raise ValueError("observed must have exactly the declared observation dimensions")
        indexes = _dimension_indexes(observed, self.observation_dims)
        for dimension, expected, actual in zip(self.observation_dims, self._observation_indexes, indexes, strict=True):
            if not expected.identical(actual):
                raise ValueError(f"observed coordinate index for dimension {dimension!r} does not match sample rows")
        return np.asarray(observed.transpose(*self.observation_dims).values).reshape(-1)


def sample_matrix(
    array: xr.DataArray,
    *,
    sample_dims: Sequence[str],
    observation_dims: Sequence[str],
) -> SampleMatrix:
    """Arrange labelled samples into an observation-by-sample matrix.

    Parameters
    ----------
    array : xarray.DataArray
        An array whose every dimension is explicitly assigned to one axis.
        Each dimension needs a unique, non-missing coordinate index. Callers
        with positional axes must assign meaningful labels deliberately before
        conversion; matching lengths do not establish observation identity.
        Auxiliary coordinates are ignored; row identity uses dimension indexes.
    sample_dims : sequence of str
        One or more sample dimensions, for example ``("chain", "draw")``,
        ``("draw",)``, or a previously stacked ``("sample",)`` dimension.
    observation_dims : sequence of str
        Observation dimensions. Use an empty tuple for a scalar posterior node.
        Likelihood event dimensions are never assigned or flattened implicitly.

    Returns
    -------
    SampleMatrix
        Values shaped ``(n_observations, n_samples)`` and their coordinate indexes.
        The last dimension in each declared sequence varies fastest. Existing
        coordinate order is preserved; labels are not sorted. For one dimension
        its original index type is retained. With multiple dimensions, the index
        contains their Cartesian product, with previously stacked labels retained
        as tuples. Values are materialized as a NumPy array; reshaping may return
        a view or require a copy. This function does not modify the input.

    Raises
    ------
    ValueError
        Dimensions overlap, repeat, are undeclared or empty, or their indexes
        are absent, contain missing labels or do not uniquely identify entries.
    """
    if not isinstance(array, xr.DataArray):
        raise TypeError("array must be an xarray.DataArray")
    samples = _dimension_names(sample_dims, "sample_dims", allow_empty=False)
    observations = _dimension_names(observation_dims, "observation_dims", allow_empty=True)
    if set(samples) & set(observations):
        raise ValueError("sample_dims and observation_dims must not overlap")
    if len(set(array.dims)) != array.ndim:
        raise ValueError("array must not repeat dimension names")
    if set(samples + observations) != set(array.dims):
        raise ValueError("sample_dims and observation_dims must declare every array dimension exactly once")
    sample_indexes = _dimension_indexes(array, samples)
    observation_indexes = _dimension_indexes(array, observations)
    n_samples = prod(array.sizes[dimension] for dimension in samples)
    n_observations = prod(array.sizes[dimension] for dimension in observations)
    values = np.asarray(array.transpose(*observations, *samples).values).reshape(n_observations, n_samples)
    return SampleMatrix(
        values=values,
        sample_index=_axis_index(sample_indexes, samples),
        observation_index=_axis_index(observation_indexes, observations),
        sample_dims=samples,
        observation_dims=observations,
        _observation_indexes=observation_indexes,
    )
