# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small numeric transforms shared across DSE projects.

NumPy-domain helpers (standardisation, logit/inverse-logit, categorical coding).
These are deliberately distinct from the PyMC-graph ``logit`` in
:mod:`dse_research_utils.statistics.models.pymc_utils`, which operates on tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dse_research_utils.math.constants import EPSILON


def standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-SD standardisation; returns centred values if SD ~ 0."""
    mean_x = np.mean(x)
    std_x = np.std(x)
    if std_x < EPSILON:
        return x - mean_x
    return (x - mean_x) / std_x


@dataclass
class Standardiser:
    """A fitted standardisation, invertible for exact refits and back-transforms.

    Persist the instance (its ``mean``/``sd``) alongside a fitted model so a
    later refit or a natural-scale readout applies exactly the same scaling.
    """

    mean: float
    """Mean of the fitted values."""
    sd: float
    """Sample standard deviation (``ddof=1``) of the fitted values."""

    def __call__(self, x: np.ndarray | pd.Series) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.sd

    def inverse(self, z: np.ndarray | pd.Series) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.sd + self.mean


def standardise(x: np.ndarray | pd.Series) -> tuple[np.ndarray, Standardiser]:
    """Standardise ``x`` (NaN-aware, ``ddof=1``), returning the fitted transform.

    Unlike :func:`standardize`, this raises on a degenerate SD (a constant
    predictor is a modelling error, not something to silently centre) and
    returns the :class:`Standardiser` so the exact scaling can be persisted
    and inverted.
    """
    arr = np.asarray(x, dtype=float)
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Standard deviation of x must be positive.")
    return (arr - mu) / sd, Standardiser(mean=mu, sd=sd)


def haldane_logit(y: np.ndarray | pd.Series, n: int | np.ndarray) -> np.ndarray:
    """Haldane-Anscombe corrected empirical logit: ``log((y + 0.5) / (n - y + 0.5))``.

    Finite at ``y = 0`` and ``y = n``, which the raw :func:`logit` of ``y / n``
    is not; NaN counts pass through as NaN. Both consuming repositories derived
    this correction independently (three spellings in one repo alone).
    """
    y = np.asarray(y, dtype=float)
    return np.log((y + 0.5) / (np.asarray(n, dtype=float) - y + 0.5))


def logit(p: float | np.ndarray) -> float | np.ndarray:
    """NumPy logit: ``log(p / (1 - p))``. Vectorised — accepts scalars or arrays."""
    return np.log(p / (1 - p))


def invlogit(x: float | np.ndarray) -> float | np.ndarray:
    """NumPy inverse logit (logistic sigmoid): ``1 / (1 + exp(-x))``. Vectorised."""
    return 1 / (1 + np.exp(-x))


def convert_to_categorical(
    data: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """Convert input data to categorical codes."""
    if isinstance(data, pd.DataFrame):
        return data.apply(lambda col: convert_to_categorical(col))
    return data.astype("category").cat.codes
