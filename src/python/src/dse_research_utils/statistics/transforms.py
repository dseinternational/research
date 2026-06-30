# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small numeric transforms shared across DSE projects.

NumPy-domain helpers (standardisation, logit/inverse-logit, categorical coding).
These are deliberately distinct from the PyMC-graph ``logit`` in
:mod:`dse_research_utils.statistics.models.pymc_utils`, which operates on tensors.
"""

from __future__ import annotations

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


def logit(p: float) -> float:
    """NumPy logit: ``log(p / (1 - p))``."""
    return np.log(p / (1 - p))


def invlogit(x: float) -> float:
    """NumPy inverse logit (logistic sigmoid): ``1 / (1 + exp(-x))``."""
    return 1 / (1 + np.exp(-x))


def convert_to_categorical(
    data: pd.DataFrame | pd.Series,
) -> np.ndarray | pd.DataFrame | pd.Series:
    """Convert input data to categorical codes."""
    if isinstance(data, pd.DataFrame):
        return data.apply(lambda col: convert_to_categorical(col))
    return data.astype("category").cat.codes
