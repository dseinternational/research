# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass
class ModelData:
    """
    Data for a model with one or more predictor variables and an outcome variable.
    """

    X_obs: np.ndarray
    """
    Observed predictor values, shape (n, p).
    """
    y_obs: np.ndarray
    """
    Observed outcome values, shape (n,).
    """


@dataclass
class BinomialModelData(ModelData):
    """
    Data for a binomial model with one predictor variable and a count outcome variable.
    """

    n_trials: int
    """
    Number of trials (maximum score count).
    """

    def __post_init__(self) -> None:
        if isinstance(self.n_trials, bool) or not isinstance(self.n_trials, Integral) or self.n_trials <= 0:
            raise ValueError("n_trials must be a positive integer.")

        self.n_trials = int(self.n_trials)
        self.X_obs = np.asarray(self.X_obs, dtype=float)
        self.y_obs = np.asarray(self.y_obs, dtype=float)

        if self.X_obs.ndim != 2 or self.X_obs.shape[1] != 1:
            raise ValueError("X_obs must be a 2D array with shape (n, 1).")

        if self.y_obs.ndim != 1 or self.y_obs.shape[0] != self.X_obs.shape[0]:
            raise ValueError("y_obs must be a 1D array with the same number of rows as X_obs.")

        if not np.all(np.isfinite(self.y_obs)):
            raise ValueError("y_obs values must be finite.")

        if not np.all(self.y_obs == np.floor(self.y_obs)):
            raise ValueError("y_obs values must be integer counts.")

        if np.any(self.y_obs < 0) or np.any(self.y_obs > self.n_trials):
            raise ValueError("y_obs values must be in the range [0, n_trials].")

        self.y_obs = self.y_obs.astype(int)
