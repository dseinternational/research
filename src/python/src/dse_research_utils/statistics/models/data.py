# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from dataclasses import dataclass


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
        self.X_obs = np.asarray(self.X_obs)
        self.y_obs = np.asarray(self.y_obs)

        if self.X_obs.ndim != 2 or self.X_obs.shape[1] != 1:
            raise ValueError("X_obs must be a 2D array with shape (n, 1).")

        if self.y_obs.ndim != 1 or self.y_obs.shape[0] != self.X_obs.shape[0]:
            raise ValueError(
                "y_obs must be a 1D array with the same number of rows as X_obs."
            )

        if np.any(self.y_obs < 0) or np.any(self.y_obs > self.n_trials):
            raise ValueError("y_obs values must be in the range [0, n_trials].")
