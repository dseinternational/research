# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gaussian-process kernel functions (textbook forms).

Adapted from the PyMC statistical-rethinking lecture notes:
https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/16-Gaussian_Processes.html
"""

from __future__ import annotations

import numpy as np


def quadratic_distance_kernel(X0, X1, eta=1, sigma=0.5):
    # Use linear algebra identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
    X0_norm = np.sum(X0**2, axis=-1)
    X1_norm = np.sum(X1**2, axis=-1)
    squared_distances = X0_norm[:, None] + X1_norm[None, :] - 2 * X0 @ X1.T
    rho = 1 / sigma**2
    return eta**2 * np.exp(-rho * squared_distances)


def ornstein_uhlenbeck_kernel(X0, X1, eta_squared=1, rho=4):
    # ``plot_kernel_function`` and similar helpers pass 2D column vectors
    # (shape ``(n, 1)``); flatten so the pairwise-distance broadcast
    # produces the expected ``(n, n)`` matrix rather than ``(n, n, 1)``.
    X0 = np.asarray(X0).reshape(-1)
    X1 = np.asarray(X1).reshape(-1)
    distances = np.abs(X1[None, :] - X0[:, None])
    return eta_squared * np.exp(-rho * distances)


def periodic_kernel(X0, X1, eta=1, sigma=1, periodicity=0.5):
    X0 = np.asarray(X0).reshape(-1)
    X1 = np.asarray(X1).reshape(-1)
    distances = np.sin((X1[None, :] - X0[:, None]) * periodicity) ** 2
    rho = 2 / sigma**2
    return eta**2 * np.exp(-rho * distances)
