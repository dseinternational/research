# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Beta-binomial likelihood helper for bounded-count outcomes."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from dse_research_utils.math.constants import EPSILON


def beta_binomial_from_p(
    name: str,
    p: pt.TensorVariable,
    n_trials: int | np.ndarray,
    kappa: pt.TensorVariable,
    observed: np.ndarray | None = None,
    dims: tuple[str, ...] | str | None = None,
) -> pt.TensorVariable:
    """
    Register a Beta-Binomial observation node parameterised by ``(p, kappa)``.

    ``alpha = p * kappa``, ``beta = (1 - p) * kappa``, with ``p`` clipped away
    from 0/1 to avoid NaNs during sampling. Use this form when the caller
    already holds the success probability (e.g. as a named ``Deterministic``,
    or after applying a non-logit link); :func:`beta_binomial_from_logit`
    wraps it for the plain logit link.
    """
    p_clip = pm.math.clip(p, EPSILON, 1 - EPSILON)
    alpha = p_clip * kappa
    beta = (1 - p_clip) * kappa
    return pm.BetaBinomial(
        name,
        n=n_trials,
        alpha=alpha,
        beta=beta,
        observed=observed,
        dims=dims,
    )


def beta_binomial_from_logit(
    name: str,
    eta: pt.TensorVariable,
    n_trials: int | np.ndarray,
    kappa: pt.TensorVariable,
    observed: np.ndarray | None = None,
    dims: tuple[str, ...] | str | None = None,
) -> pt.TensorVariable:
    """
    Register a Beta-Binomial observation node parameterised by ``(logit(mu), kappa)``.

    ``mu = sigmoid(eta)``; see :func:`beta_binomial_from_p` for the
    ``(alpha, beta)`` construction and the clipping.
    """
    return beta_binomial_from_p(
        name,
        pm.math.sigmoid(eta),
        n_trials,
        kappa,
        observed=observed,
        dims=dims,
    )
