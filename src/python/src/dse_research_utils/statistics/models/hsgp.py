# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Hilbert-space Gaussian-process (HSGP) helpers.

Two constructors are exposed:

- :func:`build_hsgp_1d` — 1D HSGP with default amplitude ``HalfNormal(0.3)`` and
  ``InverseGamma(3, 1)`` lengthscale. Suitable for smooth main effects.
- :func:`build_tau_modifier` — 1D HSGP with the same tight amplitude default,
  for effect-modification GPs (e.g. an age-varying treatment effect).

Both accept ``amplitude_prior`` / ``lengthscale_prior`` (preliz distributions) so
a caller can inject its own priors; the defaults here are generic. Lengthscale
calibration follows ``pm.gp.hsgp_approx.approx_hsgp_hyperparams`` so the basis
size ``m`` and boundary factor ``c`` cover the observed input range at the chosen
lengthscale floor.
"""

from __future__ import annotations

import numpy as np
import preliz as pz
import pymc as pm
import pytensor.tensor as pt
from preliz.distributions.distributions import Continuous


def _default_amplitude_prior() -> Continuous:
    """Generic HSGP amplitude prior ``HalfNormal(0.3)`` (deliberately tight)."""
    return pz.HalfNormal(sigma=0.3)


def _default_lengthscale_prior() -> Continuous:
    """Generic HSGP lengthscale prior ``InverseGamma(3, 1)`` on standardised inputs."""
    return pz.InverseGamma(alpha=3.0, beta=1.0)


def _approx_hsgp_params(
    x: np.ndarray,
    ls_range: tuple[float, float],
    c_floor: float | None = None,
) -> tuple[list[int], list[float]]:
    """Return ``(m, L)`` sized to cover ``x`` for the given lengthscale range.

    ``c_floor`` is a minimum boundary factor. Increasing the boundary also
    increases the basis size to retain the recommended frequency coverage.
    The half-range matches PyMC's centring of inputs at their midpoint.
    """
    x = np.asarray(x, dtype=float)
    x_min, x_max = float(x.min()), float(x.max())
    try:
        lengths = np.asarray(ls_range, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("ls_range must contain two finite, positive, increasing lengthscales.") from exc
    if lengths.shape != (2,) or not np.all(np.isfinite(lengths)) or not 0 < lengths[0] < lengths[1]:
        raise ValueError("ls_range must contain two finite, positive, increasing lengthscales.")
    m, c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
        x_range=[x_min, x_max],
        lengthscale_range=lengths.tolist(),
        cov_func="expquad",
    )
    if c_floor is not None:
        new_c = max(float(c), float(c_floor))
        if new_c > c:
            # Reuse PyMC's formula before integer truncation. Scaling its
            # already-rounded m can understate the required frequency coverage.
            m, _ = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
                x_range=[x_min, x_max],
                lengthscale_range=[float(lengths[0] * c / new_c), float(lengths[1])],
                cov_func="expquad",
            )
        c = new_c
    S = (x_max - x_min) / 2.0
    return [int(m)], [float(S * c)]


def build_hsgp_1d(
    name: str,
    X: np.ndarray,
    *,
    m: int = 20,
    c: float = 1.5,
    L: float | None = None,
    center: float | None = None,
    amplitude_prior: Continuous | None = None,
    lengthscale_prior: Continuous | None = None,
    ls_range: tuple[float, float] | None = None,
) -> pt.TensorVariable:
    """
    Construct a 1D HSGP prior evaluated at ``X`` (standardised input).

    Parameters
    ----------
    name
        Variable-name prefix. The eta, ell, and GP basis weights are named
        ``"{name}__eta"``, ``"{name}__ell"`` and ``"{name}__g_unit"``.
    X
        Standardised 1D inputs, shape ``(n,)`` or ``(n, 1)``.
    m, c
        HSGP basis size and boundary factor. With ``ls_range``, ``m`` is
        calibrated and ``c`` sets a minimum boundary factor.
    L, center
        Optional fixed half-width and midpoint of the basis domain. With ``L``,
        ``c`` is ignored and ``ls_range`` must be omitted. ``center`` requires
        ``L`` and otherwise defaults to the midpoint of ``X``. Preserve ``m``,
        ``L`` and ``center`` from the full design when fitting subsets; fixing
        only ``L`` still lets the basis move with the subset's midpoint.
    amplitude_prior
        preliz distribution for eta. Defaults to ``HalfNormal(0.3)``.
    lengthscale_prior
        preliz distribution for ell. Defaults to ``InverseGamma(3, 1)``.
    ls_range
        Optional ``(ls_low, ls_high)`` on the standardised scale. When
        provided, ``m`` and ``c`` are calibrated from it.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or X.shape[1] != 1 or X.shape[0] == 0:
        raise ValueError("X must have shape (n,) or (n, 1) with at least one observation.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite values.")
    if L is None and X.min() == X.max():
        raise ValueError("X must contain finite values with a positive range.")
    if L is None and (not np.isfinite(c) or c <= 1):
        raise ValueError("c must be finite and greater than 1.")
    if center is not None and (L is None or not np.isfinite(center)):
        raise ValueError("center must be finite and requires an explicit L.")
    if ls_range is None and (isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)) or m < 1):
        raise ValueError("m must be a positive integer.")

    if L is not None:
        if ls_range is not None:
            raise ValueError("Explicit L cannot be combined with ls_range; preserve the calibrated m instead.")
        midpoint = float((X.min() + X.max()) / 2.0) if center is None else float(center)
        if not np.isfinite(L) or L <= 0 or np.max(np.abs(X - midpoint)) >= L:
            raise ValueError("L must be finite, positive and contain all X strictly within center +/- L.")
        m_val, L_val = int(m), [float(L)]
    elif ls_range is not None:
        # When the caller supplies an explicit ``c``, treat it as a
        # minimum floor on the calibrated boundary factor — otherwise
        # the kwarg was silently ignored.
        m_list, bounds = _approx_hsgp_params(X[:, 0], ls_range, c_floor=c)
        m_val = m_list[0]
        L_val = bounds
    else:
        S = float((X.max() - X.min()) / 2.0)
        m_val = int(m)
        L_val = [S * float(c)]

    amplitude = (amplitude_prior or _default_amplitude_prior()).to_pymc(f"{name}__eta")
    lengthscale = (lengthscale_prior or _default_lengthscale_prior()).to_pymc(f"{name}__ell")

    cov = pm.gp.cov.ExpQuad(1, ls=lengthscale)
    hsgp = pm.gp.HSGP(cov_func=cov, m=[m_val], L=L_val)
    if center is not None:
        # PyMC retains the midpoint of the first prior_linearized call. Seed
        # that public API with the frozen domain, then evaluate the actual rows.
        # This creates no random variables and avoids setting private GP state.
        hsgp.prior_linearized(np.array([[center - L_val[0]], [center + L_val[0]]]))
    g_unit = hsgp.prior(f"{name}__g_unit", X=X)
    return pm.Deterministic(name, amplitude * g_unit)


def build_tau_modifier(
    name: str,
    X: np.ndarray,
    *,
    m: int = 15,
    c: float = 1.5,
    L: float | None = None,
    center: float | None = None,
    amplitude_prior: Continuous | None = None,
    lengthscale_prior: Continuous | None = None,
    ls_range: tuple[float, float] | None = None,
) -> pt.TensorVariable:
    """Same API as :func:`build_hsgp_1d` but with a smaller default basis (``m=15``)."""
    return build_hsgp_1d(
        name,
        X,
        m=m,
        c=c,
        L=L,
        center=center,
        amplitude_prior=amplitude_prior or _default_amplitude_prior(),
        lengthscale_prior=lengthscale_prior,
        ls_range=ls_range,
    )
