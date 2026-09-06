# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Frozen one-dimensional HSGP geometry, independent of priors and query rows."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _domain_limits(domain: ArrayLike) -> tuple[float, float]:
    if np.ma.isMaskedArray(domain):
        raise ValueError("domain must not be a masked array")
    values = np.asarray(domain)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or not values.size or values.dtype.kind not in "iuf":
        raise ValueError("domain must be a nonempty real array of shape (n,) or (n, 1)")
    try:
        with np.errstate(over="raise", invalid="raise"):
            values = np.asarray(values, dtype=float)
    except FloatingPointError as exc:
        raise ValueError("domain must contain finite float64 values") from exc
    if not np.isfinite(values).all() or values.min() == values.max():
        raise ValueError("domain must contain finite values with a positive range")
    lo, hi = float(values.min()), float(values.max())
    if not np.isfinite(hi - lo) or not np.isfinite(hi + lo) or (hi - lo) / 2 == 0:
        raise ValueError("domain half-range and midpoint must be representable in float64; rescale the inputs")
    return lo, hi


def _validate_eigenvalues(m: int, boundary: float) -> None:
    """Check PyMC's endpoint arithmetic without allocating all basis indices."""
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            indices = np.asarray([1, m], dtype=float)
            # Match calc_eigenvalues: numerator and denominator are evaluated
            # before division, followed by squaring. A rearranged formula could
            # accept scales for which PyMC's actual computation loses the basis.
            eigenvalues = np.square((np.pi * indices) / (2 * np.float64(boundary)))
            normalization = 1 / np.sqrt(np.float64(boundary))
    except (ArithmeticError, ValueError) as exc:
        raise ValueError("HSGP basis arithmetic is not representable in float64; rescale the input domain") from exc
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0) or not np.isfinite(normalization):
        raise ValueError("HSGP eigenvalues must remain finite and positive in float64; rescale the input domain")


def _recommend_geometry(lo: float, hi: float, lengths: tuple[float, float]) -> tuple[int, float]:
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            return pm.gp.hsgp_approx.approx_hsgp_hyperparams(
                x_range=[lo, hi], lengthscale_range=list(lengths), cov_func="expquad"
            )
    except (ArithmeticError, ValueError) as exc:
        raise ValueError(
            "HSGP calibration arithmetic is not representable in float64; rescale the domain and lengthscales "
            "or revise their range"
        ) from exc


@dataclass(frozen=True)
class HSGPDesign:
    """Realised HSGP geometry that can be saved and replayed on new rows.

    The calibration label identifies the recipe, not a model identity or a
    numerical-library version. Save the actual scalars alongside the consumer's
    own input scale, covariance, prior and implementation metadata.
    """

    m: int
    """Positive number of basis functions before any drop-first option."""
    L: float
    """Positive half-width of the approximation domain, in input units."""
    center: float
    """Fixed domain midpoint, in the same input units."""
    calibration_version: str = "explicit-v1"
    """Recipe label; explicit geometry defaults to ``explicit-v1``."""

    def __post_init__(self) -> None:
        if isinstance(self.m, bool | np.bool_) or not isinstance(self.m, int | np.integer) or self.m < 1:
            raise ValueError("m must be a positive integer")
        boundary = _finite_real(self.L, "L")
        center = _finite_real(self.center, "center")
        if boundary <= 0:
            raise ValueError("L must be positive")
        if not np.isfinite(center - boundary) or not np.isfinite(center + boundary):
            raise ValueError("The HSGP domain endpoints must be finite")
        if not center - boundary < center < center + boundary or not np.isfinite(center + center):
            raise ValueError("The HSGP domain and PyMC midpoint must be representable in float64")
        if not isinstance(self.calibration_version, str) or not self.calibration_version.strip():
            raise ValueError("calibration_version must be a nonempty string")
        _validate_eigenvalues(int(self.m), boundary)
        object.__setattr__(self, "m", int(self.m))
        object.__setattr__(self, "L", boundary)
        object.__setattr__(self, "center", center)

    @classmethod
    def from_domain(cls, domain: ArrayLike, *, m: int, c: float) -> HSGPDesign:
        """Fix geometry from a declared domain, explicit basis count and factor.

        ``domain`` has shape ``(n,)`` or ``(n, 1)`` and a positive finite range.
        ``c > 1`` expands its half-range. Choose the training/prediction domain
        deliberately; do not append report-grid rows and recalibrate a saved fit.
        """
        lo, hi = _domain_limits(domain)
        factor = _finite_real(c, "c")
        if factor <= 1:
            raise ValueError("c must be greater than 1")
        return cls(m=m, L=(hi - lo) / 2 * factor, center=(lo + hi) / 2, calibration_version="domain-factor-v1")


def calibrate_hsgp_1d(
    domain: ArrayLike,
    *,
    ls_range: tuple[float, float],
    c_floor: float | None = None,
) -> HSGPDesign:
    """Calibrate ExpQuad geometry with PyMC's one-dimensional recommendation.

    Parameters
    ----------
    domain
        Finite input-scale domain with a positive range, shape ``(n,)`` or
        ``(n, 1)``. Typically the two declared training/prediction endpoints.
    ls_range
        Positive increasing lower/upper lengthscales in the same input units.
    c_floor
        Optional minimum boundary factor, strictly greater than one. None uses
        the unmodified recommendation. Expanding the boundary recalculates the
        basis count before integer truncation to retain frequency coverage.

    Returns
    -------
    HSGPDesign
        Exact realised basis count, half-width and midpoint. This does not set
        priors, create random variables, or establish approximation accuracy.

    Notes
    -----
    The recipe is for the squared-exponential (ExpQuad) covariance. The result
    depends on the installed PyMC version. Replay saved scalars instead of
    recalibrating when evaluating subsets or changing a plotting grid.
    Inputs whose half-range, recommendation or basis arithmetic cannot be
    represented in float64 are rejected with ValueError. Rescale the domain and
    its lengthscales together, or revise their range, before calibrating again.
    """
    lo, hi = _domain_limits(domain)
    if np.ma.isMaskedArray(ls_range):
        raise ValueError("ls_range must not be masked")
    try:
        entries = tuple(ls_range)
        if len(entries) != 2:
            raise ValueError("Expected two lengthscales")
        lengths = (_finite_real(entries[0], "ls_range lower bound"), _finite_real(entries[1], "ls_range upper bound"))
    except (TypeError, ValueError) as exc:
        raise ValueError("ls_range must contain two finite, positive, increasing lengthscales") from exc
    if not 0 < lengths[0] < lengths[1]:
        raise ValueError("ls_range must contain two finite, positive, increasing lengthscales")
    floor = None if c_floor is None else _finite_real(c_floor, "c_floor")
    if floor is not None and floor <= 1:
        raise ValueError("c_floor must be greater than 1")
    m, c = _recommend_geometry(lo, hi, lengths)
    if floor is not None and floor > c:
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                # Retain the existing NumPy scalar operation order for parity.
                lower = float(np.float64(lengths[0]) * c / floor)
            if not np.isfinite(lower) or lower <= 0:
                raise ValueError("The adjusted lengthscale is not finite and positive")
        except (ArithmeticError, ValueError) as exc:
            raise ValueError(
                "HSGP boundary-floor arithmetic is not representable; rescale inputs or revise c_floor"
            ) from exc
        m, _ = _recommend_geometry(lo, hi, (lower, lengths[1]))
        c = floor
    return HSGPDesign(m=m, L=(hi - lo) / 2 * c, center=(lo + hi) / 2, calibration_version="pymc-expquad-v1")


def create_hsgp(
    design: HSGPDesign,
    *,
    cov_func: pm.gp.cov.Covariance,
    parametrization: Literal["centered", "noncentered"] = "noncentered",
    drop_first: bool = False,
) -> pm.gp.HSGP:
    """Create a fresh PyMC HSGP with a frozen centre and injected covariance.

    Parameters
    ----------
    design
        Saved one-dimensional geometry. All later inputs use this fixed domain.
    cov_func
        A one-dimensional PyMC covariance with a power spectral density.
        The caller creates its lengthscale, amplitude and other model variables.
    parametrization
        PyMC coefficient parameterisation; defaults to noncentered.
    drop_first
        Whether PyMC drops the first basis function. At least one must remain.

    Returns
    -------
    pymc.gp.HSGP
        A new object ready for ``prior_linearized``, ``prior`` or ``conditional``
        after a prior. Construction creates no random variables or model names.

    Notes
    -----
    A public ``prior_linearized`` call on the centre alone fixes PyMC's midpoint
    without private-state mutation or rounding the centre through ``center +/- L``.
    Inputs to subsequent PyMC calls must be finite, in the declared input units,
    and inside ``center +/- L``. Those mutable/symbolic inputs remain the caller's
    responsibility. This factory chooses no priors, dimensions or variable names.
    """
    if not isinstance(design, HSGPDesign):
        raise TypeError("design must be an HSGPDesign")
    if not isinstance(cov_func, pm.gp.cov.Covariance) or cov_func.input_dim != 1:
        raise ValueError("cov_func must be a one-dimensional PyMC covariance")
    if not np.array_equal(cov_func.active_dims, [0]):
        raise ValueError("cov_func must act on the single input column")
    if parametrization not in ("centered", "noncentered"):
        raise ValueError("parametrization must be centered or noncentered")
    if not isinstance(drop_first, bool) or (drop_first and design.m == 1):
        raise ValueError("drop_first must be boolean and must leave at least one basis function")
    gp = pm.gp.HSGP(
        m=[design.m], L=[design.L], cov_func=cov_func, parametrization=parametrization, drop_first=drop_first
    )
    gp.prior_linearized(np.array([[design.center]]))
    return gp
