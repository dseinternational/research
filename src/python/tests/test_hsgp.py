# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest

pm = pytest.importorskip("pymc")

from dse_research_utils.statistics.models.hsgp import (  # noqa: E402
    _approx_hsgp_params,
    build_hsgp_1d,
    build_tau_modifier,
)


@pytest.mark.parametrize("offset,c_floor", [(3.0, 1.5), (0.0, 8.0)])
def test_calibrated_basis_recovers_covariance_at_shortest_lengthscale(offset, c_floor):
    x = np.linspace(-1, 1, 31) + offset
    m, bounds = _approx_hsgp_params(x, (0.2, 0.3), c_floor=c_floor)
    cov = pm.gp.cov.ExpQuad(1, ls=0.2)
    gp = pm.gp.HSGP(m=m, L=bounds, cov_func=cov)
    phi, sqrt_psd = gp.prior_linearized(x[:, None])
    weighted_basis = (phi * sqrt_psd).eval()
    approximate = weighted_basis @ weighted_basis.T
    expected = np.exp(-0.5 * ((x[:, None] - x[None, :]) / 0.2) ** 2)
    np.testing.assert_allclose(approximate, expected, atol=0.01, rtol=0)


@pytest.mark.parametrize("ls_range", [None, (0.2, 0.3)])
def test_hsgp_covariance_does_not_depend_on_input_origin(monkeypatch, ls_range):
    captured = []
    original = pm.gp.HSGP.prior

    def capture(self, name, X, **kwargs):
        captured.append(self)
        return original(self, name, X=X, **kwargs)

    monkeypatch.setattr(pm.gp.HSGP, "prior", capture)
    x = np.linspace(-1, 1, 11)
    for offset in (0.0, 3.0):
        with pm.Model():
            build_hsgp_1d("f", x + offset, ls_range=ls_range)
    np.testing.assert_allclose(captured[0].L.eval(), captured[1].L.eval())


@pytest.mark.parametrize("x", [[], [1, 1], [0, np.nan], [[1, 2], [3, 4]]])
def test_hsgp_rejects_invalid_inputs(x):
    with pytest.raises(ValueError, match="X"), pm.Model():
        build_hsgp_1d("f", x)


@pytest.mark.parametrize("ls_range", [0.2, (0.2,), (0.2, 0.3, 0.4), (0, 0.3), (-1, 0.3), (0.3, 0.2), (0.2, np.nan)])
def test_hsgp_rejects_invalid_lengthscale_ranges(ls_range):
    with pytest.raises(ValueError, match="ls_range"), pm.Model():
        build_hsgp_1d("f", [-1, 1], ls_range=ls_range)


@pytest.mark.parametrize("x,c_floor,expected_m", [([-1, 1], 8.0, 70), ([-2.5, 1.9], 1.5, 28)])
def test_boundary_floor_uses_unrounded_pymc_basis_formula(x, c_floor, expected_m):
    m, _ = _approx_hsgp_params(np.asarray(x), (0.2, 0.3), c_floor=c_floor)
    assert m == [expected_m]


@pytest.mark.parametrize("builder", [build_hsgp_1d, build_tau_modifier])
def test_frozen_hsgp_design_preserves_curve_on_subset(builder):
    x = np.array([-2.4712, -1.0, 0.0, 0.5, 1.9])
    center = (x.min() + x.max()) / 2
    boundary = (x.max() - x.min()) / 2 * 1.5
    curves = []
    for values in (x, x[1:], x[1:2]):
        with pm.Model() as model:
            curve = builder("f", values, m=15, L=boundary, center=center)
        point = model.initial_point()
        point["f__g_unit_hsgp_coeffs"] = np.linspace(-1, 1, 15)
        value_curve = model.replace_rvs_by_values([curve])[0]
        curves.append(model.compile_fn(value_curve, inputs=model.value_vars)(point))
    np.testing.assert_allclose(curves[0][1:], curves[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(curves[0][1:2], curves[2], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("m", [0, -1, 1.5, True])
def test_hsgp_rejects_invalid_basis_sizes(m):
    with pytest.raises(ValueError, match="m must be a positive integer"), pm.Model():
        build_hsgp_1d("f", [-1, 1], m=m)


@pytest.mark.parametrize(
    "kwargs",
    [{"L": 0}, {"L": np.inf}, {"L": 0.5}, {"center": 0}, {"L": 2, "center": np.nan}, {"L": 2, "ls_range": (0.2, 0.3)}],
)
def test_hsgp_rejects_invalid_frozen_design(kwargs):
    with pytest.raises(ValueError), pm.Model():
        build_hsgp_1d("f", [-1, 1], **kwargs)


def test_build_hsgp_1d_registers_named_nodes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    with pm.Model() as model:
        build_hsgp_1d("f_age", x, m=10)
    names = set(model.named_vars)
    assert {"f_age", "f_age__eta", "f_age__ell"}.issubset(names)
    assert any(n.startswith("f_age__g_unit") for n in names)
    assert "f_age" in {d.name for d in model.deterministics}
    logp = model.compile_logp()(model.initial_point())
    assert np.isfinite(logp)


def test_build_tau_modifier_smaller_default_basis():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    with pm.Model() as model:
        build_tau_modifier("g_tau", x)
    # Default m=15 basis; node naming matches build_hsgp_1d.
    assert {"g_tau", "g_tau__eta", "g_tau__ell"}.issubset(set(model.named_vars))
