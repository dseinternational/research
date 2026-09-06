# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Frozen geometry agrees with the consumers' declared domains and GP graphs."""

import json
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from pytensor.tensor.linalg import solve

from dse_research_utils.statistics.models.hsgp_design import HSGPDesign, calibrate_hsgp_1d, create_hsgp


def test_geometry_can_round_trip_as_plain_scalars_without_recalibration():
    design = HSGPDesign.from_domain([-2.4712, -1, 0, 1.9], m=np.int64(15), c=1.5)
    assert design.m == 15
    assert design.L == (1.9 + 2.4712) / 2 * 1.5
    assert design.center == (-2.4712 + 1.9) / 2
    assert HSGPDesign(**json.loads(json.dumps(asdict(design)))) == design
    with pytest.raises(FrozenInstanceError):
        design.m = 10


@pytest.mark.parametrize("domain", [[-2.5, 1.9], [0, 2], [-1, 1]])
@pytest.mark.parametrize("floor", [None, 1.5, 8.0])
def test_calibration_matches_current_pymc_and_consumer_boundary_formula(domain, floor):
    lo, hi = domain
    lengths = [0.2, 0.3]
    m, c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(domain, lengths, "expquad")
    if floor is not None and floor > c:
        m, _ = pm.gp.hsgp_approx.approx_hsgp_hyperparams(domain, [lengths[0] * c / floor, lengths[1]], "expquad")
        c = floor
    result = calibrate_hsgp_1d(domain, ls_range=tuple(lengths), c_floor=floor)
    assert (result.m, result.L, result.center) == (m, (hi - lo) / 2 * c, (hi + lo) / 2)


def test_query_rows_do_not_enter_calibration_or_move_retained_basis_values():
    design = calibrate_hsgp_1d([-2.5, 1.9], ls_range=(0.2, 0.8))
    x = np.array([-2.0, -0.7, 0.1, 1.5])
    bases = []
    for rows in (x, x[1:], x[1:2], np.r_[x, [-2.4, 1.8]]):
        gp = create_hsgp(design, cov_func=pm.gp.cov.ExpQuad(1, ls=0.4))
        phi, sqrt_psd = gp.prior_linearized(rows[:, None])
        bases.append((phi * sqrt_psd).eval())
    np.testing.assert_array_equal(bases[0][1:], bases[1])
    np.testing.assert_array_equal(bases[0][1:2], bases[2])
    np.testing.assert_array_equal(bases[0], bases[3][: len(x)])


def test_public_center_initialisation_preserves_exact_stored_midpoint():
    # Reconstructing center through center +/- L changes low bits for this pair.
    center, boundary = -0.2856, 3.2784
    assert ((center - boundary) + (center + boundary)) / 2 != center
    design = HSGPDesign(12, boundary, center)
    gp = create_hsgp(design, cov_func=pm.gp.cov.ExpQuad(1, ls=0.5))
    reference = pm.gp.HSGP(m=[12], L=[boundary], cov_func=pm.gp.cov.ExpQuad(1, ls=0.5))
    reference._X_center = np.array([center])  # Current VG reference, never used by the shared implementation.
    x = np.array([[-2.1], [0.5], [1.7]])
    for actual, expected in zip(gp.prior_linearized(x), reference.prior_linearized(x), strict=True):
        np.testing.assert_array_equal(actual.eval(), expected.eval())


@pytest.mark.parametrize("parametrization", ["centered", "noncentered"])
@pytest.mark.parametrize("drop_first", [False, True])
def test_injected_covariance_preserves_named_variables_dims_logp_and_anchored_curve(parametrization, drop_first):
    design = HSGPDesign(10, 3.2, -0.3)
    x = np.array([-1.7, -1.1, 0.2, 1.3, 0.0])
    outcomes = []
    for shared in (False, True):
        with pm.Model(coords={"all_id": np.arange(len(x))}) as model:
            # VG owns these names, the bounded lengthscale and prior order.
            ell_unit = pm.Beta("ell_unit", alpha=2, beta=3)
            ell = pm.Deterministic("ell", 0.2 + 0.8 * ell_unit)
            eta = pm.HalfNormal("eta", sigma=0.4)
            cov = pm.gp.cov.ExpQuad(1, ls=ell)
            before = list(model.named_vars)
            if shared:
                gp = create_hsgp(design, cov_func=cov, parametrization=parametrization, drop_first=drop_first)
            else:
                gp = pm.gp.HSGP(
                    m=[design.m], L=[design.L], cov_func=cov, parametrization=parametrization, drop_first=drop_first
                )
                gp._X_center = np.array([design.center])
            assert list(model.named_vars) == before
            g = gp.prior("g_unit", X=x[:, None], dims="all_id")
            # Consumer-owned projection uses observed rows only, then anchors.
            basis = pt.as_tensor_variable(np.column_stack([np.ones(len(x)), x]))
            observed_basis = basis[:4]
            coef = solve(observed_basis.T @ observed_basis + 1e-6 * pt.eye(2), observed_basis.T @ g[:4], assume_a="pos")
            residual = g - basis @ coef
            curve = pm.Deterministic("latent", 0.2 + eta * (residual - residual[-1]), dims="all_id")
        point = model.initial_point()
        point["g_unit_hsgp_coeffs"] = np.linspace(-1, 1, design.m - drop_first)
        # In the centered form the lengthscale enters the coefficient prior,
        # not the curve conditional on those coefficients. Logp is checked too.
        compiled = model.compile_fn(
            model.replace_rvs_by_values([curve]), inputs=model.value_vars, on_unused_input="ignore"
        )
        outcomes.append(
            (
                list(model.named_vars),
                [rv.name for rv in model.free_RVs],
                model.named_vars_to_dims,
                compiled(point),
                model.compile_logp()(point),
            )
        )
    assert outcomes[0][:3] == outcomes[1][:3]
    np.testing.assert_array_equal(outcomes[0][3], outcomes[1][3])
    assert outcomes[0][4] == outcomes[1][4]
    assert outcomes[1][3][0][-1] == 0.2


def test_basis_supports_symbolic_mutable_inputs_without_recentring():
    design = HSGPDesign(8, 3, 0.25)
    with pm.Model() as model:
        x = pm.Data("x", np.array([[-1.0], [0.0], [1.0]]))
        gp = create_hsgp(design, cov_func=pm.gp.cov.Matern52(1, ls=0.5))
        phi, spectrum = gp.prior_linearized(x)
        assert model.free_RVs == []
    expected = (phi * spectrum).eval()
    with model:
        pm.set_data({"x": np.array([[0.0], [1.0]])})
    np.testing.assert_array_equal((phi * spectrum).eval(), expected[1:])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m": True},
        {"m": 0},
        {"m": 1.5},
        {"L": 0},
        {"L": float("inf")},
        {"L": True},
        {"center": float("nan")},
        {"center": 1e308},
        {"calibration_version": ""},
        {"calibration_version": None},
    ],
)
def test_invalid_saved_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError):
        HSGPDesign(**{"m": 10, "L": 2, "center": 0, **kwargs})


@pytest.mark.parametrize(
    "domain",
    [
        [],
        [1, 1],
        [0, np.nan],
        [0, np.inf],
        [[0, 1], [2, 3]],
        [False, True],
        np.ma.array([0, 1], mask=[False, True]),
        [-1e308, 1e308],
    ],
)
def test_invalid_domain_is_rejected(domain):
    with pytest.raises(ValueError, match="domain"):
        calibrate_hsgp_1d(domain, ls_range=(0.2, 0.8))


@pytest.mark.parametrize("floor", [0, 1, -2, np.nan, np.inf, True])
def test_invalid_calibration_floor_rejected(floor):
    with pytest.raises(ValueError, match="c_floor"):
        calibrate_hsgp_1d([-1, 1], ls_range=(0.2, 0.8), c_floor=floor)


def test_factory_rejects_multidimensional_covariance_and_empty_basis():
    design = HSGPDesign(1, 2, 0)
    with pytest.raises(ValueError, match="one-dimensional"):
        create_hsgp(design, cov_func=pm.gp.cov.ExpQuad(2, ls=1))
    with pytest.raises(ValueError, match="single input column"):
        create_hsgp(design, cov_func=pm.gp.cov.ExpQuad(1, ls=1, active_dims=[1]))
    with pytest.raises(ValueError, match="at least one"):
        create_hsgp(design, cov_func=pm.gp.cov.ExpQuad(1, ls=1), drop_first=True)


@pytest.mark.parametrize("lengths", [(True, 2.0), (0.2, True), ("0.2", 0.8), (0.2, "0.8"), (0.2, 10**1000)])
def test_lengthscales_are_real_numbers_without_boolean_or_string_coercion(lengths):
    with pytest.raises(ValueError, match="ls_range"):
        calibrate_hsgp_1d([-1, 1], ls_range=lengths)


def test_underflowing_domain_half_range_is_rejected_before_calibration(monkeypatch):
    def unexpected_calibration(*args, **kwargs):
        pytest.fail("A zero half-range must be rejected before asking PyMC to divide by it")

    monkeypatch.setattr(pm.gp.hsgp_approx, "approx_hsgp_hyperparams", unexpected_calibration)
    with pytest.raises(ValueError, match=r"half-range.*rescale"):
        calibrate_hsgp_1d([0, np.nextafter(0.0, 1.0)], ls_range=(0.2, 0.8))


@pytest.mark.parametrize(
    ("domain", "lengths", "floor"),
    [
        ([-1, 1], (np.nextafter(0.0, 1.0), 1.0), None),
        ([-1, 1], (0.2, 0.8), 1e308),
        ([-1e150, 1e150], (1e-180, 1e-170), None),
        ([-1e-150, 1e-150], (1e150, 1e160), None),
    ],
)
def test_unrepresentable_recommendations_raise_a_clear_rescaling_error(domain, lengths, floor):
    with pytest.raises(ValueError, match="rescale"):
        calibrate_hsgp_1d(domain, ls_range=lengths, c_floor=floor)


@pytest.mark.parametrize("boundary", [1e-200, 1e200, 1e308])
def test_saved_geometry_rejects_overflowing_or_zero_squared_frequencies(boundary):
    with pytest.raises(ValueError, match="rescale"):
        HSGPDesign(m=3, L=boundary, center=0)


def test_largest_frequency_is_checked_as_well_as_the_first():
    HSGPDesign(m=1, L=1e-153, center=0)
    with pytest.raises(ValueError, match="rescale"):
        HSGPDesign(m=100, L=1e-153, center=0)


def test_frequency_validation_does_not_allocate_one_entry_per_basis_function(monkeypatch):
    def reject_allocation(*args, **kwargs):
        pytest.fail("Saved geometry validation must not allocate the basis")

    monkeypatch.setattr(np, "arange", reject_allocation)
    design = HSGPDesign(m=10**12, L=2.0, center=0)
    assert design.m == 10**12
    with pytest.raises(ValueError, match="rescale"):
        HSGPDesign(m=10**1000, L=2.0, center=0)


@pytest.mark.parametrize("scale", [1e-140, 1e140])
def test_small_and_large_representable_scales_retain_the_same_gp_function(scale):
    bases = []
    for factor in (1.0, scale):
        design = calibrate_hsgp_1d([-factor, factor], ls_range=(0.2 * factor, 0.8 * factor))
        gp = create_hsgp(design, cov_func=pm.gp.cov.ExpQuad(1, ls=np.float64(0.5 * factor)))
        phi, spectrum = gp.prior_linearized(np.array([[-0.5], [0], [0.5]]) * factor)
        basis = (phi * spectrum).eval()
        assert np.isfinite(basis).all()
        assert np.any(basis != 0)
        bases.append(basis)
    np.testing.assert_allclose(bases[0], bases[1], rtol=1e-12, atol=1e-13)
