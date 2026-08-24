# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ArviZ subplot-budget helpers and the categorical palette."""

from __future__ import annotations

import arviz as az
import numpy as np
import xarray as xr

from dse_research_utils.plot.diagnostics_mcmc import (
    capped_plot_var_names,
    plot_required_subplots,
    plot_variable_count,
)
from dse_research_utils.plot.styles import categorical_palette


def _trace_with_scalar_and_vector_parameters() -> xr.DataTree:
    posterior = xr.Dataset(
        data_vars={
            "alpha": (("chain", "draw"), np.ones((1, 2))),
            "beta": (("chain", "draw", "coef"), np.ones((1, 2, 2))),
        },
        coords={"chain": [0], "draw": [0, 1], "coef": ["intercept", "slope"]},
        attrs={"sample_dims": ["chain", "draw"]},
    )
    return xr.DataTree.from_dict({"posterior": posterior})


def _trace_with_large_observed_diagnostic() -> xr.DataTree:
    posterior = xr.Dataset(
        data_vars={
            "alpha": (("chain", "draw"), np.ones((1, 2))),
            "kappa_obs": (("chain", "draw", "obs"), np.ones((1, 2, 100))),
        },
        coords={"chain": [0], "draw": [0, 1], "obs": range(100)},
        attrs={"sample_dims": ["chain", "draw"]},
    )
    return xr.DataTree.from_dict({"posterior": posterior})


def test_plot_required_subplots_counts_non_sample_dimensions():
    trace = _trace_with_scalar_and_vector_parameters()
    assert plot_required_subplots(trace, ["alpha", "beta"], squared=True) == 9


def test_capped_plot_var_names_limits_pair_plot_grid():
    trace = _trace_with_scalar_and_vector_parameters()
    with az.rc_context({"plot.max_subplots": 8}):
        var_names = capped_plot_var_names(trace, ["alpha", "beta"], squared=True)
    assert var_names == ["alpha"]
    assert plot_required_subplots(trace, var_names, squared=True) <= 8


def test_capped_plot_var_names_keeps_pair_plot_vars_when_limit_is_sufficient():
    trace = _trace_with_scalar_and_vector_parameters()
    with az.rc_context({"plot.max_subplots": 9}):
        assert capped_plot_var_names(trace, ["alpha", "beta"], squared=True) == ["alpha", "beta"]


def test_capped_plot_var_names_skips_large_observed_diagnostic():
    trace = _trace_with_large_observed_diagnostic()
    with az.rc_context({"plot.max_subplots": 40}):
        var_names = capped_plot_var_names(trace, ["alpha", "kappa_obs"])
    assert var_names == ["alpha"]
    assert plot_variable_count(trace, "kappa_obs") == 100


def test_categorical_palette_distinct_and_auto_widens():
    small = categorical_palette(4)
    assert len(small) == 4
    assert len(set(small)) == 4
    wide = categorical_palette(15)  # auto-widens to tab20
    assert len(wide) == 15
    assert len(set(wide)) == 15


def test_categorical_palette_samples_continuous_colormaps_evenly():
    # A continuous colormap used as qualitative must be sampled, not cycled
    # (cycling would return near-identical low-index colours).
    colours = categorical_palette(5, palette="viridis")
    assert len(set(colours)) == 5
    assert colours[0] != colours[-1]


def test_categorical_palette_explicit_qualitative_palette_cycles():
    colours = categorical_palette(12, palette="tab10")
    assert len(colours) == 12
    assert colours[0] == colours[10]  # cycled past the 10-colour palette
