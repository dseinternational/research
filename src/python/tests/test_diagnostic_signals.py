# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public numerical diagnostics retain missing signals and named draw axes."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dse_research_utils.statistics.diagnostics import bfmi_per_chain, diagnostic_extrema


def test_bfmi_uses_named_axes_and_keeps_chain_order():
    energy = xr.DataArray(
        [[0.0, 1.0, 0.0, 2.0], [1.0, 2.0, 4.0, 3.0]],
        dims=("chain", "draw"),
        coords={"chain": [7, 2], "draw": [4, 5, 6, 7]},
    )
    expected = [float(np.sum(np.diff(values) ** 2) / np.sum((values - values.mean()) ** 2)) for values in energy.values]
    for array in (energy, energy.transpose("draw", "chain")):
        trace = SimpleNamespace(sample_stats=xr.Dataset({"energy": array}))
        np.testing.assert_allclose(bfmi_per_chain(trace), expected)


def test_bfmi_single_chain_constant_energy_and_missing_energy():
    trace = SimpleNamespace(sample_stats=xr.Dataset({"energy": ("draw", [1, 1, 1])}))
    assert np.isnan(bfmi_per_chain(trace)[0])
    assert bfmi_per_chain(SimpleNamespace()) is None
    assert bfmi_per_chain(SimpleNamespace(sample_stats=xr.Dataset())) is None


def test_existing_summary_retains_unrounded_extrema_and_unavailable_names():
    summary = pd.DataFrame(
        {
            "r_hat": [1.01004, "unavailable", 1.0],
            "ess_bulk": [399.8, 500.0, 600.0],
            "ess_tail": [550.0, np.nan, 700.0],
            "mean": [999, 999, 999],
        },
        index=["boundary", "unknown", "healthy"],
    )
    original = summary.copy(deep=True)
    assert diagnostic_extrema(summary) == (1.01004, 399.8, ("unknown",))
    pd.testing.assert_frame_equal(summary, original)


def test_missing_diagnostic_column_does_not_hide_unavailable_rows():
    summary = pd.DataFrame({"r_hat": [1.0], "ess_bulk": [500]}, index=["a"])
    assert diagnostic_extrema(summary) == (1.0, 500.0, ("a",))
    max_rhat, min_ess, unavailable = diagnostic_extrema(pd.DataFrame(index=["a"]))
    assert np.isnan(max_rhat) and np.isnan(min_ess)
    assert unavailable == ("a",)
    with pytest.raises(ValueError, match="No parameters"):
        diagnostic_extrema(pd.DataFrame())
