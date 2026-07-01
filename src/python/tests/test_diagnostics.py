# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the shared convergence gate and its report banner."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from dse_research_utils.statistics.diagnostics import (
    BFMI_THRESHOLD,
    ESS_THRESHOLD,
    RHAT_MAX,
    _bfmi_per_chain,
    convergence_banner_markdown,
    write_diagnostics_summary,
)


def _make_trace(*, n_div: int = 0, seed: int = 0) -> xr.DataTree:
    """A small, well-mixed synthetic trace as an ArviZ-style DataTree.

    Two scalar parameters over 2 chains x 800 iid draws (so R-hat ~ 1.0 and
    ESS well above the gate), plus a ``sample_stats`` group carrying ``diverging``
    and ``energy`` (white-noise energy -> BFMI ~ 2).
    """
    rng = np.random.default_rng(seed)
    shape = (2, 800)
    posterior = xr.Dataset(
        {
            "mu": (("chain", "draw"), rng.normal(0.0, 1.0, size=shape)),
            "sigma": (("chain", "draw"), rng.normal(2.0, 0.5, size=shape)),
        },
        coords={"chain": range(shape[0]), "draw": range(shape[1])},
    )
    diverging = np.zeros(shape, dtype=bool)
    if n_div:
        flat = diverging.reshape(-1)
        flat[:n_div] = True
        diverging = flat.reshape(shape)
    sample_stats = xr.Dataset(
        {
            "diverging": (("chain", "draw"), diverging),
            "energy": (("chain", "draw"), rng.normal(size=shape)),
        },
        coords={"chain": range(shape[0]), "draw": range(shape[1])},
    )
    return xr.DataTree.from_dict({"posterior": posterior, "sample_stats": sample_stats})


def test_bfmi_per_chain_matches_reference():
    rng = np.random.default_rng(0)
    energy = rng.normal(size=(2, 500))
    ss = xr.Dataset({"energy": (("chain", "draw"), energy)})
    trace = SimpleNamespace(sample_stats=ss)
    bf = _bfmi_per_chain(trace)
    assert bf is not None and len(bf) == 2
    for c in range(2):
        e = energy[c]
        ref = float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2))
        assert bf[c] == pytest.approx(ref)


def test_bfmi_per_chain_handles_missing_energy():
    trace = SimpleNamespace(sample_stats=xr.Dataset({}))
    assert _bfmi_per_chain(trace) is None


def test_bfmi_per_chain_nan_for_degenerate_chain():
    # A chain with constant (zero-variance) energy has an undefined BFMI: the
    # denominator Sum((E - mean)^2) is 0, so the helper returns NaN for it.
    energy = np.vstack([np.full(500, 2.0), np.random.default_rng(0).normal(size=500)])
    ss = xr.Dataset({"energy": (("chain", "draw"), energy)})
    bf = _bfmi_per_chain(SimpleNamespace(sample_stats=ss))
    assert bf is not None and len(bf) == 2
    assert not np.isfinite(bf[0])  # constant-energy chain -> NaN
    assert np.isfinite(bf[1])


def _make_trace_degenerate_bfmi(seed: int = 1) -> xr.DataTree:
    """Well-mixed posterior, but chain 0 has constant energy (NaN BFMI)."""
    rng = np.random.default_rng(seed)
    shape = (2, 800)
    posterior = xr.Dataset(
        {
            "mu": (("chain", "draw"), rng.normal(0.0, 1.0, size=shape)),
            "sigma": (("chain", "draw"), rng.normal(2.0, 0.5, size=shape)),
        },
        coords={"chain": range(shape[0]), "draw": range(shape[1])},
    )
    energy = rng.normal(size=shape)
    energy[0, :] = 3.14  # constant -> zero energy variance -> NaN BFMI
    sample_stats = xr.Dataset(
        {
            "diverging": (("chain", "draw"), np.zeros(shape, dtype=bool)),
            "energy": (("chain", "draw"), energy),
        },
        coords={"chain": range(shape[0]), "draw": range(shape[1])},
    )
    return xr.DataTree.from_dict({"posterior": posterior, "sample_stats": sample_stats})


def test_write_diagnostics_summary_degenerate_bfmi_fails_and_serialises(tmp_path):
    # A non-finite per-chain BFMI must fail the gate (order-independently), not
    # slip through because a healthy chain sorts first under the builtin min().
    payload = write_diagnostics_summary(_make_trace_degenerate_bfmi(), str(tmp_path))
    assert payload["checks"]["bfmi"] is False
    assert payload["passed"] is False

    # ...and must serialise as valid JSON (None, never a bare ``NaN`` token).
    text = (tmp_path / "diagnostics_summary.json").read_text()
    assert "NaN" not in text
    reloaded = json.loads(text)
    assert reloaded["bfmi_per_chain"][0] is None
    assert reloaded["bfmi_per_chain"][1] is not None

    # The banner tolerates the None entry rather than raising on ``f"{None:.2f}"``.
    assert "n/a" in convergence_banner_markdown(payload)


def test_write_diagnostics_summary_passes_on_clean_trace(tmp_path):
    tables: dict = {}
    payload = write_diagnostics_summary(_make_trace(), str(tmp_path), tables=tables)

    # File written, and the payload mirrored into the supplied tables cache.
    written = json.loads((tmp_path / "diagnostics_summary.json").read_text())
    assert written == payload
    assert tables["diagnostics_summary"] == payload

    assert payload["passed"] is True
    assert payload["divergences"] == 0
    assert payload["checks"] == {"rhat": True, "ess": True, "divergences": True, "bfmi": True}
    assert payload["max_rhat"] <= RHAT_MAX
    assert payload["min_ess"] >= ESS_THRESHOLD
    assert min(payload["bfmi_per_chain"]) >= BFMI_THRESHOLD
    assert payload["thresholds"] == {
        "rhat_max": RHAT_MAX,
        "ess_threshold": ESS_THRESHOLD,
        "bfmi_threshold": BFMI_THRESHOLD,
    }


def test_write_diagnostics_summary_flags_divergences(tmp_path):
    payload = write_diagnostics_summary(_make_trace(n_div=5), str(tmp_path))
    assert payload["divergences"] == 5
    assert payload["checks"]["divergences"] is False
    assert payload["passed"] is False


def test_convergence_banner_none_is_placeholder():
    md = convergence_banner_markdown(None)
    assert ".callout-note" in md
    assert "No `diagnostics_summary.json`" in md


def test_convergence_banner_pass_renders_tip():
    summary = {
        "passed": True,
        "thresholds": {"rhat_max": RHAT_MAX, "ess_threshold": ESS_THRESHOLD, "bfmi_threshold": BFMI_THRESHOLD},
        "divergences": 0,
        "max_rhat": 1.003,
        "min_ess": 1200.0,
        "bfmi_per_chain": [1.9, 2.1],
        "rhat_failing": [],
        "ess_failing": [],
    }
    md = convergence_banner_markdown(summary)
    assert ".callout-tip" in md and "PASS" in md
    assert "Divergences:** 0" in md
    assert "provisional" not in md


def test_convergence_banner_fail_renders_warning_and_note():
    summary = {
        "passed": False,
        "thresholds": {"rhat_max": RHAT_MAX, "ess_threshold": ESS_THRESHOLD, "bfmi_threshold": BFMI_THRESHOLD},
        "divergences": 3,
        "max_rhat": 1.08,
        "min_ess": 120.0,
        "bfmi_per_chain": [0.2, 0.4],
        "rhat_failing": ["mu"],
        "ess_failing": ["mu", "sigma"],
    }
    md = convergence_banner_markdown(summary)
    assert ".callout-warning" in md and "REVIEW" in md
    assert "Parameters needing attention" in md
    assert "provisional" in md
    # dev_note can be suppressed.
    assert "provisional" not in convergence_banner_markdown(summary, dev_note=False)
