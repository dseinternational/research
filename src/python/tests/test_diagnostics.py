# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the shared convergence gate and its report banner."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dse_research_utils.statistics.diagnostics import (
    BFMI_THRESHOLD,
    ESS_THRESHOLD,
    RHAT_MAX,
    _bfmi_per_chain,
    convergence_banner_markdown,
    style_diagnostics_table,
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


def _clean_sample_stats(shape: tuple[int, int], *, seed: int = 99) -> xr.Dataset:
    """A healthy ``sample_stats`` group: no divergences, white-noise energy.

    Lets a trace engineered to fail a *single* gate (R-hat or ESS) pass the other
    checks, so the regression assertions below isolate the check under test.
    """
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "diverging": (("chain", "draw"), np.zeros(shape, dtype=bool)),
            "energy": (("chain", "draw"), rng.normal(size=shape)),
        },
        coords={"chain": range(shape[0]), "draw": range(shape[1])},
    )


def _make_trace_high_rhat(seed: int = 0) -> xr.DataTree:
    """A trace whose single parameter has a true max R-hat in (1.01, 1.05).

    Half the chains are shifted by a fixed offset, inflating the between-chain
    variance so split R-hat lands around 1.04 -- comfortably above the 1.01 gate
    yet close enough that 2-significant-figure rounding collapses it to 1.0.
    """
    rng = np.random.default_rng(seed)
    shape = (4, 800)
    mu = rng.normal(0.0, 1.0, size=shape)
    mu[0] += 0.5
    mu[1] += 0.5
    posterior = xr.Dataset(
        {"mu": (("chain", "draw"), mu)},
        coords={"chain": range(shape[0]), "draw": range(shape[1])},
    )
    return xr.DataTree.from_dict({"posterior": posterior, "sample_stats": _clean_sample_stats(shape)})


def _make_trace_low_ess(seed: int = 0) -> xr.DataTree:
    """A trace whose single parameter has a true min ESS just under 400.

    A strongly autocorrelated (AR(1), rho=0.82) chain drives ESS to ~395 -- below
    the 400 gate, but inside the 2-significant-figure band that rounds *up* to 400.
    """
    rng = np.random.default_rng(seed)
    shape = (4, 1000)
    rho = 0.82
    scale = np.sqrt(1.0 - rho**2)
    x = np.empty(shape)
    x[:, 0] = rng.normal(size=shape[0])
    for t in range(1, shape[1]):
        x[:, t] = rho * x[:, t - 1] + scale * rng.normal(size=shape[0])
    posterior = xr.Dataset(
        {"mu": (("chain", "draw"), x)},
        coords={"chain": range(shape[0]), "draw": range(shape[1])},
    )
    return xr.DataTree.from_dict({"posterior": posterior, "sample_stats": _clean_sample_stats(shape)})


def test_gate_does_not_round_borderline_rhat_through(tmp_path):
    # Regression for dseinternational/research#65: az.summary must be called with
    # round_to="none" (the string). With the buggy round_to=None, arviz-stats
    # rounds R-hat to 2 significant figures, so this ~1.04 R-hat would report as
    # 1.0 and wrongly pass the <= 1.01 gate with an empty rhat_failing list.
    payload = write_diagnostics_summary(_make_trace_high_rhat(), str(tmp_path))
    assert payload["max_rhat"] > RHAT_MAX  # unrounded value survives, not 1.0
    assert payload["checks"]["rhat"] is False
    assert payload["rhat_failing"] == ["mu"]
    assert payload["passed"] is False


def test_gate_does_not_round_borderline_ess_through(tmp_path):
    # Regression for dseinternational/research#65: a true min ESS of ~395 rounds
    # up to 400 under 2-significant-figure rounding and would wrongly pass the
    # >= 400 gate. With round_to="none" the unrounded value fails the gate.
    payload = write_diagnostics_summary(_make_trace_low_ess(), str(tmp_path))
    assert payload["min_ess"] < ESS_THRESHOLD  # unrounded value survives, not 400
    assert payload["checks"] == {"rhat": True, "ess": False, "divergences": True, "bfmi": True}
    assert payload["ess_failing"] == ["mu"]
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


def _flagged_cells(styler) -> set[tuple[int, int]]:
    """(row, col) positions the styler flags red, parsed from the rendered CSS.

    ``Styler.to_html`` merges cells sharing a style into a single comma-separated
    rule, so the red rule's selector list carries every flagged cell id.
    """
    html = styler.to_html()
    match = re.search(r"([^{}]*)\{\s*color: #b00", html)
    if not match:
        return set()
    return {(int(r), int(c)) for r, c in re.findall(r"row(\d+)_col(\d+)", match.group(1))}


def _diagnostics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "r_hat": [1.001, 1.05],  # second row (col 0) fails > 1.01
            "ess_bulk": [1200.0, 200.0],  # second row (col 1) fails < 400
            "ess_tail": [1500.0, 1800.0],  # both pass
        },
        index=["mu", "sigma"],
    )


def test_style_diagnostics_table_flags_out_of_threshold_cells():
    styler = style_diagnostics_table(_diagnostics_frame())
    # Exactly the two failing cells: r_hat 1.05 (row1,col0) and ess_bulk 200 (row1,col1).
    assert _flagged_cells(styler) == {(1, 0), (1, 1)}
    assert "Reported convergence diagnostics" in styler.to_html()


def test_style_diagnostics_table_clean_frame_has_no_flags():
    clean = pd.DataFrame(
        {"r_hat": [1.001, 1.004], "ess_bulk": [900.0, 1100.0], "ess_tail": [1000.0, 1200.0]},
        index=["mu", "sigma"],
    )
    assert _flagged_cells(style_diagnostics_table(clean)) == set()


def test_style_diagnostics_table_tolerates_missing_columns():
    # A frame carrying only r_hat must not error on the absent ESS columns.
    only_rhat = pd.DataFrame({"r_hat": [1.02, 1.005]}, index=["a", "b"])
    assert _flagged_cells(style_diagnostics_table(only_rhat)) == {(0, 0)}


def test_style_diagnostics_table_respects_custom_thresholds():
    styler = style_diagnostics_table(_diagnostics_frame(), rhat_max=1.10, ess_threshold=100)
    # With looser thresholds nothing is flagged.
    assert _flagged_cells(styler) == set()
