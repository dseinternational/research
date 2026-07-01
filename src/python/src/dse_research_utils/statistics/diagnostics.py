# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared MCMC convergence gate and its report banner.

A single, robust pass/fail convergence summary built from stable ArviZ surfaces
(divergences, R-hat, ESS, per-chain BFMI), written as ``diagnostics_summary.json``
next to a fit's other artefacts, plus a Markdown renderer so every DSE report shows
the same banner. The JSON schema is produced by :func:`write_diagnostics_summary` and
consumed by :func:`convergence_banner_markdown`; keeping both here keeps that contract
in one place.

Thresholds follow the shared workflow standard (R-hat <= 1.01, ESS >= 400,
divergences = 0, BFMI >= 0.3; Vehtari et al. 2021 for R-hat).
"""

from __future__ import annotations

import json
import os

import arviz as az
import numpy as np
from rich import print as rprint

# Convergence-gate thresholds.
RHAT_MAX = 1.01
ESS_THRESHOLD = 400
BFMI_THRESHOLD = 0.3


def _bfmi_per_chain(trace) -> list[float] | None:
    """Per-chain BFMI from the sampler energy (Betancourt 2016).

    ``arviz.bfmi`` was removed in the 1.x split, so compute it directly:
    ``BFMI = sum((E_t - E_{t-1})**2) / sum((E_t - mean(E))**2)`` per chain. Returns
    ``None`` if the energy trace is unavailable.
    """
    try:
        energy = np.atleast_2d(np.asarray(trace.sample_stats["energy"].values))
        out: list[float] = []
        for e in energy:
            num = float(np.sum(np.diff(e) ** 2))
            den = float(np.sum((e - e.mean()) ** 2))
            out.append(num / den if den > 0 else float("nan"))
        return out
    except Exception:
        return None


def write_diagnostics_summary(
    trace,
    output_dir: str,
    *,
    var_names: list[str] | None = None,
    tables: dict | None = None,
) -> dict:
    """Emit ``diagnostics_summary.json`` -- the report's pass/fail convergence gate.

    One robust summary built from stable ArviZ surfaces: divergences from
    ``sample_stats``, R-hat / ESS from ``az.summary`` over the named (or all scalar)
    parameters, and per-chain BFMI. Written unconditionally so a report's banner
    always renders, even when a diagnostic could not be computed.

    Parameters
    ----------
    trace
        An ArviZ ``InferenceData`` / ``DataTree`` with ``posterior`` and
        ``sample_stats`` groups.
    output_dir : str
        Directory to write ``diagnostics_summary.json`` into (created if absent).
    var_names : list of str, optional
        Restrict the R-hat / ESS scan to these variables; ``None`` scans the trace's
        default (scalar) parameters.
    tables : dict, optional
        When given, the payload is also stored under ``tables["diagnostics_summary"]``
        (mirrors a fit context's table cache).

    Returns
    -------
    dict
        The payload written to JSON: ``passed``, the per-check booleans, the raw
        ``divergences`` / ``max_rhat`` / ``min_ess`` / ``bfmi_per_chain`` values, the
        failing-parameter lists, and the ``thresholds`` used.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_div: int | None = None
    try:
        ss = trace.sample_stats
        if "diverging" in ss:
            n_div = int(np.asarray(ss["diverging"].values).sum())
    except Exception:
        n_div = None

    max_rhat: float | None = None
    min_ess: float | None = None
    rhat_failing: list[str] = []
    ess_failing: list[str] = []
    try:
        # Evaluate the gate on unrounded diagnostics; presentation rounding is
        # applied by convergence_banner_markdown, not here (else a borderline
        # R-hat such as 1.01004 would round to 1.0100 and slip through the gate).
        s = az.summary(trace, var_names=var_names, round_to=None, ci_kind="eti")
        if "r_hat" in s:
            max_rhat = float(np.nanmax(s["r_hat"].values))
            rhat_failing = [str(i) for i in s.index[s["r_hat"] > RHAT_MAX]]
        ess_cols = [c for c in ("ess_bulk", "ess_tail") if c in s]
        if ess_cols:
            ess_min_row = s[ess_cols].min(axis=1)
            min_ess = float(np.nanmin(ess_min_row.values))
            ess_failing = [str(i) for i in s.index[ess_min_row < ESS_THRESHOLD]]
    except Exception as exc:  # pragma: no cover - defensive
        rprint(f"[yellow]R-hat/ESS summary for the gate failed: {exc}[/yellow]")

    bfmi = _bfmi_per_chain(trace)
    # Order-independent and NaN-safe. A degenerate chain (zero energy variance ->
    # non-finite BFMI) must not be able to pass the gate just because a healthy
    # chain happens to sort first under the builtin ``min()`` (NaN comparisons are
    # all False, so ``min`` is order-dependent on a list containing NaN). Require
    # every chain to have a finite BFMI at or above the threshold.
    bfmi_ok = bool(bfmi) and all(np.isfinite(b) and b >= BFMI_THRESHOLD for b in bfmi)
    # Non-finite BFMI does not serialise as valid JSON (json emits a bare ``NaN``
    # token); store ``None`` for those chains instead.
    bfmi_json = (
        [None if (b is None or not np.isfinite(b)) else float(b) for b in bfmi]
        if bfmi is not None
        else None
    )

    checks = {
        "rhat": bool(max_rhat is not None and max_rhat <= RHAT_MAX),
        "ess": bool(min_ess is not None and min_ess >= ESS_THRESHOLD),
        "divergences": bool(n_div == 0),
        "bfmi": bfmi_ok,
    }
    passed = all(checks.values())

    payload = {
        "passed": passed,
        "checks": checks,
        "divergences": n_div,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "bfmi_per_chain": bfmi_json,
        "rhat_failing": rhat_failing,
        "ess_failing": ess_failing,
        "thresholds": {
            "rhat_max": RHAT_MAX,
            "ess_threshold": ESS_THRESHOLD,
            "bfmi_threshold": BFMI_THRESHOLD,
        },
    }
    with open(os.path.join(output_dir, "diagnostics_summary.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    if tables is not None:
        tables["diagnostics_summary"] = payload
    verdict = "[green]PASS[/green]" if passed else "[red]REVIEW[/red]"
    rprint(
        f"  Convergence gate: {verdict} "
        f"(divergences={n_div}, max R-hat={max_rhat}, min ESS={min_ess})"
    )
    return payload


def convergence_banner_markdown(summary: dict | None, *, dev_note: bool = True) -> str:
    """Render the convergence gate as a Quarto callout (Markdown ``asis`` block).

    Centralised so every report shows an identical banner. ``summary`` is the dict
    written by :func:`write_diagnostics_summary` (typically loaded back from
    ``diagnostics_summary.json``). Returns a ``callout-tip`` when the gate passed, a
    ``callout-warning`` when at least one diagnostic is outside threshold, and a
    ``callout-note`` placeholder when ``summary`` is ``None``/empty.

    Parameters
    ----------
    summary : dict or None
        The ``diagnostics_summary.json`` payload, or ``None`` if absent.
    dev_note : bool, default True
        Append the "estimates are provisional until the gate passes" note when the
        gate has not passed (short development-config chains often miss the ESS
        target).

    Returns
    -------
    str
        Markdown for a Quarto callout block.
    """
    if not summary:
        return (
            ":::{.callout-note}\n"
            "No `diagnostics_summary.json` was found for this fit.\n"
            ":::"
        )
    passed = summary.get("passed")
    thr = summary.get("thresholds", {})
    box = "callout-tip" if passed else "callout-warning"
    head = (
        "Convergence: PASS — diagnostics within thresholds"
        if passed
        else "Convergence: REVIEW — at least one diagnostic is outside threshold"
    )
    mr, me = summary.get("max_rhat"), summary.get("min_ess")
    bf = summary.get("bfmi_per_chain")
    bf_s = (
        ", ".join("n/a" if x is None else f"{x:.2f}" for x in bf) if bf else "n/a"
    )
    lines = [f'::: {{.{box} title="{head}"}}', ""]
    lines.append(f"- **Divergences:** {summary.get('divergences')} (target 0)")
    lines.append(
        f"- **Max R-hat:** {mr:.4f} (target ≤ {thr.get('rhat_max')})"
        if mr is not None
        else "- **Max R-hat:** n/a"
    )
    lines.append(
        f"- **Min ESS:** {me:.0f} (target ≥ {thr.get('ess_threshold')})"
        if me is not None
        else "- **Min ESS:** n/a"
    )
    lines.append(f"- **BFMI per chain:** {bf_s} (target ≥ {thr.get('bfmi_threshold')})")
    fails = []
    if summary.get("rhat_failing"):
        fails.append("R-hat — " + ", ".join(summary["rhat_failing"]))
    if summary.get("ess_failing"):
        fails.append("ESS — " + ", ".join(summary["ess_failing"]))
    if fails:
        lines += ["", "**Parameters needing attention:** " + "; ".join(fails)]
    if not passed and dev_note:
        lines += [
            "",
            "Treat the estimates below as provisional until the gate passes "
            "(short `dev`-config chains often miss the ESS target — refit in "
            "`reporting` config before interpreting).",
        ]
    lines += ["", ":::"]
    return "\n".join(lines)
