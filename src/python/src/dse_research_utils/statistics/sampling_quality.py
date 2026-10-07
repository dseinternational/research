# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Extract unrounded sampling diagnostics without applying a pass/fail rule.

The result contains maximum R-hat, minimum effective sample size, minimum
per-chain BFMI, divergence count and names with unavailable diagnostics.
Callers choose variables, thresholds and how to handle missing information.

ArviZ summaries must use the string ``round_to="none"``. Passing None uses
the configured rounding default, which can conceal threshold exceedances.
Divergences are reduced from the numeric sample-statistics array. BFMI comes
from ``statistics.diagnostics.bfmi_per_chain`` in named chain/draw order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np

from dse_research_utils.statistics.diagnostics import bfmi_per_chain as _bfmi_per_chain
from dse_research_utils.statistics.diagnostics import diagnostic_extrema


@dataclass(frozen=True)
class SamplingQuality:
    """Unrounded sampling-quality signals for one trace."""

    max_rhat: float
    """Largest R-hat over the summarised variables (NaNs skipped)."""
    min_ess: float
    """Smallest of bulk-ESS and tail-ESS over the summarised variables."""
    min_bfmi: float | None
    """Smallest per-chain BFMI, or ``None`` when it cannot be computed."""
    n_divergences: int | None
    """Total divergent transitions, or ``None`` when ``sample_stats`` lacks them."""
    unassessable: tuple[str, ...] = ()
    """Summarised variables whose R-hat or ESS is non-finite.

    Extrema skip NaN, so finite extrema do not establish that every row
    was assessable. Callers must consider these names separately.
    """

    def summary_line(self) -> str:
        """One-line human-readable rendering for logs and prototype scripts."""
        bfmi = "n/a" if self.min_bfmi is None else f"{self.min_bfmi:.2f}"
        div = "n/a" if self.n_divergences is None else str(self.n_divergences)
        return f"max R-hat {self.max_rhat:.4f}, min ESS {self.min_ess:.0f}, min BFMI {bfmi}, divergences {div}"


def sampling_quality(trace: Any, *, var_names: list[str] | None = None) -> SamplingQuality:
    """Read the four sampling-quality signals off ``trace``, unrounded.

    Parameters
    ----------
    trace
        An ArviZ ``InferenceData`` (or DataTree-backed equivalent) with a ``posterior``
        group; ``sample_stats`` is used for divergences and BFMI when present.
    var_names : list of str, optional
        Restrict the R-hat / ESS summary to these variables. ``None`` summarises
        ArviZ's default variable selection, which can include deterministics.
        Supply the project's selected variables when its gate requires them.

    Returns
    -------
    SamplingQuality
        The extracted signals. Exceptions from ArviZ propagate; callers that must
        tolerate a failed diagnostic calculation should catch them and decide what an
        uncheckable fit means for them.
    """
    # ``round_to="none"`` must be the string — see the module docstring.
    summ = az.summary(trace, var_names=var_names, round_to="none", kind="diagnostics")
    # Extrema skip NaN; keep unavailable row names for the caller's gate.
    max_rhat, min_ess, unassessable = diagnostic_extrema(summ)

    n_div: int | None = None
    sample_stats = getattr(trace, "sample_stats", None)
    if sample_stats is not None and "diverging" in sample_stats:
        n_div = int(np.asarray(sample_stats["diverging"].values).sum())

    bfmi = _bfmi_per_chain(trace)
    min_bfmi = float(np.min(bfmi)) if bfmi is not None and len(bfmi) > 0 and np.all(np.isfinite(bfmi)) else None

    return SamplingQuality(
        max_rhat=max_rhat,
        min_ess=min_ess,
        min_bfmi=min_bfmi,
        n_divergences=n_div,
        unassessable=unassessable,
    )
