# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ROPE-anchored report card: the DSE house-style direction/magnitude convention.

Turns a pair of posterior draw arrays — an effect on its native (e.g. logit) scale
and the matching interpretable (e.g. items) scale — into the shared ``rope_summary``
schema: a median with equal-tailed and highest-density intervals, the probability of
direction, and the region-of-practical-equivalence (ROPE) magnitude quantities judged
against a minimally-important difference ``delta``. Keeping the schema here means every
DSE report that anchors a claim on a ROPE emits the same columns and reads the same way.

This is the *convention*, not a model: the caller is responsible for producing the two
draw arrays (the average marginal effect, a contrast, etc.) — see each report's model
helpers.
"""

from __future__ import annotations

import numpy as np

from dse_research_utils.statistics.evidence import evidence_label, favoured_direction
from dse_research_utils.statistics.intervals import eti_bands, hdi_1d


def rope_card(
    effect_draws: np.ndarray,
    items: np.ndarray,
    *,
    delta: float,
    ci_prob: float = 0.89,
) -> dict[str, float | str]:
    """Assemble the ROPE report card from native-scale and items-scale effect draws.

    The formatting core shared across DSE ROPE reports so they emit an identical
    ``rope_summary`` schema. ``effect_draws`` are the native-scale (e.g. logit) effect
    draws ``(S,)`` — used for the ``pd`` direction probability — and ``items`` the
    matching interpretable-scale (e.g. items) average marginal effect per draw ``(S,)``.
    ``delta`` is the ROPE half-width (the minimally-important difference) on the items
    scale.

    The point estimate on each scale is the posterior **median** because it is
    transformation-invariant across the native and items scales. The ``tau_logit_*``
    keys are named for the ITT ``tau`` estimand that first used this schema and are kept
    verbatim across effect families so one CSV layout serves a whole suite.

    ``*_hpdi_lo`` / ``*_hpdi_hi`` add the highest-density interval at ``ci_prob`` for
    each scale — a per-scale sensitivity companion to the equal-tailed ``*_lo`` /
    ``*_hi`` fields, kept alongside them (the HPDI is not transformation-invariant, so
    it is reported per scale).

    Parameters
    ----------
    effect_draws : numpy.ndarray
        Native-scale effect draws, shape ``(S,)``.
    items : numpy.ndarray
        Items-scale (interpretable) effect draws, shape ``(S,)``.
    delta : float
        ROPE half-width on the items scale (the minimally-important difference).
    ci_prob : float, default 0.89
        Coverage of the headline equal-tailed and HPDI intervals (the ArviZ 1.x
        default, ``rcParams["stats.ci_prob"]``). Reports pass their own convention
        explicitly (e.g. the ITT suite uses 0.95).

    Returns
    -------
    dict
        The single-row ``rope_summary`` schema: medians, 50%/90%/headline equal-tailed
        bands and HPDIs per scale, ``delta_items``, ``pd`` (``P(effect > 0)``),
        ``prob_benefit_ge_delta`` (``P(items >= delta)``), ``prob_in_rope``
        (``P(|items| <= delta)``), ``prob_harm_ge_delta`` (``P(items <= -delta)``), the
        ``direction_label`` / ``benefit_label`` evidence labels, and the
        :func:`favoured_direction` fields.
    """
    effect_draws = np.asarray(effect_draws, dtype=float)
    items = np.asarray(items, dtype=float)
    if effect_draws.shape != items.shape:
        raise ValueError(
            "effect_draws and items must be the same shape (paired per draw), got "
            f"{effect_draws.shape} and {items.shape}"
        )
    if not 0.0 < ci_prob <= 1.0:
        raise ValueError(f"ci_prob must be in (0, 1], got {ci_prob!r}")
    if delta < 0:
        raise ValueError(f"delta (a ROPE half-width) must be non-negative, got {delta!r}")
    # Drop draws that are non-finite on either scale, keeping the two arrays paired
    # (matching the non-finite handling in the interval helpers). NaNs would otherwise
    # count as "not > 0" for pd and propagate through np.quantile.
    finite = np.isfinite(effect_draws) & np.isfinite(items)
    effect_draws = effect_draws[finite]
    items = items[finite]
    if effect_draws.size == 0:
        raise ValueError("no finite (effect_draws, items) draw pairs remain")
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    pd_ = float(np.mean(effect_draws > 0))
    p_benefit = float(np.mean(items >= delta))
    tau_hpdi_lo, tau_hpdi_hi = hdi_1d(effect_draws, ci_prob)
    items_hpdi_lo, items_hpdi_hi = hdi_1d(items, ci_prob)
    tau_b90 = eti_bands(effect_draws, probs=(0.9,))
    items_b90 = eti_bands(items, probs=(0.9,))
    return {
        "tau_logit_median": float(np.median(effect_draws)),
        "tau_logit_lo50": float(np.quantile(effect_draws, 0.25)),
        "tau_logit_hi50": float(np.quantile(effect_draws, 0.75)),
        "tau_logit_lo90": tau_b90["lo90"],
        "tau_logit_hi90": tau_b90["hi90"],
        "tau_logit_lo": float(np.quantile(effect_draws, lo_q)),
        "tau_logit_hi": float(np.quantile(effect_draws, hi_q)),
        "tau_logit_hpdi_lo": tau_hpdi_lo,
        "tau_logit_hpdi_hi": tau_hpdi_hi,
        "items_median": float(np.median(items)),
        "items_lo50": float(np.quantile(items, 0.25)),
        "items_hi50": float(np.quantile(items, 0.75)),
        "items_lo90": items_b90["lo90"],
        "items_hi90": items_b90["hi90"],
        "items_lo": float(np.quantile(items, lo_q)),
        "items_hi": float(np.quantile(items, hi_q)),
        "items_hpdi_lo": items_hpdi_lo,
        "items_hpdi_hi": items_hpdi_hi,
        "delta_items": float(delta),
        "pd": pd_,
        "prob_benefit_ge_delta": p_benefit,
        "prob_in_rope": float(np.mean(np.abs(items) <= delta)),
        "prob_harm_ge_delta": float(np.mean(items <= -delta)),
        "direction_label": evidence_label(pd_),
        "benefit_label": evidence_label(p_benefit),
        **favoured_direction(pd_),
    }
