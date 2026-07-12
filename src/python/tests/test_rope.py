# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the shared ROPE report card."""

import numpy as np
import pytest

from dse_research_utils.statistics.evidence import evidence_label, favoured_direction
from dse_research_utils.statistics.intervals import hdi_1d
from dse_research_utils.statistics.rope import rope_card


def test_rope_card_probabilities_match_definitions():
    rng = np.random.default_rng(0)
    # A positive-leaning effect and its items-scale image.
    effect = rng.normal(0.4, 1.0, size=20_000)
    items = rng.normal(1.5, 2.0, size=20_000)
    delta = 1.0
    card = rope_card(effect, items, delta=delta, ci_prob=0.95)

    assert card["pd"] == pytest.approx(float(np.mean(effect > 0)))
    assert card["prob_benefit_ge_delta"] == pytest.approx(float(np.mean(items >= delta)))
    assert card["prob_in_rope"] == pytest.approx(float(np.mean(np.abs(items) <= delta)))
    assert card["prob_harm_ge_delta"] == pytest.approx(float(np.mean(items <= -delta)))
    assert card["delta_items"] == delta


def test_rope_card_point_estimates_are_medians():
    effect = np.array([-2.0, -1.0, 0.0, 1.0, 5.0])
    items = np.array([-1.0, 0.0, 2.0, 3.0, 4.0])
    card = rope_card(effect, items, delta=0.5, ci_prob=0.9)
    assert card["tau_logit_median"] == pytest.approx(np.median(effect))
    assert card["items_median"] == pytest.approx(np.median(items))


def test_rope_card_hpdi_matches_hdi_1d():
    rng = np.random.default_rng(1)
    effect = rng.lognormal(size=8000) - 1.0
    items = rng.normal(size=8000)
    card = rope_card(effect, items, delta=0.25, ci_prob=0.95)
    lo, hi = hdi_1d(effect, 0.95)
    assert card["tau_logit_hpdi_lo"] == pytest.approx(lo)
    assert card["tau_logit_hpdi_hi"] == pytest.approx(hi)
    ilo, ihi = hdi_1d(items, 0.95)
    assert card["items_hpdi_lo"] == pytest.approx(ilo)
    assert card["items_hpdi_hi"] == pytest.approx(ihi)


def test_rope_card_eti_headline_and_bands():
    rng = np.random.default_rng(2)
    items = rng.normal(size=10_000)
    effect = rng.normal(size=10_000)
    card = rope_card(effect, items, delta=0.1, ci_prob=0.9)
    lo_q, hi_q = 0.05, 0.95  # ci_prob=0.9 equal-tailed
    assert card["items_lo"] == pytest.approx(float(np.quantile(items, lo_q)))
    assert card["items_hi"] == pytest.approx(float(np.quantile(items, hi_q)))
    assert card["items_lo50"] == pytest.approx(float(np.quantile(items, 0.25)))
    assert card["items_hi50"] == pytest.approx(float(np.quantile(items, 0.75)))


def test_rope_card_labels_follow_the_evidence_ladder():
    rng = np.random.default_rng(3)
    # Strong positive effect: pd near 1, meaningful benefit likely.
    effect = rng.normal(2.0, 0.5, size=20_000)
    items = rng.normal(3.0, 1.0, size=20_000)
    card = rope_card(effect, items, delta=1.0, ci_prob=0.95)
    assert card["direction_label"] == evidence_label(card["pd"])
    assert card["benefit_label"] == evidence_label(card["prob_benefit_ge_delta"])
    # favoured_direction fields are present and consistent with pd.
    fav = favoured_direction(card["pd"])
    assert card["favoured_direction"] == fav["favoured_direction"]
    assert card["favoured_direction_label"] == fav["favoured_direction_label"]


def test_rope_card_negative_effect_reads_as_harm():
    rng = np.random.default_rng(4)
    effect = rng.normal(-2.0, 0.5, size=20_000)
    items = rng.normal(-3.0, 1.0, size=20_000)
    card = rope_card(effect, items, delta=1.0, ci_prob=0.95)
    assert card["pd"] < 0.5
    assert card["favoured_direction"] == "negative"
    # Harm mass is substantial; benefit mass is negligible.
    assert card["prob_harm_ge_delta"] > 0.5
    assert card["prob_benefit_ge_delta"] < 0.05


def test_rope_card_accepts_list_inputs():
    card = rope_card([-1.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], delta=0.5, ci_prob=0.9)
    assert isinstance(card["pd"], float)
    assert card["pd"] == pytest.approx(0.75)


def test_rope_card_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        rope_card(np.zeros(10), np.zeros(9), delta=0.5, ci_prob=0.9)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_rope_card_rejects_invalid_ci_prob(bad):
    with pytest.raises(ValueError, match="ci_prob"):
        rope_card(np.zeros(10), np.zeros(10), delta=0.5, ci_prob=bad)


def test_rope_card_rejects_negative_delta():
    with pytest.raises(ValueError, match="delta"):
        rope_card(np.zeros(10), np.zeros(10), delta=-0.1, ci_prob=0.9)


def test_rope_card_drops_non_finite_pairs():
    rng = np.random.default_rng(7)
    effect = rng.normal(size=1000)
    items = rng.normal(1.0, 1.5, size=1000)
    # A card over data with a few non-finite entries must equal the card over the
    # finite subset (non-finite pairs are dropped jointly, keeping the pairing).
    e2, i2 = effect.copy(), items.copy()
    e2[3] = np.nan
    i2[10] = np.inf
    e2[20] = -np.inf
    keep = [k for k in range(1000) if k not in {3, 10, 20}]
    bad = rope_card(e2, i2, delta=0.2, ci_prob=0.95)
    clean = rope_card(effect[keep], items[keep], delta=0.2, ci_prob=0.95)
    assert bad["pd"] == pytest.approx(clean["pd"])
    assert bad["items_median"] == pytest.approx(clean["items_median"])
    assert bad["prob_benefit_ge_delta"] == pytest.approx(clean["prob_benefit_ge_delta"])


def test_rope_card_all_non_finite_raises():
    with pytest.raises(ValueError, match="no finite"):
        rope_card([np.nan, np.inf], [1.0, 2.0], delta=0.5, ci_prob=0.9)
