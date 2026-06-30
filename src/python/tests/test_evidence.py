# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the evidence-strength helpers."""

import pytest

from dse_research_utils.statistics.evidence import evidence_label, odds_string


@pytest.mark.parametrize(
    ("prob", "expected"),
    [
        (0.50, "inconclusive"),
        (0.749, "inconclusive"),
        (0.75, "suggestive"),
        (0.90, "suggestive"),
        (0.91, "moderate"),
        (0.965, "moderate"),
        (0.97, "strong"),
        (0.985, "strong"),
        (0.99, "very strong"),
        (0.9999, "very strong"),
    ],
)
def test_evidence_label_ladder_boundaries(prob, expected):
    assert evidence_label(prob) == expected


@pytest.mark.parametrize(
    ("prob", "expected"),
    [
        (0.5, "1:1"),
        (0.75, "3:1"),
        (0.90, "9:1"),
        (0.95, "19:1"),
        (0.20, "1:4"),
        (0.10, "1:9"),
    ],
)
def test_odds_string_round_odds(prob, expected):
    assert odds_string(prob) == expected


def test_odds_string_clamps_extremes():
    # No ZeroDivision / inf at the boundaries.
    assert odds_string(1.0).endswith(":1")
    assert odds_string(0.0).startswith("1:")


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0, float("nan")])
def test_evidence_label_rejects_out_of_range(bad):
    # The label is only meaningful for a probability; out-of-range / NaN inputs
    # are an upstream mistake and should fail fast rather than mislabel.
    with pytest.raises(ValueError):
        evidence_label(bad)


@pytest.mark.parametrize("ok", [0.0, 1.0])
def test_evidence_label_accepts_closed_interval_endpoints(ok):
    assert evidence_label(ok) in {"inconclusive", "very strong"}
