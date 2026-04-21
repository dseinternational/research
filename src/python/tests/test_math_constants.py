# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from dse_research_utils.math.constants import EPSILON


def test_epsilon_is_small_positive() -> None:
    assert 0 < EPSILON < 1e-6


def test_epsilon_value() -> None:
    assert EPSILON == 1e-10
