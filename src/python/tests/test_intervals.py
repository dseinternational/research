# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import math

import numpy as np
import pytest

from dse_research_utils.statistics.intervals import eti_1d, eti_bands, hdi_1d


class TestHdi1d:
    def test_empty_input_returns_nan(self) -> None:
        lo, hi = hdi_1d([])
        assert math.isnan(lo) and math.isnan(hi)

    def test_all_non_finite_returns_nan(self) -> None:
        lo, hi = hdi_1d([np.nan, np.inf, -np.inf])
        assert math.isnan(lo) and math.isnan(hi)

    def test_strips_non_finite_values(self) -> None:
        lo, hi = hdi_1d([1.0, 2.0, np.nan, 3.0, np.inf], hdi_prob=0.9)
        assert math.isfinite(lo) and math.isfinite(hi)

    def test_very_low_prob_returns_single_point(self) -> None:
        # hdi_prob * n < 1 → collapse to the smallest sorted sample
        lo, hi = hdi_1d([5.0, 3.0, 4.0], hdi_prob=0.01)
        assert lo == hi == 3.0

    def test_monotonic_symmetric_distribution(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=0.0, scale=1.0, size=20_000)
        lo, hi = hdi_1d(samples, hdi_prob=0.95)
        # 95% HDI of standard normal is approx [-1.96, 1.96]; Monte-Carlo tolerance
        assert lo == pytest.approx(-1.96, abs=0.1)
        assert hi == pytest.approx(1.96, abs=0.1)

    def test_narrower_for_lower_prob(self) -> None:
        rng = np.random.default_rng(7)
        samples = rng.normal(size=10_000)
        lo_90, hi_90 = hdi_1d(samples, hdi_prob=0.90)
        lo_50, hi_50 = hdi_1d(samples, hdi_prob=0.50)
        assert (hi_50 - lo_50) < (hi_90 - lo_90)

    def test_bounds_are_ordered(self) -> None:
        rng = np.random.default_rng(1)
        samples = rng.normal(size=500)
        lo, hi = hdi_1d(samples, hdi_prob=0.9)
        assert lo <= hi

    def test_returns_native_floats(self) -> None:
        lo, hi = hdi_1d([1.0, 2.0, 3.0, 4.0, 5.0], hdi_prob=0.6)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_full_probability_returns_range(self) -> None:
        assert hdi_1d([3.0, 1.0, 2.0], hdi_prob=1.0) == (1.0, 3.0)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.1])
    def test_rejects_invalid_probability(self, bad: float) -> None:
        with pytest.raises(ValueError, match="hdi_prob"):
            hdi_1d([1.0, 2.0, 3.0], hdi_prob=bad)


class TestEti1d:
    def test_empty_input_returns_nan(self) -> None:
        lo, hi = eti_1d([])
        assert math.isnan(lo) and math.isnan(hi)

    def test_strips_non_finite(self) -> None:
        lo, hi = eti_1d([1.0, 2.0, 3.0, np.nan], eti_prob=0.9)
        assert math.isfinite(lo) and math.isfinite(hi)

    def test_matches_percentiles(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(size=5000)
        lo, hi = eti_1d(samples, eti_prob=0.8)
        expected_lo, expected_hi = np.percentile(samples, [10.0, 90.0])
        assert lo == pytest.approx(expected_lo)
        assert hi == pytest.approx(expected_hi)

    def test_bounds_are_ordered(self) -> None:
        rng = np.random.default_rng(2)
        samples = rng.normal(size=500)
        lo, hi = eti_1d(samples, eti_prob=0.9)
        assert lo <= hi

    def test_full_probability_returns_range(self) -> None:
        assert eti_1d([3.0, 1.0, 2.0], eti_prob=1.0) == (1.0, 3.0)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.1])
    def test_rejects_invalid_probability(self, bad: float) -> None:
        with pytest.raises(ValueError, match="eti_prob"):
            eti_1d([1.0, 2.0, 3.0], eti_prob=bad)


class TestEtiBands:
    def test_default_bands_and_keys(self) -> None:
        rng = np.random.default_rng(3)
        samples = rng.normal(size=5000)
        bands = eti_bands(samples)
        assert set(bands) == {"lo50", "hi50", "lo90", "hi90", "lo95", "hi95"}

    def test_bands_match_eti_1d(self) -> None:
        rng = np.random.default_rng(4)
        samples = rng.normal(size=5000)
        bands = eti_bands(samples, probs=(0.9,))
        lo, hi = eti_1d(samples, eti_prob=0.9)
        assert bands["lo90"] == pytest.approx(lo)
        assert bands["hi90"] == pytest.approx(hi)

    def test_bands_are_nested(self) -> None:
        rng = np.random.default_rng(5)
        samples = rng.normal(size=5000)
        bands = eti_bands(samples)
        assert bands["lo50"] >= bands["lo90"] >= bands["lo95"]
        assert bands["hi50"] <= bands["hi90"] <= bands["hi95"]

    def test_percentage_rounding_in_keys(self) -> None:
        bands = eti_bands([1.0, 2.0, 3.0, 4.0], probs=(0.945,))
        assert set(bands) == {"lo94", "hi94"}

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.1])
    def test_rejects_invalid_probability(self, bad: float) -> None:
        with pytest.raises(ValueError, match="probs"):
            eti_bands([1.0, 2.0, 3.0], probs=(bad,))

    def test_strips_non_finite(self) -> None:
        finite = [1.0, 2.0, 3.0, 4.0, 5.0]
        with_bad = [1.0, np.nan, 2.0, np.inf, 3.0, -np.inf, 4.0, 5.0]
        assert eti_bands(with_bad) == eti_bands(finite)

    def test_all_non_finite_returns_nan_bands(self) -> None:
        bands = eti_bands([np.nan, np.inf, -np.inf])
        assert set(bands) == {"lo50", "hi50", "lo90", "hi90", "lo95", "hi95"}
        assert all(math.isnan(v) for v in bands.values())
