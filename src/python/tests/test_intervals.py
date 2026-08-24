# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import math

import numpy as np
import pytest

from dse_research_utils.statistics.intervals import (
    DEFAULT_CI_PROB,
    INNER_CI_PROB,
    bands,
    eti_1d,
    eti_bands,
    hdi_1d,
    interval_1d,
    summarise_bands,
)


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

    def test_default_coverage_is_0_89(self) -> None:
        # The default matches ArviZ's rcParams["stats.ci_prob"] = 0.89.
        rng = np.random.default_rng(8)
        samples = rng.normal(size=10_000)
        assert hdi_1d(samples) == hdi_1d(samples, hdi_prob=0.89)
        assert hdi_1d(samples) != hdi_1d(samples, hdi_prob=0.90)


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

    def test_default_coverage_is_0_89(self) -> None:
        # The default matches ArviZ's rcParams["stats.ci_prob"] = 0.89.
        rng = np.random.default_rng(9)
        samples = rng.normal(size=10_000)
        assert eti_1d(samples) == eti_1d(samples, eti_prob=0.89)
        assert eti_1d(samples) != eti_1d(samples, eti_prob=0.90)


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


class TestConstants:
    def test_default_ci_prob_matches_primitive_defaults(self) -> None:
        rng = np.random.default_rng(10)
        samples = rng.normal(size=5000)
        assert DEFAULT_CI_PROB == 0.89
        assert eti_1d(samples) == eti_1d(samples, eti_prob=DEFAULT_CI_PROB)

    def test_inner_ci_prob_is_the_50_percent_band(self) -> None:
        assert INNER_CI_PROB == 0.50

    def test_interval_kind_importable_from_models_reporting(self) -> None:
        # The alias moved here from statistics.models.reporting; both import
        # paths must keep working.
        from dse_research_utils.statistics.intervals import IntervalKind as from_intervals
        from dse_research_utils.statistics.models.reporting import IntervalKind as from_reporting

        assert from_intervals is from_reporting


class TestInterval1d:
    def test_eti_matches_primitive(self) -> None:
        rng = np.random.default_rng(11)
        samples = rng.normal(size=5000)
        assert interval_1d(samples, 0.9, "eti") == eti_1d(samples, eti_prob=0.9)

    def test_hdi_matches_primitive(self) -> None:
        rng = np.random.default_rng(12)
        samples = rng.lognormal(size=5000)
        assert interval_1d(samples, 0.9, "hdi") == hdi_1d(samples, hdi_prob=0.9)

    def test_default_is_eti_at_default_coverage(self) -> None:
        rng = np.random.default_rng(13)
        samples = rng.normal(size=5000)
        assert interval_1d(samples) == eti_1d(samples, eti_prob=DEFAULT_CI_PROB)

    def test_empty_input_returns_nan(self) -> None:
        lo, hi = interval_1d([])
        assert math.isnan(lo) and math.isnan(hi)

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            interval_1d([1.0, 2.0, 3.0], 0.9, "quantile")  # type: ignore[arg-type]


class TestBands:
    def test_shape_and_values_default_sample_axis(self) -> None:
        rng = np.random.default_rng(14)
        samples = rng.normal(size=(3, 4000))  # grid on axis 0, draws on axis 1
        out = bands(samples, 0.9)
        assert out.shape == (3, 2)
        for i in range(3):
            lo, hi = eti_1d(samples[i, :], eti_prob=0.9)
            assert out[i, 0] == pytest.approx(lo)
            assert out[i, 1] == pytest.approx(hi)

    def test_sample_axis_zero(self) -> None:
        rng = np.random.default_rng(15)
        samples = rng.normal(size=(4000, 3))  # draws on axis 0, grid on axis 1
        out = bands(samples, 0.9, sample_axis=0)
        assert out.shape == (3, 2)
        lo, hi = eti_1d(samples[:, 1], eti_prob=0.9)
        assert out[1, 0] == pytest.approx(lo)
        assert out[1, 1] == pytest.approx(hi)

    def test_hdi_kind_dispatch(self) -> None:
        rng = np.random.default_rng(16)
        samples = rng.lognormal(size=(2, 4000))
        out = bands(samples, 0.9, "hdi")
        lo, hi = hdi_1d(samples[0, :], hdi_prob=0.9)
        assert out[0, 0] == pytest.approx(lo)
        assert out[0, 1] == pytest.approx(hi)

    def test_nan_rows_yield_nan_bounds(self) -> None:
        samples = np.full((2, 10), np.nan)
        out = bands(samples)
        assert np.isnan(out).all()

    def test_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            bands(np.zeros(5))

    def test_rejects_bad_sample_axis(self) -> None:
        with pytest.raises(ValueError, match="sample_axis"):
            bands(np.zeros((2, 3)), sample_axis=2)


class TestSummariseBands:
    def test_columns_and_grid(self) -> None:
        rng = np.random.default_rng(17)
        samples = rng.normal(size=(3, 4000))
        df = summarise_bands(samples, np.array([1.0, 2.0, 3.0]), grid_name="age_months")
        assert list(df.columns) == [
            "age_months",
            "median",
            "ci50_lo",
            "ci50_hi",
            "ci_lo",
            "ci_hi",
            "interval_kind",
        ]
        assert df["age_months"].tolist() == [1.0, 2.0, 3.0]
        assert (df["interval_kind"] == "eti").all()

    def test_bands_are_nested_around_the_median(self) -> None:
        rng = np.random.default_rng(18)
        samples = rng.normal(size=(3, 4000))
        df = summarise_bands(samples, np.arange(3.0))
        assert (df["ci_lo"] <= df["ci50_lo"]).all()
        assert (df["ci50_lo"] <= df["median"]).all()
        assert (df["median"] <= df["ci50_hi"]).all()
        assert (df["ci50_hi"] <= df["ci_hi"]).all()

    def test_one_dimensional_input_is_a_single_grid_point(self) -> None:
        rng = np.random.default_rng(19)
        samples = rng.normal(size=4000)
        df = summarise_bands(samples, np.array([7.0]))
        assert len(df) == 1
        assert df.loc[0, "grid"] == 7.0

    def test_hdi_kind_is_recorded(self) -> None:
        rng = np.random.default_rng(20)
        samples = rng.lognormal(size=(2, 4000))
        df = summarise_bands(samples, np.arange(2.0), kind="hdi")
        assert (df["interval_kind"] == "hdi").all()
        lo, hi = hdi_1d(samples[0, :], hdi_prob=DEFAULT_CI_PROB)
        assert df.loc[0, "ci_lo"] == pytest.approx(lo)
        assert df.loc[0, "ci_hi"] == pytest.approx(hi)
