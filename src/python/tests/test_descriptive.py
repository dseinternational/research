# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest

from dse_research_utils.statistics.descriptive import describe, describe_all


@pytest.fixture
def normal_samples() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=500)


class TestDescribe:
    def test_rejects_multidim_arrays(self) -> None:
        with pytest.raises(ValueError, match="1-dimensional"):
            describe(np.ones((3, 3)), alpha=0.05)

    def test_accepts_list(self) -> None:
        result = describe([1.0, 2.0, 3.0, 4.0, 5.0], alpha=0.05)
        assert isinstance(result, pd.Series)
        assert result["count"] == 5
        assert result["mean"] == pytest.approx(3.0)

    def test_accepts_pandas_series(self) -> None:
        result = describe(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), alpha=0.05)
        assert result["count"] == 5

    def test_has_expected_keys(self, normal_samples: np.ndarray) -> None:
        result = describe(normal_samples, alpha=0.05)
        for key in (
            "count",
            "mean",
            "std",
            "skew",
            "kurtosis",
            "n_non_na",
            "shapiro_stat",
            "shapiro_pvalue",
            "shapiro_normality",
            "anderson_stat",
            "anderson_pvalue",
        ):
            assert key in result.index

    def test_normality_flag_matches_pvalue(self, normal_samples: np.ndarray) -> None:
        result = describe(normal_samples, alpha=0.05)
        assert bool(result["shapiro_normality"]) == (result["shapiro_pvalue"] > 0.05)

    def test_clearly_non_normal_data_rejected(self) -> None:
        # Heavy skew from an exponential distribution; shapiro should reject.
        rng = np.random.default_rng(123)
        samples = rng.exponential(scale=1.0, size=200)
        result = describe(samples, alpha=0.05)
        assert bool(result["shapiro_normality"]) is False

    def test_handles_na_values(self) -> None:
        series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        result = describe(series, alpha=0.05)
        assert result["n_non_na"] == 4


class TestDescribeAll:
    def test_applies_over_columns(self, normal_samples: np.ndarray) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "a": normal_samples,
                "b": rng.normal(size=len(normal_samples)),
            }
        )
        result = describe_all(df, alpha=0.05)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"a", "b"}
