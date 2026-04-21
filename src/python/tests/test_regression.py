# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest

from dse_research_utils.statistics.regression import linear_regression_ols


class TestLinearRegressionOls:
    def test_recovers_known_line(self) -> None:
        x = np.arange(100, dtype=float)
        y = 3.0 + 2.0 * x  # deterministic line, no noise
        result = linear_regression_ols(x, y)
        assert result["intercept"] == pytest.approx(3.0)
        assert result["slope"] == pytest.approx(2.0)
        assert result["r_squared"] == pytest.approx(1.0)
        assert result["residual_standard_error"] == pytest.approx(0.0, abs=1e-8)

    def test_fitted_matches_observed_for_deterministic(self) -> None:
        x = np.arange(10, dtype=float)
        y = 1.0 + 0.5 * x
        result = linear_regression_ols(x, y)
        np.testing.assert_allclose(result["fitted"], y, atol=1e-8)
        np.testing.assert_allclose(result["residuals"], 0.0, atol=1e-8)

    def test_rejects_non_1d(self) -> None:
        x = np.ones((3, 3))
        y = np.ones(3)
        with pytest.raises(ValueError, match="1D arrays"):
            linear_regression_ols(x, y)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            linear_regression_ols(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_rejects_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least two observations"):
            linear_regression_ols(np.array([1.0]), np.array([2.0]))

    def test_returns_statsmodels_objects(self) -> None:
        import statsmodels.regression.linear_model as sm_lm

        result = linear_regression_ols(np.arange(10, dtype=float), np.arange(10, dtype=float))
        assert isinstance(result["model"], sm_lm.OLS)
        assert hasattr(result["results"], "params")

    def test_r_squared_noisy_data_between_zero_and_one(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=200)
        y = 0.5 * x + rng.normal(scale=0.5, size=200)
        result = linear_regression_ols(x, y)
        assert 0.0 <= result["r_squared"] <= 1.0
