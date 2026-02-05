# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
from scipy.stats import shapiro, anderson


def describe(series: list[float] | np.ndarray | pd.Series, alpha: float) -> pd.Series:
    """
    Compute descriptive statistics and normality tests for a given series.

    Parameters
    ----------
    series : list[float] | np.ndarray | pd.Series
        Input data series.
    alpha : float
        Significance level for normality tests.

    Returns
    -------
    pd.Series
        A series containing the descriptive statistics and normality test results.
    """
    if isinstance(series, np.ndarray):
        if series.ndim != 1:
            raise ValueError("Only 1-dimensional arrays are supported.")
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    stats = series.describe()
    stats["skew"] = series.skew()
    stats["kurtosis"] = series.kurtosis()

    series_non_na = series.dropna()
    stats["n_non_na"] = len(series_non_na)

    shapiro_result = shapiro(series_non_na)
    stats["shapiro_stat"] = shapiro_result.statistic
    stats["shapiro_pvalue"] = shapiro_result.pvalue
    stats["shapiro_normality"] = shapiro_result.pvalue > alpha

    anderson_result = anderson(series_non_na, method="interpolate")
    stats["anderson_stat"] = anderson_result.statistic
    stats["anderson_pvalue"] = anderson_result.pvalue

    return stats


def describe_all(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Compute descriptive statistics and normality tests for all columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    alpha : float
        Significance level for normality tests.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the descriptive statistics and normality test results for all columns.
    """
    stats = df.apply(lambda col: describe(col, alpha))
    return stats
