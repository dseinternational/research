# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
from scipy.stats import anderson, kurtosis, shapiro, skew
from sklearn.neighbors import KernelDensity

from dse_research_utils.statistics.transforms import standardize


def differential_entropy_standardized(x, bandwidth="scott"):
    """
    Compute differential entropy of standardized data using Gaussian KDE.

    Parameters
    ----------
    x : array-like
        One-dimensional data vector.
    bandwidth : str or float
        "scott" or "silverman" for rule-of-thumb bandwidth selection,
        or provide a numeric value.

    Returns
    -------
    float
        Estimated differential entropy (in nats).
        To convert to bits, divide by np.log(2).
    """

    x = np.asarray(x).ravel()

    if len(x) < 2:
        return np.nan

    z = standardize(x)

    # Check variance after standardization
    std_z = np.std(z)
    if not np.isfinite(std_z) or std_z <= 0:
        # Data are (effectively) constant; entropy is undefined / -inf.
        return -np.inf

    # Bandwidth selection
    if isinstance(bandwidth, str):
        n = len(z)
        if bandwidth == "scott":
            bw = std_z * (n ** (-1 / 5))
        elif bandwidth == "silverman":
            bw = 0.9 * std_z * (n ** (-1 / 5))
        else:
            raise ValueError("Unknown bandwidth rule.")
    else:
        bw = float(bandwidth)

    # Safety floor in case of tiny numerical underflow
    bw_min = np.finfo(float).eps
    if not np.isfinite(bw) or bw <= 0:
        bw = bw_min

    kde = KernelDensity(kernel="gaussian", bandwidth=bw)
    kde.fit(z[:, None])

    log_f = kde.score_samples(z[:, None])
    return -np.mean(log_f)


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
        Descriptive statistics and normality test results. In addition to the
        pandas ``describe`` summary this reports ``n_non_na``, ``range``,
        ``range_std``, ``coef_var``, ``entropy`` (differential entropy of the
        standardised data), bias-corrected ``skew`` / ``kurtosis`` (only for
        ``n >= 8``), the Shapiro-Wilk test (only for ``n >= 3``) and the
        Anderson-Darling test.
    """
    if isinstance(series, np.ndarray) and series.ndim != 1:
        raise ValueError("Only 1-dimensional arrays are supported.")
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    series = series.astype("float")
    series_non_na = series.dropna()
    n = len(series_non_na)

    stats = series_non_na.describe()
    stats["n_non_na"] = n
    stats["range"] = stats["max"] - stats["min"]
    stats["range_std"] = stats["range"] / stats["std"] if stats["std"] != 0 else np.nan
    stats["coef_var"] = stats["std"] / stats["mean"] if stats["mean"] != 0 else np.nan
    stats["entropy"] = differential_entropy_standardized(series_non_na)

    # Skewness / kurtosis are unstable on very small samples; gate at n >= 8.
    # scipy bias-corrected moments match pandas' adjusted Fisher-Pearson values.
    if n < 8:
        stats["skew"] = np.nan
        stats["kurtosis"] = np.nan
    else:
        stats["skew"] = skew(series_non_na, bias=False)
        stats["kurtosis"] = kurtosis(series_non_na, fisher=True, bias=False)

    if n < 3:
        stats["shapiro_stat"] = np.nan
        stats["shapiro_pvalue"] = np.nan
        stats["shapiro_normality"] = np.nan
    else:
        shapiro_result = shapiro(series_non_na)
        stats["shapiro_stat"] = shapiro_result.statistic
        stats["shapiro_pvalue"] = shapiro_result.pvalue
        stats["shapiro_normality"] = shapiro_result.pvalue > alpha

    try:
        anderson_result = anderson(series_non_na, method="interpolate")
        stats["anderson_stat"] = anderson_result.statistic
        stats["anderson_pvalue"] = anderson_result.pvalue
    except Exception:
        stats["anderson_stat"] = np.nan
        stats["anderson_pvalue"] = np.nan

    return stats


def describe_all(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Compute descriptive statistics and normality tests for all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    alpha : float
        Significance level for normality tests.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the descriptive statistics and normality test
        results for every numeric column.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    return df[numeric_cols].apply(lambda col: describe(col, alpha))


def describe_all_grouped(
    df: pd.core.groupby.DataFrameGroupBy, alpha: float
) -> pd.DataFrame:
    """Apply :func:`describe_all` within each group of a ``groupby`` object."""
    return df.apply(lambda group: describe_all(group, alpha), include_groups=False)
