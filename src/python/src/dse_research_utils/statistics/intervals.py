import numpy as np


def hdi_1d(x, hdi_prob: float = 0.90) -> tuple[float, float]:
    """
    Compute the highest density interval (HDI) for a 1D array of samples.

    The HDI is the most "dense" part of the distribution. It has two defining characteristics:

    - Every point inside the interval has a higher probability density than any point outside of it.
    - It is the shortest possible interval that contains the required probability mass (e.g., 95%).

    Parameters
    ----------
    x : array-like
        Input samples.
    hdi_prob : float, optional
        The probability mass to include in the HDI (default is 0.90).

    Returns
    -------
    tuple
        A tuple containing the lower and upper bounds of the HDI.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n == 0:
        return np.nan, np.nan

    # Sort samples to treat as a distribution
    x = np.sort(x)

    # Calculate the number of elements that should be in the interval
    interval_idx_inc = int(np.floor(hdi_prob * n))

    if interval_idx_inc < 1:
        # If probability is too low for the sample size, return the mode-ish point
        return float(x[0]), float(x[0])

    # Calculate all possible widths for the given interval size
    low_ends = x[: n - interval_idx_inc]
    high_ends = x[interval_idx_inc:]
    widths = high_ends - low_ends

    # Find the index of the minimum width
    min_idx = np.argmin(widths)

    return float(low_ends[min_idx]), float(high_ends[min_idx])


def eti_1d(x, eti_prob=0.90):
    """
    Equal-Tailed Interval from 1D samples.

    The Equal-Tailed Interval (ETI) is a simple method for constructing a credible interval from a set of
    samples. It is defined by the quantiles of the distribution, such that the lower tail and upper tail
    each contain an equal amount of probability mass outside the interval.

    Parameters
    ----------
    x : array-like
        Input samples.
    eti_prob : float, optional
        The probability mass to include in the ETI (default is 0.90).

    Returns
    -------
    tuple
        A tuple containing the lower and upper bounds of the ETI.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan, np.nan

    # Calculate the tail probabilities
    lower_tail = (1.0 - eti_prob) / 2.0
    upper_tail = 1.0 - lower_tail

    # Use numpy.percentile to find the values at those probabilities
    # We multiply by 100 because np.percentile expects 0-100 range
    return tuple(np.percentile(x, [lower_tail * 100, upper_tail * 100]))
