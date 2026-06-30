# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gaussian-process and styled-primitive plotting helpers (matplotlib).

Adapted from the PyMC statistical-rethinking lecture utilities:
https://github.com/pymc-devs/pymc-examples/blob/main/examples/statistical_rethinking_lectures/utils.py
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def plot_scatter(xs, ys, **scatter_kwargs):
    """Draw scatter plot with consistent style (e.g. unfilled points)."""
    defaults = {"alpha": 0.6, "lw": 3, "s": 80, "color": "C0", "facecolors": "none"}

    for k, v in defaults.items():
        val = scatter_kwargs.get(k, v)
        scatter_kwargs[k] = val

    plt.scatter(xs, ys, **scatter_kwargs)


def plot_line(xs, ys, **plot_kwargs):
    """Plot line with consistent style (e.g. bordered lines)."""
    linewidth = plot_kwargs.get("linewidth", 3)
    plot_kwargs["linewidth"] = linewidth

    # Copy settings for background
    background_plot_kwargs = dict(plot_kwargs.items())
    background_plot_kwargs["linewidth"] = linewidth + 2
    background_plot_kwargs["color"] = "white"
    background_plot_kwargs.pop("label", None)  # no legend label for background

    plt.plot(xs, ys, **background_plot_kwargs, zorder=30)
    plt.plot(xs, ys, **plot_kwargs, zorder=31)


def plot_errorbar(xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3):
    """Draw thick y-error bars with consistent style."""
    if isinstance(colors, str):
        colors = [colors] * len(xs)
    for ii, (x, y, err_l, err_u) in enumerate(
        zip(xs, ys, error_lower, error_upper, strict=False)
    ):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            yerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


def plot_x_errorbar(xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3):
    """Draw thick x-error bars with consistent style."""
    if isinstance(colors, str):
        colors = [colors] * len(xs)
    for ii, (x, y, err_l, err_u) in enumerate(
        zip(xs, ys, error_lower, error_upper, strict=False)
    ):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            xerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


def plot_kernel_function(
    kernel_function, max_distance=1, resolution=100, label=None, ax=None, **line_kwargs
):
    """Plot a (stationary) kernel function over a range of distances."""
    X = np.linspace(0, max_distance, resolution)[:, None]
    covariance = kernel_function(X, X)
    distances = np.linspace(0, max_distance, resolution)
    if ax is not None:
        plt.sca(ax)
    plot_line(distances, covariance[0, :], label=label, **line_kwargs)
    plt.xlim([0, max_distance])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("|X1-X2|")
    plt.ylabel("covariance")
    if label is not None:
        plt.legend()


def plot_gaussian_process(
    X,
    samples: Iterable | None = None,
    mean=None,
    cov=None,
    X_obs=None,
    Y_obs=None,
    uncertainty_prob=0.89,
):
    """Plot GP samples and/or a mean function with an uncertainty band."""
    X = X.ravel()

    # Plot GP samples (``samples`` is optional — guard the default so the
    # function does not raise ``TypeError: 'NoneType' is not iterable`` when
    # called with only a mean/cov).
    for ii, sample in enumerate(samples or []):
        label = "GP samples" if not ii else None
        plot_line(X, sample, color=f"C{ii}", linewidth=1, label=label)

    # Add GP mean, if provided
    if mean is not None:
        mean = mean.ravel()
        plot_line(X, mean, color="k", label="GP mean")

        # Add uncertainty around mean; requires covariance matrix
        if cov is not None:
            z = stats.norm.ppf(1 - (1 - uncertainty_prob) / 2)
            uncertainty = z * np.sqrt(np.diag(cov))
            plt.fill_between(
                X,
                mean + uncertainty,
                mean - uncertainty,
                alpha=0.1,
                color="gray",
                zorder=1,
                label="GP uncertainty",
            )

    # Add any training data points, if provided
    if X_obs is not None:
        plot_scatter(X_obs, Y_obs, color="k", label="observations", zorder=100, alpha=1)

    plt.xlim([X.min(), X.max()])
    plt.ylim([-5, 5])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()


def gaussian_process_prior(X_pred, kernel_function):
    """Initialise a Gaussian-process prior distribution for the given kernel."""
    mean = np.zeros(X_pred.shape).ravel()
    cov = kernel_function(X_pred, X_pred)
    return stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)


def plot_gaussian_process_prior(kernel_function, n_samples=3, figsize=(10, 5), resolution=100):
    """Plot prior GP samples alongside the kernel function."""
    X = np.linspace(-5, 5, resolution)[:, None]

    prior = gaussian_process_prior(X, kernel_function)
    samples = prior.rvs(n_samples)

    _, axs = plt.subplots(1, 2, figsize=figsize)
    plt.sca(axs[0])
    plot_gaussian_process(X, samples=samples)

    plt.sca(axs[1])
    plot_kernel_function(kernel_function, color="k")
    plt.title("kernel function")
    return axs


def plot_2d_function(xrange, yrange, func, ax=None, **countour_kwargs):
    """Evaluate ``func(xs, ys)`` over a grid and plot the value contour.

    Parameters
    ----------
    xrange, yrange : np.ndarray
        The horizontal / vertical values to evaluate and plot.
    func : Callable
        Function of two arguments (xs, ys) returning a single value at each point.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on; defaults to the current axis.

    Returns
    -------
    contour : matplotlib.contour.QuadContourSet
    """
    resolution = len(xrange)
    xs, ys = np.meshgrid(xrange, yrange)
    xs = xs.ravel()
    ys = ys.ravel()

    value = func(xs, ys)

    if ax is not None:
        plt.sca(ax)

    return plt.contour(
        xs.reshape(resolution, resolution),
        ys.reshape(resolution, resolution),
        value.reshape(resolution, resolution),
        **countour_kwargs,
    )
