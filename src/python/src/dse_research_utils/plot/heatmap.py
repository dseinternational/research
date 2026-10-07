# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Square matrix heatmap (``imshow``) shared across DSE projects."""

from __future__ import annotations

import matplotlib.axes as mpaxes
import matplotlib.figure as mpfig
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm, Colormap

import dse_research_utils.plot.styles as plot_styles


def plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str | Colormap | None = None,
    centre: float | None = None,
    ax: mpaxes.Axes | None = None,
    tick_fontsize: float = 12,
    title_fontsize: float = 12,
    colorbar_fraction: float = 0.03,
    colorbar_pad: float = 0.025,
) -> tuple[mpfig.Figure, mpaxes.Axes]:
    """Plot a square heatmap using ``imshow``.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix of values to display.
    labels : list[str]
        Tick labels for both axes (length must match matrix dimensions).
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size. Defaults to ``(side + 1.5, side)`` where
        ``side = min(max(3, 0.6 * n), 14)``.
    cmap : str or Colormap, optional
        Matplotlib colour map. Defaults to the design tokens' sequential scale
        (:data:`~dse_research_utils.plot.styles.SEQUENTIAL_CMAP`), or their
        diverging scale (:data:`~dse_research_utils.plot.styles.DIVERGING_CMAP`)
        when ``centre`` is given.
    centre : float, optional
        A midpoint the values diverge from, such as 0 for correlations. The
        colour scale is then centred on it, symmetric about it.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None`` a new figure is created.
    tick_fontsize : float
        Font size for tick labels.
    title_fontsize : float
        Font size for the title.
    colorbar_fraction : float
        Fraction of axes given to the colorbar.
    colorbar_pad : float
        Padding between axes and colorbar.

    Returns
    -------
    fig, ax
    """
    n = len(labels)
    idx = np.arange(n)

    if ax is None:
        side = min(max(3, 0.6 * n), 14)
        sz = figsize or (side + 1.5, side)  # extra width for colorbar
        fig, ax = plt.subplots(figsize=sz)
    else:
        fig = ax.get_figure()

    with plt.rc_context(
        {
            "ytick.labelsize": tick_fontsize,
            "xtick.labelsize": tick_fontsize,
            "axes.titlesize": title_fontsize,
        }
    ):
        if centre is None:
            im = ax.imshow(matrix, cmap=cmap or plot_styles.SEQUENTIAL_CMAP)
        else:
            im = ax.imshow(matrix, cmap=cmap or plot_styles.DIVERGING_CMAP, norm=CenteredNorm(vcenter=centre))
        ax.set_title(title)
        ax.set_xticks(idx)
        ax.set_yticks(idx)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=colorbar_fraction, pad=colorbar_pad)

    return fig, ax
