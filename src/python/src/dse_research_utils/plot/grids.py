# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small-multiple grid plots. Requires the optional ``viz`` extra (seaborn)."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.figure as mpfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_histograms(
    data: pd.DataFrame,
    sharex: bool = False,
    sharey: bool = False,
    max_cols: int = 3,
    max_bins: int = 10,
    col_width: float = 4,
    col_height: float = 4,
    name_lookup: dict[str, str] | None = None,
) -> tuple[mpfig.Figure, Any]:
    """Plot histograms of all numeric columns in a DataFrame on a subplot grid.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame; only numeric columns are plotted.
    max_cols : int
        Maximum number of columns in the subplot grid.
    max_bins : int
        Maximum number of histogram bins.
    col_width, col_height : float
        Per-subplot width / height in inches.
    name_lookup : dict, optional
        Maps a column name to a display label; falls back to the column name.

    Returns
    -------
    (fig, axes)

    Notes
    -----
    Requires ``seaborn`` (install the ``viz`` extra: ``pip install
    dse-research-utils[viz]``).
    """
    try:
        import seaborn as sns
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "plot_histograms requires seaborn; install the 'viz' extra "
            "(pip install dse-research-utils[viz])."
        ) from exc

    df = data.select_dtypes(include=["number"])
    n_cols = df.shape[1]
    cols = min(max_cols, n_cols)
    rows = math.ceil(n_cols / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(col_width * cols, col_height * rows),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )

    fig.tight_layout(pad=4.0, w_pad=2.0, h_pad=4.0)

    axes_flat = axes.flatten()

    for ax, v in zip(axes_flat, df.columns, strict=False):
        series = df[v].dropna().to_numpy(dtype=np.float64)
        if series.size == 0:
            ax.set_visible(False)
            continue
        spread = float(series.max() - series.min())
        bins = min(max_bins, max(2, int(spread)))

        sns.histplot(data=series, bins=bins, kde=True, ax=ax)

        if name_lookup is not None:
            name = name_lookup.get(v)
            name = name if name is not None else v
        else:
            name = v

        ax.set_title(f"{name} distribution")
        ax.set_xlabel(name)
        ax.set_ylabel("Count")

    for ax in axes_flat[n_cols:]:
        ax.set_visible(False)

    return fig, axes
