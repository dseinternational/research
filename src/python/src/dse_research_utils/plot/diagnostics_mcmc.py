# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later


import arviz as az
import matplotlib.pyplot as plt
import os

from matplotlib.figure import Figure


def plot_kde_pair(
    data,
    var_names: list[str],
    output_dir: str | None = None,
    filename: str | None = None,
) -> Figure:
    """
    KDE pair plot with divergences.

    Returns
    -------
    matplotlib.figure.Figure
    """
    az.plot_pair(
        data,
        var_names=var_names,
        kind="kde",
        marginals=True,
        divergences=True,
    )

    if output_dir and filename:
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.svg"))

    return plt.gcf()
