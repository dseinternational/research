# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from preliz.distributions.distributions import Continuous

import dse_research_utils.plot.styles as plot_styles


def plot_distribution(
    dist: Continuous,
    output_dir: str | None = None,
    filename: str | None = None,
) -> Figure:
    plt.figure(figsize=plot_styles.FIGSIZE_MD)
    dist.plot_pdf()
    if output_dir is not None and filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.svg"))
    return plt.gcf()
