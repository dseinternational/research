# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import matplotlib.pyplot as plt
import dse_research_utils.plot.styles as plot_styles

from matplotlib.figure import Figure
from preliz.distributions.distributions import Continuous


def plot_distribution(dist: Continuous, output_dir: str, filename: str) -> Figure:
    plt.figure(figsize=plot_styles.FIGSIZE_MD)
    dist.plot_pdf()
    if output_dir is not None and filename is not None:
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.svg"))
    return plt.gcf()
