# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from preliz.distributions.distributions import Continuous

import dse_research_utils.plot.io as plot_io
import dse_research_utils.plot.styles as plot_styles


def plot_distribution(
    dist: Continuous,
    output_dir: str | None = None,
    filename: str | None = None,
) -> Figure:
    plt.figure(figsize=plot_styles.FIGSIZE_MD)
    dist.plot_pdf()
    if output_dir is not None and filename is not None:
        # bbox_inches=None and close=False keep the historical cropping and the
        # returned-figure contract of this function.
        plot_io.save_styled_figure(output_dir, filename, bbox_inches=None, close=False)
    return plt.gcf()
