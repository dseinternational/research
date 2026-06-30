# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np

from dse_research_utils.plot.heatmap import plot_heatmap


def test_plot_heatmap_returns_fig_ax_with_labels():
    m = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1], [0.2, 0.1, 1.0]])
    labels = ["a", "b", "c"]
    fig, ax = plot_heatmap(m, labels, "Title")
    assert ax.get_title() == "Title"
    assert [t.get_text() for t in ax.get_xticklabels()] == labels
    assert [t.get_text() for t in ax.get_yticklabels()] == labels
    assert ax.images  # an imshow image was drawn
    plt.close(fig)
