# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import pytest

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


def test_plot_heatmap_defaults_to_the_sequential_scale():
    fig, ax = plot_heatmap(np.eye(2), ["a", "b"], "Title")
    assert ax.images[0].get_cmap().name == "dse_sequential"
    plt.close(fig)


def test_plot_heatmap_centres_a_diverging_scale():
    m = np.array([[1.0, -0.2], [-0.2, 0.6]])
    fig, ax = plot_heatmap(m, ["a", "b"], "Correlation", centre=0.0)
    image = ax.images[0]
    assert image.get_cmap().name == "dse_diverging"
    assert image.norm.vcenter == 0.0
    # Symmetric about the centre, so equal and opposite values take mirrored colours.
    assert image.norm(1.0) == pytest.approx(1 - image.norm(-1.0))
    plt.close(fig)
