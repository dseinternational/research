# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dse_research_utils.ml.kernels import quadratic_distance_kernel
from dse_research_utils.plot.gaussian_process import (
    gaussian_process_prior,
    plot_2d_function,
    plot_errorbar,
    plot_gaussian_process,
    plot_gaussian_process_prior,
    plot_kernel_function,
    plot_line,
    plot_x_errorbar,
)
from dse_research_utils.plot.io import save_figure


def test_gaussian_process_accepts_numpy_samples():
    fig, ax = plt.subplots()
    try:
        x = np.linspace(-1, 1, 5)
        samples = np.array([x, x**2])
        plot_gaussian_process(x, samples=samples)
        assert len(ax.lines) == 4
        np.testing.assert_allclose(ax.lines[1].get_ydata(), samples[0])
        np.testing.assert_allclose(ax.lines[3].get_ydata(), samples[1])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("n_samples", [1, 3])
def test_gaussian_process_prior_plot_renders_samples(n_samples):
    try:
        axes = plot_gaussian_process_prior(quadratic_distance_kernel, n_samples=n_samples, resolution=10)
        assert len(axes[0].lines) == 2 * n_samples
    finally:
        plt.close("all")


def test_plot_line_accepts_explicit_zorder():
    fig, ax = plt.subplots()
    try:
        plot_line([0, 1], [0, 1], zorder=5)
        assert ax.lines[0].get_zorder() < ax.lines[1].get_zorder() == 5
    finally:
        plt.close(fig)


def test_contour_accepts_rectangular_grid():
    fig, ax = plt.subplots()
    try:
        contour = plot_2d_function(np.arange(3.0), np.arange(5.0), lambda x, y: x + y, ax=ax, levels=[2.0])
        vertices = contour.allsegs[0][0]
        np.testing.assert_allclose(vertices.sum(axis=1), 2.0)
    finally:
        plt.close(fig)


def test_plot_graph_accepts_custom_layout():
    nx = pytest.importorskip("networkx")
    from dse_research_utils.plot.graphs import plot_graph

    fig = plt.figure()
    try:
        graph = nx.path_graph(3, create_using=nx.DiGraph)
        expected = nx.circular_layout(graph)
        actual = plot_graph(graph, layout=nx.circular_layout)
        for node in graph:
            np.testing.assert_allclose(actual[node], expected[node])
    finally:
        plt.close(fig)


def test_plot_line_draws_two_layers():
    fig, ax = plt.subplots()
    plot_line(np.arange(5), np.arange(5), label="x")
    # bordered style draws a white background line + the foreground line.
    assert len(ax.lines) == 2
    plt.close(fig)


def test_errorbar_rejects_mismatched_lengths():
    fig, _ax = plt.subplots()
    with pytest.raises(ValueError, match="zip\\(\\) argument"):
        plot_errorbar([1, 2], [1, 2], [0.1], [0.2, 0.3])
    with pytest.raises(ValueError, match="same length"):
        plot_x_errorbar([1, 2], [1, 2], [0.1, 0.2], [0.2, 0.3], colors=["C0"])
    plt.close(fig)


def test_plot_kernel_function_runs():
    fig, ax = plt.subplots()
    plot_kernel_function(quadratic_distance_kernel, ax=ax)
    assert ax.get_xlabel() == "|X1-X2|"
    plt.close(fig)


def test_gaussian_process_prior_is_multivariate_normal():
    X = np.linspace(-2, 2, 10)[:, None]
    prior = gaussian_process_prior(X, quadratic_distance_kernel)
    assert prior.rvs(3).shape == (3, 10)


def test_save_figure_writes_to_output_dir(tmp_path):
    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    save_figure("fig.png", tmp_path)
    plt.close(fig)
    assert (tmp_path / "fig.png").exists()


def test_save_figure_reports_through_shared_console(tmp_path, captured_console):
    """The save message must not bypass the shared console (#91)."""
    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    save_figure("fig.png", tmp_path)
    plt.close(fig)
    assert "saving figure to" in captured_console.export_text()


def test_plot_histograms_optional_seaborn(tmp_path):
    pytest.importorskip("seaborn")
    from dse_research_utils.plot.grids import plot_histograms

    df = pd.DataFrame({"a": np.arange(20.0), "b": np.linspace(0, 5, 20)})
    fig, axes = plot_histograms(df)
    assert axes.shape[0] >= 1
    plt.close(fig)


def test_plot_histograms_rejects_no_numeric_columns():
    pytest.importorskip("seaborn")
    from dse_research_utils.plot.grids import plot_histograms

    with pytest.raises(ValueError, match="numeric column"):
        plot_histograms(pd.DataFrame({"label": ["a", "b"]}))


def test_plot_histograms_rejects_invalid_layout_arguments():
    from dse_research_utils.plot.grids import plot_histograms

    with pytest.raises(ValueError, match="max_cols"):
        plot_histograms(pd.DataFrame({"a": [1.0, 2.0]}), max_cols=0)
