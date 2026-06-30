# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dse_research_utils.ml.kernels import quadratic_distance_kernel
from dse_research_utils.plot.gaussian_process import (
    gaussian_process_prior,
    plot_kernel_function,
    plot_line,
)
from dse_research_utils.plot.io import save_figure


def test_plot_line_draws_two_layers():
    fig, ax = plt.subplots()
    plot_line(np.arange(5), np.arange(5), label="x")
    # bordered style draws a white background line + the foreground line.
    assert len(ax.lines) == 2
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


def test_plot_histograms_optional_seaborn(tmp_path):
    pytest.importorskip("seaborn")
    from dse_research_utils.plot.grids import plot_histograms

    df = pd.DataFrame({"a": np.arange(20.0), "b": np.linspace(0, 5, 20)})
    fig, axes = plot_histograms(df)
    assert axes.shape[0] >= 1
    plt.close(fig)
