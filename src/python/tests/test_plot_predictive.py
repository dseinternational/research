# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from dse_research_utils.plot.predictive import plot_prior_samples_binomial


def _make_samples(n_x: int = 30, n_samples: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, n_x)
    y_samples = rng.uniform(0.0, 1.0, size=(n_x, n_samples))
    x_obs = rng.uniform(0.0, 10.0, size=15)
    y_obs = rng.integers(0, 100, size=15)
    return x, y_samples, x_obs, y_obs


def test_returns_figure() -> None:
    x, y_samples, x_obs, y_obs = _make_samples()
    fig = plot_prior_samples_binomial(
        x=x,
        y_samples=y_samples,
        x_observed=x_obs,
        y_observed=y_obs,
        n_curves=5,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_random_seed_is_deterministic() -> None:
    x, y_samples, x_obs, y_obs = _make_samples()
    fig1 = plot_prior_samples_binomial(
        x=x,
        y_samples=y_samples,
        x_observed=x_obs,
        y_observed=y_obs,
        n_curves=3,
        random_seed=42,
    )
    ax1 = fig1.axes[0]
    ys1 = [tuple(line.get_ydata()) for line in ax1.lines]
    plt.close(fig1)

    fig2 = plot_prior_samples_binomial(
        x=x,
        y_samples=y_samples,
        x_observed=x_obs,
        y_observed=y_obs,
        n_curves=3,
        random_seed=42,
    )
    ax2 = fig2.axes[0]
    ys2 = [tuple(line.get_ydata()) for line in ax2.lines]
    plt.close(fig2)

    assert ys1 == ys2


def test_saves_primary_and_report_figs(tmp_path) -> None:
    x, y_samples, x_obs, y_obs = _make_samples()
    out = tmp_path / "out"
    report = tmp_path / "report"
    fig = plot_prior_samples_binomial(
        x=x,
        y_samples=y_samples,
        x_observed=x_obs,
        y_observed=y_obs,
        n_curves=2,
        output_dir=str(out),
        filename="prior",
        report_figs_dir=str(report),
        random_seed=0,
    )
    assert (out / "prior.png").exists()
    assert (out / "prior.svg").exists()
    assert (report / "prior.png").exists()
    plt.close(fig)
