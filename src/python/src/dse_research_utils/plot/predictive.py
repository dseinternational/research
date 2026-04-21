# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import dse_research_utils.plot.styles as plot_styles


def plot_prior_samples_binomial(
    x: np.ndarray,
    y_samples: np.ndarray,
    x_observed: np.ndarray | pd.Series,
    y_observed: np.ndarray | pd.Series,
    n_trials: int = 100,
    n_curves: int = 500,
    x_label: str = "x",
    y_label: str = "y",
    filename: str | None = None,
    output_dir: str | None = None,
    report_figs_dir: str | None = None,
    random_seed: int | None = None,
) -> Figure:

    plt.figure(figsize=plot_styles.FIGSIZE_XL)

    n_samples = y_samples.shape[1]

    rng = np.random.default_rng(random_seed)
    idx = rng.integers(0, n_samples, n_curves)

    for i in idx:
        counts = y_samples[:, i] * n_trials
        plt.plot(x, counts, c=plot_styles.COLOUR_ORANGE, alpha=0.1, lw=1)

    plt.scatter(
        x_observed,
        y_observed,
        c=plot_styles.COLOUR_BLUE,
        alpha=0.4,
        label="Observed data",
    )

    plt.xlim(x.min() - 1, x.max() + 1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if filename is not None and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.svg"))
        if report_figs_dir is not None:
            os.makedirs(report_figs_dir, exist_ok=True)
            plt.savefig(os.path.join(report_figs_dir, f"{filename}.png"), dpi=300)

    return plt.gcf()


def _plot_predictive_checks(
    data,
    group="posterior",
    num_pp_samples=None,
    random_seed=None,
    output_dir=None,
    filename=None,
) -> Figure:

    az.plot_ppc(
        data,
        group=group,
        num_pp_samples=num_pp_samples,
        random_seed=random_seed,
        figsize=plot_styles.FIGSIZE_MD,
    )

    if output_dir is not None and filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.svg"))

    return plt.gcf()


def plot_prior_predictive_checks(
    data,
    num_pp_samples=None,
    random_seed=None,
    output_dir=None,
    filename=None,
) -> Figure:

    return _plot_predictive_checks(
        data,
        "prior",
        num_pp_samples,
        random_seed,
        output_dir,
        filename,
    )


def plot_posterior_predictive_checks(
    data,
    num_pp_samples=None,
    random_seed=None,
    output_dir=None,
    filename=None,
) -> Figure:

    return _plot_predictive_checks(
        data,
        "posterior",
        num_pp_samples,
        random_seed,
        output_dir,
        filename,
    )
