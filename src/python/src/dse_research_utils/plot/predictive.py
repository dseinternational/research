# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import inspect
from typing import Any

import arviz_plots as azp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arviz_plots import PlotCollection
from matplotlib.figure import Figure

import dse_research_utils.plot.io as plot_io
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
        # bbox_inches=None and close=False keep the historical cropping and the
        # returned-figure contract of this function.
        plot_io.save_styled_figure(output_dir, filename, bbox_inches=None, close=False)
        if report_figs_dir is not None:
            plot_io.save_styled_figure(report_figs_dir, filename, bbox_inches=None, close=False, svg=False)

    return plt.gcf()


def _plot_predictive_checks(
    data: Any,
    group: str = "posterior",
    num_pp_samples: int | None = None,
    random_seed: int | None = None,
    output_dir: str | None = None,
    filename: str | None = None,
) -> PlotCollection:

    kwargs: dict = {}
    if num_pp_samples is not None:
        kwargs["num_samples"] = num_pp_samples
    seed_via_argument = False
    if random_seed is not None:
        try:
            seed_via_argument = "random_seed" in inspect.signature(azp.plot_ppc_dist).parameters
        except TypeError, ValueError:
            seed_via_argument = False
        if seed_via_argument:
            kwargs["random_seed"] = random_seed

    if random_seed is not None and not seed_via_argument:
        rng_state = np.random.get_state()
        np.random.seed(random_seed)
        try:
            pc = azp.plot_ppc_dist(
                data,
                group=f"{group}_predictive",
                **kwargs,
            )
        finally:
            np.random.set_state(rng_state)
    else:
        pc = azp.plot_ppc_dist(
            data,
            group=f"{group}_predictive",
            **kwargs,
        )

    if output_dir is not None and filename is not None:
        # close=False keeps the returned collection displayable in notebooks.
        plot_io.save_plotcollection(pc, output_dir, filename, close=False)

    return pc


def plot_prior_predictive_checks(
    data: Any,
    num_pp_samples: int | None = None,
    random_seed: int | None = None,
    output_dir: str | None = None,
    filename: str | None = None,
) -> PlotCollection:

    return _plot_predictive_checks(
        data,
        "prior",
        num_pp_samples,
        random_seed,
        output_dir,
        filename,
    )


def plot_posterior_predictive_checks(
    data: Any,
    num_pp_samples: int | None = None,
    random_seed: int | None = None,
    output_dir: str | None = None,
    filename: str | None = None,
) -> PlotCollection:

    return _plot_predictive_checks(
        data,
        "posterior",
        num_pp_samples,
        random_seed,
        output_dir,
        filename,
    )
