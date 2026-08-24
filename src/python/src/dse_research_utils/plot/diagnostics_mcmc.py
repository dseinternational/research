# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later


from typing import Any

import arviz as az
import arviz_plots as azp
import numpy as np
from arviz_plots import PlotCollection

import dse_research_utils.plot.io as plot_io


def plot_variable_count(trace: Any, var_name: str) -> int:
    """Number of subplot panels one posterior variable needs (non-sample dims).

    The sample dims are resolved from ``posterior.attrs["sample_dims"]`` or
    ArviZ's ``rcParams["data.sample_dims"]`` rather than hard-coding
    ``{"chain", "draw"}``.
    """
    posterior = trace.posterior
    if hasattr(posterior, "dataset"):
        posterior = posterior.dataset

    sample_dims = posterior.attrs.get("sample_dims", az.rcParams["data.sample_dims"])
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    sample_dims = set(sample_dims)
    variable = posterior[var_name]
    plot_dims = [dim for dim in variable.dims if dim not in sample_dims]
    return int(np.prod([variable.sizes[dim] for dim in plot_dims]))


def capped_plot_var_names(
    trace: Any,
    var_names: list[str],
    *,
    squared: bool = False,
) -> list[str]:
    """Return the first variables that fit within ArviZ's subplot limit.

    ``squared=True`` budgets for a pair plot, where ``k`` selected panels
    render as a ``k x k`` grid, so the per-variable budget is
    ``floor(sqrt(max_subplots))``.
    """
    max_subplots = az.rcParams.get("plot.max_subplots")
    if not isinstance(max_subplots, int):
        return list(var_names)

    max_plot_items = max_subplots
    if squared:
        max_plot_items = int(np.floor(np.sqrt(max_subplots)))

    selected_var_names: list[str] = []
    selected_plot_items = 0
    for var_name in var_names:
        plot_items = plot_variable_count(trace, var_name)
        if selected_plot_items + plot_items <= max_plot_items:
            selected_var_names.append(var_name)
            selected_plot_items += plot_items

    return selected_var_names


def plot_required_subplots(
    trace: Any,
    var_names: list[str],
    *,
    squared: bool = False,
) -> int:
    """Total panels ``var_names`` would need (``squared=True`` for pair plots)."""
    n_plot_items = sum(plot_variable_count(trace, var_name) for var_name in var_names)
    if squared:
        return n_plot_items**2
    return n_plot_items


def plot_kde_pair(
    data: Any,
    var_names: list[str],
    output_dir: str | None = None,
    filename: str | None = None,
) -> PlotCollection:
    """
    KDE pair plot with divergences.

    Returns
    -------
    arviz_plots.PlotCollection
    """
    pc = azp.plot_pair(
        data,
        var_names=var_names,
        marginal=True,
        marginal_kind="kde",
        visuals={"divergence": True},
    )

    if output_dir is not None and filename is not None:
        # close=False keeps the returned collection displayable in notebooks.
        plot_io.save_plotcollection(pc, output_dir, filename, close=False)

    return pc
