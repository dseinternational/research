# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later


from typing import Any

import arviz_plots as azp
from arviz_plots import PlotCollection

import dse_research_utils.plot.io as plot_io


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
