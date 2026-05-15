# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later


import os

import arviz_plots as azp
from arviz_plots import PlotCollection


def plot_kde_pair(
    data,
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
        os.makedirs(output_dir, exist_ok=True)
        pc.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        pc.savefig(os.path.join(output_dir, f"{filename}.svg"))

    return pc
