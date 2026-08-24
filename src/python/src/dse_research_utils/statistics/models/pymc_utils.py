# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import arviz as az
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable

from dse_research_utils.console.console import get_console
from dse_research_utils.console.sections import section_header, subsection
from dse_research_utils.math.constants import EPSILON

if TYPE_CHECKING:
    from graphviz import Digraph
    from xarray import DataTree


def logit(p: Any, eps: float = EPSILON) -> Any:
    p = pm.math.clip(p, eps, 1 - eps)
    return pm.math.log(p) - pt.log1p(-p)


def report_model_summary(model: pm.Model) -> None:
    """
    Print a summary of the PyMC model.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to summarize.
    """
    console = get_console()
    section_header("PyMC Model summary")

    subsection("Free random variables")
    for rv in model.free_RVs:
        console.print(f"  {rv.name} [{rv.type}]")

    subsection("Deterministic random variables")
    for urv in model.deterministics:
        console.print(f"  {urv.name} [{urv.type}]")

    subsection("Observed random variables")
    for rv in model.observed_RVs:
        console.print(f"  {rv.name} [{rv.type}]")


def model_to_graphviz(model: pm.Model, *, dpi: int | None = None) -> Digraph:
    """
    Export the PyMC model to a Graphviz dot file.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to export.
    dpi : int, optional
        Raster resolution recorded on the graph (e.g. 150 for a publication
        PNG); ``None`` leaves Graphviz's default.

    Returns
    -------
    Digraph
        The Graphviz Digraph object representing the model.
    """
    try:
        import graphviz as _graphviz  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "model_to_graphviz requires graphviz; install the 'graphs' extra "
            "(pip install dse-research-utils[graphs]) and the system Graphviz binaries."
        ) from exc

    dg = pm.model_to_graphviz(model)
    dg.graph_attr["fontname"] = "Helvetica"
    dg.node_attr["fontname"] = "Helvetica"
    dg.edge_attr["fontname"] = "Helvetica"
    if dpi is not None:
        dg.graph_attr["dpi"] = str(dpi)
    return dg


def get_variables_dict(model: pm.Model) -> dict[str, TensorVariable]:
    """
    Get a dictionary of all variables in the PyMC model.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to extract variables from.

    Returns
    -------
    dict[str, TensorVariable]
        A dictionary mapping variable names to their corresponding PyMC variable objects.
    """
    variables = {}
    for rv in model.free_RVs:
        variables[rv.name] = rv
    for urv in model.deterministics:
        variables[urv.name] = urv
    for rv in model.observed_RVs:
        variables[rv.name] = rv
    return variables


def get_summary_diagnostics(
    model: pm.Model,
    trace: DataTree,
    round_to: int = 3,
    hdi_prob: float = 0.89,
) -> pd.DataFrame:
    """
    Get summary diagnostics for the PyMC model.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to analyze.
    trace : xarray.DataTree
        The trace object containing the samples from the model.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the summary diagnostics for each variable.
    """
    var_names = [var.name for var in model.unobserved_RVs if var.size.eval() <= 2]

    diagnostics_df = az.summary(
        trace,
        var_names=var_names,
        round_to=round_to,
        ci_prob=hdi_prob,
        ci_kind="hdi",
    )

    return diagnostics_df
