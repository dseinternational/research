# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import arviz as az
import pandas as pd
import psutil
import pymc as pm
import pytensor.tensor as pt
from arviz import InferenceData
from graphviz import Digraph
from pytensor.tensor.variable import TensorVariable

from dse_research_utils.console.console import get_console
from dse_research_utils.console.sections import section_header, subsection
from dse_research_utils.math.constants import EPSILON


def logit(p, eps=EPSILON):
    p = pm.math.clip(p, eps, 1 - eps)
    return pm.math.log(p) - pt.log1p(-p)


def get_available_cores() -> int:
    """
    Get the number of available cores to use for PyMC sampling.

    Returns
    -------
    int
        The number of CPU cores to use, which is the maximum of 4 and
        the total number of physical CPUs minus one.
    """
    n_cpus = psutil.cpu_count(logical=False)
    if n_cpus is None:
        n_cpus = psutil.cpu_count(logical=True) or 2
    return max(4, n_cpus - 1)


def report_model_summary(model: pm.Model):
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


def model_to_graphviz(model: pm.Model) -> Digraph:
    """
    Export the PyMC model to a Graphviz dot file.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to export.
    Returns
    -------
    Digraph
        The Graphviz Digraph object representing the model.
    """
    dg = pm.model_to_graphviz(model)
    dg.graph_attr["fontname"] = "Helvetica"
    dg.node_attr["fontname"] = "Helvetica"
    dg.edge_attr["fontname"] = "Helvetica"
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
    trace: InferenceData,
    round_to: int = 3,
    hdi_prob: float = 0.89,
) -> pd.DataFrame:
    """
    Get summary diagnostics for the PyMC model.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to analyze.
    trace : InferenceData
        The trace object containing the samples from the model.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the summary diagnostics for each variable.
    """
    var_names = [var.name for var in model.unobserved_RVs if var.size.eval() <= 2]

    diagnostics_df = az.summary(trace, var_names=var_names, round_to=round_to, hdi_prob=hdi_prob)

    return diagnostics_df
