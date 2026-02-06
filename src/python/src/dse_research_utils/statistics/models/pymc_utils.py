# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytensor.tensor as pt
import pymc as pm
import psutil

from graphviz import Digraph

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
    print("PyMC Model summary:")
    print("===================")

    print()
    print("Free random variables:")
    print("----------------------")
    for rv in model.free_RVs:
        print(f"  {rv.name} [{rv.type}]")

    print()
    print("Deterministic random variables:")
    print("-------------------------------")
    for urv in model.deterministics:
        print(f"  {urv.name} [{urv.type}]")

    print()
    print("Observed random variables:")
    print("--------------------------")
    for rv in model.observed_RVs:
        print(f"  {rv.name} [{rv.type}]")

    model.basic_RVs


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
