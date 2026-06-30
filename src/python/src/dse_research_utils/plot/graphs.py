# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Directed-graph plotting helpers.

Requires the optional ``graphs`` extra (graphviz + networkx). ``graphviz`` also
needs the system Graphviz binaries on PATH, which is why these are not a base
dependency. Adapted from the PyMC statistical-rethinking lecture utilities:
https://github.com/pymc-devs/pymc-examples/blob/main/examples/statistical_rethinking_lectures/utils.py
"""

from __future__ import annotations

import numpy as np


def draw_causal_graph(edge_list, node_props=None, edge_props=None, graph_direction="UD"):
    """Draw a causal (directed) graph with graphviz; returns the ``Digraph``."""
    try:
        import graphviz as gr
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "draw_causal_graph requires graphviz; install the 'graphs' extra "
            "(pip install dse-research-utils[graphs]) and the system Graphviz binaries."
        ) from exc

    g = gr.Digraph(graph_attr={"rankdir": graph_direction})

    edge_props = {} if edge_props is None else edge_props
    for e in edge_list:
        props = edge_props.get(e, {})
        g.edge(e[0], e[1], **props)

    if node_props is not None:
        for name, props in node_props.items():
            g.node(name=name, **props)
    return g


def plot_graph(graph, **graph_kwargs):
    """Draw a network graph with networkx.

    Parameters
    ----------
    graph : networkx.DiGraph | np.ndarray
        If an ndarray, it is treated as an adjacency matrix of a directed graph.

    Returns
    -------
    The node layout (``pos``) used, for consistent re-plotting.
    """
    try:
        import networkx as nx
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "plot_graph requires networkx; install the 'graphs' extra "
            "(pip install dse-research-utils[graphs])."
        ) from exc

    # convert to networkx.DiGraph, if needed
    G = (
        nx.from_numpy_array(graph, create_using=nx.DiGraph)
        if isinstance(graph, np.ndarray)
        else graph
    )

    # Some layout functions (e.g. networkx's spring_layout) consume the global
    # numpy RNG. Snapshot it so we can seed deterministically for the layout call
    # and then restore the caller's RNG state.
    rng_state = np.random.get_state()
    try:
        np.random.seed(123)  # legacy global RNG, needed by networkx layout
        if "layout" in graph_kwargs:
            graph_kwargs["pos"] = graph_kwargs["layout"](G)

        default_graph_kwargs = {
            "node_color": "C0",
            "node_size": 500,
            "arrowsize": 30,
            "width": 3,
            "alpha": 0.7,
            "connectionstyle": "arc3,rad=0.1",
            "pos": nx.kamada_kawai_layout(G),
        }
        for k, v in default_graph_kwargs.items():
            if k not in graph_kwargs:
                graph_kwargs[k] = v

        nx.draw(G, **graph_kwargs)
    finally:
        np.random.set_state(rng_state)
    # return the node layout for consistent graphing
    return graph_kwargs["pos"]
