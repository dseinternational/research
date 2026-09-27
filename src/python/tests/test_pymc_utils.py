# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import pymc as pm
import pytest

from dse_research_utils.statistics.models.pymc_utils import model_to_graphviz

pytest.importorskip("graphviz")


def test_model_to_graphviz_uses_noto_sans_with_generic_fallback() -> None:
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1, observed=[0.0, 1.0])

    dg = model_to_graphviz(model, dpi=150)

    for attrs in (dg.graph_attr, dg.node_attr, dg.edge_attr):
        assert attrs["fontname"] == "Noto Sans,sans-serif"
    assert dg.graph_attr["dpi"] == "150"
    assert 'fontname="Noto Sans,sans-serif"' in dg.source
