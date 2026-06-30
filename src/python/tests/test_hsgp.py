# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest

pm = pytest.importorskip("pymc")

from dse_research_utils.statistics.models.hsgp import (  # noqa: E402
    build_hsgp_1d,
    build_tau_modifier,
)


def test_build_hsgp_1d_registers_named_nodes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    with pm.Model() as model:
        f = build_hsgp_1d("f_age", x, m=10)
    names = set(model.named_vars)
    assert {"f_age", "f_age__eta", "f_age__ell"}.issubset(names)
    assert any(n.startswith("f_age__g_unit") for n in names)
    assert "f_age" in {d.name for d in model.deterministics}
    logp = model.compile_logp()(model.initial_point())
    assert np.isfinite(logp)


def test_build_tau_modifier_smaller_default_basis():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    with pm.Model() as model:
        build_tau_modifier("g_tau", x)
    # Default m=15 basis; node naming matches build_hsgp_1d.
    assert {"g_tau", "g_tau__eta", "g_tau__ell"}.issubset(set(model.named_vars))
