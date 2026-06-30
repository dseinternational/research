# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest

pm = pytest.importorskip("pymc")

from dse_research_utils.statistics.models.likelihood import (  # noqa: E402
    beta_binomial_from_logit,
)


def test_beta_binomial_from_logit_registers_observed_node():
    rng = np.random.default_rng(0)
    n_trials = 20
    observed = rng.integers(0, n_trials + 1, size=30)
    with pm.Model() as model:
        eta = pm.Normal("eta", 0.0, 1.0, shape=30)
        kappa = pm.HalfNormal("kappa", 5.0)
        y = beta_binomial_from_logit("y", eta, n_trials, kappa, observed=observed)
    assert "y" in model.named_vars
    assert y in model.observed_RVs
    # logp is finite at the initial point (mu clipped away from 0/1).
    logp = model.compile_logp()(model.initial_point())
    assert np.isfinite(logp)
