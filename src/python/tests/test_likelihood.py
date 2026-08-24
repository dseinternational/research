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


def test_beta_binomial_from_p_matches_logit_form():
    from dse_research_utils.statistics.models.likelihood import beta_binomial_from_p

    rng = np.random.default_rng(1)
    n_trials = 20
    observed = rng.integers(0, n_trials + 1, size=30)
    eta_values = rng.normal(size=30)

    with pm.Model() as m_logit:
        eta = pm.Normal("eta", 0.0, 1.0, shape=30)
        kappa = pm.HalfNormal("kappa", 5.0)
        beta_binomial_from_logit("y", eta, n_trials, kappa, observed=observed)

    with pm.Model() as m_p:
        eta = pm.Normal("eta", 0.0, 1.0, shape=30)
        kappa = pm.HalfNormal("kappa", 5.0)
        beta_binomial_from_p("y", pm.math.sigmoid(eta), n_trials, kappa, observed=observed)

    # The two parameterisations build the same likelihood: identical logp at
    # the same point.
    point = m_logit.initial_point()
    point["eta"] = eta_values
    assert m_logit.compile_logp()(point) == pytest.approx(m_p.compile_logp()(point))


def test_beta_binomial_from_p_clips_boundary_probabilities():
    from dse_research_utils.statistics.models.likelihood import beta_binomial_from_p

    observed = np.array([0, 5, 10])
    with pm.Model() as model:
        kappa = pm.HalfNormal("kappa", 5.0)
        # Constant p at the exact boundary would give alpha = 0 without the clip.
        beta_binomial_from_p("y", pm.math.constant([0.0, 0.5, 1.0]), 10, kappa, observed=observed)
    logp = model.compile_logp()(model.initial_point())
    assert np.isfinite(logp)
