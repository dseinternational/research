# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the shared PSIS-LOO / ELPD helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from dse_research_utils.statistics.loo import (
    ELPD_DIFF_INCONCLUSIVE,
    as_dataset,
    elpd_diff_paired,
    elpd_total_se,
    elpd_verdict,
    group_names,
    has_group,
    loo_summary_row,
    pareto_k_bands,
    pareto_k_reliability,
    pareto_k_values,
    reff_or_default,
    sampled_parameter_names,
    sampled_parameter_reff,
)


def _posterior_dataset(seed: int = 0, n_chains: int = 2) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    shape = (n_chains, 400)
    return xr.Dataset(
        {
            "mu": (("chain", "draw"), rng.normal(size=shape)),
            "sigma": (("chain", "draw"), rng.normal(2.0, 0.5, size=shape)),
            "derived": (("chain", "draw"), rng.normal(size=shape)),
        },
        coords={"chain": range(shape[0]), "draw": range(shape[1])},
    )


def _tree(posterior: xr.Dataset) -> xr.DataTree:
    return xr.DataTree.from_dict({"posterior": posterior})


class TestContainerShims:
    def test_as_dataset_passthrough(self):
        ds = _posterior_dataset()
        assert as_dataset(ds) is ds

    def test_as_dataset_from_datatree_node(self):
        tree = _tree(_posterior_dataset())
        node = tree["posterior"]
        out = as_dataset(node)
        assert isinstance(out, xr.Dataset)
        assert set(out.data_vars) == {"mu", "sigma", "derived"}

    def test_group_names_from_datatree(self):
        tree = xr.DataTree.from_dict({"posterior": _posterior_dataset(), "sample_stats": xr.Dataset({})})
        assert {"posterior", "sample_stats"} <= group_names(tree)
        assert has_group(tree, "posterior")
        assert not has_group(tree, "log_likelihood")

    def test_group_names_tolerates_objects_without_groups(self):
        assert group_names(object()) == set()


class TestSampledParameterReff:
    def test_names_argument_wins(self):
        assert sampled_parameter_names(object(), names=["a", "b"]) == ["a", "b"]

    def test_attr_reader_supplies_names(self):
        assert sampled_parameter_names(object(), attr_reader=lambda t: ["mu"]) == ["mu"]

    def test_lookup_error_when_nothing_available(self):
        with pytest.raises(LookupError, match="sampled parameters"):
            sampled_parameter_names(object(), attr_reader=lambda t: None)

    def test_reff_restricted_to_named_parameters(self):
        pytest.importorskip("arviz_stats")
        tree = _tree(_posterior_dataset())
        reff = sampled_parameter_reff(tree, names=["mu", "sigma"])
        assert reff > 0.0
        # iid draws -> ESS close to n_samples -> reff near (often above) 1.
        assert reff == pytest.approx(1.0, abs=0.35)

    def test_reff_single_chain_is_one(self):
        tree = _tree(_posterior_dataset(n_chains=1))
        assert sampled_parameter_reff(tree, names=["mu"]) == 1.0

    def test_reff_missing_parameter_raises(self):
        tree = _tree(_posterior_dataset())
        with pytest.raises(KeyError, match="absent"):
            sampled_parameter_reff(tree, names=["nonexistent"])

    def test_reff_or_default_warns_and_returns_none(self):
        messages: list[str] = []
        tree = _tree(_posterior_dataset())
        out = reff_or_default(tree, attr_reader=lambda t: None, label="vg10", warn=messages.append)
        assert out is None
        assert len(messages) == 1
        assert "vg10" in messages[0]
        assert "posterior-wide" in messages[0]

    def test_reff_or_default_pins_when_possible(self):
        pytest.importorskip("arviz_stats")
        tree = _tree(_posterior_dataset())
        out = reff_or_default(tree, attr_reader=lambda t: ["mu", "sigma"])
        assert out is not None and out > 0


def _fake_elpd(
    *,
    elpd: float = -100.0,
    se: float = 5.0,
    p: float = 3.0,
    k: tuple[float, ...] = (0.1, 0.4, 0.8),
    good_k: float = 0.7,
    one_x_style: bool = True,
) -> SimpleNamespace:
    arr = xr.DataArray(np.asarray(k, dtype=float))
    ns = SimpleNamespace(elpd=elpd, se=se, p=p, good_k=good_k)
    if one_x_style:
        ns.diagnostics = arr  # ArviZ 1.x name
    else:
        ns.pareto_k = arr  # ArviZ 0.x name
    return ns


class TestParetoK:
    def test_pareto_k_values_both_attribute_conventions(self):
        np.testing.assert_allclose(pareto_k_values(_fake_elpd(one_x_style=True)), [0.1, 0.4, 0.8])
        np.testing.assert_allclose(pareto_k_values(_fake_elpd(one_x_style=False)), [0.1, 0.4, 0.8])

    def test_reliability_reductions(self):
        rel = pareto_k_reliability([0.1, 0.4, 0.8, 1.2], good_k=0.7)
        assert rel["max"] == pytest.approx(1.2)
        assert rel["n_above"] == 2
        assert rel["share_above"] == pytest.approx(0.5)
        assert rel["reliable"] is False
        rel_ok = pareto_k_reliability([0.1, 0.2], good_k=0.7)
        assert rel_ok["reliable"] is True

    def test_bands(self):
        bands = pareto_k_bands([0.1, 0.68, 0.72, 0.9, 1.5], good_k=0.7)
        assert bands == {"good": 2, "bad": 2, "very_bad": 1}


class TestLooSummaryRow:
    def test_default_threshold_is_the_fits_good_k(self):
        row = loo_summary_row(_fake_elpd(k=(0.1, 0.65, 0.8), good_k=0.6), label="m1")
        assert row["label"] == "m1"
        assert row["elpd_loo"] == pytest.approx(-100.0)
        assert row["se"] == pytest.approx(5.0)
        assert row["p_loo"] == pytest.approx(3.0)
        assert row["k_threshold"] == pytest.approx(0.6)
        assert row["pareto_k_above"] == 2  # 0.65 and 0.8 exceed the fit's own good_k
        assert row["n_observations"] == 3
        assert row["reff"] is None
        assert "looic" not in row

    def test_explicit_threshold_and_unit_name_and_looic(self):
        row = loo_summary_row(
            _fake_elpd(),
            label="m2",
            reff=0.9,
            unit_name="n_subjects",
            k_threshold=0.7,
            include_looic=True,
        )
        assert row["pareto_k_above"] == 1
        assert row["n_subjects"] == 3
        assert row["reff"] == pytest.approx(0.9)
        assert row["looic"] == pytest.approx(200.0)
        assert row["looic_se"] == pytest.approx(10.0)

    def test_missing_good_k_falls_back_to_0_7(self):
        loo = _fake_elpd()
        del loo.good_k
        row = loo_summary_row(loo, label="m3")
        assert row["k_threshold"] == pytest.approx(0.7)


class TestElpdConventions:
    def test_total_se_matches_formula(self):
        rng = np.random.default_rng(0)
        pw = rng.normal(size=250)
        assert elpd_total_se(pw) == pytest.approx(np.sqrt(250) * np.std(pw, ddof=1))

    def test_paired_diff(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=100)
        b = a - 0.3  # model a better by 0.3 per point
        out = elpd_diff_paired(a, b)
        assert out["diff"] == pytest.approx(30.0)
        assert out["dse"] == pytest.approx(0.0, abs=1e-9)  # constant difference
        assert out["n"] == 100

    def test_paired_diff_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal shapes"):
            elpd_diff_paired([1.0, 2.0], [1.0])

    def test_verdict(self):
        assert elpd_verdict(None) == "unavailable"
        assert elpd_verdict(float("nan")) == "unavailable"
        assert "inconclusive" in elpd_verdict(3.9)
        assert "inconclusive" in elpd_verdict(-3.9)
        assert "discriminating" in elpd_verdict(4.0)
        assert "discriminating" in elpd_verdict(-12.0)
        assert ELPD_DIFF_INCONCLUSIVE == 4.0
