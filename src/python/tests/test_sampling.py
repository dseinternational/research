# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from dse_research_utils.statistics.models.sampling import (
    CHAINS_DEV,
    CHAINS_REP,
    CHAINS_REP_LITE,
    CHAINS_TEST,
    SAMPLES_DEV,
    SAMPLES_REP,
    SAMPLES_REP_LITE,
    SAMPLES_TEST,
    TARGET_ACCEPT_DEV,
    TARGET_ACCEPT_REP,
    TARGET_ACCEPT_REP_LITE,
    TARGET_ACCEPT_TEST,
    TUNES_DEV,
    TUNES_REP,
    TUNES_REP_LITE,
    TUNES_TEST,
    SamplingConfiguration,
    get_sampling_configuration,
)


class TestGetSamplingConfiguration:
    @pytest.mark.parametrize("name", ["dev", "development"])
    def test_dev_aliases(self, name: str) -> None:
        cfg = get_sampling_configuration(name, random_seed=7)
        assert cfg.chains == CHAINS_DEV
        assert cfg.draws == SAMPLES_DEV
        assert cfg.tune == TUNES_DEV
        assert cfg.target_accept == TARGET_ACCEPT_DEV
        assert cfg.random_seed == 7

    @pytest.mark.parametrize("name", ["test", "testing"])
    def test_test_aliases(self, name: str) -> None:
        cfg = get_sampling_configuration(name)
        assert cfg.chains == CHAINS_TEST
        assert cfg.draws == SAMPLES_TEST
        assert cfg.tune == TUNES_TEST
        assert cfg.target_accept == TARGET_ACCEPT_TEST

    @pytest.mark.parametrize("name", ["reporting", "report", "rep"])
    def test_reporting_aliases(self, name: str) -> None:
        cfg = get_sampling_configuration(name)
        assert cfg.chains == CHAINS_REP
        assert cfg.draws == SAMPLES_REP
        assert cfg.tune == TUNES_REP
        assert cfg.target_accept == TARGET_ACCEPT_REP

    @pytest.mark.parametrize("name", ["rep-lite", "reporting-lite", "rep_lite"])
    def test_rep_lite_aliases(self, name: str) -> None:
        cfg = get_sampling_configuration(name)
        assert cfg.chains == CHAINS_REP_LITE
        assert cfg.draws == SAMPLES_REP_LITE
        assert cfg.tune == TUNES_REP_LITE
        assert cfg.target_accept == TARGET_ACCEPT_REP_LITE

    def test_unknown_config_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid sampling configuration"):
            get_sampling_configuration("nonsense")

    def test_default_is_dev(self) -> None:
        cfg = get_sampling_configuration()
        assert cfg.chains == CHAINS_DEV
        assert cfg.draws == SAMPLES_DEV

    def test_cores_bounded_by_chains(self) -> None:
        cfg = get_sampling_configuration("dev")
        assert cfg.cores <= CHAINS_DEV
        assert cfg.cores >= 1

    def test_returns_sampling_configuration_dataclass(self) -> None:
        cfg = get_sampling_configuration("dev")
        assert isinstance(cfg, SamplingConfiguration)


def test_target_accept_increases_with_rigour() -> None:
    assert TARGET_ACCEPT_DEV < TARGET_ACCEPT_TEST < TARGET_ACCEPT_REP
    assert 0.5 < TARGET_ACCEPT_DEV < 1.0
    assert 0.5 < TARGET_ACCEPT_REP < 1.0


def test_rep_lite_is_lighter_but_keeps_reporting_target_accept() -> None:
    lite = get_sampling_configuration("rep-lite")
    rep = get_sampling_configuration("rep")
    # Fewer total posterior draws than full reporting...
    assert lite.chains * lite.draws < rep.chains * rep.draws
    # ...but keeps rep's divergence defence (unlike test, which lowers it).
    assert lite.target_accept == rep.target_accept
    assert lite.target_accept == TARGET_ACCEPT_REP_LITE
