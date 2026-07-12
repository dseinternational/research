# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import pytest

from dse_research_utils.statistics.models.reporting import ReportingConfiguration


class TestReportingConfiguration:
    def test_default_ci_prob(self) -> None:
        cfg = ReportingConfiguration(
            model_name="vg01",
            config_name="age-spoken-td",
            output_root_dir="/tmp/out",
        )
        assert cfg.ci_prob == 0.89

    def test_default_interval_kind_is_hdi(self) -> None:
        cfg = ReportingConfiguration(
            model_name="vg01",
            config_name="age-spoken-td",
            output_root_dir="/tmp/out",
        )
        assert cfg.interval_kind == "hdi"

    def test_model_label(self) -> None:
        cfg = ReportingConfiguration(
            model_name="vg01",
            config_name="age-spoken-td",
            output_root_dir="/tmp/out",
        )
        assert cfg.model_label == "vg01-age-spoken-td"

    def test_models_dir(self) -> None:
        cfg = ReportingConfiguration(
            model_name="vg01",
            config_name="cfg",
            output_root_dir="/tmp/out",
        )
        assert cfg.models_dir == os.path.join("/tmp/out", "models")

    def test_output_dir(self) -> None:
        cfg = ReportingConfiguration(
            model_name="vg01",
            config_name="cfg",
            output_root_dir="/tmp/out",
        )
        assert cfg.output_dir == os.path.join("/tmp/out", "models", "vg01-cfg")

    def test_custom_ci_prob_and_eti(self) -> None:
        cfg = ReportingConfiguration(
            model_name="m",
            config_name="c",
            output_root_dir="/tmp/out",
            ci_prob=0.95,
            interval_kind="eti",
        )
        assert cfg.ci_prob == 0.95
        assert cfg.interval_kind == "eti"

    def test_rejects_out_of_range_ci_prob(self) -> None:
        with pytest.raises(ValueError):
            ReportingConfiguration(
                model_name="m",
                config_name="c",
                output_root_dir="/tmp/out",
                ci_prob=1.5,
            )

    def test_rejects_unknown_interval_kind(self) -> None:
        with pytest.raises(ValueError):
            ReportingConfiguration(
                model_name="m",
                config_name="c",
                output_root_dir="/tmp/out",
                interval_kind="quantile",  # type: ignore[arg-type]
            )

    def test_hdi_field_is_gone(self) -> None:
        # The legacy ``hdi`` field was renamed to ``ci_prob`` (with an explicit
        # ``interval_kind``); constructing with it must now fail rather than silently
        # store a shadow attribute.
        with pytest.raises(TypeError):
            ReportingConfiguration(
                model_name="m",
                config_name="c",
                output_root_dir="/tmp/out",
                hdi=0.95,  # type: ignore[call-arg]
            )
