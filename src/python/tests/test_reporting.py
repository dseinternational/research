# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from dse_research_utils.statistics.models.reporting import ReportingConfiguration


class TestReportingConfiguration:
    def test_default_hdi(self) -> None:
        cfg = ReportingConfiguration(
            model_name="vg01",
            config_name="age-spoken-td",
            output_root_dir="/tmp/out",
        )
        assert cfg.hdi == 0.89

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

    def test_custom_hdi(self) -> None:
        cfg = ReportingConfiguration(
            model_name="m",
            config_name="c",
            output_root_dir="/tmp/out",
            hdi=0.95,
        )
        assert cfg.hdi == 0.95
