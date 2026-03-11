# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from attr import dataclass


@dataclass
class ReportingConfiguration:
    model_name: str
    """
    Name of the model (e.g. "vg01").
    """
    config_name: str
    """
    Name of the configuration (e.g. "age-spoken-td").
    """
    output_root_dir: str
    """
    Root output directory.
    """
    hdi: float = 0.89
    """
    High density interval width for reporting (e.g., 0.95 for 95% HDI).
    """

    @property
    def models_dir(self) -> str:
        """
        Directory for saving model outputs (plots, data, report).
        """
        return os.path.join(self.output_root_dir, "models")

    @property
    def model_label(self) -> str:
        """
        Label for the model.
        """
        return f"{self.model_name}-{self.config_name}"

    @property
    def output_dir(self) -> str:
        """
        Directory for saving model outputs (plots, data).
        """
        return os.path.join(self.models_dir, self.model_label)
