# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from dataclasses import dataclass

from dse_research_utils.statistics.intervals import IntervalKind


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
    ci_prob: float = 0.89
    """
    Coverage probability of the reported credible interval (e.g. 0.95 for a 95%
    interval). This is the interval *mass*; :attr:`interval_kind` selects whether it
    is summarised as an equal-tailed or a highest-density interval.
    """
    interval_kind: IntervalKind = "hdi"
    """
    Credible-interval convention used to summarise :attr:`ci_prob` mass — ``"eti"``
    (equal-tailed) or ``"hdi"`` (highest density). Reports read this back to keep the
    interval kind consistent between the tables, plots, and the diagnostics summary.
    """

    def __post_init__(self) -> None:
        if not 0.0 < self.ci_prob <= 1.0:
            raise ValueError(f"ci_prob must be in (0, 1], got {self.ci_prob!r}")
        if self.interval_kind not in ("eti", "hdi"):
            raise ValueError(f"interval_kind must be 'eti' or 'hdi', got {self.interval_kind!r}")

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
