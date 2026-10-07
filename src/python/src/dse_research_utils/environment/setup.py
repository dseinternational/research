# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from dse_research_utils.environment.info import report_environment_info
from dse_research_utils.plot.styles import set_matplotlib_default_style


def init() -> None:
    """Apply the default matplotlib style."""
    set_matplotlib_default_style()


def init_script() -> None:
    """Apply the default matplotlib style for a script without reporting system details."""
    init()


def init_workbook() -> None:
    """Apply the default matplotlib style and print environment information for a notebook."""
    init()
    report_environment_info()
