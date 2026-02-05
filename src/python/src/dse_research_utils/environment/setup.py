# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from dse_research_utils.plot.styles import set_matplotlib_default_style
from dse_research_utils.environment.info import report_environment_info

def init():
    """Initialize the environment."""
    set_matplotlib_default_style()


def init_script():
    """Initialize the script environment."""
    init()


def init_workbook():
    """Initialize the workbook environment."""
    init()
    report_environment_info()
