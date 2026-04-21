# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from io import StringIO

import pytest
from rich.console import Console

from dse_research_utils.console import console as console_module


@pytest.fixture
def captured_console():
    """Install a StringIO-backed console for the duration of a test."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        highlight=False,
        width=240,
        force_terminal=False,
        record=True,
        color_system=None,
    )
    console_module.set_console(console)
    try:
        yield console
    finally:
        console_module.set_console(None)
