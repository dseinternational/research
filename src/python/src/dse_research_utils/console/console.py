# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared rich ``Console`` singleton and thin print wrappers.

All ``dse_research_utils.console`` helpers route their output through
:func:`get_console` so that rendering is consistent (highlight off, no soft
wrap surprises) and tests can replace the singleton via
:func:`set_console`.
"""

import codecs
import contextlib
import sys
from typing import TextIO

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from dse_research_utils.console.styles import STYLE_SECTION

_console: Console | None = None


def _relax_encoding_errors(stream: TextIO | None) -> None:
    """Use replacement characters for text streams that are not UTF-8.

    The console helpers emit symbols that some output encodings cannot represent.
    Setting ``errors="replace"`` avoids UnicodeEncodeError on those streams.
    UTF-8 streams keep their existing error handler. Streams that cannot be
    reconfigured, such as StringIO and some test capture streams, are unchanged.

    Parameters
    ----------
    stream : TextIO or None
        The stream to inspect, typically ``sys.stdout``.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        canonical = codecs.lookup(getattr(stream, "encoding", "") or "").name
    except LookupError:
        canonical = ""
    if canonical == "utf-8":
        return
    with contextlib.suppress(OSError, ValueError):  # stream detached or already closed
        reconfigure(errors="replace")


def get_console() -> Console:
    """
    Return the module-level rich ``Console`` singleton.

    The console is created lazily the first time this is called, with
    ``highlight=False`` so that numeric and path values render uniformly
    rather than receiving rich's default auto-highlighting.

    It also sets replacement handling on non-UTF-8 output streams through
    :func:`_relax_encoding_errors`.

    Returns
    -------
    Console
        The shared console instance used by every helper in this
        subpackage.
    """
    global _console
    if _console is None:
        _relax_encoding_errors(sys.stdout)
        _console = Console(highlight=False)
    return _console


def set_console(console: Console | None) -> None:
    """
    Replace the module-level console singleton.

    For tests, a ``Console`` bound to ``StringIO`` captures
    rendered output for assertions. Pass ``None`` to reset to the default.

    Parameters
    ----------
    console : Console or None
        The console to install, or ``None`` to clear so the next
        :func:`get_console` call recreates the default.
    """
    global _console
    _console = console


def print_table(table: Table) -> None:
    """Print a rich ``Table`` via the shared console."""
    get_console().print(table)


def print_panel(panel: Panel) -> None:
    """Print a rich ``Panel`` via the shared console."""
    get_console().print(panel)


def print_rule(text: str, *, style: str = STYLE_SECTION) -> None:
    """
    Print a horizontal rule with styled label text.

    Parameters
    ----------
    text : str
        Label shown inside the rule.
    style : str
        Rich style applied to both the label and the rule itself.
    """
    get_console().print(Rule(Text(text, style=style), style=style))
