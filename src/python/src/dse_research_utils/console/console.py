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
    """
    Switch a text stream that is not UTF-8 to ``errors="replace"``.

    Windows still defaults ``sys.stdout`` to the legacy cp1252 code page, and
    rich raises rather than degrading characters the stream cannot encode. The
    helpers here emit ``✓``, ``—`` and ``…`` at runtime, so on such a console a
    single completed section aborts the caller's pipeline with
    ``UnicodeEncodeError`` — after the work, not before it. Relaxing the error
    handler substitutes ``?`` for those characters instead.

    Streams that are already UTF-8 — every Linux and macOS default, and Windows
    under ``PYTHONUTF8=1`` — keep their strict handler, so nothing that renders
    correctly today changes. Streams that cannot be reconfigured at all
    (``StringIO``, pytest's capture objects) are left alone.

    PEP 686 makes UTF-8 mode the default in Python 3.15, which retires this.

    Parameters
    ----------
    stream : TextIO or None
        The stream to inspect; typically ``sys.stdout``.
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

    Creating it also relaxes the error handler on a ``sys.stdout`` that cannot
    encode the characters these helpers emit — see
    :func:`_relax_encoding_errors` — so that a legacy Windows code page
    degrades the output rather than aborting the run.

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

    Intended for tests — passing a ``Console`` bound to ``StringIO`` captures
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
