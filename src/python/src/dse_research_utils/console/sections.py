# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Top-level structural output helpers: banners, section headers, timed sections.

These helpers print directly to the shared console. Callers that need to
build output without emitting it (e.g. for tests or composition) should use
the table builders in :mod:`dse_research_utils.console.tables` and
:mod:`dse_research_utils.console.summary` instead.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager

from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from dse_research_utils.console.console import get_console
from dse_research_utils.console.format import format_duration
from dse_research_utils.console.styles import (
    STYLE_BANNER_BORDER,
    STYLE_BANNER_SUBTITLE,
    STYLE_BANNER_TITLE,
    STYLE_CHECK,
    STYLE_SECTION,
    STYLE_SUBSECTION,
)


def banner(
    title: str,
    subtitle: str | None = None,
    *,
    style: str = STYLE_BANNER_TITLE,
    border_style: str = STYLE_BANNER_BORDER,
) -> None:
    """
    Print a framed banner with centered title and optional subtitle.

    Use once per pipeline run to mark the top of output.

    Parameters
    ----------
    title : str
        The headline text, centered inside the panel.
    subtitle : str or None
        Optional dim subtitle printed below the title.
    style : str
        Rich style for the title text.
    border_style : str
        Rich style for the panel border.
    """
    body = Text(title, style=style, justify="center")
    if subtitle:
        body.append("\n")
        body.append(subtitle, style=STYLE_BANNER_SUBTITLE)
    console = get_console()
    console.print()
    console.print(Panel(body, border_style=border_style, expand=True))


def section_header(title: str, *, style: str = STYLE_SECTION) -> None:
    """
    Print a rule-based section divider with a styled label.

    Parameters
    ----------
    title : str
        Label displayed inside the rule.
    style : str
        Rich style for the rule and label.
    """
    console = get_console()
    console.print()
    console.print(Rule(Text(title, style=style), style=style))


def subsection(title: str, *, style: str = STYLE_SUBSECTION) -> None:
    """
    Print a minor sub-section marker (no rule, just a styled line).

    Parameters
    ----------
    title : str
        The sub-section label.
    style : str
        Rich style for the label.
    """
    console = get_console()
    console.print()
    console.print(Text(title, style=style))


@contextmanager
def timed_section(
    title: str,
    *,
    timings: dict[str, float] | None = None,
    key: str | None = None,
    style: str = STYLE_SECTION,
) -> Iterator[None]:
    """
    Context manager that emits a section header, times the block, and reports
    the elapsed duration on exit.

    Parameters
    ----------
    title : str
        Section label; also used as the timings key when ``key`` is ``None``.
    timings : dict of str to float, optional
        If provided, the elapsed seconds are recorded under ``key or title``.
        Useful for later rendering with
        :func:`dse_research_utils.console.summary.pipeline_summary`.
    key : str or None
        Override the timings key. Defaults to ``title``.
    style : str
        Rich style passed through to :func:`section_header`.
    """
    section_header(title, style=style)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if timings is not None:
            timings[key or title] = elapsed
        get_console().print(Text(f"  ✓ {title} — {format_duration(elapsed)}", style=STYLE_CHECK))
