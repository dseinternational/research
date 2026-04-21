# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Composed end-of-pipeline renderings: run headers, run summaries, pipeline
timing tables, artefact listings.

These helpers return :class:`rich.panel.Panel` or :class:`rich.table.Table`
objects so callers can print them via the shared console or compose them
further.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dse_research_utils.console.format import format_duration, format_value
from dse_research_utils.console.styles import (
    STYLE_BANNER_BORDER,
    STYLE_KEY_COLUMN,
    STYLE_MUTED,
    STYLE_SECTION,
    STYLE_TABLE_HEADER,
    STYLE_TABLE_TITLE,
)


def run_header_panel(
    title: str,
    details: Mapping[str, Any],
    *,
    border_style: str = STYLE_BANNER_BORDER,
) -> Panel:
    """
    Build the banner panel printed at the start of a run.

    Parameters
    ----------
    title : str
        Headline; rendered in bold at the top of the panel body.
    details : mapping of str to Any
        Key/value pairs rendered one per line as ``[dim]key:[/dim] value``.
        Values pass through
        :func:`dse_research_utils.console.format.format_value`.
    border_style : str
        Rich style for the panel border.

    Returns
    -------
    Panel
        A rich panel ready to print.
    """
    lines = [f"[bold]{title}[/bold]"]
    if details:
        lines.append("")
        for key, value in details.items():
            lines.append(f"[dim]{key}:[/dim] {format_value(value)}")
    return Panel("\n".join(lines), border_style=border_style, padding=(1, 2))


def run_summary_panel(
    title: str = "Done",
    artefacts_dir: Path | str | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
    border_style: str = STYLE_BANNER_BORDER,
) -> Panel:
    """
    Build the banner panel printed at the end of a run.

    Parameters
    ----------
    title : str
        Headline; rendered in bold green. Defaults to ``"Done"``.
    artefacts_dir : Path, str or None
        If provided, rendered as an absolute path after the title.
    extra : mapping of str to Any, optional
        Additional key/value lines rendered below the artefacts path.
    border_style : str
        Rich style for the panel border.

    Returns
    -------
    Panel
        A rich panel ready to print.
    """
    lines = [f"[bold green]{title}[/bold green]"]
    if artefacts_dir is not None:
        path = Path(artefacts_dir).resolve()
        lines.append("")
        lines.append(f"[dim]Artefacts:[/dim] {path}")
    if extra:
        for key, value in extra.items():
            lines.append(f"[dim]{key}:[/dim] {format_value(value)}")
    return Panel("\n".join(lines), border_style=border_style, padding=(1, 2))


def pipeline_summary(
    timings: Mapping[str, float],
    *,
    title: str = "Pipeline timings",
    total_label: str = "Total",
) -> Table:
    """
    Build a per-stage timing table with durations and percentage of total.

    Parameters
    ----------
    timings : mapping of str to float
        Stage name to elapsed seconds. Typically populated by
        :func:`dse_research_utils.console.sections.timed_section`.
    title : str
        Table title.
    total_label : str
        Label for the final total-row section.

    Returns
    -------
    Table
        A rich table ready to print. When ``timings`` is empty the table
        still renders with its header but no body rows.
    """
    table = Table(
        title=title,
        title_style=STYLE_SECTION,
        title_justify="left",
        show_header=True,
        header_style=STYLE_TABLE_HEADER,
        expand=False,
        pad_edge=False,
    )
    table.add_column("Stage", style=STYLE_KEY_COLUMN)
    table.add_column("Duration", justify="right")
    table.add_column("% total", justify="right", style=STYLE_MUTED)

    if not timings:
        return table

    total = sum(timings.values())
    for stage, seconds in timings.items():
        pct = (seconds / total * 100) if total > 0 else 0.0
        table.add_row(stage, format_duration(seconds), f"{pct:5.1f}%")

    table.add_section()
    table.add_row(
        Text(total_label, style="bold"),
        Text(format_duration(total), style="bold"),
        "",
    )
    return table


def artefact_summary(
    output_dir: Path | str,
    *,
    title: str = "Artefacts",
) -> Table:
    """
    Walk ``output_dir`` recursively and render a table of file counts by
    extension.

    Parameters
    ----------
    output_dir : Path or str
        Directory to walk. If the directory does not exist, a header-only
        table is returned.
    title : str
        Table title.

    Returns
    -------
    Table
        A rich table ready to print.
    """
    path = Path(output_dir)
    table = Table(
        title=title,
        title_style=STYLE_TABLE_TITLE,
        title_justify="left",
        show_header=True,
        header_style=STYLE_TABLE_HEADER,
        expand=False,
        pad_edge=False,
    )
    table.add_column("Type", style=STYLE_KEY_COLUMN)
    table.add_column("Count", justify="right")

    if not path.exists():
        return table

    counts: dict[str, int] = {}
    for entry in path.rglob("*"):
        if entry.is_file():
            ext = entry.suffix.lower() or "(no ext)"
            counts[ext] = counts.get(ext, 0) + 1

    for ext, count in sorted(counts.items()):
        table.add_row(ext, str(count))
    return table
