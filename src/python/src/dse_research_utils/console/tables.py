# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Table builders for console output.

Functions in this module return a :class:`rich.table.Table` so that callers
can compose or defer printing. Use ``print_*`` helpers when immediate output
is wanted.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd
from rich.table import Table

from dse_research_utils.console.console import get_console
from dse_research_utils.console.format import format_value
from dse_research_utils.console.styles import (
    STYLE_KEY_COLUMN,
    STYLE_MUTED,
    STYLE_TABLE_HEADER,
    STYLE_TABLE_TITLE,
    STYLE_VALUE_COLUMN,
)


def key_value_table(
    pairs: Mapping[str, Any] | Iterable[tuple[str, Any]],
    *,
    title: str | None = None,
    key_header: str = "Key",
    value_header: str = "Value",
    precision: int = 4,
) -> Table:
    """
    Build a two-column key/value table.

    Parameters
    ----------
    pairs : mapping or iterable of (str, Any)
        The items to render, preserving iteration order.
    title : str or None
        Optional title rendered above the header.
    key_header : str
        Header label for the key column.
    value_header : str
        Header label for the value column.
    precision : int
        Forwarded to :func:`dse_research_utils.console.format.format_value`.

    Returns
    -------
    Table
        A rich table ready to print.
    """
    items = list(pairs.items() if isinstance(pairs, Mapping) else pairs)

    table = Table(
        title=title,
        title_style=STYLE_TABLE_TITLE,
        title_justify="left",
        show_header=True,
        header_style=STYLE_TABLE_HEADER,
        expand=False,
        pad_edge=False,
    )
    table.add_column(key_header, style=STYLE_KEY_COLUMN, no_wrap=True)
    table.add_column(value_header, style=STYLE_VALUE_COLUMN)
    for key, value in items:
        table.add_row(str(key), format_value(value, precision=precision))
    return table


def print_key_value_table(
    pairs: Mapping[str, Any] | Iterable[tuple[str, Any]],
    *,
    title: str | None = None,
    key_header: str = "Key",
    value_header: str = "Value",
    precision: int = 4,
) -> None:
    """Build and print a key/value table via the shared console."""
    get_console().print(
        key_value_table(
            pairs,
            title=title,
            key_header=key_header,
            value_header=value_header,
            precision=precision,
        )
    )


def metrics_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    columns: Sequence[str] | None = None,
    precision: int = 4,
) -> Table:
    """
    Build a multi-column metrics table from a sequence of row mappings.

    The first column is left-aligned and no-wrap (typically a metric name);
    subsequent columns are right-aligned (typically numeric).

    Parameters
    ----------
    rows : sequence of mappings
        Each mapping is one table row. When ``columns`` is omitted the keys
        from the first row define the column order.
    title : str or None
        Optional table title.
    columns : sequence of str, optional
        Column order. Defaults to the keys of the first row.
    precision : int
        Forwarded to :func:`dse_research_utils.console.format.format_value`.

    Returns
    -------
    Table
        A rich table ready to print. When ``rows`` is empty, a title-only
        table is returned.
    """
    table = Table(
        title=title,
        title_style=STYLE_TABLE_TITLE,
        title_justify="left",
        show_header=True,
        header_style=STYLE_TABLE_HEADER,
        show_lines=False,
        expand=False,
        pad_edge=False,
    )
    if not rows:
        return table

    cols = list(columns) if columns is not None else list(rows[0].keys())
    for i, col in enumerate(cols):
        justify = "left" if i == 0 else "right"
        table.add_column(col, justify=justify, no_wrap=(i == 0))

    for row in rows:
        rendered = [format_value(row.get(col), precision=precision) for col in cols]
        table.add_row(*rendered)
    return table


def params_table(
    params: Mapping[str, Any],
    *,
    title: str | None = None,
    precision: int = 6,
) -> Table:
    """
    Render a parameter dict as a two-column ``param`` / ``value`` table.

    Thin wrapper over :func:`metrics_table` with a fixed column layout;
    ``precision`` defaults to ``6`` to keep hyperparameter values readable.
    """
    rows = [{"param": k, "value": v} for k, v in params.items()]
    return metrics_table(
        rows,
        title=title,
        columns=["param", "value"],
        precision=precision,
    )


def dataframe_table(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    columns: Sequence[str] | None = None,
    max_rows: int | None = 50,
    truncation: str = "head_tail",
    rank_column: str | None = None,
    show_index: bool = True,
    precision: int = 4,
) -> Table:
    """
    Render a pandas ``DataFrame`` as a rich ``Table``.

    Parameters
    ----------
    df : pandas.DataFrame
        The frame to render. Row order is preserved.
    title : str or None
        Optional table title.
    columns : sequence of str, optional
        Subset (and order) of columns to render. Defaults to all columns.
    max_rows : int or None
        If set, truncate long frames to this many rows. ``None`` renders
        every row.
    truncation : str
        ``"head"`` keeps the first ``max_rows`` and adds a trailing ellipsis
        row (matches LRP ``ranked_dataframe_table`` behaviour); ``"head_tail"``
        splits into head + tail with a caption noting the truncation.
    rank_column : str or None
        When set, prepend a 1-based rank column with this header (and suppress
        the index column).
    show_index : bool
        When ``True`` and ``rank_column`` is ``None``, render the frame index
        as the first column.
    precision : int
        Forwarded to :func:`dse_research_utils.console.format.format_value`.

    Returns
    -------
    Table
        A rich table ready to print.

    Raises
    ------
    ValueError
        If ``truncation`` is not ``"head"`` or ``"head_tail"``.
    """
    if truncation not in {"head", "head_tail"}:
        raise ValueError(f"truncation must be 'head' or 'head_tail', got {truncation!r}")

    table = Table(
        title=title,
        title_style=STYLE_TABLE_TITLE,
        title_justify="left",
        show_header=True,
        header_style=STYLE_TABLE_HEADER,
        show_lines=False,
        expand=False,
        pad_edge=False,
    )
    if df.empty:
        return table

    cols = list(columns) if columns is not None else list(df.columns)
    truncated = False
    hidden = 0
    display_df = df
    if max_rows is not None and len(df) > max_rows:
        truncated = True
        hidden = len(df) - max_rows
        if truncation == "head_tail":
            half = max(1, max_rows // 2)
            display_df = pd.concat([df.head(half), df.tail(max_rows - half)])
        else:
            display_df = df.head(max_rows)

    n_prefix_cols = 0
    if rank_column is not None:
        table.add_column(rank_column, justify="right", style=STYLE_MUTED, no_wrap=True)
        n_prefix_cols = 1
    elif show_index:
        index_name = str(df.index.name) if df.index.name is not None else ""
        table.add_column(index_name, style=STYLE_KEY_COLUMN, no_wrap=True)
        n_prefix_cols = 1
    for i, col in enumerate(cols):
        justify = "left" if i == 0 and n_prefix_cols == 0 else "right"
        table.add_column(str(col), justify=justify, no_wrap=(i == 0 and n_prefix_cols == 0))

    for rank, (idx, row) in enumerate(display_df.iterrows(), start=1):
        cells = [format_value(row[c], precision=precision) for c in cols]
        if rank_column is not None:
            cells = [str(rank), *cells]
        elif show_index:
            cells = [str(idx), *cells]
        table.add_row(*cells)

    if truncated:
        if truncation == "head":
            filler = ["…"] * (len(cols) + n_prefix_cols)
            table.add_row(*filler)
            table.caption = f"(+{hidden} more row{'s' if hidden != 1 else ''})"
        else:
            table.caption = f"… showing {max_rows} of {len(df)} rows (head/tail) …"

    return table
