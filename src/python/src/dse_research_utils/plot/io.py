# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Figure save / notebook-display helpers.

The house figure-artifact policy: always write ``<name>.png`` (the artifact
report templates reference — raster keeps output pages quick to browse); also
write an ``<name>.svg`` sibling, optionally capped in size (very large SVGs are
what make report viewers slow); optionally write ``<name>.csv`` of the data
behind the plot. Both matplotlib figures and ``arviz_plots`` ``PlotCollection``
objects route through here so a single change propagates everywhere.

Consistent house style (fonts, colours, grid, DPI) comes from
``set_matplotlib_default_style``; these helpers only standardise *saving*.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from dse_research_utils.console.console import get_console
from dse_research_utils.plot.styles import DPI_FILE

SVG_MAX_BYTES = 2 * 1024 * 1024
"""Suggested SVG size cap: comfortably above a typical vector figure and well
below the multi-megabyte grids best kept raster-only. Pass it as
``svg_max_bytes`` to opt a call site into the cap; the default is uncapped."""


def save_figure(
    filename: str,
    output_dir: str | os.PathLike,
    *,
    format: str = "png",
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> None:
    """Save the current matplotlib figure to ``output_dir / filename``."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    figure_path = out / filename
    get_console().print(f"saving figure to {figure_path}")
    plt.savefig(figure_path, format=format, dpi=dpi, bbox_inches=bbox_inches)


def display_image(filename: str, output_dir: str | os.PathLike, width: int = 600) -> Any:
    """Display an image from ``output_dir / filename`` in a notebook.

    Requires IPython (install the ``notebook`` extra:
    ``pip install dse-research-utils[notebook]``).
    """
    try:
        from IPython.display import Image, display
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "display_image requires IPython; install the 'notebook' extra "
            "(pip install dse-research-utils[notebook])."
        ) from exc

    return display(Image(filename=str(Path(output_dir) / filename), width=width))


def _stem(name: str) -> str:
    """Return ``name`` without a trailing ``.png``/``.svg`` extension.

    Callers may pass either a historical ``"trace_plot.png"`` or a bare
    ``"trace_plot"`` stem.
    """
    for ext in (".png", ".svg"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _write_svg_sibling(save: Any, base: str, svg_max_bytes: int | None = None) -> None:
    """Write ``base + '.svg'`` via ``save(path)``, dropping it if over the cap.

    ``save`` is a one-arg callable (``fig.savefig`` or ``pc.savefig``) so this
    works for both matplotlib figures and ``arviz_plots`` collections. With
    ``svg_max_bytes=None`` the SVG is kept regardless of size. Guarded so an
    SVG-backend hiccup never costs the (already-written) PNG.
    """
    svg = base + ".svg"
    try:
        save(svg)
        if svg_max_bytes is not None and os.path.getsize(svg) > svg_max_bytes:
            os.remove(svg)
    except Exception as exc:  # pragma: no cover - defensive
        get_console().print(f"[yellow]SVG sibling for {os.path.basename(base)} skipped: {exc}[/yellow]")
        if os.path.exists(svg):
            with contextlib.suppress(OSError):
                os.remove(svg)


def save_plot_data(output_dir: str | os.PathLike, name: str, data: Any, *, index: bool = False) -> str:
    """Write the data behind a plot as ``<name>.csv``, returning the path.

    ``name`` may be a stem or carry a ``.png``/``.svg`` extension; ``data`` is
    coerced to a :class:`pandas.DataFrame` if it is not one already.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{_stem(name)}.csv")
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    df.to_csv(path, index=index)
    return path


def save_styled_figure(
    output_dir: str | os.PathLike,
    name: str,
    *,
    fig: Any | None = None,
    dpi: float = DPI_FILE,
    bbox_inches: str | None = "tight",
    close: bool = True,
    svg: bool = True,
    svg_max_bytes: int | None = None,
    data: Any | None = None,
) -> str:
    """Save a matplotlib figure as PNG (+ SVG sibling, + optional data CSV).

    Parameters
    ----------
    output_dir : path-like
        Directory to write into (created if needed).
    name : str
        Output stem; may carry a ``.png``/``.svg`` extension, which is stripped.
    fig : Figure, optional
        Figure to save (default: the current figure).
    dpi : float, optional
        PNG resolution (default :data:`~dse_research_utils.plot.styles.DPI_FILE`).
    bbox_inches : str or None, optional
        Passed to ``savefig`` (default ``"tight"``; pass ``None`` for the
        matplotlib default cropping).
    close : bool, optional
        Close the figure after saving (default ``True``). Pass ``False`` when the
        caller returns the figure for further use.
    svg : bool, optional
        Also write an SVG sibling (default ``True``).
    svg_max_bytes : int or None, optional
        Drop the SVG sibling if it exceeds this size; ``None`` (default) keeps it
        regardless. :data:`SVG_MAX_BYTES` is the suggested cap.
    data : optional
        When given, also written as ``<name>.csv`` via :func:`save_plot_data`.

    Returns
    -------
    str
        The PNG path.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.gcf() if fig is None else fig
    base = os.path.join(output_dir, _stem(name))
    png = base + ".png"
    fig.savefig(png, dpi=dpi, bbox_inches=bbox_inches)
    if svg:
        _write_svg_sibling(
            lambda p: fig.savefig(p, format="svg", bbox_inches=bbox_inches),
            base,
            svg_max_bytes,
        )
    if data is not None:
        save_plot_data(output_dir, name, data)
    if close:
        plt.close(fig)
    return png


def _pc_figure(pc: Any) -> Any:
    """Best-effort matplotlib ``Figure`` behind an ``arviz_plots`` collection."""
    try:
        return pc.viz["figure"].item()
    except Exception:  # pragma: no cover - defensive
        try:
            return plt.gcf()
        except Exception:
            return None


def save_plotcollection(
    pc: Any,
    output_dir: str | os.PathLike,
    name: str,
    *,
    suptitle: str | None = None,
    dpi: float = DPI_FILE,
    svg: bool = True,
    svg_max_bytes: int | None = None,
    close: bool = True,
    data: Any | None = None,
) -> None:
    """Save an ``arviz_plots`` ``PlotCollection`` as PNG (+ SVG sibling).

    Optionally adds a figure-level ``suptitle`` (ArviZ plots render untitled) and
    emits the SVG through ``pc.savefig`` so the collection lays out correctly.
    With ``close=True`` (the default) every open pyplot figure is closed after
    saving; pass ``close=False`` when the caller returns ``pc`` for display.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, _stem(name))
    if suptitle:
        fig = _pc_figure(pc)
        if fig is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                fig.suptitle(suptitle)
    pc.savefig(base + ".png", dpi=dpi)
    if svg:
        _write_svg_sibling(lambda p: pc.savefig(p), base, svg_max_bytes)
    if data is not None:
        save_plot_data(output_dir, name, data)
    if close:
        plt.close("all")
