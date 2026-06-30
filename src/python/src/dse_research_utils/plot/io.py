# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Figure save / notebook-display helpers."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(
    filename: str,
    output_dir: str | os.PathLike,
    *,
    format: str = "png",
    dpi: int = 300,
    bbox_inches: str = "tight",
):
    """Save the current matplotlib figure to ``output_dir / filename``."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    figure_path = out / filename
    print(f"saving figure to {figure_path}")
    plt.savefig(figure_path, format=format, dpi=dpi, bbox_inches=bbox_inches)


def display_image(filename: str, output_dir: str | os.PathLike, width: int = 600):
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
