# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Value and duration formatters for console rendering.

:func:`format_value` is the single source of truth for how cell values in
key/value tables, metric tables and dataframe tables render. Callers that
need different behaviour can format values upstream and pass strings
through.
"""

import math
from pathlib import Path
from typing import Any

import pandas as pd

EM_DASH: str = "—"


def format_value(value: Any, *, precision: int = 4) -> str:
    """
    Format a scalar (or simple collection) for console display.

    Rules applied in order:

    - ``None`` and pandas missing sentinels (``pd.NA``, ``pd.NaT``, NaN)
      render as an em-dash.
    - ``bool`` renders as ``"yes"`` / ``"no"`` (handled before ``int`` because
      ``bool`` is a subclass of ``int``).
    - ``float`` uses scientific notation when ``|v| >= 1e4`` or
      ``0 < |v| < 1e-3``; otherwise ``g`` format with ``precision``.
    - ``int`` uses thousands separators (``1,234,567``).
    - :class:`pathlib.Path` renders as ``str(path)``.
    - ``list`` / ``tuple`` recurse and join with ``, ``.
    - Anything else falls back to ``str(value)``.

    Parameters
    ----------
    value : Any
        The value to format.
    precision : int
        Significant digits for floats (default ``4``).

    Returns
    -------
    str
        The formatted string.
    """
    if value is None:
        return EM_DASH
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return EM_DASH
        abs_v = abs(value)
        if abs_v >= 1e4 or (abs_v != 0 and abs_v < 1e-3):
            return f"{value:.{precision}e}"
        return f"{value:.{precision}g}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return ", ".join(format_value(v, precision=precision) for v in value)
    try:
        if pd.isna(value):
            return EM_DASH
    except TypeError, ValueError:
        pass
    return str(value)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.

    Units adapt to magnitude:

    - ``< 1 s`` → ``"N ms"``
    - ``< 60 s`` → ``"N.NN s"``
    - ``< 1 h`` → ``"Nm NN.Ns"``
    - otherwise → ``"Nh NNm NN.Ns"``

    Parameters
    ----------
    seconds : float
        Elapsed time in seconds.

    Returns
    -------
    str
        The formatted duration.
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes):02d}m {secs:04.1f}s"
