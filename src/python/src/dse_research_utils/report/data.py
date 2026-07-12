# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Report data-access helpers with graceful degradation before a fit exists.

Every DSE report chapter reads a fitted model's artefacts (summary CSVs, JSON gate
payloads, figures) the same way, and must render a visible "pending fit" placeholder —
not raise — when a model has not been fitted yet. This module supplies that shared
pattern so no report re-implements it.

Reports differ only in *where* a model's artefacts live, so :class:`ReportData` is
constructed with a resolver mapping ``(model_id, config)`` to the model's output
directory; the readers on top of it are identical across reports. The pure helpers
:func:`show_or_pending` and :func:`num` need no directory context and are module-level.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


def num(value: Any, fmt: str = "{:.0f}") -> str:
    """Format a value for prose, or an em dash when it is ``None``.

    Parameters
    ----------
    value : Any
        The value to format, or ``None`` when the underlying artefact is absent.
    fmt : str, default ``"{:.0f}"``
        A ``str.format`` template applied when ``value`` is not ``None``.

    Returns
    -------
    str
        The formatted value, or ``"—"`` when ``value`` is ``None``.
    """
    return "—" if value is None else fmt.format(value)


def show_or_pending(df: Any, what: str, *, hint: str = "its output artefacts are present") -> Any:
    """Return ``df`` if present, else a visible "pending fit" Markdown placeholder.

    The graceful-degradation primitive for report chapters: a chapter that references a
    not-yet-fitted model renders an explanatory callout rather than failing to build.

    Parameters
    ----------
    df : Any
        A loaded artefact (typically a :class:`pandas.DataFrame`), or ``None`` when it
        is not present on disk yet.
    what : str
        Human-readable name of the thing that will appear once fitted (e.g. "the ITT
        treatment-effect table").
    hint : str
        Trailing clause describing where the artefacts appear once the fit has run.

    Returns
    -------
    Any
        ``df`` unchanged when it is not ``None``; otherwise an
        :class:`IPython.display.Markdown` placeholder callout, or the placeholder as a
        plain Markdown string when IPython is not installed (so a non-notebook build
        degrades instead of raising).
    """
    if df is not None:
        return df
    message = (
        f"> **Pending fit** — {what} will appear here once the model has been "
        f"fitted and {hint}."
    )
    try:
        from IPython.display import Markdown
    except ModuleNotFoundError:
        return message
    return Markdown(message)


class ReportData:
    """Reader for a report's per-model artefacts, keyed off a directory resolver.

    Parameters
    ----------
    resolve_model_dir : callable
        Maps ``(model_id, config)`` to the directory holding that model's artefacts.
        ``config`` is the resolved configuration name (see ``default_config``); a
        resolver whose layout does not depend on the configuration may ignore it.
    default_config : str or None, default None
        Configuration used when a reader is called without an explicit ``config``.
    """

    def __init__(
        self,
        resolve_model_dir: Callable[[str, str | None], Path],
        *,
        default_config: str | None = None,
    ) -> None:
        self._resolve_model_dir = resolve_model_dir
        self.default_config = default_config

    def model_dir(self, model_id: str, config: str | None = None) -> Path:
        """Directory holding ``model_id``'s artefacts at the given (or default) config."""
        resolved = self.default_config if config is None else config
        return Path(self._resolve_model_dir(model_id, resolved))

    def load_summary(self, model_id: str, name: str, config: str | None = None) -> pd.DataFrame | None:
        """A summary / diagnostic CSV for a fitted model, or ``None`` if not present."""
        path = self.model_dir(model_id, config) / f"{name}.csv"
        return pd.read_csv(path) if path.exists() else None

    def load_json(self, model_id: str, name: str, config: str | None = None) -> Any | None:
        """A parsed JSON artefact for a model (e.g. the convergence gate), or ``None``."""
        path = self.model_dir(model_id, config) / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def fig(self, model_id: str, filename: str, config: str | None = None) -> str:
        """Path to a figure produced for a model, as a string (for embedding)."""
        return str(self.model_dir(model_id, config) / filename)

    def summary_value(
        self, model_id: str, name: str, column: str, config: str | None = None
    ) -> Any | None:
        """The first-row value of ``column`` in a summary, or ``None`` if unavailable."""
        df = self.load_summary(model_id, name, config)
        if df is None or df.empty or column not in df.columns:
            return None
        return df[column].iloc[0]

    def value_at(
        self,
        model_id: str,
        name: str,
        column: str,
        *,
        at: float,
        key: str = "age_months",
        config: str | None = None,
    ) -> Any | None:
        """Value of ``column`` in a summary at the row whose ``key`` is nearest ``at``.

        Returns ``None`` when the summary, the ``key`` column, or ``column`` is absent.
        """
        df = self.load_summary(model_id, name, config)
        if df is None or key not in df.columns or column not in df.columns:
            return None
        row = df.iloc[(df[key] - at).abs().argmin()]
        return row[column]
