# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configurable output-root resolver shared by the consuming repositories.

Model traces and reporting-quality artefacts are large (a reporting-config
``trace.nc`` exceeds 10 GB), so ephemeral scratch-disk VM runs need to redirect
them off the repo disk without breaking the established relative layout, report
rendering, uploads, comparisons, or scripts that read previous runs. Each repo
declares an :class:`OutputRoot` with its own environment-variable name and
repo-local default; the resolution *policy* — CLI override > environment
variable > default, resolved at call time — lives here so it cannot drift
between repositories.
"""

from __future__ import annotations

import os
from pathlib import Path


class OutputRoot:
    """A call-time-resolved output root with a fixed precedence.

    Precedence: an explicit override set via :meth:`set` (typically the parsed
    ``--output-dir``) > the configured environment variable > the repo-local
    default. Paths are resolved with ``Path.expanduser().resolve()``.

    Parameters
    ----------
    env_var : str
        Name of the environment variable carrying the root (e.g.
        ``"DSE_VOCAB_GROWTH_OUTPUT_DIR"``).
    default : path-like
        The repo-local default root (e.g. ``<repo>/output``), used when neither
        an override nor the environment variable is set. Not normalised, so the
        default's exact spelling is preserved for path comparisons.
    """

    def __init__(self, env_var: str, default: str | os.PathLike[str]) -> None:
        self.env_var = env_var
        self.default = Path(default)
        self._override: Path | None = None

    def set(self, path: str | os.PathLike[str] | None) -> Path:
        """Set (or clear) the process-wide override — highest precedence.

        Pass the parsed ``--output-dir`` value, or ``None`` to clear the
        override and fall back to the environment variable / default. Returns
        the resolved output root. Call once, early in a command, before any
        output path is resolved.
        """
        self._override = Path(path).expanduser().resolve() if path else None
        return self.resolve()

    def resolve(self) -> Path:
        """Resolve the output root at call time (see class docstring)."""
        if self._override is not None:
            return self._override
        env_value = os.environ.get(self.env_var)
        if env_value:
            return Path(env_value).expanduser().resolve()
        return self.default

    def is_overridden(self) -> bool:
        """True when the resolved root differs from the repo-local default."""
        return self.resolve() != self.default

    def describe(self) -> str:
        """One-line description of the resolved root and its source, for run logs."""
        if self._override is not None:
            source = "--output-dir"
        elif os.environ.get(self.env_var):
            source = self.env_var
        else:
            source = "repo-local default"
        return f"{self.resolve()}  (source: {source})"
