# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Disk-space preflight for scripts that write large artefacts.

Model traces exceed 10 GB at reporting configurations, so a full volume should
fail fast at the start of a run rather than after a multi-hour sample.
"""

from __future__ import annotations

import os
import shutil

from dse_research_utils.console.console import get_console


def _normalise(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def free_space_gb(path: str | os.PathLike[str]) -> float:
    """Free space (GiB) on the volume backing ``path``.

    Walks up to the nearest existing parent if ``path`` does not exist yet, so it
    works for an output directory that is about to be created.
    """
    target = _normalise(path)
    while not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    return shutil.disk_usage(target).free / (1024**3)


def preflight_disk(
    min_gb: float,
    path: str | os.PathLike[str],
    *,
    label: str = "operation",
    output_root: str | os.PathLike[str] | None = None,
) -> float:
    """Report free disk space and raise ``RuntimeError`` if below ``min_gb`` (GiB).

    Call at the start of any script that writes large artefacts so a full volume
    fails fast rather than after a multi-hour sample. Returns the free space in
    GiB when the check passes.

    Parameters
    ----------
    min_gb : float
        Minimum free space required, in GiB.
    path : path-like
        The location the artefacts will be written to (an existing parent is used
        when it does not exist yet).
    label : str, optional
        Human-readable name of the operation, used in the log line and the error.
    output_root : path-like, optional
        The caller's resolved output root. When given and ``path`` resolves to it,
        a ``[output]`` line names the root so redirected runs are obvious in job
        logs (surfaced only in that case, so the line cannot disagree with the
        ``[disk]`` target when a caller passes a subdirectory).
    """
    target = _normalise(path)
    free = free_space_gb(target)
    drive = os.path.splitdrive(target)[0] or target
    # markup=False: the [output] / [disk] log prefixes must render literally,
    # not parse as rich markup tags.
    if output_root is not None and target == _normalise(output_root):
        get_console().print(f"[output] resolved output root: {os.fspath(output_root)}", markup=False)
    get_console().print(f"[disk] {free:.1f} GiB free on {drive} (need >= {min_gb:.0f} GiB for {label})", markup=False)
    if free < min_gb:
        raise RuntimeError(
            f"Insufficient disk space for {label}: {free:.1f} GiB free on {drive}, "
            f"need >= {min_gb:.0f} GiB. Free space and retry."
        )
    return free
