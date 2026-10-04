# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""File replacement without exposing partly written output to readers."""

from __future__ import annotations

import os
import stat
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

# Windows refuses a replacement while another handle has the destination open
# without delete sharing (Python's own ``open`` does not grant it) or while
# another replacement of it is in progress. Both clear once that handle closes.
_RETRYABLE_WINERRORS = frozenset({5, 32})  # ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION
_REPLACE_RETRY_DELAYS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def _replace(source: Path, destination: Path) -> None:
    """Replace ``destination``, retrying briefly on transient Windows refusals."""
    for delay in (*_REPLACE_RETRY_DELAYS, None):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if delay is None or getattr(exc, "winerror", None) not in _RETRYABLE_WINERRORS:
                raise
            time.sleep(delay)


def default_file_mode(directory: str | os.PathLike[str]) -> int:
    """Read the permission bits of a new ordinary file without changing umask.

    Parameters
    ----------
    directory
        Existing directory in which to create and remove an empty probe file.

    Returns
    -------
    int
        Mode bits from an exclusive file creation with requested mode 0o666.
        On POSIX these reflect the process umask and directory creation rules.
        They do not describe or preserve access-control entries.

    Raises
    ------
    OSError
        If creating, inspecting or removing the probe fails.
    """
    probe = Path(directory) / f".mode-{uuid4().hex}"
    descriptor = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        return stat.S_IMODE(os.fstat(descriptor).st_mode)
    finally:
        try:
            os.close(descriptor)
        finally:
            probe.unlink()


def atomic_write(
    path: str | os.PathLike[str],
    write_temporary: Callable[[Path], object],
    *,
    mode: int | Literal["default"] | None = None,
) -> None:
    """Write one file through a sibling temporary file and replace its destination.

    Parameters
    ----------
    path
        Destination file. Missing parent directories are created. An existing
        destination symlink is replaced, without changing the file it points to.
    write_temporary
        Callback receiving the absolute path of an existing empty temporary file
        in the destination directory. All destination suffixes are preserved so
        writers can infer formats such as ``.csv.gz`` or ``.npy``. The callback
        must write this file, close its handles before returning, and leave a
        regular file at this path. Its return value is ignored.
    mode
        Optional permission bits applied after the callback. ``"default"``
        uses :func:`default_file_mode` in the destination directory. None
        retains the callback's permissions, including the owner-only initial
        permissions when the callback has not changed them. An explicit mode
        overrides copied permissions; it does not preserve access-control
        entries or inherit an existing destination's mode.

    Raises
    ------
    ValueError
        If mode is outside the permission-bit range, or the callback leaves a
        directory, symlink or other non-regular file.
    TypeError
        If mode is neither an integer, "default", nor None.
    OSError
        If creating, writing or replacing a file fails. Callback exceptions also
        propagate. On failure before replacement, the old destination is intact
        and the temporary file is removed when possible.

    Notes
    -----
    The temporary file starts with owner-only read/write permissions on POSIX.
    With mode=None, replacement retains the temporary file's permissions and
    metadata, including changes made by the callback (for example, by
    ``shutil.copy2``). Permissions of an existing destination are not inherited.
    Parent directories created by this function are not rolled back after failure.

    Replacement is an ``os.replace`` on the destination filesystem. Concurrent
    readers see a complete old or new file; concurrent writers can overwrite one
    another. On Windows, a replacement refused because another handle holds the
    destination open is retried for up to about one second before the error
    propagates. This does not lock a read-modify-write sequence, commit a bundle of
    files, or guarantee durability after power loss. The callback is trusted code
    and must not move or modify the destination itself.
    """
    # Do not resolve the final component: that would follow a destination symlink
    # and replace its target instead of the requested directory entry.
    if mode is not None and mode != "default":
        if isinstance(mode, bool) or not isinstance(mode, int):
            raise TypeError("mode must be permission bits, 'default', or None")
        if not 0 <= mode <= 0o7777:
            raise ValueError("mode must lie between 0 and 0o7777")
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=".tmp-",
        suffix="".join(destination.suffixes),
    )
    temporary = Path(temporary_name)
    try:
        # Closing this handle before the callback also permits replacement on
        # Windows, where an open handle can prevent a writer from opening it.
        os.close(fd)
        write_temporary(temporary)
        if not stat.S_ISREG(temporary.lstat().st_mode):
            raise ValueError("The writer must leave a regular file at the temporary path.")
        if mode is not None:
            temporary.chmod(default_file_mode(destination.parent) if mode == "default" else mode)
        _replace(temporary, destination)
    except BaseException as exc:
        try:
            # Reject a directory without traversing or deleting its contents.
            if temporary.is_dir() and not temporary.is_symlink():
                temporary.rmdir()
            else:
                temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            exc.add_note(f"Could not remove temporary path {temporary}: {cleanup_error}")
        raise
