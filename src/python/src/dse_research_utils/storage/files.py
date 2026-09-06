# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""File replacement without exposing partly written output to readers."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path


def atomic_write(path: str | os.PathLike[str], write_temporary: Callable[[Path], object]) -> None:
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

    Raises
    ------
    ValueError
        If the callback leaves a directory, symlink or other non-regular file.
    OSError
        If creating, writing or replacing a file fails. Callback exceptions also
        propagate. On failure before replacement, the old destination is intact
        and the temporary file is removed when possible.

    Notes
    -----
    The temporary file starts with owner-only read/write permissions on POSIX.
    Replacement retains the temporary file's permissions and metadata, including
    changes made by the callback (for example, by ``shutil.copy2``). Permissions
    of an existing destination are not inherited. Parent directories created by
    this function are not rolled back after failure.

    Replacement uses one ``os.replace`` on the destination filesystem. Concurrent
    readers see a complete old or new file; concurrent writers can overwrite one
    another. This does not lock a read-modify-write sequence, commit a bundle of
    files, or guarantee durability after power loss. The callback is trusted code
    and must not move or modify the destination itself.
    """
    # Do not resolve the final component: that would follow a destination symlink
    # and replace its target instead of the requested directory entry.
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
        os.replace(temporary, destination)
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
