# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Promote completed directory trees with explicit exclusion and retained backups."""

from __future__ import annotations

import errno
import os
import stat
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DirectoryPromotion:
    """The paths left by a successful directory promotion."""

    destination: Path
    """Absolute path now containing the former staged tree."""

    backup: Path | None
    """Retained former destination, or None when no destination existed."""


@dataclass(frozen=True)
class _DirectoryPath:
    path: Path
    info: os.stat_result | None
    parents: tuple[os.stat_result, ...]


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _directory_path(value: str | os.PathLike[str], label: str) -> _DirectoryPath:
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"{label} must not contain parent-traversal components")
    path = path.absolute()
    if path == path.parent:
        raise ValueError(f"{label} must not be a filesystem root")
    parents: list[os.stat_result] = []
    for parent in path.parents:
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} must not have a symlinked ancestor: {parent}")
        if not stat.S_ISDIR(info.st_mode):
            raise NotADirectoryError(errno.ENOTDIR, f"{label} needs existing directory parents", str(parent))
        parents.append(info)
    try:
        info = path.lstat()
    except FileNotFoundError:
        if label == "staged":
            raise
        info = None
    if info is not None:
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} must not be a symlink: {path}")
        if label == "backup":
            raise FileExistsError(errno.EEXIST, "backup must be an unused path", str(path))
        if not stat.S_ISDIR(info.st_mode):
            raise NotADirectoryError(errno.ENOTDIR, f"{label} must be a directory", str(path))
    return _DirectoryPath(path, info, tuple(parents))


def _validate_relationships(paths: tuple[_DirectoryPath, ...]) -> None:
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left.path.is_relative_to(right.path) or right.path.is_relative_to(left.path):
                raise ValueError("staged, destination and backup must be distinct, non-nested paths")
            if left.info is not None and right.info is not None and _identity(left.info) == _identity(right.info):
                raise ValueError("staged and destination must not refer to the same directory")
            # Parent identities also detect aliases through mounted directory
            # trees, which Path.resolve cannot identify as symlinks.
            if _identity(left.parents[0]) == _identity(right.parents[0]) and os.path.normcase(
                left.path.name
            ) == os.path.normcase(right.path.name):
                raise ValueError("Promotion paths must not alias the same directory entry")
            if (left.info is not None and _identity(left.info) in {_identity(info) for info in right.parents}) or (
                right.info is not None and _identity(right.info) in {_identity(info) for info in left.parents}
            ):
                raise ValueError("Promotion paths must not be nested through directory aliases")
    devices = {item.parents[0].st_dev for item in paths}
    devices.update(item.info.st_dev for item in paths if item.info is not None)
    if len(devices) != 1:
        raise OSError(errno.EXDEV, "Promotion paths and their parents must be on the same filesystem")


def _require_absent(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise FileExistsError(errno.EEXIST, "Refusing to overwrite a path created during promotion", str(path))


def _restore_backup(original: _DirectoryPath, backup: Path, error: BaseException) -> None:
    """Attempt recovery without replacing the primary promotion exception."""
    try:
        try:
            backup_info = backup.lstat()
        except FileNotFoundError as missing_backup:
            try:
                destination_info = original.path.lstat()
            except FileNotFoundError as missing_destination:
                raise RuntimeError(
                    "Neither the backup nor the original destination is present"
                ) from missing_destination
            if (
                original.info is None
                or not stat.S_ISDIR(destination_info.st_mode)
                or _identity(destination_info) != _identity(original.info)
            ):
                raise RuntimeError(
                    "The backup is missing and the destination no longer identifies the original directory"
                ) from missing_backup
            return
        if (
            original.info is None
            or not stat.S_ISDIR(backup_info.st_mode)
            or _identity(backup_info) != _identity(original.info)
        ):
            raise RuntimeError("The backup no longer identifies the original destination; refusing to move it")
        _require_absent(original.path)
        os.rename(backup, original.path)
    except BaseException as rollback_error:
        error.add_note(
            f"Could not restore the previous directory from {backup} to {original.path}: "
            f"{type(rollback_error).__name__}: {rollback_error}. Preserve these paths for recovery."
        )


def _promote_locked(
    staged: str | os.PathLike[str], destination: str | os.PathLike[str], backup: str | os.PathLike[str]
) -> DirectoryPromotion:
    source = _directory_path(staged, "staged")
    final = _directory_path(destination, "destination")
    previous = _directory_path(backup, "backup")
    _validate_relationships((source, final, previous))
    try:
        if final.info is not None:
            _require_absent(previous.path)
            os.rename(final.path, previous.path)
        _require_absent(final.path)
        os.rename(source.path, final.path)
    except BaseException as error:
        if final.info is not None:
            _restore_backup(final, previous.path, error)
        error.add_note(
            f"Directory promotion failed with staged={source.path}, destination={final.path}, backup={previous.path}."
        )
        raise
    return DirectoryPromotion(final.path, previous.path if final.info is not None else None)


def promote_directory(
    staged: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    backup: str | os.PathLike[str],
    lock: AbstractContextManager[object],
) -> DirectoryPromotion:
    """Move a completed tree into place and retain any previous destination.

    Parameters
    ----------
    staged
        Existing directory containing the completed output. The caller validates
        completeness and closes any handles that would prevent a rename.
    destination
        Directory to create or replace. All parent directories must exist.
    backup
        Explicit unused path for the previous destination, with an existing
        parent. It is validated even when no destination exists. No backup is
        created in that case, and the result's ``backup`` is None.
    lock
        Context manager held during validation and all renames. The caller
        chooses a lock shared by every writer of these paths, their parents and
        the staged tree. A thread lock covers only cooperating threads; use an
        appropriate process lock when multiple processes write. A null context
        is appropriate only when the caller already provides exclusive access.
        Errors are raised even if this context manager tries to suppress them.

    Returns
    -------
    DirectoryPromotion
        Absolute destination and retained backup paths. No backup cleanup is
        attempted. The caller chooses retention and any recursive deletion after
        this operation succeeds; a later cleanup failure is a separate outcome.

    Raises
    ------
    ValueError
        Paths are aliased, nested, symlinked, contain parent traversal, or name
        a filesystem root. Symlinked ancestors are also rejected; callers may
        deliberately resolve trusted parent paths before calling.
    OSError
        A parent or staged directory is missing, a path has an unsafe type, the
        backup is already occupied, filesystems differ, or a rename fails.
        Rename failures preserve the original exception. If moving the staged
        tree fails after the old destination was backed up, recovery is attempted
        only while the destination remains absent and the backup still identifies
        the original directory. A rollback failure is attached as an exception
        note, and recoverable directories are left for caller inspection.

    Notes
    -----
    Replacing an existing directory takes two renames. Readers can observe a gap
    between moving the old destination away and promoting the new tree. This is
    not an atomic directory exchange, a reader lock, or a durability guarantee
    after power loss. No directories are created, traversed for cleanup, or
    deleted. Permissions and tree contents are retained by the renames.

    Filesystem checks run while holding the supplied lock, before the first
    rename. A known cross-filesystem move is rejected before the old destination
    changes; later operating-system failures still require rollback. The absent
    target checks do not protect against actors that ignore the exclusion
    contract or alter mount points during the operation.

    Context-manager errors also propagate. If exiting the lock fails after a
    successful promotion, its exception receives a note identifying the completed
    destination and retained backup. If exiting fails while handling a promotion
    error, the promotion error remains primary and the lock error is noted.
    """
    failure: BaseException | None = None
    result: DirectoryPromotion | None = None
    try:
        with lock:
            try:
                result = _promote_locked(staged, destination, backup)
            except BaseException as error:
                failure = error
                raise
    except BaseException as context_error:
        if failure is not None and context_error is not failure:
            failure.add_note(f"Exiting the lock also failed: {type(context_error).__name__}: {context_error}")
            raise failure from context_error
        if failure is None and result is not None:
            context_error.add_note(
                f"Directory promotion completed before the lock exit failed: "
                f"destination={result.destination}, backup={result.backup}."
            )
        raise
    if failure is not None:
        raise failure
    assert result is not None
    return result
