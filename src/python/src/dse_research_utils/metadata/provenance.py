# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Raw provenance facts without project-specific manifest or publication rules."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal

_HASH_CHUNK_BYTES = 1024 * 1024
_GIT_REPOSITORY_ENV = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
    }
)


@dataclass(frozen=True)
class GitSnapshot:
    """Working-tree facts, with unavailable fields represented by ``None``."""

    state: Literal["available", "partial", "unavailable"]
    """Whether all, some, or none of the requested Git facts were obtained."""

    commit: str | None = None
    """Full HEAD object ID; absent for an unborn branch or an unavailable query."""

    branch: str | None = None
    """Branch name; absent for detached HEAD or an unavailable query."""

    detached: bool | None = None
    """Whether HEAD is detached from a branch."""

    unborn: bool | None = None
    """Whether the current branch has no commit yet."""

    dirty: bool | None = None
    """Any staged, unstaged, unmerged or untracked change reported by Git."""

    untracked: bool | None = None
    """Whether Git reported untracked file/directory entries in this worktree."""

    error: (
        Literal["invalid_path", "git_unavailable", "timeout", "os_error", "git_error", "incomplete_output"] | None
    ) = None
    """Failure category; excludes stderr, filesystem paths and remote URLs."""

    returncode: int | None = None
    """Git's nonzero exit code, when a command ran but failed."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a file's binary contents with bounded memory.

    Parameters
    ----------
    path : str or path-like
        File to read. Open and read errors propagate to the caller.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256, without a prefix. Filenames and other
        metadata are not included. Callers own multi-file ordering and scope.

    Notes
    -----
    The file is not locked. Callers requiring an immutable input must arrange
    that separately; this hashes the bytes read, not a transactional snapshot.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(distributions: Iterable[str] | Mapping[str, str]) -> dict[str, str | None]:
    """Read installed distribution versions without importing their packages.

    Parameters
    ----------
    distributions : iterable of str or mapping of str to str
        Distribution names, or a mapping from caller-chosen labels to names.
        There is no default package set.

    Returns
    -------
    dict of str to str or None
        Versions under the requested names or labels. Missing, unreadable or
        malformed metadata gives ``None``. Results are not cached, so changes
        to installed metadata are visible on the next call.
    """
    if isinstance(distributions, str | bytes):
        raise TypeError("distributions must be an iterable of names or a label-to-name mapping, not one string")
    items = distributions.items() if isinstance(distributions, Mapping) else ((name, name) for name in distributions)
    versions: dict[str, str | None] = {}
    for label, distribution in items:
        try:
            versions[label] = metadata.version(distribution)
        except metadata.PackageNotFoundError, OSError, ValueError:
            versions[label] = None
    return versions


def _snapshot_from_status(output: bytes) -> GitSnapshot:
    commit = branch = None
    detached = unborn = None
    dirty = untracked = False
    valid_status = output.endswith(b"\0")
    records = iter(output.split(b"\0"))
    for record in records:
        if record.startswith(b"# branch.oid "):
            value = record.removeprefix(b"# branch.oid ")
            if value == b"(initial)":
                unborn = True
            elif len(value) in (40, 64) and all(char in b"0123456789abcdef" for char in value):
                commit, unborn = value.decode("ascii"), False
        elif record.startswith(b"# branch.head "):
            value = record.removeprefix(b"# branch.head ")
            if value == b"(detached)":
                detached = True
            elif value:
                try:
                    branch = value.decode("utf-8")
                except UnicodeDecodeError:
                    pass
                else:
                    detached = False
        elif record.startswith((b"1 ", b"u ")):
            dirty = True
        elif record.startswith(b"2 "):
            dirty = True
            # In -z output, a rename/copy record is followed by the old path.
            # That path may itself resemble a status record or a branch header.
            if not next(records, b""):
                valid_status = False
        elif record.startswith(b"? "):
            dirty = untracked = True
        elif record and not record.startswith((b"# ", b"! ")):
            valid_status = False
    complete = detached is not None and unborn is not None and valid_status
    state: Literal["available", "partial", "unavailable"] = "available" if complete else "partial"
    if not output:
        state = "unavailable"
    return GitSnapshot(
        state=state,
        commit=commit,
        branch=branch,
        detached=detached,
        unborn=unborn,
        dirty=dirty if valid_status else None,
        untracked=untracked if valid_status else None,
        error=None if complete else "incomplete_output",
    )


def git_snapshot(repo_root: str | os.PathLike[str], *, timeout: float = 5.0) -> GitSnapshot:
    """Read Git working-tree facts from an explicit directory in one bounded call.

    Parameters
    ----------
    repo_root : str or path-like
        Directory within the intended working tree. Git's normal parent search
        applies, but inherited repository/index overrides are removed.
    timeout : float, default 5.0
        Positive finite timeout in seconds for the single Git status command.

    Returns
    -------
    GitSnapshot
        Facts and a failure category when needed. Unborn branches and detached
        HEAD are valid states, not failures. A nonexistent directory, missing
        Git, timeout or command failure is unavailable. A successful command
        with incomplete status output retains its known facts as partial.
        Bare repositories are unavailable because working-tree status fails.

    Notes
    -----
    Dirty includes untracked entries, but excludes ignored files. Git optional
    locks and the filesystem-monitor hook are disabled. This function does not
    lock the repository or promise a transactional snapshot during concurrent
    changes. It returns no local paths, remote URLs or raw command errors.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    root = Path(repo_root)
    if not root.is_dir():
        return GitSnapshot(state="unavailable", error="invalid_path")
    environ = {key: value for key, value in os.environ.items() if key not in _GIT_REPOSITORY_ENV}
    try:
        result = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "status",
                "--porcelain=v2",
                "--branch",
                "--untracked-files=normal",
                "--ignore-submodules=none",
                "-z",
            ],
            cwd=root,
            env=environ,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError:
        return GitSnapshot(state="unavailable", error="git_unavailable")
    except subprocess.TimeoutExpired:
        return GitSnapshot(state="unavailable", error="timeout")
    except OSError:
        return GitSnapshot(state="unavailable", error="os_error")
    if result.returncode:
        return GitSnapshot(state="unavailable", error="git_error", returncode=result.returncode)
    return _snapshot_from_status(result.stdout)
