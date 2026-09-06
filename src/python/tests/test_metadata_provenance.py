# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import hashlib
import os
import shutil
import subprocess
from importlib import metadata

import pytest

from dse_research_utils.metadata import provenance
from dse_research_utils.metadata.provenance import git_snapshot, package_versions, sha256_file


def _git(root, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key not in provenance._GIT_REPOSITORY_ENV},
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("Git is needed for real-repository fixtures")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Provenance test")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "config", "core.autocrlf", "false")
    return root


def _commit(root):
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "test: create fixture")
    return _git(root, "rev-parse", "HEAD")


@pytest.mark.parametrize("content", [b"", bytes(range(256)), b"\0binary\xff\r\n" * 200_000])
def test_sha256_hashes_exact_binary_contents(tmp_path, content):
    path = tmp_path / "input.bin"
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()
    renamed = path.rename(tmp_path / "another-name.bin")
    assert sha256_file(str(renamed)) == hashlib.sha256(content).hexdigest()


def test_sha256_reads_bounded_chunks(monkeypatch):
    class Source:
        def __enter__(self):
            self.reads = []
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size):
            assert 0 < size <= 1024 * 1024
            self.reads.append(size)
            return b"part" if len(self.reads) < 3 else b""

    source = Source()
    monkeypatch.setattr(provenance, "open", lambda *_args: source, raising=False)
    assert sha256_file("unused") == hashlib.sha256(b"partpart").hexdigest()
    assert len(source.reads) == 3


def test_sha256_propagates_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "absent")
    with pytest.raises(OSError):
        sha256_file(tmp_path)


def test_package_versions_reads_installed_distribution_without_display_fallback():
    assert package_versions(["pytest"]) == {"pytest": metadata.version("pytest")}
    assert package_versions(["dse-no-such-distribution-for-test"]) == {"dse-no-such-distribution-for-test": None}


@pytest.mark.parametrize("distribution", ["pytest", b"pytest"])
def test_package_versions_rejects_a_single_string(distribution):
    with pytest.raises(TypeError, match="not one string"):
        package_versions(distribution)


def test_package_versions_accepts_label_mapping_and_does_not_cache(monkeypatch):
    available = {"some-distribution": "1.2.3"}
    monkeypatch.setattr(provenance.metadata, "version", available.__getitem__)
    assert package_versions({"display label": "some-distribution"}) == {"display label": "1.2.3"}
    available["some-distribution"] = "2.0"
    assert package_versions(iter(["some-distribution"])) == {"some-distribution": "2.0"}
    assert package_versions([]) == {}


@pytest.mark.parametrize("error", [metadata.PackageNotFoundError("missing"), OSError("unreadable"), ValueError("bad")])
def test_unavailable_package_metadata_is_none(monkeypatch, error):
    def unavailable(_name):
        raise error

    monkeypatch.setattr(provenance.metadata, "version", unavailable)
    assert package_versions(["package"]) == {"package": None}


def test_unborn_branch_is_available(repo):
    snapshot = git_snapshot(repo)
    assert snapshot.state == "available"
    assert snapshot.commit is None
    assert snapshot.branch == "main"
    assert snapshot.unborn is True
    assert snapshot.detached is False
    assert snapshot.dirty is False
    assert snapshot.untracked is False
    assert snapshot.error is None


def test_clean_committed_and_detached_states(repo):
    commit = _commit(repo)
    snapshot = git_snapshot(repo)
    assert snapshot.state == "available"
    assert snapshot.commit == commit
    assert snapshot.branch == "main"
    assert snapshot.unborn is False
    assert snapshot.detached is False
    assert snapshot.dirty is False
    _git(repo, "checkout", "--detach", "HEAD")
    detached = git_snapshot(repo)
    assert detached.state == "available"
    assert detached.commit == commit
    assert detached.branch is None
    assert detached.detached is True
    assert detached.unborn is False


@pytest.mark.parametrize("staged", [False, True])
def test_dirty_includes_tracked_edits(repo, staged):
    _commit(repo)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    if staged:
        _git(repo, "add", "tracked.txt")
    snapshot = git_snapshot(repo)
    assert snapshot.state == "available"
    assert snapshot.dirty is True
    assert snapshot.untracked is False


def test_untracked_names_are_not_parsed_as_status_headers(repo):
    _commit(repo)
    name = "# branch.head (detached)"
    if os.name != "nt":
        name += "\n? odd"
    (repo / name).write_bytes(b"untracked")
    snapshot = git_snapshot(repo)
    assert snapshot.branch == "main"
    assert snapshot.detached is False
    assert snapshot.dirty is True
    assert snapshot.untracked is True


def test_rename_old_path_is_not_parsed_as_a_status_header(repo):
    original = "# branch.head (detached)"
    (repo / original).write_bytes(b"tracked")
    _git(repo, "add", original)
    _git(repo, "commit", "-m", "test: name resembles status header")
    _git(repo, "mv", original, "renamed")
    snapshot = git_snapshot(repo)
    assert snapshot.state == "available"
    assert snapshot.branch == "main"
    assert snapshot.detached is False
    assert snapshot.dirty is True
    assert snapshot.untracked is False


def test_ignored_files_do_not_make_snapshot_dirty(repo):
    _commit(repo)
    (repo / ".git" / "info" / "exclude").write_text("ignored\n", encoding="utf-8")
    (repo / "ignored").write_bytes(b"ignored")
    assert git_snapshot(repo).dirty is False


def test_linked_worktree_uses_its_own_head_and_status(repo, tmp_path):
    expected = _commit(repo)
    worktree = tmp_path / "linked"
    _git(repo, "worktree", "add", "--detach", str(worktree))
    (worktree / "untracked.txt").write_bytes(b"only here")
    snapshot = git_snapshot(worktree)
    assert snapshot.state == "available"
    assert snapshot.commit == expected
    assert snapshot.detached is True
    assert snapshot.dirty is snapshot.untracked is True
    assert git_snapshot(repo).dirty is False


def test_unmerged_files_make_snapshot_dirty(repo):
    _commit(repo)
    _git(repo, "checkout", "-b", "other")
    (repo / "tracked.txt").write_text("other\n", encoding="utf-8")
    _git(repo, "commit", "-am", "test: edit on other branch")
    _git(repo, "checkout", "main")
    (repo / "tracked.txt").write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "test: conflicting edit")
    with pytest.raises(subprocess.CalledProcessError):
        _git(repo, "merge", "other")
    snapshot = git_snapshot(repo)
    assert snapshot.state == "available"
    assert snapshot.branch == "main"
    assert snapshot.dirty is True
    assert snapshot.untracked is False


def test_explicit_root_ignores_inherited_repository_and_index_overrides(repo, tmp_path, monkeypatch):
    expected = _commit(repo)
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "--initial-branch=elsewhere")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    monkeypatch.setenv("GIT_INDEX_FILE", str(other / ".git" / "index"))
    monkeypatch.setenv("GIT_COMMON_DIR", str(other / ".git"))
    snapshot = git_snapshot(repo)
    assert snapshot.state == "available"
    assert snapshot.commit == expected
    assert snapshot.branch == "main"
    assert snapshot.dirty is False


def test_snapshot_does_not_refresh_index_or_enable_fsmonitor(repo, monkeypatch):
    _commit(repo)
    real_run = provenance.subprocess.run
    calls = []

    def capture(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return real_run(arguments, **kwargs)

    monkeypatch.setattr(provenance.subprocess, "run", capture)
    assert git_snapshot(repo, timeout=0.7).state == "available"
    assert len(calls) == 1
    arguments, kwargs = calls[0]
    assert "--no-optional-locks" in arguments
    assert "core.fsmonitor=false" in arguments
    assert kwargs["timeout"] == 0.7
    assert kwargs["cwd"] == repo


def test_non_repository_and_bare_repository_are_unavailable(tmp_path, repo):
    snapshot = git_snapshot(tmp_path)
    assert snapshot.state == "unavailable"
    assert snapshot.error == "git_error"
    assert snapshot.returncode != 0
    assert snapshot.dirty is None
    bare = tmp_path / "bare.git"
    _git(repo, "init", "--bare", str(bare))
    assert git_snapshot(bare).error == "git_error"


@pytest.mark.parametrize("kind", ["absent", "file"])
def test_invalid_path_is_structured_unavailable(tmp_path, kind):
    path = tmp_path / kind
    if kind == "file":
        path.write_bytes(b"not a directory")
    snapshot = git_snapshot(path)
    assert snapshot.state == "unavailable"
    assert snapshot.error == "invalid_path"
    assert snapshot.returncode is None


@pytest.mark.parametrize(
    ("exception", "reason"),
    [
        (FileNotFoundError("sensitive path"), "git_unavailable"),
        (PermissionError("sensitive path"), "os_error"),
        (subprocess.TimeoutExpired("sensitive command", 1), "timeout"),
    ],
)
def test_process_failures_are_structured_and_do_not_disclose_errors(tmp_path, monkeypatch, exception, reason):
    def fail(*_args, **_kwargs):
        raise exception

    monkeypatch.setattr(provenance.subprocess, "run", fail)
    snapshot = git_snapshot(tmp_path)
    assert snapshot.state == "unavailable"
    assert snapshot.error == reason
    assert snapshot.commit is snapshot.branch is snapshot.dirty is None
    assert "sensitive" not in repr(snapshot)


def test_git_failure_does_not_disclose_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 128, b"", b"private paths and URLs"),
    )
    snapshot = git_snapshot(tmp_path)
    assert snapshot.error == "git_error"
    assert snapshot.returncode == 128
    assert "private" not in repr(snapshot)


def test_partial_output_retains_available_facts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"# branch.head main\0? file\0", b""),
    )
    snapshot = git_snapshot(tmp_path)
    assert snapshot.state == "partial"
    assert snapshot.error == "incomplete_output"
    assert snapshot.branch == "main"
    assert snapshot.detached is False
    assert snapshot.commit is None
    assert snapshot.unborn is None
    assert snapshot.dirty is snapshot.untracked is True


def test_unrecognised_status_record_does_not_report_clean(tmp_path, monkeypatch):
    output = b"# branch.head main\0# branch.oid (initial)\0unrecognised status\0"
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, b""),
    )
    snapshot = git_snapshot(tmp_path)
    assert snapshot.state == "partial"
    assert snapshot.branch == "main"
    assert snapshot.unborn is True
    assert snapshot.dirty is snapshot.untracked is None


@pytest.mark.parametrize("output", [b"", b"# branch.head main\0# branch.oid (initial)"])
def test_truncated_status_cannot_establish_complete_or_clean_snapshot(tmp_path, monkeypatch, output):
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, b""),
    )
    snapshot = git_snapshot(tmp_path)
    assert snapshot.state != "available"
    assert snapshot.error == "incomplete_output"
    assert snapshot.dirty is snapshot.untracked is None


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_timeout_must_be_finite_and_positive(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        git_snapshot(tmp_path, timeout=timeout)
