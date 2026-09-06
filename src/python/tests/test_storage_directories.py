# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import errno
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, suppress
from pathlib import Path
from threading import Barrier, Lock

import pytest

from dse_research_utils.storage.directories import DirectoryPromotion, promote_directory


def _tree(path, content):
    path.mkdir(parents=True)
    (path / "result.txt").write_text(content, encoding="utf-8")
    return path


def _content(path):
    return (path / "result.txt").read_text(encoding="utf-8")


def test_existing_destination_is_retained_and_tree_metadata_moves(tmp_path):
    source = _tree(tmp_path / "staging" / "new", "new result")
    source.chmod(0o750)
    old = _tree(tmp_path / "published" / "model", "old result")
    backup = tmp_path / "previous" / "model-old"
    backup.parent.mkdir()
    original_source = source.stat()
    original_destination = old.stat()

    result = promote_directory(source, old, backup=backup, lock=Lock())

    assert result == DirectoryPromotion(old, backup)
    assert _content(old) == "new result"
    assert _content(backup) == "old result"
    assert not source.exists()
    assert old.stat().st_ino == original_source.st_ino
    assert backup.stat().st_ino == original_destination.st_ino
    assert stat.S_IMODE(old.stat().st_mode) == stat.S_IMODE(original_source.st_mode)


def test_new_destination_does_not_create_a_backup(tmp_path, monkeypatch):
    source = _tree(tmp_path / "new", "new result")
    monkeypatch.chdir(tmp_path)
    result = promote_directory("new", "published", backup="unused", lock=nullcontext())
    assert result == DirectoryPromotion(tmp_path / "published", None)
    assert _content(result.destination) == "new result"
    assert not source.exists()
    assert not (tmp_path / "unused").exists()


def test_successful_promotion_stays_successful_when_caller_cleanup_fails(tmp_path, monkeypatch):
    source = _tree(tmp_path / "new", "new result")
    destination = _tree(tmp_path / "published", "old result")
    backup = tmp_path / "previous"

    def deny_cleanup(path):
        raise PermissionError("retention cleanup denied")

    monkeypatch.setattr(shutil, "rmtree", deny_cleanup)
    result = promote_directory(source, destination, backup=backup, lock=Lock())
    with pytest.raises(PermissionError, match="retention cleanup denied"):
        shutil.rmtree(result.backup)
    assert result.destination == destination
    assert _content(result.destination) == "new result"
    assert _content(result.backup) == "old result"


@pytest.mark.parametrize("failure", [PermissionError("destination in use"), KeyboardInterrupt()])
def test_failed_second_rename_restores_old_destination_and_preserves_primary(tmp_path, monkeypatch, failure):
    source = _tree(tmp_path / "new", "new result")
    destination = _tree(tmp_path / "published", "old result")
    backup = tmp_path / "previous"
    rename = os.rename
    calls = []
    lock = Lock()

    def fail_promotion(start, end):
        assert lock.locked()
        calls.append((start, end))
        if start == source:
            assert not destination.exists()
            assert _content(backup) == "old result"
            raise failure
        rename(start, end)

    monkeypatch.setattr("dse_research_utils.storage.directories.os.rename", fail_promotion)
    with pytest.raises(type(failure)) as caught:
        promote_directory(source, destination, backup=backup, lock=lock)
    assert caught.value is failure
    assert calls == [(destination, backup), (source, destination), (backup, destination)]
    assert _content(destination) == "old result"
    assert _content(source) == "new result"
    assert not backup.exists()
    assert not lock.locked()


def test_failed_first_rename_leaves_both_trees_in_place(tmp_path, monkeypatch):
    source = _tree(tmp_path / "new", "new result")
    destination = _tree(tmp_path / "published", "old result")
    backup = tmp_path / "previous"

    def fail_rename(start, end):
        assert (start, end) == (destination, backup)
        raise PermissionError("backup move denied")

    monkeypatch.setattr("dse_research_utils.storage.directories.os.rename", fail_rename)
    with pytest.raises(PermissionError, match="backup move denied"):
        promote_directory(source, destination, backup=backup, lock=Lock())
    assert _content(source) == "new result"
    assert _content(destination) == "old result"
    assert not backup.exists()


def test_failed_initial_publication_preserves_staged_output(tmp_path, monkeypatch):
    source = _tree(tmp_path / "new", "new result")
    destination = tmp_path / "published"
    backup = tmp_path / "previous"

    def fail_rename(start, end):
        assert (start, end) == (source, destination)
        raise OSError("initial publication denied")

    monkeypatch.setattr("dse_research_utils.storage.directories.os.rename", fail_rename)
    with pytest.raises(OSError, match="initial publication denied"):
        promote_directory(source, destination, backup=backup, lock=Lock())
    assert _content(source) == "new result"
    assert not destination.exists()
    assert not backup.exists()


def test_failed_rollback_retains_backup_and_does_not_hide_the_promotion_error(tmp_path, monkeypatch):
    source = _tree(tmp_path / "new", "new result")
    destination = _tree(tmp_path / "published", "old result")
    backup = tmp_path / "previous"
    primary = PermissionError("promotion denied")
    rollback = OSError("rollback denied")
    rename = os.rename

    def fail_promotion_and_rollback(start, end):
        if start == source:
            raise primary
        if start == backup:
            raise rollback
        rename(start, end)

    monkeypatch.setattr("dse_research_utils.storage.directories.os.rename", fail_promotion_and_rollback)
    with pytest.raises(PermissionError, match="promotion denied") as caught:
        promote_directory(source, destination, backup=backup, lock=Lock())
    assert caught.value is primary
    assert any("rollback denied" in note and str(backup) in note for note in primary.__notes__)
    assert not destination.exists()
    assert _content(backup) == "old result"
    assert _content(source) == "new result"


def test_rollback_does_not_overwrite_an_unexpected_destination(tmp_path, monkeypatch):
    source = _tree(tmp_path / "new", "new result")
    destination = _tree(tmp_path / "published", "old result")
    backup = tmp_path / "previous"
    rename = os.rename
    primary = OSError("promotion failed")

    def interfere_with_promotion(start, end):
        if start == source:
            _tree(destination, "unexpected writer")
            raise primary
        assert start != backup, "Rollback must not replace an occupied destination"
        rename(start, end)

    monkeypatch.setattr("dse_research_utils.storage.directories.os.rename", interfere_with_promotion)
    with pytest.raises(OSError) as caught:
        promote_directory(source, destination, backup=backup, lock=Lock())
    assert caught.value is primary
    assert any("FileExistsError" in note for note in primary.__notes__)
    assert _content(destination) == "unexpected writer"
    assert _content(backup) == "old result"
    assert _content(source) == "new result"


def test_rollback_does_not_move_a_replaced_backup(tmp_path, monkeypatch):
    source = _tree(tmp_path / "new", "new result")
    destination = _tree(tmp_path / "published", "old result")
    backup = tmp_path / "previous"
    moved_old = tmp_path / "moved-old"
    rename = os.rename

    def interfere_with_backup(start, end):
        if start == source:
            rename(backup, moved_old)
            _tree(backup, "different tree")
            raise OSError("promotion failed")
        rename(start, end)

    monkeypatch.setattr("dse_research_utils.storage.directories.os.rename", interfere_with_backup)
    with pytest.raises(OSError, match="promotion failed") as caught:
        promote_directory(source, destination, backup=backup, lock=Lock())
    assert any("no longer identifies" in note for note in caught.value.__notes__)
    assert not destination.exists()
    assert _content(backup) == "different tree"
    assert _content(moved_old) == "old result"


@pytest.mark.parametrize("replacement", [False, True])
def test_missing_backup_reports_when_the_original_destination_cannot_be_found(tmp_path, monkeypatch, replacement):
    source = _tree(tmp_path / "new", "new result")
    destination = _tree(tmp_path / "published", "old result")
    backup = tmp_path / "previous"
    moved_old = tmp_path / "moved-old"
    rename = os.rename
    primary = OSError("promotion failed")

    def remove_backup_before_failure(start, end):
        if start == source:
            rename(backup, moved_old)
            if replacement:
                _tree(destination, "unexpected writer")
            raise primary
        rename(start, end)

    monkeypatch.setattr("dse_research_utils.storage.directories.os.rename", remove_backup_before_failure)
    with pytest.raises(OSError, match="promotion failed") as caught:
        promote_directory(source, destination, backup=backup, lock=Lock())
    assert caught.value is primary
    assert any("Could not restore" in note and str(backup) in note for note in primary.__notes__)
    assert _content(moved_old) == "old result"
    assert _content(source) == "new result"
    assert destination.exists() == replacement
    if replacement:
        assert _content(destination) == "unexpected writer"


@pytest.mark.parametrize("stage", ["preflight", "promotion"])
def test_context_manager_cannot_suppress_a_failed_promotion(tmp_path, monkeypatch, stage):
    source = _tree(tmp_path / "new", "new result")
    destination = _tree(tmp_path / "published", "old result")
    backup = tmp_path / "previous"
    rename = os.rename
    if stage == "preflight":
        _tree(backup, "retained history")

    def fail_promotion(start, end):
        if start == source:
            raise PermissionError("promotion denied")
        rename(start, end)

    monkeypatch.setattr("dse_research_utils.storage.directories.os.rename", fail_promotion)
    with pytest.raises(OSError):
        promote_directory(source, destination, backup=backup, lock=suppress(OSError))
    assert _content(source) == "new result"
    assert _content(destination) == "old result"


@pytest.mark.parametrize("promotion_fails", [False, True])
def test_lock_exit_failure_preserves_the_primary_error_or_reports_completed_promotion(
    tmp_path, monkeypatch, promotion_fails
):
    source = _tree(tmp_path / "new", "new result")
    destination = _tree(tmp_path / "published", "old result")
    backup = tmp_path / "previous"
    rename = os.rename
    primary = PermissionError("promotion denied")
    exit_error = RuntimeError("lock exit failed")

    class FailingExit:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise exit_error

    def maybe_fail_promotion(start, end):
        if start == source and promotion_fails:
            raise primary
        rename(start, end)

    monkeypatch.setattr("dse_research_utils.storage.directories.os.rename", maybe_fail_promotion)
    expected = primary if promotion_fails else exit_error
    with pytest.raises(type(expected)) as caught:
        promote_directory(source, destination, backup=backup, lock=FailingExit())
    assert caught.value is expected
    if promotion_fails:
        assert any("Exiting the lock also failed" in note for note in primary.__notes__)
        assert _content(destination) == "old result"
        assert _content(source) == "new result"
        assert not backup.exists()
    else:
        assert any("promotion completed" in note and str(destination) in note for note in exit_error.__notes__)
        assert _content(destination) == "new result"
        assert _content(backup) == "old result"
        assert not source.exists()


@pytest.mark.parametrize("destination_exists", [False, True])
def test_backup_must_be_unused_even_for_a_new_destination(tmp_path, destination_exists):
    source = _tree(tmp_path / "new", "new result")
    destination = tmp_path / "published"
    if destination_exists:
        _tree(destination, "old result")
    backup = _tree(tmp_path / "previous", "retained history")
    with pytest.raises(FileExistsError, match="unused"):
        promote_directory(source, destination, backup=backup, lock=Lock())
    assert _content(source) == "new result"
    assert _content(backup) == "retained history"
    assert destination.exists() == destination_exists


@pytest.mark.parametrize("missing_parent", ["destination", "backup"])
def test_missing_parents_are_not_created(tmp_path, missing_parent):
    source = _tree(tmp_path / "new", "new result")
    destination = tmp_path / "published"
    backup = tmp_path / "previous"
    if missing_parent == "destination":
        destination = tmp_path / "missing" / "published"
    else:
        backup = tmp_path / "missing" / "previous"
    with pytest.raises(FileNotFoundError):
        promote_directory(source, destination, backup=backup, lock=Lock())
    assert not (tmp_path / "missing").exists()
    assert _content(source) == "new result"


@pytest.mark.parametrize("kind", ["staged", "destination", "backup"])
def test_regular_files_are_not_treated_as_directory_trees(tmp_path, kind):
    paths = {"staged": tmp_path / "new", "destination": tmp_path / "published", "backup": tmp_path / "previous"}
    if kind != "staged":
        _tree(paths["staged"], "new result")
    paths[kind].write_text("keep file", encoding="utf-8")
    with pytest.raises((NotADirectoryError, FileExistsError)):
        promote_directory(paths["staged"], paths["destination"], backup=paths["backup"], lock=Lock())
    assert paths[kind].read_text(encoding="utf-8") == "keep file"


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
@pytest.mark.parametrize("kind", ["staged", "destination", "backup"])
@pytest.mark.parametrize("dangling", [False, True])
def test_final_symlinks_are_rejected_without_touching_their_targets(tmp_path, kind, dangling):
    paths = {"staged": tmp_path / "new", "destination": tmp_path / "published", "backup": tmp_path / "previous"}
    target = tmp_path / "target"
    if not dangling:
        _tree(target, "target result")
    if kind != "staged":
        _tree(paths["staged"], "new result")
    paths[kind].symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        promote_directory(paths["staged"], paths["destination"], backup=paths["backup"], lock=Lock())
    assert paths[kind].is_symlink()
    assert target.exists() != dangling


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
def test_symlinked_ancestor_is_rejected_before_renaming(tmp_path):
    source = _tree(tmp_path / "real" / "new", "new result")
    (tmp_path / "alias").symlink_to("real", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked ancestor"):
        promote_directory(tmp_path / "alias" / "new", tmp_path / "published", backup=tmp_path / "previous", lock=Lock())
    assert _content(source) == "new result"


@pytest.mark.parametrize(
    ("staged", "destination", "backup"),
    [
        ("new", "new", "backup"),
        ("new", "new/published", "backup"),
        ("new", "published", "new/backup"),
        ("outer/new", "outer", "backup"),
        ("new", "published", "published/backup"),
        ("new", "published", "published"),
    ],
)
def test_equal_and_nested_paths_are_rejected_before_mutation(tmp_path, staged, destination, backup):
    source = _tree(tmp_path / staged, "new result")
    final = tmp_path / destination
    if final != source and not final.exists():
        final.mkdir(parents=True)
    with pytest.raises((ValueError, FileExistsError)):
        promote_directory(source, final, backup=tmp_path / backup, lock=Lock())
    assert _content(source) == "new result"


def test_parent_traversal_is_rejected_even_if_it_would_normalize_to_a_sibling(tmp_path):
    source = _tree(tmp_path / "new", "new result")
    with pytest.raises(ValueError, match="parent-traversal"):
        promote_directory(source, tmp_path / "new" / ".." / "published", backup=tmp_path / "previous", lock=Lock())
    assert _content(source) == "new result"


def test_cross_filesystem_preflight_does_not_move_the_old_destination(tmp_path, monkeypatch):
    source = _tree(tmp_path / "new", "new result")
    destination = _tree(tmp_path / "published", "old result")
    backup_parent = tmp_path / "different-filesystem"
    backup_parent.mkdir()
    original_stat = Path.lstat

    def other_device(path, *args, **kwargs):
        info = original_stat(path, *args, **kwargs)
        if path == backup_parent:
            fields = list(info)
            fields[stat.ST_DEV] += 1
            return os.stat_result(fields)
        return info

    def reject_rename(*args):
        pytest.fail("Known cross-filesystem paths must fail before any rename")

    monkeypatch.setattr(Path, "lstat", other_device)
    monkeypatch.setattr("dse_research_utils.storage.directories.os.rename", reject_rename)
    with pytest.raises(OSError) as caught:
        promote_directory(source, destination, backup=backup_parent / "old", lock=Lock())
    assert caught.value.errno == errno.EXDEV
    assert _content(source) == "new result"
    assert _content(destination) == "old result"


def test_same_directory_identity_under_distinct_paths_is_rejected(tmp_path, monkeypatch):
    source = _tree(tmp_path / "new", "new result")
    destination = _tree(tmp_path / "published", "old result")
    original_stat = Path.lstat
    source_info = original_stat(source)

    def aliased_identity(path, *args, **kwargs):
        info = original_stat(path, *args, **kwargs)
        if path == destination:
            fields = list(info)
            fields[stat.ST_DEV] = source_info.st_dev
            fields[stat.ST_INO] = source_info.st_ino
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(Path, "lstat", aliased_identity)
    with pytest.raises(ValueError, match="same directory"):
        promote_directory(source, destination, backup=tmp_path / "previous", lock=Lock())
    assert _content(source) == "new result"
    assert _content(destination) == "old result"


def test_preflight_checks_are_inside_the_supplied_exclusion_context(tmp_path, monkeypatch):
    source = _tree(tmp_path / "new", "new result")
    lock = Lock()
    original_stat = Path.lstat
    inspected = []

    def locked_stat(path, *args, **kwargs):
        assert lock.locked()
        inspected.append(path)
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", locked_stat)
        promote_directory(source, tmp_path / "published", backup=tmp_path / "previous", lock=lock)
    assert source in inspected
    assert tmp_path / "published" in inspected
    assert tmp_path / "previous" in inspected
    assert not lock.locked()


def test_concurrent_promotions_share_exclusion_and_retain_every_version(tmp_path):
    destination = _tree(tmp_path / "published", "original")
    sources = [_tree(tmp_path / name, name) for name in ("first", "second")]
    backups = [tmp_path / "previous-first", tmp_path / "previous-second"]
    lock = Lock()
    barrier = Barrier(2)

    def promote(index):
        barrier.wait(timeout=5)
        return promote_directory(sources[index], destination, backup=backups[index], lock=lock)

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(promote, range(2)))
    assert {result.backup for result in results} == set(backups)
    assert {_content(destination), *(_content(path) for path in backups)} == {"original", "first", "second"}
    assert not any(path.exists() for path in sources)


def test_concurrent_reuse_of_one_backup_fails_without_overwriting_history(tmp_path):
    destination = _tree(tmp_path / "published", "original")
    sources = [_tree(tmp_path / name, name) for name in ("first", "second")]
    backup = tmp_path / "previous"
    lock = Lock()
    barrier = Barrier(2)

    def promote(index):
        barrier.wait(timeout=5)
        try:
            promote_directory(sources[index], destination, backup=backup, lock=lock)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as workers:
        successes = list(workers.map(promote, range(2)))
    assert sorted(successes) == [False, True]
    winner = successes.index(True)
    assert _content(destination) == sources[winner].name
    assert _content(backup) == "original"
    assert _content(sources[1 - winner]) == sources[1 - winner].name


def test_lock_is_required_even_when_destination_is_absent(tmp_path):
    source = _tree(tmp_path / "new", "new result")
    with pytest.raises(TypeError):
        promote_directory(source, tmp_path / "published", backup=tmp_path / "previous")
    assert _content(source) == "new result"
