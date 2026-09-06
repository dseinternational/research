# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import gzip
import json
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pandas as pd
import pytest

from dse_research_utils.storage.files import atomic_write


def test_json_writer_keeps_encoding_and_old_file_until_callback_finishes(tmp_path):
    destination = tmp_path / "summary.json"
    destination.write_text("old content", encoding="utf-8")
    payload = {"label": "café", "values": [None, 0.5]}

    def writer(temporary):
        assert temporary.is_absolute()
        assert temporary.parent == destination.parent
        assert temporary.read_bytes() == b""
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        assert destination.read_text() == "old content"

    atomic_write(destination, writer)
    assert destination.read_text(encoding="utf-8") == json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_creates_parents_and_accepts_bare_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    atomic_write("summary.json", lambda temporary: temporary.write_text("{}"))
    atomic_write("nested/results/data.csv", lambda temporary: temporary.write_text("value\n1\n"))
    assert (tmp_path / "summary.json").read_text() == "{}"
    assert (tmp_path / "nested/results/data.csv").read_text() == "value\n1\n"


def test_pandas_infers_compression_from_preserved_suffixes(tmp_path):
    destination = tmp_path / "data.csv.gz"
    frame = pd.DataFrame({"value": [1, 2]})
    atomic_write(destination, lambda temporary: frame.to_csv(temporary, index=False))
    with gzip.open(destination, "rt") as handle:
        assert handle.read() == "value\n1\n2\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_numpy_writes_the_staged_file_without_appending_another_suffix(tmp_path):
    destination = tmp_path / "draws.npy"
    values = np.arange(12).reshape(3, 4)
    atomic_write(destination, lambda temporary: np.save(temporary, values))
    np.testing.assert_array_equal(np.load(destination), values)
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("error", [RuntimeError("writer failed"), KeyboardInterrupt()])
def test_partial_write_is_removed_and_original_error_propagates(tmp_path, existing, error):
    destination = tmp_path / "result.json"
    if existing:
        destination.write_bytes(b"old result")

    def writer(temporary):
        temporary.write_bytes(b"partial")
        raise error

    with pytest.raises(type(error)) as caught:
        atomic_write(destination, writer)
    assert caught.value is error
    assert destination.exists() == existing
    if existing:
        assert destination.read_bytes() == b"old result"
    assert list(tmp_path.iterdir()) == ([destination] if existing else [])


def test_replace_failure_preserves_destination_and_removes_temporary(tmp_path, monkeypatch):
    destination = tmp_path / "result.json"
    destination.write_bytes(b"old result")

    def fail_replace(source, target):
        assert source.read_bytes() == b"new result"
        assert target == destination
        raise PermissionError("destination is in use")

    monkeypatch.setattr("dse_research_utils.storage.files.os.replace", fail_replace)
    with pytest.raises(PermissionError, match="destination is in use"):
        atomic_write(destination, lambda temporary: temporary.write_bytes(b"new result"))
    assert destination.read_bytes() == b"old result"
    assert list(tmp_path.iterdir()) == [destination]


def test_cleanup_failure_does_not_hide_writer_error(tmp_path, monkeypatch):
    staged = []
    error = ValueError("bad output")

    def writer(temporary):
        staged.append(temporary)
        raise error

    def fail_unlink(*args, **kwargs):
        raise PermissionError("cleanup denied")

    with monkeypatch.context() as patch:
        patch.setattr("pathlib.Path.unlink", fail_unlink)
        with pytest.raises(ValueError, match="bad output") as caught:
            atomic_write(tmp_path / "result.json", writer)
    assert caught.value is error
    assert "cleanup denied" in error.__notes__[0]
    assert not (tmp_path / "result.json").exists()
    staged[0].unlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions and symlinks")
def test_replacement_keeps_copy_metadata_and_replaces_destination_symlink(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n")
    source.chmod(0o640)
    os.utime(source, ns=(1_000_000_000, 2_000_000_000))
    old_target = tmp_path / "old.csv"
    old_target.write_text("old content")
    destination = tmp_path / "result.csv"
    destination.symlink_to(old_target)

    atomic_write(destination, lambda temporary: shutil.copy2(source, temporary))

    assert not destination.is_symlink()
    assert destination.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    assert destination.stat().st_mtime_ns == source.stat().st_mtime_ns
    assert old_target.read_text() == "old content"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
def test_default_staged_permissions_are_not_inherited_from_destination(tmp_path):
    destination = tmp_path / "result.json"
    destination.write_text("old")
    destination.chmod(0o644)
    atomic_write(destination, lambda temporary: temporary.write_text("new"))
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


@pytest.mark.parametrize("staged_kind", ["missing", "directory", "symlink"])
def test_writer_must_leave_a_regular_file(tmp_path, staged_kind):
    if staged_kind == "symlink" and os.name == "nt":
        pytest.skip("Creating symlinks requires privileges on Windows")
    destination = tmp_path / "result.json"
    destination.write_text("old")

    def writer(temporary):
        temporary.unlink()
        if staged_kind == "directory":
            temporary.mkdir()
        elif staged_kind == "symlink":
            temporary.symlink_to(destination)

    expected = FileNotFoundError if staged_kind == "missing" else ValueError
    with pytest.raises(expected):
        atomic_write(destination, writer)
    assert destination.read_text() == "old"
    assert list(tmp_path.iterdir()) == [destination]


def test_concurrent_writers_use_distinct_temporary_files(tmp_path):
    destination = tmp_path / "result.bin"
    destination.write_bytes(b"old")
    barrier = Barrier(2, timeout=10)
    staged = []
    outputs = [b"a" * 8192, b"b" * 8192]

    def run(payload):
        def writer(temporary):
            staged.append(temporary)
            temporary.write_bytes(payload)
            barrier.wait()
            # Neither writer can replace the old file until both have finished
            # writing. The second barrier keeps this assertion race-free.
            assert destination.read_bytes() == b"old"
            barrier.wait()

        atomic_write(destination, writer)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, outputs))
    assert len(set(staged)) == 2
    assert destination.read_bytes() in outputs
    assert list(tmp_path.iterdir()) == [destination]
