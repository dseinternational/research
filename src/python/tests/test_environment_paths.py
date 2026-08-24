# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the shared output-root resolver and disk preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from dse_research_utils.environment.disk import free_space_gb, preflight_disk
from dse_research_utils.environment.paths import OutputRoot

ENV_VAR = "DSE_TEST_OUTPUT_DIR"


@pytest.fixture
def root(tmp_path, monkeypatch) -> OutputRoot:
    monkeypatch.delenv(ENV_VAR, raising=False)
    return OutputRoot(ENV_VAR, tmp_path / "output")


def test_default_root_when_nothing_configured(root, tmp_path):
    assert root.resolve() == tmp_path / "output"
    assert not root.is_overridden()
    assert "repo-local default" in root.describe()


def test_env_var_beats_default(root, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    monkeypatch.setenv(ENV_VAR, str(scratch))
    assert root.resolve() == scratch.expanduser().resolve()
    assert root.is_overridden()
    assert ENV_VAR in root.describe()


def test_set_beats_env_var(root, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "scratch"))
    override = tmp_path / "cli"
    resolved = root.set(override)
    assert resolved == override.expanduser().resolve()
    assert root.resolve() == resolved
    assert "--output-dir" in root.describe()


def test_set_none_clears_the_override(root, tmp_path):
    root.set(tmp_path / "cli")
    assert root.set(None) == tmp_path / "output"
    assert not root.is_overridden()


def test_resolution_happens_at_call_time(root, tmp_path, monkeypatch):
    # The common case: a script parses its args after import, then resolves.
    assert root.resolve() == tmp_path / "output"
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "late"))
    assert root.resolve() == (tmp_path / "late").expanduser().resolve()


def test_free_space_gb_walks_up_to_an_existing_parent(tmp_path):
    missing = tmp_path / "not" / "yet" / "created"
    assert free_space_gb(missing) == pytest.approx(free_space_gb(tmp_path), rel=0.01)
    assert free_space_gb(tmp_path) > 0


def test_preflight_disk_passes_and_returns_free_space(tmp_path, captured_console):
    free = preflight_disk(0.001, tmp_path, label="test artefacts")
    assert free > 0
    out = captured_console.export_text()
    assert "[disk]" in out
    assert "test artefacts" in out


def test_preflight_disk_raises_below_threshold(tmp_path):
    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        preflight_disk(10.0**9, tmp_path, label="impossible")


def test_preflight_disk_names_the_output_root_only_when_checking_it(tmp_path, captured_console):
    preflight_disk(0.001, tmp_path, output_root=tmp_path)
    assert "[output] resolved output root" in captured_console.export_text()


def test_preflight_disk_omits_output_line_for_subdirectories(tmp_path, captured_console):
    sub = tmp_path / "sub"
    sub.mkdir()
    preflight_disk(0.001, sub, output_root=tmp_path)
    out = captured_console.export_text()
    assert "[output]" not in out
    assert "[disk]" in out


def test_preflight_accepts_path_objects(tmp_path):
    assert preflight_disk(0.001, Path(tmp_path)) > 0


def test_resolve_symlinks_false_preserves_the_link_path(tmp_path, monkeypatch):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError, NotImplementedError:  # pragma: no cover - needs privilege
        pytest.skip("symlink creation not permitted in this environment")

    resolving = OutputRoot(ENV_VAR, tmp_path / "output")
    preserving = OutputRoot(ENV_VAR, tmp_path / "output", resolve_symlinks=False)
    monkeypatch.setenv(ENV_VAR, str(link))

    assert resolving.resolve() == target.resolve()
    assert preserving.resolve() == link
