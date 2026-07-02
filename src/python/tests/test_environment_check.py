# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

from dse_research_utils.environment.check import check_env_core, load_core, main

CORE = load_core()


def _write_env(tmp_path: Path, conda_deps: list[str]) -> Path:
    lines = ["name: some-repo", "channels:", "  - conda-forge", "dependencies:"]
    lines += [f"  - {dep}" for dep in conda_deps]
    # A representative pip layer, which the checker must ignore.
    lines += ["  - pip", "  - pip:", "      - dse-research-utils[io]", "      - -e ./"]
    path = tmp_path / "environment.yml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_core_has_expected_packages() -> None:
    assert "pymc" in CORE
    assert "numpy" in CORE
    assert CORE["numpy"].startswith("numpy>=")


def test_matching_env_has_no_problems(tmp_path: Path) -> None:
    # Exact core specs plus a repo-specific add-on (which must be ignored).
    env = _write_env(tmp_path, [*CORE.values(), "lightgbm>=4.6.0"])
    assert check_env_core(env) == []


def test_missing_core_package_is_reported(tmp_path: Path) -> None:
    env = _write_env(tmp_path, [spec for name, spec in CORE.items() if name != "pymc"])
    problems = check_env_core(env)
    assert any("pymc" in problem and "missing" in problem for problem in problems)


def test_drifted_spec_is_reported(tmp_path: Path) -> None:
    env = _write_env(tmp_path, ["numpy>=2.0" if name == "numpy" else spec for name, spec in CORE.items()])
    problems = check_env_core(env)
    assert any("numpy" in problem for problem in problems)


def test_main_returns_zero_on_match(tmp_path: Path, capsys) -> None:
    env = _write_env(tmp_path, list(CORE.values()))
    assert main([str(env)]) == 0
    assert "matches" in capsys.readouterr().out


def test_main_returns_one_on_drift(tmp_path: Path, capsys) -> None:
    env = _write_env(tmp_path, [spec for name, spec in CORE.items() if name != "nutpie"])
    assert main([str(env)]) == 1
    assert "drift" in capsys.readouterr().out
