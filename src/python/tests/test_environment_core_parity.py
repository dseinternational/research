# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check that current Python floors cover the retained conda baseline.

``data/environment-core.yml`` is retained, deprecated, so the consuming repos
that have not yet migrated off conda keep a working ``dse-check-env``. That
file retains the baseline shipped in 0.15.1. It cannot track newer PyPI floors
because conda-forge does not yet provide the same compatible package set. Every
package in that baseline must remain declared in ``pyproject.toml``, with a floor
at least as high, whether it sits in the base dependencies or in an extra.

Remove this module together with ``environment-core.yml`` and ``dse-check-env``
in a separate cleanup when support for legacy consumers ends.
"""

import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

from dse_research_utils.environment.check import load_core

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Conda packages with no PyPI counterpart: the interpreter itself, and the
# system Graphviz binaries (the PyPI `graphviz` package is bindings only and is
# the conda `python-graphviz`).
_CONDA_ONLY = {"python", "graphviz"}
_CONDA_TO_PYPI = {"python-graphviz": "graphviz"}


def _floor(spec: str) -> str | None:
    """Return the '>=' lower bound of a requirement spec, or None if unbounded."""
    # Drop any environment marker first: the `boosting-cpu` extra splits xgboost
    # on `sys_platform`, and without this the marker would be read as part of the
    # version (">=3.3.0; sys_platform == 'darwin'").
    spec = spec.split(";", 1)[0]
    for clause in spec.split(","):
        clause = clause.strip()
        if clause.startswith(">="):
            return clause[2:].strip()
    return None


def _pyproject_floors() -> dict[str, str | None]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    specs = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        specs.extend(extra)

    floors: dict[str, str | None] = {}
    for spec in specs:
        spec = spec.strip()
        name = spec
        for i, ch in enumerate(spec):
            if ch in "<>=!~[ ":
                name = spec[:i]
                break
        floors[name.replace("_", "-").lower()] = _floor(spec.removeprefix(name))
    return floors


def _core_floors() -> dict[str, str | None]:
    floors: dict[str, str | None] = {}
    for name, spec in load_core().items():
        if name in _CONDA_ONLY:
            continue
        pypi_name = _CONDA_TO_PYPI.get(name, name)
        floors[pypi_name] = _floor(spec.removeprefix(name))
    return floors


@pytest.mark.parametrize("package", sorted(_core_floors()))
def test_current_floor_covers_retained_conda_baseline(package: str) -> None:
    """A move to the current Python requirements must not lower a legacy floor."""
    pyproject = _pyproject_floors()
    assert package in pyproject, (
        f"'{package}' is in the retained conda core but is declared nowhere in "
        f"pyproject.toml — consuming repos on conda would get a floor this package no longer states"
    )
    current_floor = pyproject[package]
    legacy_floor = _core_floors()[package]
    assert current_floor is not None and legacy_floor is not None
    assert Version(current_floor) >= Version(legacy_floor), (
        f"'{package}' minimum regressed: retained conda baseline says >={legacy_floor}, "
        f"pyproject.toml says >={current_floor}"
    )
