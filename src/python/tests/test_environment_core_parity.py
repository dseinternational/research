# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard the transition away from the conda-forge core.

``data/environment-core.yml`` is retained, deprecated, so the consuming repos
that have not yet migrated off conda keep a working ``dse-check-env``. That
leaves two declarations of the same floors for as long as the transition runs,
and this test stops them drifting apart: every package in the retained core must
carry the same lower bound as ``pyproject.toml``, whether it sits in the base
dependencies or in one of the extras.

Delete this module together with ``environment-core.yml`` and ``dse-check-env``
once every consuming repository is on uv.
"""

import tomllib
from pathlib import Path

import pytest

from dse_research_utils.environment.check import load_core

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Conda packages with no PyPI counterpart: the interpreter itself, and the
# system Graphviz binaries (the PyPI `graphviz` package is bindings only and is
# the conda `python-graphviz`).
_CONDA_ONLY = {"python", "graphviz"}
_CONDA_TO_PYPI = {"python-graphviz": "graphviz"}


def _floor(spec: str) -> str | None:
    """Return the '>=' lower bound of a requirement spec, or None if unbounded."""
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
def test_retained_conda_core_floor_matches_pyproject(package: str) -> None:
    """Each retained conda core package is declared in pyproject with the same floor."""
    pyproject = _pyproject_floors()
    assert package in pyproject, (
        f"'{package}' is in the retained conda core but is declared nowhere in "
        f"pyproject.toml — consuming repos on conda would get a floor this package no longer states"
    )
    assert pyproject[package] == _core_floors()[package], (
        f"'{package}' floor drift: conda core says >={_core_floors()[package]}, "
        f"pyproject.toml says >={pyproject[package]}"
    )
