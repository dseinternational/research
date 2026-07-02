# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Drift check for the shared conda-forge "compiled core" of DSE data-science envs.

Every repo's ``environment.yml`` must pin the canonical core packages (shipped
with this package as ``data/environment-core.yml``) to identical specs, so the
compiled scientific stack cannot drift across repos. Run as a console script::

    dse-check-env path/to/environment.yml

Exits non-zero and prints a report if the environment omits a core package or
pins one to a different spec than the canonical core. Packages beyond the core
(repo-specific add-ons, the whole pip layer) are ignored.
"""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

_CORE_RESOURCE = "data/environment-core.yml"
_NAME_DELIMITERS = "<>=!~ "


def _require_yaml():
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - trivial guard
        raise SystemExit(
            "PyYAML is required for dse-check-env (it ships with dse-research-utils; "
            "in a bare venv run `pip install pyyaml`)."
        ) from exc
    return yaml


def _package_name(spec: str) -> str:
    """Return the package name from a conda spec (e.g. 'numpy>=2.4.6,<2.5' -> 'numpy')."""
    for i, ch in enumerate(spec):
        if ch in _NAME_DELIMITERS:
            return spec[:i]
    return spec


def _conda_specs(dependencies: list) -> dict[str, str]:
    """Map package name -> full spec string for the conda (non-pip) dependencies.

    Dict entries such as ``{'pip': [...]}`` are the pip layer and are skipped.
    """
    specs: dict[str, str] = {}
    for dep in dependencies:
        if isinstance(dep, str):
            spec = dep.strip()
            specs[_package_name(spec).strip()] = spec
    return specs


def _load_yaml_deps(text: str) -> list:
    yaml = _require_yaml()
    data = yaml.safe_load(text) or {}
    return data.get("dependencies", []) or []


def load_core() -> dict[str, str]:
    """Load the canonical core specs shipped with this package."""
    text = files("dse_research_utils").joinpath(_CORE_RESOURCE).read_text(encoding="utf-8")
    return _conda_specs(_load_yaml_deps(text))


def check_env_core(env_path: str | Path, core: dict[str, str] | None = None) -> list[str]:
    """Return a list of human-readable drift problems for ``env_path`` (empty = OK)."""
    core = load_core() if core is None else core
    env_specs = _conda_specs(_load_yaml_deps(Path(env_path).read_text(encoding="utf-8")))
    problems: list[str] = []
    for name, core_spec in sorted(core.items()):
        actual = env_specs.get(name)
        if actual is None:
            problems.append(f"missing core package '{name}' (canonical: {core_spec})")
        elif actual != core_spec:
            problems.append(f"'{name}' pinned as '{actual}' but the canonical core requires '{core_spec}'")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dse-check-env",
        description="Check a repo environment.yml against the canonical DSE conda-forge core.",
    )
    parser.add_argument("env_file", help="Path to the repo's environment.yml")
    args = parser.parse_args(argv)

    problems = check_env_core(args.env_file)
    if problems:
        print(f"environment core drift in {args.env_file}:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{args.env_file}: conda-forge core matches the canonical spec")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
