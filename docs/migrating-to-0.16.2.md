> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Upgrade to 0.16.2

This change prepares version 0.16.2. The Python API and Python 3.14 requirement are unchanged. Frank Buckley will publish `v0.16.2` on the merged release commit after its checks pass. Downstream tag upgrades must wait until that tag exists.

## Library requirements

The following minimum versions have changed since the `v0.16.1` tag. This includes dependency updates merged before the 0.16.2 release PR. A downstream repository must satisfy the base requirements when it selects 0.16.2. The DuckDB requirement applies when it selects the `columnar` extra.

| Package            | Previous minimum | New minimum | Scope    |
| ------------------ | ---------------- | ----------- | -------- |
| azure-identity     | 1.25.3           | 1.26.0      | Base     |
| azure-storage-blob | 12.30.3          | 12.31.0     | Base     |
| pytensor           | 3.3.2            | 3.3.3       | Base     |
| xarray             | 2026.7.0         | 2026.9.0    | Base     |
| duckdb             | 1.5.5            | 1.5.6       | columnar |

Numba remains at 0.67.0 because [PyTensor 3.3.3's dependency metadata](https://pypi.org/pypi/pytensor/3.3.3/json) requires `numba<=0.67.0`. Numba 0.68.0 cannot be installed with that release. NumPy retains its `<2.6` limit and PyTensor retains `<3.4`.

## Repository environment

The development minimum moves to Ruff 0.16.10. The repository uses cspell 10.3.6, and its npm lock refreshes spelling dictionaries. These tools do not become dependencies of the installed Python library.

The Python lock updates 22 external packages since `v0.16.1`, including indirect dependencies. A downstream repository does not inherit this lock. Its own resolver chooses versions that satisfy the library requirements and its other constraints.

## Downstream baselines

The default-branch declarations checked on 3 October 2026 use the following tags. Preserve each project's selected extras when updating it.

| Repository                                                                                                              | Current tag | Selected extras                                                   |
| ----------------------------------------------------------------------------------------------------------------------- | ----------- | ----------------------------------------------------------------- |
| [language-reading-predictors](https://github.com/dseinternational/language-reading-predictors/blob/main/pyproject.toml) | v0.16.1     | boosting, columnar, dependence, graphs, io, notebook, tuning, viz |
| [vocabulary-growth](https://github.com/dseinternational/vocabulary-growth/blob/main/pyproject.toml)                     | v0.16.1     | columnar, graphs, io, jax, notebook, viz                          |
| [us-birth-certificates](https://github.com/dspopulations/us-birth-certificates/blob/main/pyproject.toml)                | v0.15.2     | boosting, columnar, dependence, graphs, io, jax, notebook, tuning |

## Upgrade after release

1. Confirm that the release PR is merged, its checks pass and `v0.16.2` resolves to the intended release commit.
2. Change the research source tag in the downstream `pyproject.toml` to `v0.16.2`. Preserve its selected extras and other dependency constraints.
3. Resolve the new library requirements and install the locked environment.

```bash
uv lock --upgrade-package dse-research-utils
uv lock --check
uv sync --locked
```

The new minimums force changes where the old lock selected an older version. Other packages can remain at their locked versions. To adopt all available compatible dependency updates, use `uv lock --upgrade` and review the full lock diff. Do not copy the research lock or add duplicate declarations for the library's dependencies.

4. Run the downstream repository's documented tests and other checks, including model compilation, sampling, plotting and storage checks where applicable. Verify that both the installed distribution and `dse_research_utils.__version__` report 0.16.2. Record the resolved tag commit and package versions in the downstream PR.
5. Keep existing fitted results and their recorded environments. Dependency upgrades can affect numerical results, so a library test pass does not establish that a refitted analysis will reproduce its earlier results.

Projects upgrading from 0.15.2 or earlier must also apply the [0.16.0 upgrade notes](migrating-to-0.16.md) and [0.16.1 upgrade notes](migrating-to-0.16.1.md). For `us-birth-certificates`, this includes reviewing the new plot fonts and symbol fallback.
