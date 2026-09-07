> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words hsgp -->

# Python utilities

`dse-research-utils` — the shared library for [Down Syndrome Education International](https://www.down-syndrome.org/) research projects.

This package's `pyproject.toml` is the canonical source of the dependency floors shared across DSE research repositories. Consuming repositories depend on the library and inherit those floors transitively rather than restating package versions of their own.

## Install

After the `v0.14.0` tag is published, install the consolidated helpers with the command below. The [0.14.0 upgrade notes](../../docs/migrating-to-0.14.md) explain release preparation and consumer checks. The older `v0.13.0` tag does not contain the additions from PR #101.

```bash
uv add "dse-research-utils @ git+https://github.com/dseinternational/research.git@v0.14.0#subdirectory=src/python"
```

The base install carries the modelling stack (PyMC, PyTensor, nutpie, ArviZ, PreliZ), the numerics core, and the netCDF engine (h5netcdf, h5py) that `InferenceData.to_netcdf` requires. Optional layers are extras:

| Extra          | Adds                                    | For                                                |
| -------------- | --------------------------------------- | -------------------------------------------------- |
| `viz`          | seaborn                                 | `plot.grids` histogram grids                       |
| `graphs`       | graphviz, networkx                      | `plot.graphs` — also needs the system `dot` binary |
| `notebook`     | jupyter, jupytext                       | notebook workflows; `plot.io.display_image`        |
| `dependence`   | dcor                                    | `ml.feature_dependence.distance_corr_matrix`       |
| `tuning`       | optuna, optuna-integration              | hyper-parameter search                             |
| `io`           | orjson, tabulate                        | fast JSON and table rendering                      |
| `jax`          | jax, numpyro                            | JAX/NumPyro sampler backends                       |
| `boosting`     | lightgbm, xgboost, shap                 | gradient boosting and explanation                  |
| `boosting-cpu` | lightgbm, xgboost-cpu, shap             | CPU-only boosting; uses xgboost on macOS           |
| `columnar`     | duckdb, polars, pyreadstat              | columnar and statistical data formats              |
| `storage`      | zarr                                    | zarr as an alternative to the netCDF core          |
| `all`          | all compatible extras; takes `boosting` | development environments                           |

Helpers that need an extra lazy-import it and raise a clear error when it is absent. `boosting` and `boosting-cpu` are mutually exclusive; retain the variant already chosen by the consuming project.

## Shared file operations

`storage.files.atomic_write` writes one complete file through a temporary file beside its destination. `metadata.provenance` provides `git_snapshot`, `package_versions` and `sha256_file` without choosing a project's manifest schema. See the [usage and migration guide](../../docs/shared-file-provenance.md) for examples, failure handling and compatibility requirements.

## Shared statistical arrays

`statistics.samples` aligns labelled predictive and observed arrays. `statistics.predictive` computes per-observation predictive checks, and `statistics.log_likelihood` aggregates factors into explicitly chosen evaluation units. See the [statistical array guide](../../docs/shared-predictive-arrays.md) for numerical conventions and consumer migration examples.

## Shared report assets and feature groups

`report.assets` inspects direct HTML resources and checks upload inventories and HTTP availability. `ml.feature_groups` builds and cuts clustering trees from existing dissimilarity matrices. See the [usage and migration guide](../../docs/shared-assets-and-feature-groups.md) for scope, numerical conventions and consumer adapters.

## Directory promotion, report reads and evaluation

`storage.directories` promotes completed trees with an explicit lock and retained backup. `report.readers` distinguishes present, missing and invalid files. `statistics.array_intervals` reduces explicit sample axes, `ml.permutation` supports separate held-out and pooled evaluation, and `statistics.models.hsgp_design` records and replays fixed Gaussian-process geometry. The [combined migration guide](../../docs/consolidation-migration.md) covers these APIs and the rules each consumer retains.

## Development

Work from the repository root, which is the uv workspace root:

```bash
uv sync                                 # create .venv
uv run pytest                           # tests
uv run ruff check src/python            # lint
uv build --package dse-research-utils   # wheel + sdist
```
