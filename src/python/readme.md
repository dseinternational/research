> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words HSGP BFMI MCMC PSIS ELPD docstrings prob -->

# Python utilities

`dse-research-utils` is the shared Python library for [Down Syndrome Education International](https://www.down-syndrome.org/) research projects. It requires Python 3.14 or later.

## Install

Install the published `v0.17.0` tag:

```bash
uv add "dse-research-utils @ git+https://github.com/dseinternational/research.git@v0.17.0#subdirectory=src/python"
```

For an existing installation, follow the [upgrade procedure and version notes](../../docs/README.md#upgrade-a-consuming-project). Preserve the extras the project already uses.

This package's `pyproject.toml` defines the shared minimum dependency versions. Consuming projects inherit these requirements and resolve their own lock files. Declare any additional requirements that their own code needs, without copying the library's version list.

The base install includes PyMC, PyTensor, nutpie, ArviZ, PreliZ, numerical libraries and the h5netcdf/h5py engine used to save traces. Optional extras add:

| Extra          | Packages                                    | Use                                                   |
| -------------- | ------------------------------------------- | ----------------------------------------------------- |
| `viz`          | seaborn                                     | Histogram grids                                       |
| `graphs`       | graphviz, networkx                          | Graph plotting; also requires the system `dot` binary |
| `notebook`     | jupyter, jupytext                           | Notebooks and image display                           |
| `dependence`   | dcor                                        | Distance correlation                                  |
| `tuning`       | optuna, optuna-integration                  | Parameter search                                      |
| `io`           | orjson, tabulate                            | JSON and table output                                 |
| `jax`          | jax, numpyro                                | Alternative sampling backends                         |
| `boosting`     | lightgbm, xgboost, shap                     | Gradient boosting and explanation                     |
| `boosting-cpu` | lightgbm, xgboost-cpu, shap                 | CPU-only boosting; uses xgboost on macOS              |
| `columnar`     | duckdb, polars, pyreadstat                  | Columnar and statistical data formats                 |
| `storage`      | zarr                                        | Alternative array storage                             |
| `all`          | All compatible extras, including `boosting` | Full contributor environment                          |

Add extras inside the dependency name, for example `dse-research-utils[graphs,viz]`, in the installation command. `boosting` and `boosting-cpu` provide the same `xgboost` import and must not be combined. Helpers import optional packages when needed and report a missing dependency if it is absent.

## System requirements for plots

Graph plotting requires Graphviz's `dot` executable. Install Graphviz with `brew install graphviz`, `sudo apt install graphviz` or `winget install Graphviz.Graphviz`.

The default plot style uses the Noto Sans and Noto Sans Math system fonts. Install them with `brew install --cask font-noto-sans font-noto-sans-math` on macOS, `sudo apt install fonts-noto-core` on Debian or Ubuntu, or the Google Fonts downloads on Windows. The [font guide](../../docs/migrating-to-0.16.md#install-the-fonts) covers cache refresh and figure checks.

Plain-text symbols fall back to Noto Sans Math when installed, then to matplotlib's bundled DejaVu Sans. Without Noto Sans, ordinary text uses the next installed font in the style's list. Missing math fonts can produce font warnings and different output. Install the same fonts on workstations and CI machines when figures must match.

## Package map

Modules live under `dse_research_utils`. Import helpers from their defining module; `__init__.py` files provide no re-exports.

| Area                  | Purpose                                                                                               | Usage guide                                                                                                                                                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `console`             | Shared Rich console, tables and run summaries                                                         | Module docstrings                                                                                                                                                                                                                      |
| `environment`         | Script/notebook setup, output paths and disk checks                                                   | Module docstrings                                                                                                                                                                                                                      |
| `math`                | Numerical constants                                                                                   | Module docstrings                                                                                                                                                                                                                      |
| `metadata`, `storage` | File writes, directory promotion, provenance and Azure uploads                                        | [Files and provenance](../../docs/shared-file-provenance.md), [directory promotion](../../docs/consolidation-migration.md#promote-a-completed-directory)                                                                               |
| `ml`                  | Feature dependence, grouping, search, cross-validation and permutation scores                         | [Feature groups](../../docs/shared-assets-and-feature-groups.md#reuse-a-dissimilarity-matrix-for-feature-grouping), [permutation scoring](../../docs/consolidation-migration.md#keep-held-out-and-pooled-permutation-scoring-explicit) |
| `plot`                | Figure styles, saving and statistical plots                                                           | Module docstrings and [font guide](../../docs/migrating-to-0.16.md)                                                                                                                                                                    |
| `report`              | Model artefact reads, nearest-row lookup and asset checks                                             | [Report reads](../../docs/consolidation-migration.md#read-file-state-before-applying-report-rules), [assets](../../docs/shared-assets-and-feature-groups.md)                                                                           |
| `statistics`          | Intervals, predictive checks, likelihood aggregation, diagnostics, model helpers and sampling presets | [Statistical arrays](../../docs/shared-predictive-arrays.md), [array intervals and HSGP geometry](../../docs/consolidation-migration.md), [diagnostic reductions](../../docs/migrating-to-0.17.md)                                     |

The helpers keep study choices in the caller. Projects select observation units, priors, interval coverage, feature cut thresholds, scoring populations, missing-value rules and criteria for accepting or publishing a fit.

## Reporting and sampling defaults

Shared interval helpers and `ReportingConfiguration.ci_prob` default to 0.89 coverage. `ReportingConfiguration.interval_kind` defaults to `"hdi"` for a highest-density interval; several interval functions default to `"eti"` for an equal-tailed interval. Pass both coverage and kind explicitly when a report needs one convention throughout.

`statistics.models.sampling.get_sampling_configuration` supplies these presets. Draws and tuning steps are per chain. MCMC means Markov chain Monte Carlo, the method used to draw samples from a posterior distribution.

| Accepted names                           | Chains | Draws | Tuning steps | `target_accept` |
| ---------------------------------------- | ------ | ----- | ------------ | --------------- |
| `dev`, `development`                     | 2      | 500   | 500          | 0.85            |
| `test`, `testing`                        | 4      | 2,000 | 2,000        | 0.90            |
| `rep-lite`, `reporting-lite`, `rep_lite` | 4      | 4,000 | 4,000        | 0.95            |
| `reporting`, `report`, `rep`             | 6      | 6,000 | 6,000        | 0.95            |

The default seed is 47. Worker count is capped by the chain count and available cores. A preset does not guarantee convergence or sufficient precision. Check the fitted model's diagnostics.

For contributor setup and checks, use the [repository readme](../../README.md#develop-in-this-repository).
