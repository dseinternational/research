> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words HSGP docstrings prob fontlist cachedir CMAP cmap dsegroup edgecolor -->

# Python utilities

`dse-research-utils` is the shared Python library for [Down Syndrome Education International](https://www.down-syndrome.org/) research projects. It requires Python 3.14 or later.

## Install

Install the `v0.19.4` release from your project's root.

```bash
uv add "dse-research-utils @ git+https://github.com/dseinternational/research.git@v0.19.4#subdirectory=src/python"
```

For an existing installation, follow the [upgrade procedure](../../docs/README.md#upgrade-a-consuming-project) and [0.19.4 notes](../../docs/migrating-to-0.19.md).

The base install includes PyMC, PyTensor, nutpie, ArviZ, PreliZ, numerical libraries and the h5netcdf/h5py engine used to save traces. See `pyproject.toml` in the target tag for dependency minimums. Add extras inside the dependency name in the install command, for example `dse-research-utils[graphs,viz]`.

| Extra          | Packages                                    | Use                                             |
| -------------- | ------------------------------------------- | ----------------------------------------------- |
| `viz`          | seaborn                                     | Histogram grids                                 |
| `graphs`       | graphviz, networkx                          | Graph plotting with the system `dot` executable |
| `notebook`     | jupyter, jupytext                           | Notebooks and image display                     |
| `dependence`   | dcor                                        | Distance correlation                            |
| `tuning`       | optuna, optuna-integration                  | Parameter search                                |
| `io`           | orjson, tabulate                            | JSON and table output                           |
| `jax`          | jax, numpyro                                | Alternative sampling backends                   |
| `boosting`     | lightgbm, xgboost, shap                     | Gradient boosting and explanation               |
| `boosting-cpu` | lightgbm, xgboost-cpu, shap                 | CPU-only boosting (xgboost on macOS)            |
| `columnar`     | duckdb, polars, pyreadstat                  | Columnar and statistical data formats           |
| `storage`      | zarr                                        | Alternative array storage                       |
| `all`          | All compatible extras, including `boosting` | All optional library features                   |

Choose one of `boosting` and `boosting-cpu`. Both provide the `xgboost` import. Helpers report missing optional packages when called.

The `jax` and `all` extras require AVX2 and FMA instructions on x86-64 CPUs. Apple silicon uses ARM64 builds. [JAX 0.12 release notes](https://docs.jax.dev/en/latest/changelog.html#jax-0-12-0-october-19-2026)

The `columnar` and `all` extras require DuckDB 2.0 or later. Check file and Python API compatibility before upgrading from 1.x. [Database and data-import changes](../../docs/migrating-to-0.19.md#databases-and-data-imports)

## System requirements for plots

Graph plotting requires Graphviz's `dot` executable. Install Graphviz with `brew install graphviz`, `sudo apt install graphviz` or `winget install Graphviz.Graphviz`.

The default plot style uses the Noto Sans and Noto Sans Math system fonts. Install them with `brew install --cask font-noto-sans font-noto-sans-math` on macOS, `sudo apt install fonts-noto-core` on Debian or Ubuntu, or the [Noto Sans](https://fonts.google.com/noto/specimen/Noto+Sans) and [Noto Sans Math](https://fonts.google.com/noto/specimen/Noto+Sans+Math) downloads on Windows.

Plain-text symbols fall back to Noto Sans Math when installed, then to matplotlib's bundled DejaVu Sans. Without Noto Sans, ordinary text uses the next installed font in the style's list. Missing math fonts can produce font warnings and different output. Install the same fonts on workstations and CI machines when figures must match.

If matplotlib cannot find newly installed fonts, use the command below to locate its cache directory. Delete `fontlist-*.json` there, then restart running kernels.

```bash
uv run python -c "import matplotlib; print(matplotlib.get_cachedir())"
```

Regenerate affected figures and model graphs. Check labels, legends and symbols for clipping, overlap or missing glyphs.

## Plot colours

`dse_research_utils.plot.styles` reads the DSE design tokens from `plot/design_tokens/colors.json`. To update the tokens, copy a released file from `dsegroup/apps-common` and update `design-tokens.pin.json` with it.

- Use `CHART_COLOURS` in order for categorical series. It is also the default colour cycle. `categorical_palette(n)` returns up to six hex colours and raises `ValueError` above six. For more categories, group the smallest categories, use small multiples, label series directly or pass a named matplotlib `palette`.
- Ordered values take `sequential_palette(n)`, from least to most, or `diverging_palette(n)`, from the low or negative end in orange to the high or positive end in blue, for three to five values. Every categorical colour and ordered step reaches 3:1 against the white background. Separate neighbouring segments, such as the parts of a stacked bar, with `edgecolor=BACKGROUND_COLOUR`.
- Continuous scales are `SEQUENTIAL_CMAP` and `DIVERGING_CMAP`, registered as `dse_sequential` and `dse_diverging`, with `_r` reversals. The sequential scale is the default `image.cmap`. `plot_heatmap(..., centre=0.0)` centres the diverging scale, for example for correlations.
- Use `TEXT_COLOUR` for text, `MUTED_TEXT_COLOUR` for secondary text and reference lines, `LINE_COLOUR` for grid lines, and `BACKGROUND_COLOUR` for backgrounds.

Accessing a deprecated `COLOUR_*` name emits `DeprecationWarning`. Blue, orange, green and purple map to `CHART_COLOURS[0]` through `[3]`. Other names retain their legacy values. [Deprecated named hues](../../docs/migrating-to-0.18.md#deprecated-named-hues)

## Package map

Import helpers from their defining module under `dse_research_utils`.

| Area                  | Purpose                                                                                               | Usage guide                                                                                                                                                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `console`             | Shared Rich console, tables and run summaries                                                         | Module docstrings                                                                                                                                                                                                                                 |
| `environment`         | Script/notebook setup, output paths and disk checks                                                   | Module docstrings                                                                                                                                                                                                                                 |
| `math`                | Numerical constants                                                                                   | Module docstrings                                                                                                                                                                                                                                 |
| `metadata`, `storage` | File writes, directory promotion, provenance and Azure uploads                                        | [Files and provenance](../../docs/shared-file-provenance.md), [directory promotion](../../docs/shared-directory-report-evaluation.md#promote-a-completed-directory)                                                                               |
| `ml`                  | Feature dependence, grouping, search, cross-validation and permutation scores                         | [Feature groups](../../docs/shared-assets-and-feature-groups.md#reuse-a-dissimilarity-matrix-for-feature-grouping), [permutation scoring](../../docs/shared-directory-report-evaluation.md#keep-held-out-and-pooled-permutation-scoring-explicit) |
| `plot`                | Figure styles, saving and statistical plots                                                           | Module docstrings, [plot colours](#plot-colours) and [font setup](#system-requirements-for-plots)                                                                                                                                                 |
| `report`              | Model artefact reads, nearest-row lookup and asset checks                                             | [Report reads](../../docs/shared-directory-report-evaluation.md#read-file-state-before-applying-report-rules), [assets](../../docs/shared-assets-and-feature-groups.md)                                                                           |
| `statistics`          | Intervals, predictive checks, likelihood aggregation, diagnostics, model helpers and sampling presets | [Statistical arrays](../../docs/shared-predictive-arrays.md), [array intervals and HSGP geometry](../../docs/shared-directory-report-evaluation.md), [diagnostic reductions](../../docs/migrating-to-0.17.md)                                     |

## Reporting and sampling defaults

`ReportingConfiguration` defaults to 0.89 coverage (`ci_prob`) and a highest-density interval (`interval_kind="hdi"`). Helpers such as `interval_1d` default to an equal-tailed interval (`kind="eti"`) at the same coverage. Pass both settings explicitly when tables and figures must agree.

`statistics.models.sampling.get_sampling_configuration` defaults to `dev`. Draws and tuning steps are per chain.

| Accepted names                           | Chains | Draws | Tuning steps | `target_accept` |
| ---------------------------------------- | ------ | ----- | ------------ | --------------- |
| `dev`, `development`                     | 2      | 500   | 500          | 0.85            |
| `test`, `testing`                        | 4      | 2,000 | 2,000        | 0.90            |
| `rep-lite`, `reporting-lite`, `rep_lite` | 4      | 4,000 | 4,000        | 0.95            |
| `reporting`, `report`, `rep`             | 6      | 6,000 | 6,000        | 0.95            |

The default seed is 47. Worker count is capped by the chain count and available cores. Check convergence and precision using the fitted model's diagnostics.

For contributor setup and checks, use the [repository readme](../../README.md#develop-in-this-repository).
