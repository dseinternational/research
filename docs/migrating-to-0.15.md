> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

<!-- cspell:words abseil libarrow libabseil ANOVA lockfiles NSGAII Chainer -->

# Migrating to 0.15.0

Version 0.15.0 is a dependency release. It changes no library API and adds no packages. It raises the shared floors in `src/python/pyproject.toml` — the compiled core every consuming repository inherits transitively — and lifts the NumPy ceiling that has capped the stack since 0.11.0. Downstream projects adopt it by updating one tag and re-locking; the work that needs judgement is checking the two upstream releases that changed behaviour, NumPy 2.5 and Optuna 5.0.

## Release sequence

The version change prepares the release. It does not publish `v0.15.0`. Merge the release-preparation PR, check CI on the merged commit, then tag that commit as `v0.15.0`. The existing `v0.14.0` tag points to the code before this dependency update and must remain there.

```bash
uv sync --locked
uv run pytest
uv run ruff check src/python
uv run ruff format --check src/python
uv build --package dse-research-utils
npm ci && npm run spellcheck && npm run format:check
```

Check that both built distributions report `0.15.0`. Git tags are the installation source used by the consuming repositories; merging this PR does not update their pins or lockfiles.

## The NumPy ceiling lifts

`numpy>=2.4.6,<2.5` becomes `numpy>=2.4.6,<2.6`. The chain that held it has moved: PyTensor 3.3.1 admits `numba<=0.67.0`, and numba 0.67.0 relaxed to `numpy<2.6`. The library's `pytensor` floor rises to `>=3.3.1` accordingly, because 3.3.1 is what makes the wider ceiling reachable rather than merely declared.

The floor deliberately stays at `2.4.6`. A floor states the minimum supported version, and PyTensor 3.3.0 with numba 0.66 and NumPy 2.4.x remains a legal solution for a project that has not moved.

Two places record this cap, and both changed together: the specification in `src/python/pyproject.toml` and the Dependabot ignore rule in `.github/dependabot.yml`, now `numpy >=2.6.0`. A consuming repository carries its own copy of that ignore rule and must update it in the same PR as the pin, or Dependabot will keep proposing a widening no resolver can use.

The retained conda core (`environment-core.yml`, deprecated) keeps `numpy<2.5`. conda-forge still ships PyTensor 3.3.0, so the older ceiling is still correct there. The parity test compares floors only, so the two files legitimately differ on the cap.

## Optuna 5.0 changes tuning defaults

The `tuning` extra moves from `optuna>=4.9.0` to `optuna>=5.0.0`. Nothing in this library imports Optuna — the extra exists for consumers — but 5.0 is a major release that changes what a study does by default:

- `TPESampler` enables multivariate TPE and the constant-liar strategy, and replaces `NSGAIISampler` as the multi-objective default.
- `PedAnovaImportanceEvaluator` replaces f-ANOVA as the default parameter-importance evaluator.
- `optuna.multi_objective` and the AllenNLP, Chainer and MXNet integration wrappers are removed; `constraints_func` is deprecated in favour of `trial.set_constraint()`; `RDBStorage` and `JournalStorage` normalise trial timestamps to UTC.

A project that resumes a stored study, or that compares a new search against recorded results, should re-read the [5.0 release notes](https://github.com/optuna/optuna/releases/tag/v5.0.0) before treating the two as comparable. Search results from 4.x and 5.x are not interchangeable evidence.

## Floors raised

| Package                        | 0.14.0     | 0.15.0     | Layer                      |
| ------------------------------ | ---------- | ---------- | -------------------------- |
| `numpy` (ceiling)              | `<2.5`     | `<2.6`     | core                       |
| `pytensor`                     | `>=3.2.2`  | `>=3.3.1`  | core                       |
| `statsmodels`                  | `>=0.14.6` | `>=0.15.0` | core                       |
| `preliz`                       | `>=0.27.1` | `>=0.28.0` | core                       |
| `arviz-stats`                  | `>=1.3.0`  | `>=1.3.2`  | core                       |
| `arviz-plots`                  | `>=1.3.0`  | `>=1.3.1`  | core                       |
| `optuna`, `optuna-integration` | `>=4.9.0`  | `>=5.0.0`  | `tuning`                   |
| `xgboost`, `xgboost-cpu`       | `>=3.3.0`  | `>=3.4.1`  | `boosting`, `boosting-cpu` |
| `polars`                       | `>=1.43.2` | `>=1.44.1` | `columnar`                 |
| `pyreadstat`                   | `>=1.3.5`  | `>=1.3.6`  | `columnar`                 |
| `orjson`                       | `>=3.11.9` | `>=3.12.0` | `io`                       |
| `seaborn`                      | `>=0.13`   | `>=0.13.2` | `viz`                      |
| `networkx`                     | `>=3.6`    | `>=3.6.1`  | `graphs`                   |

Floors held deliberately, because conda-forge cannot yet satisfy a higher one while the retained core is still in use: `scipy>=1.18.0` (conda-forge has 1.18.0, PyPI 1.18.1), `pymc>=6.3.1` (conda-forge 6.3.1, PyPI 6.3.2) and `jax>=0.10.2` (conda-forge jaxlib is still 0.10.2). `pyarrow>=24.0.0` is held for the same reason it always has been: libarrow 25.x needs a newer libabseil than any conda-forge jaxlib build pins. Resolving fresh on PyPI still installs the current version of each.

Repository tooling also moved: `hatchling>=1.32.0`, `build>=1.6.0`, `hatch>=1.18.0`, `ruff>=0.16.6`, `plotly>=7.0.0`, cspell 10.3.0 and the .NET SDK pin to 10.0.401. These are development-only and reach no consumer.

## Update a consumer once

After the tag exists:

```bash
uv add "dse-research-utils @ git+https://github.com/dseinternational/research.git@v0.15.0#subdirectory=src/python"
```

Retain the project's existing extras. A project that records its Git source in `tool.uv.sources` can change only the tag:

```toml
[tool.uv.sources]
dse-research-utils = { git = "https://github.com/dseinternational/research.git", tag = "v0.15.0", subdirectory = "src/python" }
```

Then run `uv lock` and `uv sync --locked`, and update the repository's own `numpy` Dependabot ignore rule to `>=2.6.0` in the same PR. Commit the pin, the lockfile and the ignore-rule change together.

Projects upgrading from before `0.14.0` must also apply the [0.14.0 migration requirements](migrating-to-0.14.md) and, from before `0.13.0`, the [0.13.0 requirements](migrating-to-0.13.md). This release does not remove them.

## What to check downstream

A dependency floor change does not refit a model. Before accepting results produced on the new stack:

- Re-run the project's own test suite and its convergence gate. NumPy 2.5, numba 0.67 and PyTensor 3.3.1 are a different numerical stack from the one 0.14.0 locked, even though this library's 1190 tests pass unchanged on it.
- Treat stored Optuna studies as belonging to the sampler that produced them, per the section above.
- Check `statsmodels` 0.15.0 against any regression output the project reports; the minor release is the largest single library move here after Optuna.
- Do not compare a saved fit against a new one across this upgrade without re-establishing that the implementation identity is unchanged.
