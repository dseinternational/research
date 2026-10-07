> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
> Updated by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words ANOVA lockfiles NSGAII Chainer -->

# Migrating to 0.15.0

Version 0.15.0 raises minimum dependency versions and permits NumPy 2.5. It changes no library API and adds no packages. Review the numerical stack and Optuna 5 defaults when adopting it.

## The NumPy ceiling lifts

`numpy>=2.4.6,<2.5` becomes `numpy>=2.4.6,<2.6`. PyTensor 3.3.1 permits `numba<=0.67.0`, and Numba 0.67.0 supports `numpy<2.6`. The library therefore requires `pytensor>=3.3.1,<3.4` to permit this combination.

The NumPy minimum remains `2.4.6`; the wider limit permits 2.5 without requiring it. PyTensor must satisfy the new `>=3.3.1,<3.4` requirement.

If a consuming repository has dependency-update ignore rules for NumPy or Numba, review them against the supported ranges. Old rules that exclude NumPy 2.5 or Numba 0.67 can prevent supported updates.

## Optuna 5.0 changes tuning defaults

The `tuning` extra moves from `optuna>=4.9.0` to `optuna>=5.0.0`. The library does not import Optuna itself. Consumers that use it should review these default changes:

- `TPESampler` enables multivariate TPE and the constant-liar strategy, and replaces `NSGAIISampler` as the multi-objective default.
- `PedAnovaImportanceEvaluator` replaces f-ANOVA as the default parameter-importance evaluator.
- `optuna.multi_objective` and the AllenNLP, Chainer and MXNet integration wrappers are removed; `constraints_func` is deprecated in favour of `trial.set_constraint()`; `RDBStorage` and `JournalStorage` normalise trial timestamps to UTC.

Review the [Optuna 5.0 release notes](https://github.com/optuna/optuna/releases/tag/v5.0.0) before resuming a stored study or comparing searches. Record the sampler settings and evaluator used; a default change can alter the comparison.

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

This release retained `scipy>=1.18.0`, `pymc>=6.3.1`, `jax>=0.10.2` and `pyarrow>=24.0.0`. These minimums permit later compatible releases when the environment is resolved.

## Upgrade

Follow the [shared upgrade procedure](README.md#upgrade-a-consuming-project), with `v0.15.0` as the target tag. Apply the [0.14.0](migrating-to-0.14.md) and [0.13.0](migrating-to-0.13.md) requirements if upgrading from before those versions.

## What to check downstream

A dependency floor change does not refit a model. Before accepting results produced on the new stack:

- Re-run the project's own test suite and its convergence gate. NumPy 2.5, numba 0.67 and PyTensor 3.3.1 are a different numerical stack from the one 0.14.0 locked, so passing shared-library tests alone is insufficient.
- Treat stored Optuna studies as belonging to the sampler that produced them, per the section above.
- Check any reported regression outputs against `statsmodels` 0.15.0.
- Do not compare a saved fit against a new one across this upgrade without re-establishing that the implementation identity is unchanged.
