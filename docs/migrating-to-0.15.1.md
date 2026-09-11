> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Upgrade to 0.15.1

<!-- cspell:ignore fonttools tqdm wrapt -->

This change prepares version 0.15.1. Downstream upgrades must wait until the research PR is merged, its checks pass and `v0.15.1` is tagged on the merged release commit. Preparing this file does not publish that tag.

## What changes

The research lock file refreshes the following packages. The library's dependency floors and extras do not change. A downstream repository does not inherit this lock file when it installs the library from Git, so changing the library tag alone does not ensure these package updates.

| Package      | Previous lock | Updated lock |
| ------------ | ------------- | ------------ |
| build        | 1.6.0         | 1.6.1        |
| fonttools    | 4.64.0        | 4.65.0       |
| pure-eval    | 0.2.3         | 0.2.4        |
| ruff         | 0.16.6        | 0.16.7       |
| scikit-learn | 1.9.0         | 1.9.1        |
| tqdm         | 4.70.0        | 4.70.1       |
| uv           | 0.12.11       | 0.12.13      |
| wrapt        | 2.4.0         | 2.4.1        |

## Confirmed downstream baselines

The default branches checked on 11 September 2026 all specify `v0.15.0` and lock `dse-research-utils` 0.15.0 to research commit `e818caf52c0a73aed4ba5286bdfdc58c7867ea69`.

| Repository                  | Checked commit                                                                                                           | Existing extras                                                   |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| language-reading-predictors | [ac58282](https://github.com/dseinternational/language-reading-predictors/tree/ac5828228fb8dd1dcddf8d198de9fd097ea94baf) | boosting, columnar, dependence, graphs, io, notebook, tuning, viz |
| vocabulary-growth           | [82cd646](https://github.com/dseinternational/vocabulary-growth/tree/82cd646f715834d31e922d4b4b66973e28b3d2fc)           | columnar, graphs, io, jax, notebook, viz                          |
| us-birth-certificates       | [61f5aea](https://github.com/dspopulations/us-birth-certificates/tree/61f5aea98297c60a3a20458edbb41e60a9f07273)          | boosting, columnar, dependence, graphs, io, jax, notebook, tuning |

All three lock NumPy 2.5.3, numba 0.67.0, PyTensor 3.3.1 and PyMC 6.3.2. Those versions also remain unchanged in the research lock. All three contain the previous versions of fonttools, pure-eval, ruff, scikit-learn, tqdm and wrapt listed above. None contains build or uv as a locked package. These are dated baselines, not a claim about later commits.

## Downstream steps after release

1. Confirm that the upstream PR is merged, its checks pass and `v0.15.1` resolves to the intended release commit.
2. Change only the research source tag in the downstream `pyproject.toml` to `v0.15.1`. Preserve its current extras and other dependency constraints.
3. Refresh the downstream lock explicitly, then review every resolved change.

```bash
uv lock --upgrade-package dse-research-utils --upgrade-package scikit-learn --upgrade-package fonttools --upgrade-package pure-eval --upgrade-package tqdm --upgrade-package wrapt --upgrade-package ruff
uv lock --check
uv sync --locked
```

The resolver may select newer compatible releases by the time this runs. Record the actual package versions and the resolved research tag commit in the downstream PR. Do not copy the research lock file or add its repository-only build and uv dependencies downstream.

4. Run each downstream repository's documented lint, formatting, documentation and test checks, including its separate slow tests where required. Confirm that the research package reports version 0.15.1 and that the selected extras still import.
5. Keep existing fitted results and their recorded environments intact. This dependency update does not authorize refitting models or publishing revised research results.

## Upstream validation

The refreshed lock passed lock consistency, environment sync, Python lint and formatting, package build, Markdown formatting and spelling checks. The Windows test run recorded 1,173 passes, 12 skips and four failures, with three binary-hash parameter cases excluded because their generated test identifiers exceed the Windows environment-variable limit. The remaining failures concern a filename containing `?`, two symbolic-link cases and concurrent file replacement. These failures have not been compared against the old lock, so they are not established as pre-existing. The Linux PR checks must pass before release.
