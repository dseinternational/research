> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words HSGP -->

# Documentation

Start with the [repository readme](../README.md) for contributor setup or the [Python readme](../src/python/readme.md) for installation, extras and the package map.

## Usage guides

These APIs were introduced in 0.14.0. File-permission options require 0.17.0 or later.

| Guide                                                                   | Covers                                                                                                    |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [File writes and provenance](shared-file-provenance.md)                 | Single-file replacement, permission options, Git state, package versions and file hashes                  |
| [Sample arrays and predictive summaries](shared-predictive-arrays.md)   | Labelled observations, predictive intervals and ranks, likelihood aggregation                             |
| [Report assets and feature groups](shared-assets-and-feature-groups.md) | Local HTML resources, upload inventories, HTTP availability and clustering                                |
| [Directory, report and evaluation helpers](consolidation-migration.md)  | Directory promotion, file reads, array intervals, fixed Gaussian-process geometry and permutation scoring |

## Upgrade a consuming project

Release tags through `v0.17.0` are published. Choose the target release and read all notes between the project's current version and that target. Notes describe the changes in their named release; the package's current `pyproject.toml` may contain later dependency updates.

1. Change the Git tag in the consuming project's `pyproject.toml`. Preserve its selected extras and any additional project constraints. For a source declared in `tool.uv.sources`, update the `tag` field in that entry.
2. Resolve the new package version and install the project's own locked environment:

   ```bash
   uv lock --upgrade-package dse-research-utils
   uv lock --check
   uv sync --locked
   ```

3. Review the lockfile changes. Minimum requirements can force dependency updates, but the resolver may also select later compatible releases. This repository's lockfile and development groups are not inherited. Use `uv lock --upgrade` only when an update of the whole environment is intended.
4. Run the consuming project's documented checks and compare affected outputs. Record the resolved Git commit and installed package versions. Check both the installed distribution version and `dse_research_utils.__version__`.
5. Apply the project's fit-compatibility rules before resuming or publishing stored fits. Keep historical manifests and recorded environments intact. A dependency update does not refit models, and passing library tests does not establish compatibility of every saved result.

Commit the dependency declaration, lockfile and any required adapters together. Adapters should preserve the project's schemas, labels, precision, missing-value rules and reporting decisions unless the upgrade deliberately changes them.

## Version notes

| Version                          | Changes to review                                                              |
| -------------------------------- | ------------------------------------------------------------------------------ |
| [0.17.0](migrating-to-0.17.md)   | Public diagnostic reductions, nullable diagnostics and file-permission options |
| [0.16.2](migrating-to-0.16.2.md) | Dependency minimums                                                            |
| [0.16.1](migrating-to-0.16.1.md) | Font fallback for plain-text symbols                                           |
| [0.16.0](migrating-to-0.16.md)   | Noto fonts and ArviZ minimums                                                  |
| [0.15.2](migrating-to-0.15.2.md) | Dependency minimums, including NumPy and PyTensor                              |
| [0.15.1](migrating-to-0.15.1.md) | Repository lock refresh without new package requirements                       |
| [0.15.0](migrating-to-0.15.md)   | Wider NumPy limit and Optuna 5 defaults                                        |
| [0.14.0](migrating-to-0.14.md)   | Shared file, statistical, reporting and evaluation APIs                        |
| [0.13.0](migrating-to-0.13.md)   | Corrected HSGP boundaries, diagnostic decisions and upload paths               |

## Completed reviews

The superseded working notes are retained in Git history. The merged pull requests record the changes and review evidence:

- [Research #100](https://github.com/dseinternational/research/pull/100) corrected the statistical, plotting and reporting defects released in 0.13.0.
- [Research #101](https://github.com/dseinternational/research/pull/101) added the shared helpers released in 0.14.0.
- [Research #117](https://github.com/dseinternational/research/pull/117) added the public diagnostic and permission APIs released in 0.17.0.
- [Language and reading #704](https://github.com/dseinternational/language-reading-predictors/pull/704), [vocabulary growth #384](https://github.com/dseinternational/vocabulary-growth/pull/384) and [US births #128](https://github.com/dspopulations/us-birth-certificates/pull/128) adopted the October refactors.
- [Language and reading #706](https://github.com/dseinternational/language-reading-predictors/pull/706), [vocabulary growth #386](https://github.com/dseinternational/vocabulary-growth/pull/386), [vocabulary growth #387](https://github.com/dseinternational/vocabulary-growth/pull/387) and [US births #130](https://github.com/dspopulations/us-birth-certificates/pull/130) adopted the public diagnostic and permission helpers.
