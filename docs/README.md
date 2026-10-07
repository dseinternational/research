> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words HSGP -->

# Documentation

Start with the [repository readme](../README.md) for contributor setup or the [Python readme](../src/python/readme.md) for installation, extras and the package map.

## Usage guides

These APIs were introduced in 0.14.0. File-permission options require 0.17.0 or later.

| Guide                                                                             | Covers                                                                                                    |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [File writes and provenance](shared-file-provenance.md)                           | Single-file replacement, permission options, Git state, package versions and file hashes                  |
| [Sample arrays and predictive summaries](shared-predictive-arrays.md)             | Labelled observations, predictive intervals and ranks, likelihood aggregation                             |
| [Report assets and feature groups](shared-assets-and-feature-groups.md)           | Local HTML resources, upload inventories, HTTP availability and clustering                                |
| [Directory, report and evaluation helpers](shared-directory-report-evaluation.md) | Directory promotion, file reads, array intervals, fixed Gaussian-process geometry and permutation scoring |

## Upgrade a consuming project

The current release is `v0.17.0`. Read its [upgrade notes](migrating-to-0.17.md) and check the dependency requirements in that tag's `src/python/pyproject.toml`.

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

## Completed reviews

The superseded working notes are retained in Git history. The merged pull requests record the changes and review evidence:

- [Research #100](https://github.com/dseinternational/research/pull/100) corrected the statistical, plotting and reporting defects released in 0.13.0.
- [Research #101](https://github.com/dseinternational/research/pull/101) added the shared helpers released in 0.14.0.
- [Research #117](https://github.com/dseinternational/research/pull/117) added the public diagnostic and permission APIs released in 0.17.0.
- [Language and reading #704](https://github.com/dseinternational/language-reading-predictors/pull/704), [vocabulary growth #384](https://github.com/dseinternational/vocabulary-growth/pull/384) and [US births #128](https://github.com/dspopulations/us-birth-certificates/pull/128) adopted the October refactors.
- [Language and reading #706](https://github.com/dseinternational/language-reading-predictors/pull/706), [vocabulary growth #386](https://github.com/dseinternational/vocabulary-growth/pull/386), [vocabulary growth #387](https://github.com/dseinternational/vocabulary-growth/pull/387) and [US births #130](https://github.com/dspopulations/us-birth-certificates/pull/130) adopted the public diagnostic and permission helpers.
