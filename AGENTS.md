# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

> **Keep in sync:** `CLAUDE.md`, `AGENTS.md`, and `.github/copilot-instructions.md` must contain identical guidance (except for the first heading). When updating one, update all three.

## Overview

Shared library repository for [Down Syndrome Education International (DSE)](https://www.down-syndrome.org/) research projects. The primary artifact is `dse-research-utils`, a Python utility package (`src/python/`). A .NET utilities area (`src/dotnet/`) is a future placeholder.

## AI tool attribution

When you draft or author any of the following, you **must** prefix it with a callout that identifies the AI tool used. Place the callout as the first lines of the body:

- document drafts
- pull request descriptions
- issue descriptions
- comments on pull requests
- comments on issues

Substitute the actual tool and model you are running as (for example, `GitHub Copilot` or `Claude Code/Opus 4.8`). This requirement applies to every drafted artifact so that human reviewers can readily distinguish AI-generated content.

Use the callout syntax that matches where the text will render:

**GitHub surfaces** — pull request descriptions, issues, and their comments — use [GitHub alert syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#alerts):

```markdown
> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).
```

**Quarto documents** (`.qmd`) — GitHub alert syntax does **not** render in Quarto, so use a [Quarto callout block](https://quarto.org/docs/authoring/callouts.html) instead:

```markdown
::: {.callout-note appearance="simple"}
Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).
:::
```

Quarto supports five callout types (`note`, `tip`, `warning`, `caution`, `important`); use `note` for attribution. Drop `appearance="simple"` for the default boxed style, or add `collapse="true"` to make it collapsible.

## Markdown authoring

When drafting Markdown — repository documents, pull request descriptions, issue descriptions, or comments on either — write each paragraph as a single unwrapped line and avoid superfluous line breaks. Do not hard-wrap prose at a fixed column or scatter extra blank lines; let the renderer wrap the text. Prettier runs with `proseWrap: "preserve"`, so any manual wrapping is kept verbatim and produces noisy diffs.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for every commit — this applies to agents and humans alike.

- Subject line: `<type>[optional scope]: <description>` (for example `feat(plot): add credible-interval helper` or `fix: guard against zero variance`).
- Common types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- Keep the description in the imperative mood and concise, with no trailing period.
- Flag breaking changes with a `!` before the colon (for example `feat!:`) or a `BREAKING CHANGE:` footer.
- Add a body separated from the subject by a blank line when the change needs explanation.

## Repositories using these libraries

Current projects using these libraries include:

- [dseinternational/language-reading-predictors](https://github.com/dseinternational/language-reading-predictors)
- [dseinternational/vocabulary-growth](https://github.com/dseinternational/vocabulary-growth)
- [dspopulations/us-birth-certificates](https://github.com/dspopulations/us-birth-certificates)

To support developing across these repositories simultaneously, we typically check these out relative to this project as follows:

- ../language-reading-predictors
- ../vocabulary-growth
- ../../dspopulations/us-birth-certificates

## Commands

### Spellcheck

```bash
npm ci
npm run spellcheck      # runs cspell over all *.md files
```

### Markdown format

```bash
npm run format          # rewrites Markdown files in place
npm run format:check    # checks Markdown formatting without rewriting
```

Uses Prettier with `proseWrap: "preserve"` so existing prose line breaks are kept.

### Python — environment

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`, `winget install astral-sh.uv`, or the installer from the uv docs). uv provisions CPython 3.14 itself from `.python-version`, so there is no separate Python install step. Conda is no longer used — see `docs/migrating-to-uv.md`.

```bash
uv sync                 # create/refresh .venv from uv.lock (interpreter, library, extras, tooling)
uv sync --locked        # CI-style: fail rather than re-resolve when uv.lock is stale
```

Run anything inside the environment with `uv run <command>`, or activate `.venv` as usual.

`dse_research_utils.plot.graphs` additionally needs the system Graphviz `dot` binary, which is not a Python package: `brew install graphviz`, `apt install graphviz`, or `winget install Graphviz.Graphviz`.

### Python — dependencies

`src/python/pyproject.toml` is the single source of truth for the shared compiled core — the dependency floors every consuming repo inherits transitively by depending on `dse-research-utils`. Declare dependencies there, never by re-listing packages in a consuming repo:

- base `dependencies` — everything the library imports unconditionally, including the PyMC/PyTensor/nutpie stack
- extras — optional layers: `viz`, `graphs`, `notebook`, `dependence`, `tuning`, `io`, `jax`, `boosting`, `boosting-cpu`, `columnar`, `storage`, and `all` (every layer at once). `boosting-cpu` is the CPU-only variant of `boosting` for repos that do no GPU training — it swaps `xgboost` for `xgboost-cpu` off macOS, avoiding the CUDA payload the default Linux wheel carries. The two are mutually exclusive, so `all` takes `boosting`.

The root `pyproject.toml` is a uv workspace root that is never packaged or published; repo-only tooling and research packages live in its `dev` and `research` dependency groups. After changing any dependency run `uv lock` and commit the updated `uv.lock`.

### Python — build

```bash
uv build --package dse-research-utils   # builds wheel + sdist via hatchling
```

### Python — lint and format

```bash
uv run ruff check src/python            # lint
uv run ruff check src/python --fix      # lint with auto-fix
uv run ruff format src/python           # format
```

### Python — tests

```bash
uv run pytest                                           # full suite
uv run pytest path/to/test_file.py                      # single file
uv run pytest path/to/test_file.py::test_function_name  # single test
```

## Architecture

`src/python/src/dse_research_utils/` is structured by domain:

- **`environment/`** — system info, execution context; `init_workbook()` / `init_script()` for notebook/script setup; the configurable output-root resolver (`paths.OutputRoot`: CLI override > env var > repo default) and disk preflight (`disk.free_space_gb` / `preflight_disk`); `check.py` (the `dse-check-env` console script) is **deprecated**, retained only while consuming repos migrate off conda
- **`math/`** — constants (`EPSILON`, etc.)
- **`metadata/`** — package version introspection
- **`ml/`** — ML utilities (placeholder)
- **`plot/`** — matplotlib/ArviZ plotting helpers; the styled figure-save layer (`io.save_styled_figure` / `save_plot_data` / `save_plotcollection`: PNG + optional size-capped SVG sibling + optional data CSV); ArviZ subplot budgeting (`diagnostics_mcmc.capped_plot_var_names`); constants follow `FIGSIZE_XS`, `COLOUR_BLUE`, `DPI_NOTEBOOK` naming, plus `styles.categorical_palette`
- **`report/`** — report data-access helpers (`ReportData`, `show_or_pending`) that read a fitted model's artefacts and degrade to a visible "pending fit" placeholder before a fit exists
- **`statistics/`** — descriptive stats; credible/confidence intervals (`intervals.hdi_1d` / `eti_1d` / `eti_bands`, the `interval_1d` kind dispatcher, per-grid `bands`, and the tidy two-band `summarise_bands`); the shared evidence ladder (`evidence.py`: `evidence_label` / `odds_string` / `favoured_direction`); the ROPE report card (`rope.py`: `rope_card`); the MCMC convergence gate, banner, and styled diagnostics table (`diagnostics.py`, including the `diagnostics_assessable` check and `amend_diagnostics_summary`); unrounded sampling-quality signal extraction (`sampling_quality.py`); PSIS-LOO/ELPD helpers (`loo.py`: reff pinning, Pareto-k reductions, the canonical LOO summary row, and the `elpd_verdict` convention); and PyMC models and sampling presets

All `__init__.py` files are empty — no re-exports. Use fully-qualified absolute imports everywhere (e.g. `from dse_research_utils.math.constants import EPSILON`).

`statistics/models/reporting.ReportingConfiguration` carries the reporting `ci_prob` (credible-interval coverage) and `interval_kind` (`"eti"` equal-tailed or `"hdi"` highest-density); reports read both back so tables, plots, and the diagnostics summary agree on the interval convention. Credible-interval coverage defaults to **0.89** across the shared helpers (`hdi_1d`, `eti_1d`, `rope_card`, `ReportingConfiguration.ci_prob`, …), matching ArviZ's `rcParams["stats.ci_prob"]`; a report that wants a different width passes it explicitly (e.g. `vocabulary-growth` uses 0.90, `language-reading-predictors` uses 0.95).

Bayesian sampling presets in `statistics/models/sampling.py`, selected via `get_sampling_configuration(config)`:

- `dev` / `development` — 2 chains × 500 draws, `target_accept=0.85` (fast iteration)
- `test` / `testing` — 4 chains × 2000 draws, `target_accept=0.90`
- `rep-lite` / `reporting-lite` / `rep_lite` — 4 chains × 4000 draws, `target_accept=0.95`
- `reporting` / `report` / `rep` — 6 chains × 6000 draws, `target_accept=0.95`

## Python Conventions

**License header** — every source file starts with:

```python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
```

- **Naming**: `snake_case` for files/functions/variables; `UPPER_CASE` for constants grouped by domain (e.g. `FIGSIZE_XS`, `COLOUR_BLUE`, `DPI_NOTEBOOK`)
- **Type hints**: always on function signatures
- **Type unions**: `X | Y` syntax (not `Union[X, Y]`)
- **Docstrings**: NumPy-style (`Parameters`, `Returns`, `---` separators); dataclass fields use attribute docstrings (bare string literals after the field declaration)
- **`print`**: library code prints through the shared console — `from dse_research_utils.console.console import get_console`, then `get_console().print(...)`. It is the one place that relaxes the error handler on a stdout that is not UTF-8, so a legacy Windows code page degrades the output instead of raising `UnicodeEncodeError`. In notebooks and scripts, import `from rich import print` to override the built-in.
- **Imports**: fully-qualified absolute imports only — no relative imports, no `__init__.py` re-exports. The package root `__init__.py` is the one exception: it stores `__version__` for Hatch.
- **Dataclasses**: stdlib `@dataclass`; use `__post_init__` for validation
- **Plot functions**: create figure → render → optionally save to `output_dir` as `.png` (300 DPI) and `.svg` → `return plt.gcf()`
- **Notebooks**: call `init_workbook()` at top (prints environment info); scripts call `init_script()` (silent setup); both apply the default matplotlib style via `set_matplotlib_default_style()`

Ruff config (in `src/python/pyproject.toml`): line-length 120, target Python 3.14, rules: `F`, `E`, `W`, `I`, `UP`, `B`, `SIM`, `RUF`, `ANN` (source only; tests ignore annotation rules).

## .NET

SDK pinned to **10.0.200** (`rollForward: latestMinor`) via `global.json`. Test runner is `Microsoft.Testing.Platform`. NuGet sources: `nuget.org` (all packages) and `dseinternational` (`https://nuget.pkg.github.com/dseinternational/index.json`, `DSE.*` packages only) — package source mapping enforced with `<clear />`. No projects exist yet.
