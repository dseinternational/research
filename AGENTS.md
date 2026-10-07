> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# AGENTS.md

Keep `AGENTS.md`, `CLAUDE.md` and `.github/copilot-instructions.md` identical except for their first heading. Update all three when changing this guidance.

## Repository scope

This repository provides shared utilities for Down Syndrome Education International research projects. The Python package lives in `src/python/`; the root is its uv development workspace. The .NET area has no projects yet.

Read the [repository readme](https://github.com/dseinternational/research/blob/main/README.md) for setup and checks, the [Python readme](https://github.com/dseinternational/research/blob/main/src/python/readme.md) for the package map and defaults, and the [documentation index](https://github.com/dseinternational/research/blob/main/docs/README.md) for usage and migration guides. Check the source before changing a documented API contract.

The consumers include `dseinternational/language-reading-predictors`, `dseinternational/vocabulary-growth` and `dspopulations/us-birth-certificates`. Their locations are not fixed. When working across repositories, locate the intended checkout and read its instructions first.

Keep scientific and reporting choices in the consuming project. These include observation and likelihood units, priors, interval coverage, missing-value rules, permutation donors, fit acceptance and publication decisions. A shared helper or passing library test does not establish compatibility of a consumer's stored fits.

## AI attribution

Prefix document drafts, pull request descriptions, issue descriptions and comments on pull requests or issues with a callout that identifies the actual AI tool and model. Place it before the document title or other body text. Use this form for Markdown rendered on GitHub:

```markdown
> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).
```

For Quarto `.qmd` documents, use a Quarto callout:

```markdown
::: {.callout-note appearance="simple"}
Drafted by a LLM-based AI tool (Codex/GPT-6).
:::
```

Substitute the tool and model used for the draft. GitHub alert syntax does not render as a Quarto callout.

## Markdown and commits

Write each prose paragraph on one unwrapped line. Use only the blank lines needed to separate Markdown elements. Prettier preserves prose wrapping. `npm run format` formats tracked Markdown except `data/**/*.md`; `npm run format:check` checks it. `npm run spellcheck` uses cspell, whose configuration excludes the three agent instruction files.

Every commit must follow Conventional Commits. Use an imperative subject without a trailing period:

```text
<type>[optional scope]: <description>
```

Types include `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore` and `revert`. Mark a breaking change with `!` before the colon or a `BREAKING CHANGE:` footer. Add a body after a blank line when explanation is needed.

## Dependencies and checks

Run commands from the repository root. `uv sync --locked` installs the committed environment; `uv run` runs commands in it. The [readme](https://github.com/dseinternational/research/blob/main/README.md#develop-in-this-repository) lists Python, build and Markdown checks. Use `uv run pytest path/to/test_file.py` or `uv run pytest path/to/test_file.py::test_function_name` for focused tests.

Declare library requirements and extras in `src/python/pyproject.toml`. Keep their minimum versions there rather than copying that list into consumers. The root `pyproject.toml` is not packaged; its `dev` and `research` groups are repository tooling. Run `uv lock` after changing dependency declarations and commit `uv.lock` with them.

The `boosting` and `boosting-cpu` extras both provide `xgboost` and must not be combined. The `all` extra selects `boosting`. The [Python readme](https://github.com/dseinternational/research/blob/main/src/python/readme.md#system-requirements-for-plots) covers the Graphviz executable and plot fonts, which are system requirements rather than Python packages.

## Python conventions

Every source file starts with:

```python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
```

- Use `snake_case` for files, functions and variables, and `UPPER_CASE` for constants. Keep domain affixes such as `FIGSIZE_`, `DPI_` and `_COLOUR`.
- Annotate function signatures. Use `X | Y` for unions.
- Use NumPy-style docstrings with sections such as `Parameters` and `Returns`. Document dataclass fields with a string literal after each declaration.
- Use fully qualified absolute imports. Do not add `__init__.py` re-exports. The package root stores `__version__` for Hatch; other package initialisers contain no executable definitions.
- Use standard-library dataclasses and `__post_init__` for validation.
- Print library output through `get_console()` from `dse_research_utils.console.console`. The shared console handles characters that legacy output encodings cannot represent. Notebooks and scripts can use `from rich import print`.
- Follow existing plotting patterns. Create and draw the figure, optionally save to `output_dir` as PNG at 300 DPI and SVG, then return the figure. Keep figure lifetime explicit when using save helpers, which can close it.
- Take plot colours from `plot.styles`, which reads the DSE design tokens vendored in `plot/design_tokens/`: `CHART_COLOURS` or `categorical_palette()` for up to six categories, `sequential_palette()` or `diverging_palette()` for three to five ordered values, and `SEQUENTIAL_CMAP` or `DIVERGING_CMAP` for continuous scales. The tokens are generated by `dsegroup/apps-common` and pinned in `design-tokens.pin.json`; never edit them by hand, and bump them by copying the released file and updating the pin.
- Call `init_workbook()` at the start of notebooks and `init_script()` at the start of scripts. Both apply the default plot style; workbook setup also reports environment information.

Ruff settings in `src/python/pyproject.toml` specify Python 3.14, a 120-character line limit and the enabled rules. Tests omit annotation rules. Check lint with `uv run ruff check src/python` and formatting with `uv run ruff format --check src/python`.

Read reporting defaults from `ReportingConfiguration` and sampling presets from `statistics.models.sampling`. Pass interval coverage and kind explicitly when a report must use the same convention in its tables and figures. Preset draw counts do not guarantee convergence.

## .NET configuration

[`global.json`](https://github.com/dseinternational/research/blob/main/global.json) sets the SDK and test runner. [`NuGet.config`](https://github.com/dseinternational/research/blob/main/NuGet.config) defines package sources and mapping. The [.NET readme](https://github.com/dseinternational/research/blob/main/src/dotnet/readme.md) records the current implementation status.
