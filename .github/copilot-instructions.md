> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Copilot Instructions

Keep `AGENTS.md`, `CLAUDE.md` and `.github/copilot-instructions.md` identical except for the first heading.

## Repository scope

This repository provides shared research utilities for Down Syndrome Education International. The Python package is in `src/python/` and its uv workspace is at the repository root.

Read `README.md` for setup and checks, `src/python/readme.md` for the package map and defaults, and `docs/README.md` for usage and upgrade guides. Check the source before changing a documented API contract.

Consumers include `dseinternational/language-reading-predictors`, `dseinternational/vocabulary-growth` and `dspopulations/us-birth-certificates`. When working across repositories, locate the intended checkout and read its instructions first.

Keep observation and likelihood units, priors, interval coverage, missing-value rules, permutation donors, fit acceptance and publication decisions in the consuming project. Verify stored-fit compatibility using the project's own rules.

## AI attribution

Put an attribution callout with the actual AI tool and model before the title or body of document drafts, pull request and issue descriptions, and comments on pull requests or issues. Use this form for GitHub Markdown.

```markdown
> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).
```

For Quarto `.qmd` documents, use a Quarto callout.

```markdown
::: {.callout-note appearance="simple"}
Drafted by a LLM-based AI tool (Codex/GPT-6).
:::
```

## Markdown and commits

Write prose paragraphs on single lines and use blank lines to separate Markdown elements. See `README.md` for the format and spelling commands.

Follow Conventional Commits. Write the description as an imperative without a trailing period.

```text
<type>[optional scope]: <description>
```

Mark breaking changes with `!` before the colon or a `BREAKING CHANGE:` footer. Separate an optional explanatory body from the subject with a blank line.

## Dependencies and checks

Run commands from the repository root. Follow the setup and checks in `README.md`. For a focused test, use `uv run pytest path/to/test_file.py` or append `::test_function_name` to the file path.

- Declare shared requirements and extras in `src/python/pyproject.toml`. Consumers inherit those minimums. The root `pyproject.toml` defines the contributor environment. Run `uv lock` after changing dependency declarations and commit `uv.lock` with them.
- Use pnpm to change Markdown tooling dependencies and commit `pnpm-lock.yaml` with them.
- Use either `boosting` or `boosting-cpu`, since both provide `xgboost`. `all` includes `boosting`. See `src/python/readme.md` for system requirements for graphs and plots.

## Python conventions

Start every Python source file with these headers.

```python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
```

- Use `snake_case` for files, functions and variables, and `UPPER_CASE` for constants. Keep domain affixes such as `FIGSIZE_`, `DPI_` and `_COLOUR`.
- Annotate function signatures. Use `X | Y` for unions.
- Use NumPy-style docstrings with sections such as `Parameters` and `Returns`. Document dataclass fields with a string literal after each declaration.
- Use fully qualified absolute imports. Keep initialisers free of re-exports and executable definitions, except for `__version__` in `dse_research_utils/__init__.py`, which Hatch reads.
- Use standard-library dataclasses and `__post_init__` for validation.
- Use `get_console()` from `dse_research_utils.console.console` for library output, including on legacy encodings. Notebooks and scripts can use `from rich import print`.
- Follow existing plotting patterns. Create and draw the figure, optionally save to `output_dir` as PNG at 300 DPI and SVG, then return the figure. Keep figure lifetime explicit when using save helpers, which can close it.
- Use `plot.styles` for colours. The Python readme explains which palette to use. Update `plot/design_tokens/` by copying the released colour file from `dsegroup/apps-common` and updating `design-tokens.pin.json` together.
- Start notebooks with `init_workbook()` and scripts with `init_script()` from `environment.setup`. Both apply the default plot style. Notebooks also report environment information.
- Follow the Ruff settings in `src/python/pyproject.toml`, including the 120-character line limit and the annotation exemption for tests.
- Read reporting defaults from `ReportingConfiguration` and sampling presets from `statistics.models.sampling`. Pass interval coverage and kind explicitly when tables and figures must agree. Check convergence after sampling.

## .NET configuration

See `src/dotnet/readme.md` for the SDK, package sources and implementation status.
