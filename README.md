> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Research

**\*Shared libraries and utilities for research supported by [Down Syndrome Education International](https://www.down-syndrome.org/).**

Current projects using these libraries include:

- [dseinternational/language-reading-predictors](https://github.com/dseinternational/language-reading-predictors)
- [dseinternational/vocabulary-growth](https://github.com/dseinternational/vocabulary-growth)
- [dspopulations/us-birth-certificates](https://github.com/dspopulations/us-birth-certificates)

## Getting started

The Python environment is managed with [uv](https://docs.astral.sh/uv/). uv provisions CPython 3.14 itself, so this is the whole setup:

```bash
uv sync                                # create .venv from uv.lock
uv run pytest                          # run the test suite
uv build --package dse-research-utils  # build the wheel
```

Windows is supported natively — WSL is no longer required. Intel macOS is not supported, because numba publishes no macOS x86_64 wheels. Plotting model graphs additionally needs the system Graphviz `dot` binary (`brew install graphviz`, `apt install graphviz`, `winget install Graphviz.Graphviz`). The default plot style and model graphs use the Noto Sans and Noto Sans Math system fonts (`brew install --cask font-noto-sans font-noto-sans-math`, `apt install fonts-noto-core`, or Google Fonts on Windows). Symbols that Noto Sans lacks, such as arrows and ≤, fall back to Noto Sans Math and then to DejaVu Sans, which ships with matplotlib. Without the fonts, text uses the next installed font in `font.sans-serif` and math uses DejaVu Sans.

The [0.16.2 upgrade notes](docs/migrating-to-0.16.2.md) describe the new dependency minimums and downstream upgrade steps. The [0.16.1 upgrade notes](docs/migrating-to-0.16.1.md) describe the font fallback that draws symbols Noto Sans lacks. The [0.16.0 upgrade notes](docs/migrating-to-0.16.md) describe the switch to Noto Sans and Noto Sans Math in the default plot style and the raised ArviZ minimums. The [0.15.2 upgrade notes](docs/migrating-to-0.15.2.md) describe the earlier raised minimum dependency versions and downstream upgrade steps. The [0.15.1 upgrade notes](docs/migrating-to-0.15.1.md) record the earlier lock refresh. The [0.15.0 upgrade notes](docs/migrating-to-0.15.md) describe the lifted NumPy ceiling and the Optuna 5.0 move. The [0.14.0 upgrade notes](docs/migrating-to-0.14.md) describe the shared-helper release sequence and installation. The [shared-helper migration guide](docs/consolidation-migration.md) explains how downstream projects can adopt the shared APIs in one dependency upgrade.

## License

All source code in this repository is licensed under the GNU Affero General Public License v3.0 **(AGPL-3.0-only)**. See `LICENSE`.

AGPL-3.0 requires that if you modify and run this software to provide a network service, you must offer the corresponding source code to users of that service.
