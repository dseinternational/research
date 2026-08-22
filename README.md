# Research

**\*Shared libraries and utilities for research supported by [Down Syndrome Education International](https://www.down-syndrome.org/).**

Current projects using these libraries include:

- [dseinternational/language-reading-predictors](https://github.com/dseinternational/language-reading-predictors)
- [dseinternational/vocabulary-growth](https://github.com/dseinternational/vocabulary-growth)
- [dspopulations/us-birth-certificates](https://github.com/dspopulations/us-birth-certificates)

## Getting started

The Python environment is managed with [uv](https://docs.astral.sh/uv/); conda is no longer used. uv provisions CPython 3.14 itself, so this is the whole setup:

```bash
uv sync                                # create .venv from uv.lock
uv run pytest                          # run the test suite
uv build --package dse-research-utils  # build the wheel
```

Windows is supported natively — WSL is no longer required. Intel macOS is not supported, because numba publishes no macOS x86_64 wheels. Plotting model graphs additionally needs the system Graphviz `dot` binary (`brew install graphviz`, `apt install graphviz`, `winget install Graphviz.Graphviz`).

Repositories still on the old conda environment should follow [docs/migrating-to-uv.md](docs/migrating-to-uv.md).

## License

All source code in this repository is licensed under the GNU Affero General Public License v3.0 **(AGPL-3.0-only)**. See `LICENSE`.

AGPL-3.0 requires that if you modify and run this software to provide a network service, you must offer the corresponding source code to users of that service.
