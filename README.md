> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# DSE research utilities

Shared libraries for research supported by [Down Syndrome Education International](https://www.down-syndrome.org/). The Python package, [`dse-research-utils`](src/python/readme.md), provides statistical, plotting, reporting and storage helpers. The [.NET area](src/dotnet/readme.md) has configuration but no projects.

Projects that use the Python package include [language-reading-predictors](https://github.com/dseinternational/language-reading-predictors), [vocabulary-growth](https://github.com/dseinternational/vocabulary-growth) and [us-birth-certificates](https://github.com/dspopulations/us-birth-certificates).

## Use the library

To install the `v0.19.4` release, run this in your project's root.

```bash
uv add "dse-research-utils @ git+https://github.com/dseinternational/research.git@v0.19.4#subdirectory=src/python"
```

See the [Python readme](src/python/readme.md) for extras and system requirements, and the [documentation index](docs/README.md) for usage guides and the upgrade procedure.

## Develop in this repository

Run the contributor commands from this repository's root. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then use the locked workspace. uv uses the Python version in `.python-version`.

```bash
uv sync --locked
uv run pytest
uv run ruff check src/python
uv run ruff format --check src/python
uv build --package dse-research-utils
```

The root `pyproject.toml` defines the contributor workspace and its development groups. Each consumer resolves the package requirements from `src/python/pyproject.toml` in its own lockfile.

`uv.lock` covers Linux x86_64 and ARM64, Apple silicon macOS, and native Windows x64. CI runs the Python tests on Linux ARM64 and Windows x64.

For Markdown, use the Node.js version in `.nvmrc` and the pnpm version in `package.json`. After installing [pnpm](https://pnpm.io/installation), run the following checks.

```bash
pnpm install --frozen-lockfile
pnpm run spellcheck
pnpm run format:check
```

Use `pnpm run format` to format tracked Markdown outside `data/**/*.md`. Follow the [contributor and agent instructions](AGENTS.md) for paragraph wrapping, source conventions, AI attribution and commits.

Cspell settings live in `cspell.yaml`. Add project terms to `cspell-words.txt`, one word per line.

## Licence

The source headers and package metadata specify `AGPL-3.0-or-later`. See the [GNU Affero General Public License](LICENSE).
