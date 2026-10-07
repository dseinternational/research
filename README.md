> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# DSE research utilities

Shared libraries for research supported by [Down Syndrome Education International](https://www.down-syndrome.org/). The Python package, [`dse-research-utils`](src/python/readme.md), provides statistical, plotting, reporting and storage helpers. The [.NET area](src/dotnet/readme.md) has configuration but no projects.

Projects that use the Python package include [language-reading-predictors](https://github.com/dseinternational/language-reading-predictors), [vocabulary-growth](https://github.com/dseinternational/vocabulary-growth) and [us-birth-certificates](https://github.com/dspopulations/us-birth-certificates).

## Use the library

Install the published `v0.18.0` tag in a consuming project:

```bash
uv add "dse-research-utils @ git+https://github.com/dseinternational/research.git@v0.18.0#subdirectory=src/python"
```

The [Python readme](src/python/readme.md) lists optional extras and system requirements. The [documentation index](docs/README.md) links to usage guides and version-specific upgrade notes.

## Develop in this repository

Run these commands from the repository root. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first; it provisions Python 3.14 from `.python-version`.

```bash
uv sync --locked
uv run pytest
uv run ruff check src/python
uv run ruff format --check src/python
uv build --package dse-research-utils
```

The root `pyproject.toml` defines the contributor environment. The package requirements live in `src/python/pyproject.toml`. Consuming projects resolve their own dependencies and do not inherit this repository's lockfile or development groups.

The workspace resolves environments for Linux x86_64 and ARM64, Apple silicon macOS, and Windows x64. Windows runs natively. Intel macOS is outside the configured environments.

Use Node.js 24, as specified in `.nvmrc`, for the Markdown checks:

```bash
npm ci
npm run spellcheck
npm run format:check
```

`npm run format` formats tracked Markdown files, except `data/**/*.md`. It preserves paragraph line breaks, so write each paragraph on one line. Follow [the contributor and agent instructions](AGENTS.md) for source conventions, AI attribution and commit messages.

## Licence

The source headers and npm metadata specify `AGPL-3.0-or-later`. See the [GNU Affero General Public License](LICENSE).
