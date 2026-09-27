> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5.5).

<!-- cspell:ignore mathtext bfit fontname Pango mathcal mathbb mathbf mathrm fontset stixsans dejavusans cachedir fontlist findfont greenlet pyplot FIGSIZE STIX -->

# Upgrade to 0.16.0

Version 0.16.0 changes the fonts in the default plot style and in model graphs, and raises three ArviZ dependency minimums. The Python API and the Python 3.14 requirement are unchanged, but figures drawn with `set_matplotlib_default_style()` (and therefore `init_workbook()` and `init_script()`) and graphs from `model_to_graphviz()` look different. That visible change is why this is a minor release rather than a patch. Publish `v0.16.0` on the merged release commit after its checks pass; downstream upgrades require that tag to exist.

## Default plot fonts

Text now uses Noto Sans instead of Source Sans 3. Math text now uses Noto Sans Math instead of matplotlib's DejaVu Sans math fontset. `plot.styles` exposes the two family names as `FONT_FAMILY_DEFAULT` and `FONT_FAMILY_MATH`.

| Setting                                       | 0.15.2                         | 0.16.0                    |
| --------------------------------------------- | ------------------------------ | ------------------------- |
| `font.sans-serif` (first choice)              | Source Sans 3                  | Noto Sans                 |
| `mathtext.fontset`                            | `dejavusans` (matplotlib)      | `custom`                  |
| `mathtext.rm`, `mathtext.sf`, `mathtext.cal`  | DejaVu Sans                    | Noto Sans Math            |
| `mathtext.it`, `mathtext.bf`, `mathtext.bfit` | DejaVu Sans italic and bold    | Noto Sans italic and bold |
| `mathtext.fallback`                           | STIX Sans (built into fontset) | `stixsans`                |

Noto Sans Math has a single upright face. The italic and bold math styles therefore come from Noto Sans, the text family Noto Sans Math is designed to pair with. Variables stay italic and `\mathbf` stays bold. Operators, digits, relations, Greek capitals and `\mathrm` come from Noto Sans Math. STIX Sans, which ships with matplotlib, supplies the glyphs that mathtext cannot reach in Noto Sans Math, such as `\mathbb` letters and the larger delimiters used by `\left` and `\right`. It was also the fallback of the previous DejaVu Sans fontset.

Two rendering differences need attention:

- `\mathcal{N}` renders as an upright N. matplotlib's custom fontset cannot select the script letters in Noto Sans Math. Type the Unicode character instead (`$𝒩(0, 1)$`) for a script letter.
- Noto Sans is wider than Source Sans 3. Titles, tick labels and legends that fitted before can now clip or overlap, most visibly in `FIGSIZE_XS` and `FIGSIZE_SM` figures at the default 12 pt size.

### Model graphs

`statistics.models.pymc_utils.model_to_graphviz` now sets the graph, node and edge `fontname` to `Noto Sans,sans-serif` instead of `Helvetica`. Graphviz copies `fontname` directly into the `font-family` of SVG output, so the generic `sans-serif` gives viewers without Noto Sans a sans-serif fallback, as `Helvetica,sans-Serif` did before. PNG output reads the same family list through Pango. Noto Sans has taller line spacing than Helvetica, so nodes are slightly larger and graph layouts can shift.

### Install the fonts

The fonts are system fonts, not Python packages, so each machine that renders figures needs them installed:

```bash
brew install --cask font-noto-sans font-noto-sans-math   # macOS
sudo apt install fonts-noto-core                         # Debian and Ubuntu
```

On Windows, install Noto Sans and Noto Sans Math from Google Fonts. matplotlib caches its font list, so a newly installed font is not found until the cache is rebuilt. Delete `fontlist-*.json` from the directory printed by the following command, and restart any running kernels:

```bash
uv run python -c "import matplotlib; print(matplotlib.get_cachedir())"
```

Without the fonts, matplotlib logs a `findfont` warning and falls back to DejaVu Sans for both text and math, and Graphviz uses the system's default sans-serif font. CI images and containers that render figures therefore need the fonts installed, or their output will not match a workstation's.

### Keep the previous look

A project that needs to keep the old fonts, for example while its figures are under review, can restore them after applying the style:

```python
import matplotlib.pyplot as plt

from dse_research_utils.plot.styles import set_matplotlib_default_style

set_matplotlib_default_style()
plt.rcParams["font.sans-serif"] = ["Source Sans 3", *plt.rcParams["font.sans-serif"]]
plt.rcParams["mathtext.fontset"] = "dejavusans"
```

For model graphs, set the three `fontname` attributes on the returned `Digraph` (`graph_attr`, `node_attr` and `edge_attr`) back to `Helvetica` before rendering.

## Library requirements

These changes are part of the package metadata, so a downstream repository must satisfy them when it selects 0.16.0.

| Package     | Previous minimum | New minimum | Scope |
| ----------- | ---------------- | ----------- | ----- |
| arviz-base  | 1.3.0            | 1.3.1       | Base  |
| arviz-plots | 1.3.1            | 1.3.2       | Base  |
| arviz-stats | 1.3.2            | 1.3.3       | Base  |

Every other declared minimum was already at the latest release on PyPI when checked on 27 September 2026. NumPy keeps its `<2.6` limit and PyTensor keeps `<3.4`, because NumPy 2.5.3, numba 0.67.0 and PyTensor 3.3.2 are still the latest releases.

## Repository tooling

The development minimum moves to Ruff 0.16.9, and the research group now requires pingouin 0.7.0. These groups are not included in the library's published dependency metadata. The release also contains cspell 10.3.4.

The refreshed Python lock moves SQLAlchemy from 2.0.54 to 2.1.1. SQLAlchemy is a transitive dependency of Optuna, in the `tuning` extra, and version 2.1 drops greenlet as a default dependency. An Optuna study created, run and reloaded through SQLite storage worked on the new lock. A consumer that uses Optuna's RDB storage against another database should still run its own storage checks.

## Upgrade a downstream repository

1. Change the research source tag from `v0.15.2` to `v0.16.0` in the downstream `pyproject.toml`. Preserve its existing extras and other dependency constraints.
2. Resolve the new library requirement and install the resulting environment.

```bash
uv lock --upgrade-package dse-research-utils
uv lock --check
uv sync --locked
```

3. Install Noto Sans and Noto Sans Math on every machine and CI image that renders figures, then clear the matplotlib font cache as described above.
4. Regenerate the project's figures and model graphs and review them, particularly small figures, long titles, legends and any labels that use `\mathcal`. Saved PNG and SVG outputs will differ from those produced with 0.15.2 even where the data is unchanged.
5. Run the downstream repository's tests. Verify that the installed library version is 0.16.0, and record the resolved tag commit and package versions.

Projects upgrading from before 0.15.2 must also apply the [0.15.2 upgrade notes](migrating-to-0.15.2.md) and any earlier notes they have not yet applied.
