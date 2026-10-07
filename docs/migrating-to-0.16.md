> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5.5).
> Updated by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:ignore mathtext fontname mathcal mathbb mathbf mathrm fontset stixsans dejavusans cachedir fontlist pyplot FIGSIZE bfit STIX ylabel -->

# Upgrade to 0.16.0

Version 0.16.0 changes the default plot fonts and model-graph fonts, and raises dependency minimums. The Python API and Python 3.14 requirement are unchanged. Review figures created after `set_matplotlib_default_style()`, `init_workbook()` or `init_script()`, and graphs from `model_to_graphviz()`.

## Default plot fonts

Text uses Noto Sans instead of Source Sans 3. Math text uses Noto Sans Math through matplotlib's custom font settings. `plot.styles` exposes these family names as `FONT_FAMILY_DEFAULT` and `FONT_FAMILY_MATH`.

| Setting                                       | 0.15.2                      | 0.16.0                    |
| --------------------------------------------- | --------------------------- | ------------------------- |
| First `font.sans-serif` choice                | Source Sans 3               | Noto Sans                 |
| `mathtext.fontset`                            | `dejavusans`                | `custom`                  |
| `mathtext.rm`, `mathtext.sf`, `mathtext.cal`  | DejaVu Sans                 | Noto Sans Math            |
| `mathtext.it`, `mathtext.bf`, `mathtext.bfit` | DejaVu Sans italic and bold | Noto Sans italic and bold |
| `mathtext.fallback`                           | STIX Sans                   | `stixsans`                |

Noto Sans Math has one upright face. Noto Sans supplies italic and bold math styles. Matplotlib's bundled STIX Sans supplies math glyphs that the custom font settings cannot reach, including `\mathbb` letters and larger delimiters. With these settings, `\mathcal{N}` renders as an upright N. Use the Unicode script character, for example `$𝒩(0, 1)$`, if that is the intended label.

Font metrics change. Review titles, tick labels and legends for clipping or overlap, particularly in small figures such as `FIGSIZE_XS` and `FIGSIZE_SM`.

### Symbols in plain text

The 0.16.0 style selects one font for plain text. Noto Sans lacks symbols such as arrows and mathematical relations, so literal labels containing → or ≤ can show missing-glyph boxes. [Version 0.16.1](migrating-to-0.16.1.md) adds fallback fonts. Upgrade to 0.16.1 or later when using these labels. If remaining on 0.16.0, write the symbols with mathtext:

```python
ax.set_ylabel(r"$P(Y \leq k)$")
ax.set_title(r"Sign $\rightarrow$ speech")
```

### Model graphs

`statistics.models.pymc_utils.model_to_graphviz` sets the graph, node and edge `fontname` attributes to `Noto Sans,sans-serif` instead of `Helvetica`. Font metrics can change node sizes and layouts. SVG viewers and raster renderers choose their installed fallback when Noto Sans is unavailable.

### Install the fonts

Install the system fonts on each machine that renders figures:

```bash
brew install --cask font-noto-sans font-noto-sans-math  # macOS
sudo apt install fonts-noto-core                     # Debian and Ubuntu
```

On Windows, install [Noto Sans](https://fonts.google.com/noto/specimen/Noto+Sans) and [Noto Sans Math](https://fonts.google.com/noto/specimen/Noto+Sans+Math) from Google Fonts. If matplotlib does not find newly installed fonts, delete `fontlist-*.json` from its cache directory and restart running kernels. Find the directory with:

```bash
uv run python -c "import matplotlib; print(matplotlib.get_cachedir())"
```

Without these fonts, ordinary text uses the next installed font in the style's list, math can produce font warnings and use DejaVu Sans, and Graphviz uses its system fallback. Use the same fonts in CI and on workstations when output must match.

### Keep the previous look

To retain the earlier text and math fonts, apply this override after the default style and before creating figures. Source Sans 3 must be installed.

```python
import matplotlib.pyplot as plt

from dse_research_utils.plot.styles import set_matplotlib_default_style

set_matplotlib_default_style()
plt.rcParams["font.sans-serif"] = ["Source Sans 3", *plt.rcParams["font.sans-serif"]]
plt.rcParams["mathtext.fontset"] = "dejavusans"
```

For model graphs, restore `Helvetica` in `graph_attr`, `node_attr` and `edge_attr` on the returned `Digraph` before rendering.

## Library requirements

| Package            | Previous minimum | New minimum | Scope  |
| ------------------ | ---------------- | ----------- | ------ |
| arviz-base         | 1.3.0            | 1.3.1       | Base   |
| arviz-plots        | 1.3.1            | 1.3.2       | Base   |
| arviz-stats        | 1.3.2            | 1.3.3       | Base   |
| azure-storage-blob | 12.30.1          | 12.30.3     | Base   |
| pandas             | 3.0.5            | 3.0.6       | Base   |
| networkx           | 3.6.1            | 3.7         | graphs |
| jax                | 0.11.1           | 0.11.2      | jax    |
| numpyro            | 0.21.0           | 0.22.0      | jax    |

NumPy retains `<2.6` and PyTensor retains `<3.4`. Consumers resolve their own lock files. A consumer that uses Optuna's database storage should test creation and resumption of a study against its resolved dependencies.

## Upgrade

Follow the [shared upgrade procedure](README.md#upgrade-a-consuming-project), with `v0.16.0` as the target tag. Install the fonts, regenerate affected figures and model graphs, and review their labels and layout. Apply the [0.15.2 notes](migrating-to-0.15.2.md) and earlier requirements when upgrading from an older version.
