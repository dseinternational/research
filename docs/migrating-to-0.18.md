> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5.5).

<!-- cspell:words cmap CMAP Colormap hasattr jupytext RGBA rgba viridis -->

# Upgrade to 0.18.0

Version 0.18.0 takes the plot colours from the DSE design tokens, so a group has the same colour in a research figure as on DSE's websites. Default figure colours change. `categorical_palette` has two breaking changes, and the named `COLOUR_*` hues are deprecated.

Python 3.14 and the base dependency requirements are unchanged from `v0.17.0`. The `notebook` extra, and so `all`, now requires jupytext 1.19.6 or later; the other extras are unchanged. No statistical, reporting or storage behaviour changes, so stored fits and their compatibility rules are unaffected.

## Colours that change without code changes

The default style, applied by `set_matplotlib_default_style()`, `init_script()` and `init_workbook()`, changes these colours.

| Setting                                | 0.17.0               | 0.18.0                                                      |
| -------------------------------------- | -------------------- | ----------------------------------------------------------- |
| Colour cycle (`axes.prop_cycle`)       | matplotlib's `tab10` | `CHART_COLOURS`: blue, green, orange, purple, teal and pink |
| `image.cmap`                           | `tab10`              | `dse_sequential`                                            |
| `plot_heatmap` default scale           | `viridis`            | `dse_sequential`                                            |
| `TEXT_COLOUR`                          | `#333333`            | `#202020`                                                   |
| `LINE_COLOUR`                          | `#c0c0c0`            | `#d2d8df`, lighter: 1.4:1 rather than 1.8:1 against white   |
| Figure, axes and patch-edge background | white                | `BACKGROUND_COLOUR`, which is white                         |

Colour strings such as `"C1"` follow the cycle. `"C1"` was orange and is now green, `"C3"` was red and is now purple, and `"C6"` onwards repeat the six colours from `"C0"`. Check figures that rely on a hue's meaning, such as red for a failure, or on two `"C"` colours that now look alike.

## Changes that can need code

### Categorical palettes

`categorical_palette(n)` without a `palette` returns the first `n` of `CHART_COLOURS`.

- It raises `ValueError` for more than six series, because the design language allows six categorical colours. Group the smallest categories, use small multiples or label series directly. A named matplotlib palette keeps the old behaviour for a figure outside the design language, for example `categorical_palette(12, palette="tab20")`.
- It returns hex strings, not RGBA tuples. Matplotlib accepts either. Code that reads a colour's channels must first convert it with `matplotlib.colors.to_rgba`.

### Deprecated named hues

Reading a `COLOUR_*` name, including through `from dse_research_utils.plot.styles import COLOUR_BLUE` or `hasattr`, raises `DeprecationWarning`. Python shows the warning by default only when the code reading the name runs as `__main__`, such as a script, so a use in a package module usually appears only under pytest.

| Name                                                            | Returns in 0.18.0                        | Replace with                                                |
| --------------------------------------------------------------- | ---------------------------------------- | ----------------------------------------------------------- |
| `COLOUR_BLUE`, `COLOUR_GREEN`, `COLOUR_ORANGE`, `COLOUR_PURPLE` | `CHART_COLOURS[0]` to `CHART_COLOURS[3]` | The same `CHART_COLOURS` entry                              |
| `COLOUR_RED`, `COLOUR_YELLOW`, `COLOUR_DARK_*`                  | Their 0.17.0 values                      | A `CHART_COLOURS` entry in series order, or an ordered step |

Blue, green, orange and purple already draw in the new colours, so replacing those names changes no figure. The others need a choice. Use the next chart colour for another series, a diverging step for signed values, or `TEXT_COLOUR` for an outline.

### Reference lines and signed matrices

Draw a reference line that carries meaning, such as zero, an identity diagonal or a threshold, in `MUTED_TEXT_COLOUR`. Annotation text belongs in `TEXT_COLOUR` or `MUTED_TEXT_COLOUR`. `LINE_COLOUR` is now too light for either and is for grid lines.

The sequential default puts negative values at the quiet end of the scale. Pass `centre=0.0` to `plot_heatmap` for correlations and other signed matrices. Replace an explicit `viridis` for magnitudes with `SEQUENTIAL_CMAP`, or leave the colour map out.

## New names

`dse_research_utils.plot.styles` adds these names.

- `CHART_COLOURS`, the six categorical colours in order.
- `sequential_palette(n)` and `diverging_palette(n)`, three to five ordered steps. Sequential runs from least to most in blue. Diverging runs from the low or negative end in orange to the high or positive end in blue, with a grey middle step when `n` is odd. `SEQUENTIAL_PALETTES` and `DIVERGING_PALETTES` hold the same steps by count.
- `SEQUENTIAL_CMAP` and `DIVERGING_CMAP`, registered as `dse_sequential` and `dse_diverging` with `_r` reversals.
- `MUTED_TEXT_COLOUR` for secondary text and meaningful reference lines, and `BACKGROUND_COLOUR`.

Every categorical colour and ordered step reaches 3:1 against the white background. `plot_heatmap` adds a `centre` argument, and its `cmap` argument accepts a `Colormap`. The [Python readme](../src/python/readme.md#plot-colours) describes the colours.

## Upgrade

Follow the [shared upgrade procedure](README.md#upgrade-a-consuming-project), with `v0.18.0` as the target tag. A project that selects the `notebook` or `all` extra resolves jupytext 1.19.6 or later.

1. Find the affected calls, for example with `git grep -nE "COLOUR_|categorical_palette|plot_heatmap|viridis|tab10|\"C[0-9]\""`.
2. Replace the deprecated names and choose colours for red, yellow and the dark hues. Give one entity the same colour in every figure.
3. Run the project's tests with the warnings promoted to errors, for example `uv run pytest -W "error:styles.COLOUR:DeprecationWarning"`. Import the scripts that tests do not cover in the same way.
4. Regenerate representative figures. Check that legends, labels and markers still distinguish series, and that colour is never the only signal.

Committed figures and reports change on their next render. A dependency upgrade does not refit models.
