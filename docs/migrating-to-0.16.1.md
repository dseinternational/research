> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5.5).

<!-- cspell:ignore mathtext findfont rcParam rightarrow -->

# Upgrade to 0.16.1

Version 0.16.1 fixes a gap in the 0.16.0 default plot style ([#112](https://github.com/dseinternational/research/issues/112)). Plain text that contained an arrow, a mathematical relation or another symbol that Noto Sans lacks drew a missing-glyph box in place of the symbol. The style now draws those symbols from Noto Sans Math, then from DejaVu Sans. Figures that contain none of them render exactly as they did under 0.16.0, pixel for pixel. The dependency requirements and the Python 3.14 requirement are unchanged. Publish `v0.16.1` on the merged release commit after its checks pass; downstream upgrades require that tag to exist.

## Symbols in plain text

The [0.16.0 upgrade notes](migrating-to-0.16.md#symbols-in-plain-text) describe the gap. `set_matplotlib_default_style()`, and therefore `init_workbook()` and `init_script()`, now sets `font.family` to a list of families instead of the generic `sans-serif`:

| Machine                      | 0.16.0       | 0.16.1                                            |
| ---------------------------- | ------------ | ------------------------------------------------- |
| Noto Sans Math installed     | `sans-serif` | `["sans-serif", "Noto Sans Math", "DejaVu Sans"]` |
| Noto Sans Math not installed | `sans-serif` | `["sans-serif", "DejaVu Sans"]`                   |

matplotlib falls back to another font glyph by glyph, across the families named in `font.family`. The generic `sans-serif` still resolves to the first installed font in `font.sans-serif`, which is Noto Sans wherever it is installed, so ordinary text does not change. A symbol that font lacks comes from Noto Sans Math, which is designed to pair with Noto Sans and has the arrows, relations, operators and geometric shapes. DejaVu Sans ships with matplotlib and covers most of what remains, such as ✗ and other dingbats.

Noto Sans Math is listed only when it is installed. matplotlib logs `findfont: Font family 'Noto Sans Math' not found.` each time it lays out text with a named family that it cannot find, several hundred times for a simple figure. The list starts with the generic `sans-serif` rather than naming Noto Sans for the same reason. Without Noto Sans, text still falls back silently to the next installed font in `font.sans-serif`, and the [recipe for keeping the previous look](migrating-to-0.16.md#keep-the-previous-look) still works.

`plot.styles` gains two public names, and one setting changes:

- `default_font_families()` returns the `font.family` list for the current machine. A project that applies the style's fonts without the rest of the style can set `font.family` from it.
- `FONT_FAMILY_FALLBACK` names DejaVu Sans.
- `DEFAULT_STYLE_DICT["font.family"]` is now `["sans-serif", "DejaVu Sans"]`. Every machine has DejaVu Sans, so code that applies the dictionary directly draws these symbols from DejaVu Sans instead of as boxes. `set_matplotlib_default_style()` then adds Noto Sans Math where it is installed.

Mathtext is unaffected. Symbols such as `$\leq$` and `$\rightarrow$` still come from Noto Sans Math through the mathtext settings, so labels already rewritten as mathtext under 0.16.0 need no change.

### Weight warnings

Noto Sans Math has a single regular face, and DejaVu Sans has no medium weight. The style sets titles and axis labels in medium and figure titles in bold. matplotlib therefore logs lines like the following the first time a process needs each weight at each size, typically three lines in all:

```text
findfont: Failed to find font weight medium for Noto Sans Math, now using 400.
findfont: Failed to find font weight medium for DejaVu Sans, now using 400.
findfont: Failed to find font weight bold for Noto Sans Math, now using 400.
```

They are logged even for figures without fallback symbols, because matplotlib resolves every family in the list when it lays out text. They affect only fallback symbols, which draw at regular weight within medium or bold text.

### Override the list

A project that needs a different list sets `font.family` after applying the style, because `set_matplotlib_default_style()` resets it, and before creating the figure. Text keeps the font list that was in effect when it was created, so a list set afterwards does not reach existing titles and labels.

## Upgrade a downstream repository

1. Change the research source tag from `v0.16.0` to `v0.16.1` in the downstream `pyproject.toml`. Preserve its existing extras and other dependency constraints.
2. Resolve the new library version and install the resulting environment.

```bash
uv lock --upgrade-package dse-research-utils
uv lock --check
uv sync --locked
```

3. Remove any project-level `font.family` override that worked around this gap. An override that names Noto Sans itself, such as `["Noto Sans", "Noto Sans Math", "DejaVu Sans"]`, logs `findfont: Font family 'Noto Sans' not found.` for every text element on a machine without Noto Sans, including CI runners that install no fonts.
4. Regenerate figures whose plain text contains symbols that Noto Sans lacks, and check that no `missing from font(s) Noto Sans` warnings remain. Other figures are unchanged.
5. Run the downstream repository's tests. Verify that the installed library version is 0.16.1, and record the resolved tag commit.

Projects upgrading from 0.15.2 or earlier must also apply the [0.16.0 upgrade notes](migrating-to-0.16.md) and any earlier notes they have not yet applied.
