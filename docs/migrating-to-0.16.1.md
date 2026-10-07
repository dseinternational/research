> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5.5).
> Updated by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:ignore mathtext findfont -->

# Upgrade to 0.16.1

Version 0.16.1 fixes missing plain-text symbols in the 0.16.0 plot style. Arrows and mathematical relations that Noto Sans lacks now fall back to Noto Sans Math, then to DejaVu Sans. Dependency requirements and the Python 3.14 requirement are unchanged.

## Font fallback

`set_matplotlib_default_style()`, and therefore `init_workbook()` and `init_script()`, sets `font.family` to:

| Machine                  | Font list                                         |
| ------------------------ | ------------------------------------------------- |
| Noto Sans Math installed | `["sans-serif", "Noto Sans Math", "DejaVu Sans"]` |
| Noto Sans Math absent    | `["sans-serif", "DejaVu Sans"]`                   |

The generic `sans-serif` resolves to the first installed font in `font.sans-serif`. Matplotlib then tries the other families for missing glyphs. Noto Sans Math is included only when installed, to avoid repeated warnings about a missing named font. DejaVu Sans ships with matplotlib.

The release adds `default_font_families()` and `FONT_FAMILY_FALLBACK`. Code that applies `DEFAULT_STYLE_DICT` directly gets `["sans-serif", "DejaVu Sans"]`; `set_matplotlib_default_style()` adds Noto Sans Math when available. Mathtext settings are unchanged.

Fallback fonts may draw symbols at regular weight within medium or bold text, and matplotlib can log a font-weight warning. Ordinary text continues to use the selected text family. Inspect rendered output rather than treating every font warning as a missing symbol.

To override the list, set `font.family` after applying the style and before creating text. Existing titles and labels retain the font list used at creation. The [previous-font recipe](migrating-to-0.16.md#keep-the-previous-look) still applies.

## Upgrade

Follow the [shared upgrade procedure](README.md#upgrade-a-consuming-project), with `v0.16.1` as the target tag. Remove project overrides that only worked around the 0.16.0 symbol gap. Keep deliberate custom-font choices.

Regenerate figures with plain-text arrows or relations and check for missing-glyph warnings. Projects upgrading from 0.15.2 or earlier must also review the [0.16.0 font changes](migrating-to-0.16.md) and earlier requirements.
