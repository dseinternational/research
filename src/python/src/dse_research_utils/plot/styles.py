# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
from matplotlib import font_manager

FIGSIZE_XS = (2.913, 2.060)  # 74mm x 52mm (A8 landscape)
FIGSIZE_SM = (4.134, 2.923)  # 105mm x 74mm (A7 landscape)
FIGSIZE_MD = (5.000, 3.536)  # 127mm x 90mm
FIGSIZE_LG = (5.827, 4.120)  # 148mm x 105mm (A6 landscape)
FIGSIZE_XL = (8.268, 5.846)  # 210mm x 148.5mm (A5 landscape)
FIGSIZE_XXL = (11.693, 8.268)  # 297mm x 210mm (A4 landscape)

FIGSIZE_XS_PORTRAIT = (2.060, 2.913)  # 52mm x 74mm (A8 portrait)
FIGSIZE_SM_PORTRAIT = (2.923, 4.134)  # 74mm x 105mm (A7 portrait)
FIGSIZE_MD_PORTRAIT = (3.536, 5.000)  # 90mm x 127mm
FIGSIZE_LG_PORTRAIT = (4.120, 5.827)  # 105mm x 148mm (A6 portrait)
FIGSIZE_XL_PORTRAIT = (5.846, 8.268)  # 148.5mm x 210mm (A5 portrait)
FIGSIZE_XXL_PORTRAIT = (8.268, 11.693)  # 210mm x 297mm (A4 portrait)

DPI_NOTEBOOK = 120.0
DPI_FILE = 300.0

COLOUR_BLUE = "#1f77b4"
COLOUR_ORANGE = "#ff7f0e"
COLOUR_GREEN = "#2ca02c"
COLOUR_RED = "#d62728"
COLOUR_YELLOW = "#ffbb00"
COLOUR_PURPLE = "#9467bd"

COLOUR_DARK_BLUE = "#014b7f"
COLOUR_DARK_ORANGE = "#ef7001"
COLOUR_DARK_GREEN = "#036903"
COLOUR_DARK_RED = "#910202"
COLOUR_DARK_YELLOW = "#ad7f01"
COLOUR_DARK_PURPLE = "#503a64"

TEXT_COLOUR = "#333333"

LINE_COLOUR = "#c0c0c0"

FONT_SIZE_DEFAULT = 12

FONT_FAMILY_DEFAULT = "Noto Sans"
FONT_FAMILY_MATH = "Noto Sans Math"
FONT_FAMILY_FALLBACK = "DejaVu Sans"  # ships with matplotlib, so always installed

DEFAULT_STYLE_DICT = {
    # Figure Settings
    "figure.figsize": FIGSIZE_MD,
    "figure.dpi": DPI_NOTEBOOK,
    "savefig.dpi": DPI_FILE,
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
    "figure.titlesize": FONT_SIZE_DEFAULT,
    "figure.titleweight": "bold",
    # Font and Text
    # matplotlib resolves the generic "sans-serif" to one font, the first
    # installed entry of font.sans-serif, and falls back glyph by glyph only
    # across the families listed in font.family. Noto Sans has no arrows or
    # mathematical relations (→ ≈ ≤ ✓), so DejaVu Sans draws them.
    # set_matplotlib_default_style() puts Noto Sans Math ahead of DejaVu Sans
    # when it is installed.
    "font.family": ["sans-serif", FONT_FAMILY_FALLBACK],
    "font.sans-serif": [
        FONT_FAMILY_DEFAULT,
        "Helvetica Neue LT Std",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "sans-serif",
    ],
    # "font.stretch": "semi-condensed",
    "font.size": FONT_SIZE_DEFAULT,
    "text.color": TEXT_COLOUR,
    # Math Text
    # Noto Sans Math has a single upright face, so the italic and bold styles
    # come from Noto Sans, the text family it is designed to pair with. STIX
    # Sans supplies what Noto Sans Math cannot reach through mathtext
    # (\mathbb, sized delimiters). \mathcal renders upright.
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY_MATH,
    "mathtext.sf": FONT_FAMILY_MATH,
    "mathtext.cal": FONT_FAMILY_MATH,
    "mathtext.it": f"{FONT_FAMILY_DEFAULT}:italic",
    "mathtext.bf": f"{FONT_FAMILY_DEFAULT}:bold",
    "mathtext.bfit": f"{FONT_FAMILY_DEFAULT}:italic:bold",
    "mathtext.fallback": "stixsans",
    # Axes
    "axes.labelsize": FONT_SIZE_DEFAULT,
    "axes.titlesize": FONT_SIZE_DEFAULT,
    "axes.titleweight": "medium",
    "axes.labelweight": "medium",
    "axes.labelcolor": TEXT_COLOUR,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.facecolor": "white",
    "axes.edgecolor": LINE_COLOUR,
    "axes.linewidth": 0,
    # Grid
    "grid.linestyle": "-",
    "grid.color": LINE_COLOUR,
    "grid.linewidth": 0.25,
    "grid.alpha": 1,
    # Ticks
    "xtick.labelsize": FONT_SIZE_DEFAULT,
    "ytick.labelsize": FONT_SIZE_DEFAULT,
    "xtick.color": TEXT_COLOUR,
    "ytick.color": TEXT_COLOUR,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    # Plotting Elements (Lines, Scatters, Patches)
    "lines.solid_capstyle": "round",
    "scatter.marker": "o",
    "patch.edgecolor": "#ffffffff",  # Transparent/White edge
    "patch.linewidth": 0.75,
    "patch.antialiased": True,
    "image.cmap": "tab10",
    "pcolormesh.snap": True,
    # Legend
    "legend.fontsize": FONT_SIZE_DEFAULT,
    "legend.title_fontsize": FONT_SIZE_DEFAULT,
    "legend.frameon": True,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
}


def set_matplotlib_default_style() -> None:
    """Applies the default custom matplotlib style dictionary.

    ``font.family`` is then set from :func:`default_font_families`, so plain text
    draws the symbols Noto Sans lacks from Noto Sans Math where it is installed.
    Like any rcParam, the setting applies to text created after the call.
    """
    plt.style.use(DEFAULT_STYLE_DICT)
    plt.rcParams["font.family"] = default_font_families()


def default_font_families() -> list[str]:
    """Return the ``font.family`` list for the default style on this machine.

    matplotlib falls back to another font glyph by glyph only across the families
    named in ``font.family``, and resolves the generic ``"sans-serif"`` to a single
    font: the first installed entry of ``font.sans-serif``. Noto Sans has no
    arrows, mathematical operators, technical symbols, geometric shapes or
    dingbats, so the list continues with Noto Sans Math, which is designed to pair
    with it, and then DejaVu Sans, which ships with matplotlib.

    Noto Sans Math is listed only when it is installed, because matplotlib logs a
    warning every time it lays out text with a named family it cannot find.

    Returns
    -------
    list[str]
        ``["sans-serif", "Noto Sans Math", "DejaVu Sans"]``, or
        ``["sans-serif", "DejaVu Sans"]`` when Noto Sans Math is not installed.
    """
    families = ["sans-serif"]
    if _font_family_installed(FONT_FAMILY_MATH):
        families.append(FONT_FAMILY_MATH)
    families.append(FONT_FAMILY_FALLBACK)
    return families


def _font_family_installed(family: str) -> bool:
    # A list, because FontProperties parses a lone string as a fontconfig pattern.
    try:
        font_manager.findfont(font_manager.FontProperties(family=[family]), fallback_to_default=False)
    except ValueError:
        return False
    return True


def categorical_palette(n: int, palette: str | None = None) -> list:
    """Return ``n`` colours for categorical series.

    The default is ``tab10`` for at most ten colours, otherwise ``tab20``.
    Listed palettes cycle when ``n`` exceeds their size, so colours can repeat.
    Colormaps with at least 256 entries are sampled evenly, including endpoints.
    This function does not guarantee that all colours are visually distinct.
    """
    cmap = plt.get_cmap(palette or ("tab20" if n > 10 else "tab10"))
    if cmap.N >= 256:  # continuous colormap used as qualitative
        return [cmap(i / max(n - 1, 1)) for i in range(n)]
    return [cmap(i % cmap.N) for i in range(n)]
