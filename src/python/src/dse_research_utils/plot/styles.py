# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt

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

TEXT_COLOUR = "#333333"

LINE_COLOUR = "#c0c0c0"

FONT_SIZE_DEFAULT = 10

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
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans", "Helvetica", "Arial", "sans-serif"],
    "font.stretch": "semi-condensed",
    "font.size": FONT_SIZE_DEFAULT,
    "text.color": TEXT_COLOUR,
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


def set_matplotlib_default_style():
    """Applies the default custom matplotlib style dictionary."""
    plt.style.use(DEFAULT_STYLE_DICT)
