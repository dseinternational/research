# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt

FIGSIZE_XS = (2.913, 2.060)  # 74mm x 52mm (A8 landscape)
FIGSIZE_SM = (4.134, 2.923)  # 105mm x 74mm (A7 landscape)
FIGSIZE_MD = (5.000, 3.536) # 127mm x 90mm
FIGSIZE_LG = (5.827, 4.120)  # 148mm x 105mm (A6 landscape)
FIGSIZE_XL = (8.268, 5.846)  # 210mm x 148.5mm (A5 landscape)
FIGSIZE_XXL = (11.693, 8.268)  # 297mm x 210mm (A4 landscape)

FIGSIZE_XS_PORTRAIT = (2.060, 2.913)  # 52mm x 74mm (A8 portrait)
FIGSIZE_SM_PORTRAIT = (2.923, 4.134)  # 74mm x 105mm (A7 portrait)
FIGSIZE_MD_PORTRAIT = (3.536, 5.000) # 90mm x 127mm
FIGSIZE_LG_PORTRAIT = (4.120, 5.827)  # 105mm x 148mm (A6 portrait)
FIGSIZE_XL_PORTRAIT = (5.846, 8.268)  # 148.5mm x 210mm (A5 portrait)
FIGSIZE_XXL_PORTRAIT = (8.268, 11.693)  # 210mm x 297mm (A4 portrait)

DPI_NOTEBOOK = 120.0
DPI_FILE = 300.0

DEFAULT_STYLE_DICT = {
    # Figure Settings
    "figure.figsize": FIGSIZE_MD,
    "figure.dpi": DPI_NOTEBOOK,
    "savefig.dpi": DPI_FILE,
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
    "figure.titlesize": 10,
    "figure.titleweight": "bold",
    # Font and Text
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans", "Helvetica", "Arial", "sans-serif"],
    "font.stretch": "semi-condensed",
    "font.size": 10,
    "text.color": "#333333",
    # Axes
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "medium",
    "axes.labelweight": "medium",
    "axes.labelcolor": "#333333",
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c0c0c0",
    "axes.linewidth": 0,
    # Grid
    "grid.linestyle": "-",
    "grid.color": "#c0c0c0",
    "grid.linewidth": 0.25,
    "grid.alpha": 1,
    # Ticks
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
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
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "legend.frameon": True,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
}


def set_matplotlib_default_style():
    """Applies the default custom matplotlib style dictionary."""
    plt.style.use(DEFAULT_STYLE_DICT)
