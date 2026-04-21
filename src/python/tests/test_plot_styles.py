# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt

from dse_research_utils.plot import styles


def test_figure_sizes_are_tuples() -> None:
    for name in dir(styles):
        if name.startswith("FIGSIZE_"):
            value = getattr(styles, name)
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert all(isinstance(v, float) for v in value)
            assert all(v > 0 for v in value)


def test_portrait_sizes_are_swap_of_landscape() -> None:
    for base in ("XS", "SM", "MD", "LG", "XL", "XXL"):
        landscape = getattr(styles, f"FIGSIZE_{base}")
        portrait = getattr(styles, f"FIGSIZE_{base}_PORTRAIT")
        assert (portrait[1], portrait[0]) == landscape


def test_colour_constants_are_hex() -> None:
    for name in dir(styles):
        if name.startswith(("COLOUR_", "TEXT_COLOUR", "LINE_COLOUR")):
            value = getattr(styles, name)
            assert isinstance(value, str)
            assert value.startswith("#")
            assert len(value) in (7, 9)  # #rrggbb or #rrggbbaa


def test_set_matplotlib_default_style_applies() -> None:
    styles.set_matplotlib_default_style()
    assert plt.rcParams["figure.facecolor"] == "white"
    assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["font.size"] == float(styles.FONT_SIZE_DEFAULT)
