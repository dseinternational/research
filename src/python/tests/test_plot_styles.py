# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from dse_research_utils.plot import styles

# Symbols Noto Sans lacks: arrows, relations and a dingbat.
SYMBOL_TEXT = "average effect ≈ +1.4 items → ≤ ≥ ↔ ⇒ ≠ ✓"


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
    names = [name for name in dir(styles) if name.endswith("_COLOUR")]
    assert {"TEXT_COLOUR", "MUTED_TEXT_COLOUR", "LINE_COLOUR", "BACKGROUND_COLOUR"} <= set(names)
    for name in names:
        value = getattr(styles, name)
        assert isinstance(value, str)
        assert value.startswith("#")
        assert len(value) == 7


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("COLOUR_BLUE", styles.CHART_COLOURS[0]),
        ("COLOUR_GREEN", styles.CHART_COLOURS[1]),
        ("COLOUR_ORANGE", styles.CHART_COLOURS[2]),
        ("COLOUR_PURPLE", styles.CHART_COLOURS[3]),
        ("COLOUR_RED", "#d62728"),
        ("COLOUR_DARK_GREEN", "#036903"),
    ],
)
def test_named_hues_are_deprecated(name: str, expected: str) -> None:
    with pytest.warns(DeprecationWarning, match=name):
        assert getattr(styles, name) == expected


def test_named_hues_can_still_be_imported() -> None:
    with pytest.warns(DeprecationWarning):
        from dse_research_utils.plot.styles import COLOUR_DARK_BLUE
    assert COLOUR_DARK_BLUE == "#014b7f"


def test_unknown_attributes_raise() -> None:
    with pytest.raises(AttributeError):
        _ = styles.COLOUR_TEAL


def test_set_matplotlib_default_style_applies() -> None:
    with plt.rc_context():
        styles.set_matplotlib_default_style()
        assert plt.rcParams["figure.facecolor"] == styles.BACKGROUND_COLOUR
        assert plt.rcParams["axes.grid"] is True
        assert plt.rcParams["font.size"] == float(styles.FONT_SIZE_DEFAULT)


def test_default_style_draws_series_and_images_in_chart_colours() -> None:
    with plt.rc_context():
        styles.set_matplotlib_default_style()
        assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == list(styles.CHART_COLOURS)
        assert plt.rcParams["image.cmap"] == "dse_sequential"
        assert plt.rcParams["text.color"] == styles.TEXT_COLOUR


def test_ordered_palettes() -> None:
    assert styles.sequential_palette(3) == list(styles.SEQUENTIAL_PALETTES[3])
    assert styles.diverging_palette(5) == list(styles.DIVERGING_PALETTES[5])
    # The middle step of an odd diverging set is grey.
    middle = styles.diverging_palette(5)[2]
    assert middle[1:3] == middle[3:5] == middle[5:7]


@pytest.mark.parametrize("n", [2, 6])
def test_ordered_palettes_reject_unsupported_sizes(n: int) -> None:
    with pytest.raises(ValueError, match="3, 4, 5"):
        styles.sequential_palette(n)
    with pytest.raises(ValueError, match="3, 4, 5"):
        styles.diverging_palette(n)


def test_default_style_uses_noto_fonts() -> None:
    with plt.rc_context():
        styles.set_matplotlib_default_style()
        assert plt.rcParams["font.sans-serif"][0] == "Noto Sans"
        assert plt.rcParams["mathtext.fontset"] == "custom"
        assert plt.rcParams["mathtext.rm"] == "Noto Sans Math"
        assert plt.rcParams["mathtext.it"] == "Noto Sans:italic"


def test_default_style_renders_math_text() -> None:
    # Draws whether or not the Noto fonts are installed: findfont falls back.
    with plt.rc_context():
        styles.set_matplotlib_default_style()
        fig, ax = plt.subplots()
        ax.set_title(
            r"$\hat{R} \leq 1.01,\ \mathbf{x} \sim \mathcal{N}(\mu, \sigma^2),\ \mathbb{E}\left[\frac{a}{b}\right]$"
        )
        fig.canvas.draw()
        plt.close(fig)


def test_default_style_dict_falls_back_to_a_bundled_font() -> None:
    # Safe to apply directly: DejaVu Sans ships with matplotlib, so no named family is missing.
    assert styles.DEFAULT_STYLE_DICT["font.family"] == ["sans-serif", styles.FONT_FAMILY_FALLBACK]


def test_default_font_families_lists_noto_sans_math_only_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(styles, "_font_family_installed", lambda family: True)
    assert styles.default_font_families() == ["sans-serif", "Noto Sans Math", "DejaVu Sans"]
    monkeypatch.setattr(styles, "_font_family_installed", lambda family: False)
    assert styles.default_font_families() == ["sans-serif", "DejaVu Sans"]


def test_font_family_installed_detects_a_missing_family() -> None:
    assert styles._font_family_installed(styles.FONT_FAMILY_FALLBACK)
    assert not styles._font_family_installed("No Such Font Family")


def test_default_style_sets_the_font_family_list() -> None:
    with plt.rc_context():
        styles.set_matplotlib_default_style()
        assert plt.rcParams["font.family"] == styles.default_font_families()


def test_default_style_draws_symbols_noto_sans_lacks(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Holds with or without the Noto fonts: without them (as in CI) the text font or
    # DejaVu Sans draws the symbols. With them, the generic "sans-serif" family alone
    # drew each symbol as a missing-glyph box and only warned (issue #112).
    with plt.rc_context():
        styles.set_matplotlib_default_style()
        fig, ax = plt.subplots()
        fig.suptitle(SYMBOL_TEXT)
        ax.set_title(SYMBOL_TEXT)
        ax.set_xlabel("x ≥ 0")
        ax.plot([0, 1], label="a → b")
        ax.legend()
        with (
            warnings.catch_warnings(record=True) as caught,
            caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"),
        ):
            warnings.simplefilter("always")
            fig.canvas.draw()
            fig.savefig(tmp_path / "symbols.svg")
            fig.savefig(tmp_path / "symbols.pdf")
        plt.close(fig)
    assert not [str(w.message) for w in caught if "missing from font" in str(w.message)]
    # A named family that is not installed would be logged for every text element.
    assert not [r.getMessage() for r in caplog.records if "not found" in r.getMessage()]
