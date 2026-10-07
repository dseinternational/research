# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import pytest

from dse_research_utils.plot import styles

REPO_ROOT = Path(__file__).resolve().parents[3]
PIN = REPO_ROOT / "src/python/src/dse_research_utils/plot/design_tokens/design-tokens.pin.json"


def _luminance(colour: str) -> float:
    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(int(colour[i : i + 2], 16) / 255) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_vendored_design_tokens_match_their_pin() -> None:
    # A mismatch means the copy was edited or a bump is incomplete: copy the file at the pinned release.
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    assert pin["repository"] == "dsegroup/apps-common"
    for file in pin["files"]:
        actual = hashlib.sha256((REPO_ROOT / file["path"]).read_bytes()).hexdigest()
        assert actual == file["sha256"], f"{file['path']} differs from {pin['tag']} {file['source']}"


def test_chart_colours_are_the_six_categorical_tokens() -> None:
    assert len(styles.CHART_COLOURS) == 6
    assert len(set(styles.CHART_COLOURS)) == 6
    assert all(c.startswith("#") and len(c) == 7 for c in styles.CHART_COLOURS)


@pytest.mark.parametrize("palettes", [styles.SEQUENTIAL_PALETTES, styles.DIVERGING_PALETTES])
def test_ordered_palettes_have_three_to_five_distinct_steps(palettes: dict[int, tuple[str, ...]]) -> None:
    assert sorted(palettes) == [3, 4, 5]
    for n, steps in palettes.items():
        assert len(steps) == n
        assert len(set(steps)) == n


def test_every_chart_colour_reaches_three_to_one_on_the_background() -> None:
    colours = [*styles.CHART_COLOURS]
    for palettes in (styles.SEQUENTIAL_PALETTES, styles.DIVERGING_PALETTES):
        for steps in palettes.values():
            colours.extend(steps)
    for colour in colours:
        assert _contrast(colour, styles.BACKGROUND_COLOUR) >= 3, colour


def test_continuous_scales_are_registered() -> None:
    for name in ("dse_sequential", "dse_diverging", "dse_sequential_r", "dse_diverging_r"):
        assert name in mpl.colormaps
    # Continuous scales start from the quiet muted fill; their strongest values are the strongest steps.
    assert mpl.colors.to_hex(styles.SEQUENTIAL_CMAP(1.0)) == styles.SEQUENTIAL_PALETTES[5][-1]
    assert mpl.colors.to_hex(styles.DIVERGING_CMAP(0.0)) == styles.DIVERGING_PALETTES[5][0]
    assert mpl.colors.to_hex(styles.DIVERGING_CMAP(1.0)) == styles.DIVERGING_PALETTES[5][-1]
