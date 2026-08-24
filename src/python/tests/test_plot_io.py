# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the shared figure-artifact save helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dse_research_utils.plot.io import (
    SVG_MAX_BYTES,
    save_plot_data,
    save_plotcollection,
    save_styled_figure,
)


def _tiny_fig():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return fig


def test_save_styled_figure_writes_png_and_svg_and_closes(tmp_path):
    fig = _tiny_fig()
    png = save_styled_figure(str(tmp_path), "demo", fig=fig)
    assert (tmp_path / "demo.png").exists()
    assert (tmp_path / "demo.svg").exists()
    assert png.endswith("demo.png")
    # close=True by default -> no lingering figures
    assert plt.get_fignums() == []


def test_save_styled_figure_close_opt_out(tmp_path):
    fig = _tiny_fig()
    save_styled_figure(str(tmp_path), "kept", fig=fig, close=False)
    assert fig.number in plt.get_fignums()
    plt.close(fig)


def test_save_styled_figure_accepts_png_extension_in_name(tmp_path):
    save_styled_figure(str(tmp_path), "with_ext.png", fig=_tiny_fig())
    assert (tmp_path / "with_ext.png").exists()
    assert (tmp_path / "with_ext.svg").exists()
    assert not (tmp_path / "with_ext.png.png").exists()


def test_save_styled_figure_svg_size_cap(tmp_path):
    # Force the SVG over the cap -> it is written then dropped, PNG kept.
    save_styled_figure(str(tmp_path), "big", fig=_tiny_fig(), svg_max_bytes=1)
    assert (tmp_path / "big.png").exists()
    assert not (tmp_path / "big.svg").exists()


def test_save_styled_figure_svg_uncapped_by_default(tmp_path):
    # The default keeps the SVG regardless of size (callers opt into the cap).
    save_styled_figure(str(tmp_path), "uncapped", fig=_tiny_fig())
    assert (tmp_path / "uncapped.svg").exists()
    assert SVG_MAX_BYTES == 2 * 1024 * 1024


def test_save_styled_figure_svg_opt_out(tmp_path):
    save_styled_figure(str(tmp_path), "nosvg", fig=_tiny_fig(), svg=False)
    assert (tmp_path / "nosvg.png").exists()
    assert not (tmp_path / "nosvg.svg").exists()


def test_save_styled_figure_writes_data_csv(tmp_path):
    df = pd.DataFrame({"x": [0, 1], "y": [1, 2]})
    save_styled_figure(str(tmp_path), "withdata", fig=_tiny_fig(), data=df)
    out = pd.read_csv(tmp_path / "withdata.csv")
    assert list(out.columns) == ["x", "y"]
    assert len(out) == 2


def test_save_styled_figure_creates_output_dir(tmp_path):
    target = tmp_path / "nested" / "dir"
    save_styled_figure(str(target), "made", fig=_tiny_fig())
    assert (target / "made.png").exists()


def test_save_plot_data_stem_and_no_index(tmp_path):
    save_plot_data(str(tmp_path), "curve.png", {"a": [1, 2, 3]})
    assert (tmp_path / "curve.csv").exists()
    assert (tmp_path / "curve.csv").read_text().splitlines()[0] == "a"


class _Item:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakePlotCollection:
    """Minimal stand-in for an arviz_plots PlotCollection."""

    def __init__(self, fig):
        self._fig = fig
        self.viz = {"figure": _Item(fig)}

    def savefig(self, path, **kwargs):
        self._fig.savefig(path)


def test_save_plotcollection_png_svg_and_suptitle(tmp_path):
    fig = _tiny_fig()
    pc = _FakePlotCollection(fig)
    save_plotcollection(pc, str(tmp_path), "trace_plot.png", suptitle="My title")
    assert (tmp_path / "trace_plot.png").exists()
    assert (tmp_path / "trace_plot.svg").exists()
    assert fig._suptitle is not None and fig._suptitle.get_text() == "My title"
    # close=True by default -> no lingering figures
    assert plt.get_fignums() == []


def test_save_plotcollection_close_opt_out(tmp_path):
    fig = _tiny_fig()
    pc = _FakePlotCollection(fig)
    save_plotcollection(pc, str(tmp_path), "kept_pc", close=False)
    assert fig.number in plt.get_fignums()
    plt.close(fig)
