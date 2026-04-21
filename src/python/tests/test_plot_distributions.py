# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import matplotlib.pyplot as plt
import preliz as pz
from matplotlib.figure import Figure

from dse_research_utils.plot.distributions import plot_distribution


def test_plot_distribution_returns_figure() -> None:
    dist = pz.Normal(mu=0.0, sigma=1.0)
    fig = plot_distribution(dist)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_distribution_saves_to_disk(tmp_path) -> None:
    dist = pz.Normal(mu=0.0, sigma=1.0)
    fig = plot_distribution(dist, output_dir=str(tmp_path), filename="norm")
    assert (tmp_path / "norm.png").exists()
    assert (tmp_path / "norm.svg").exists()
    plt.close(fig)


def test_plot_distribution_creates_missing_output_dir(tmp_path) -> None:
    target = tmp_path / "nested" / "new_dir"
    assert not target.exists()
    dist = pz.Normal(mu=0.0, sigma=1.0)
    fig = plot_distribution(dist, output_dir=str(target), filename="norm")
    assert target.exists()
    assert (target / "norm.png").exists()
    plt.close(fig)


def test_plot_distribution_skips_save_without_filename(tmp_path) -> None:
    dist = pz.Normal(mu=0.0, sigma=1.0)
    fig = plot_distribution(dist, output_dir=str(tmp_path), filename=None)
    assert not any(tmp_path.iterdir())
    plt.close(fig)


def test_plot_distribution_skips_save_without_output_dir(tmp_path) -> None:
    dist = pz.Normal(mu=0.0, sigma=1.0)
    fig = plot_distribution(dist, output_dir=None, filename="norm")
    # output_dir is None — nothing should be written to tmp_path
    assert not any(tmp_path.iterdir())
    plt.close(fig)


def test_output_dir_created_only_when_saving(tmp_path) -> None:
    # Verify os.makedirs was gated on both args, not just one.
    missing = tmp_path / "never"
    dist = pz.Normal(mu=0.0, sigma=1.0)
    plt.close(plot_distribution(dist, output_dir=str(missing), filename=None))
    assert not missing.exists()
    assert os.path.isdir(str(tmp_path))
