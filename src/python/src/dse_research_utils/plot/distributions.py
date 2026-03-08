# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import os
from matplotlib.figure import Figure

from preliz.distributions.distributions import Continuous

def plot_distribution(dist: Continuous, output_dir: str, filename: str) -> Figure:
    dist.plot_pdf()
    if output_dir is not None and filename is not None:
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.svg"))
    return plt.gcf()