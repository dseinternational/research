# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from attr import dataclass

import dse_research_utils.statistics.models.pymc_utils as pymc_utils


@dataclass
class SamplingConfiguration:
    draws: int
    """
    Number of MCMC draws.
    """
    tune: int
    """
    Number of MCMC tuning steps.
    """
    chains: int
    """
    Number of MCMC chains.
    """
    cores: int
    """
    Number of CPU cores to use for sampling.
    """
    target_accept: float
    """
    Target acceptance rate for NUTS sampler.
    """
    random_seed: int
    """
    Random seed for reproducibility.
    """


CHAINS_DEV = 2
CHAINS_REP = 6
TUNES_DEV = 1000
TUNES_REP = 4000
SAMPLES_DEV = 1000
SAMPLES_REP = 6000
TARGET_ACCEPT_DEV = 0.85
TARGET_ACCEPT_REP = 0.95


def get_sampling_configuration(config: str = "dev", random_seed: int = 47) -> SamplingConfiguration:
    """
    Returns a sampling configuration.
    """
    if config == "reporting" or config == "rep":
        return SamplingConfiguration(
            draws=SAMPLES_REP,
            tune=TUNES_REP,
            chains=CHAINS_REP,
            cores=min(CHAINS_REP, pymc_utils.get_available_cores()),
            target_accept=TARGET_ACCEPT_REP,
            random_seed=random_seed,
        )

    if config == "dev" or config == "development":
        return SamplingConfiguration(
            draws=SAMPLES_DEV,
            tune=TUNES_DEV,
            chains=CHAINS_DEV,
            cores=min(CHAINS_DEV, pymc_utils.get_available_cores()),
            target_accept=TARGET_ACCEPT_DEV,
            random_seed=random_seed,
        )

    raise ValueError(f"Invalid sampling configuration: {config}")
