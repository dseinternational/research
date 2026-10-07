# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass

import psutil


@dataclass
class SamplingConfiguration:
    draws: int
    """
    Number of retained posterior draws per chain.
    """
    tune: int
    """
    Number of tuning steps per chain, excluded from the retained draws.
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
CHAINS_TEST = 4
CHAINS_REP = 6
CHAINS_REP_LITE = 4
TUNES_DEV = 500
TUNES_TEST = 2000
TUNES_REP = 6000
TUNES_REP_LITE = 4000
SAMPLES_DEV = 500
SAMPLES_TEST = 2000
SAMPLES_REP = 6000
SAMPLES_REP_LITE = 4000
TARGET_ACCEPT_DEV = 0.85
TARGET_ACCEPT_TEST = 0.90
TARGET_ACCEPT_REP = 0.95
TARGET_ACCEPT_REP_LITE = 0.95


def _get_available_cores() -> int:
    """Return the available physical core count, keeping one core spare where possible."""
    n_cpus = psutil.cpu_count(logical=False)
    if n_cpus is None:
        n_cpus = psutil.cpu_count(logical=True) or 2
    return max(1, n_cpus - 1)


def get_sampling_configuration(config: str = "dev", random_seed: int = 47) -> SamplingConfiguration:
    """Return a named sampling preset with a caller-supplied random seed.

    Parameters
    ----------
    config : str, default "dev"
        Accepted names are ``dev``/``development``, ``test``/``testing``,
        ``rep-lite``/``reporting-lite``/``rep_lite``, and
        ``reporting``/``report``/``rep``. See the Python readme for preset values.
    random_seed : int, default 47
        Seed recorded in the returned configuration.

    Returns
    -------
    SamplingConfiguration
        Draws and tuning steps per chain, chain count, worker count, acceptance
        target and seed. Worker count is capped by chains and available cores.
        These settings do not guarantee convergence or sampling precision.

    Raises
    ------
    ValueError
        If the configuration name is unknown.
    """
    if config == "reporting" or config == "report" or config == "rep":
        return SamplingConfiguration(
            draws=SAMPLES_REP,
            tune=TUNES_REP,
            chains=CHAINS_REP,
            cores=min(CHAINS_REP, _get_available_cores()),
            target_accept=TARGET_ACCEPT_REP,
            random_seed=random_seed,
        )

    if config == "rep-lite" or config == "reporting-lite" or config == "rep_lite":
        # Retain the reporting acceptance target with fewer chains and draws.
        # Check achieved effective sample sizes and convergence after fitting.
        return SamplingConfiguration(
            draws=SAMPLES_REP_LITE,
            tune=TUNES_REP_LITE,
            chains=CHAINS_REP_LITE,
            cores=min(CHAINS_REP_LITE, _get_available_cores()),
            target_accept=TARGET_ACCEPT_REP_LITE,
            random_seed=random_seed,
        )

    if config == "dev" or config == "development":
        return SamplingConfiguration(
            draws=SAMPLES_DEV,
            tune=TUNES_DEV,
            chains=CHAINS_DEV,
            cores=min(CHAINS_DEV, _get_available_cores()),
            target_accept=TARGET_ACCEPT_DEV,
            random_seed=random_seed,
        )

    if config == "test" or config == "testing":
        return SamplingConfiguration(
            draws=SAMPLES_TEST,
            tune=TUNES_TEST,
            chains=CHAINS_TEST,
            cores=min(CHAINS_TEST, _get_available_cores()),
            target_accept=TARGET_ACCEPT_TEST,
            random_seed=random_seed,
        )

    raise ValueError(f"Invalid sampling configuration: {config}")
