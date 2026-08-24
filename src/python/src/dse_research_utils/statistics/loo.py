# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared PSIS-LOO / ELPD helpers: reff pinning, summary rows, verdicts.

The consuming repositories each re-derived these pieces — the DataTree/Dataset
normalisation, the relative-efficiency pin, the LOO summary-row schema, the
Pareto-k reliability reductions, and the ELPD-difference conventions — with
small, silent divergences (a ``pareto_k_gt_0.7`` column that meant "above 0.7"
in one table and "above ArviZ's sample-size-dependent ``good_k``" in another).
This module is the single home for those conventions.

reff
----
ArviZ's ``loo`` derives its relative efficiency ``reff`` — the factor that
scales the PSIS tail length and so shapes every Pareto-k estimate — from the
mean effective sample size over **every variable in the posterior group**. A
posterior carrying observation-level deterministics therefore yields a
``reff`` that moves when the storage policy moves (one model measured 0.561 /
0.554 / 0.905 under three storage conventions). :func:`sampled_parameter_reff`
pins it to the model's free random variables instead — the parameters the
sampler actually moved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import xarray as xr

ELPD_DIFF_INCONCLUSIVE = 4.0
"""House convention: an absolute ELPD difference below this is inconclusive.

Below it, two models are treated as predictively indistinguishable rather than
ranked; see Sivula, Magnusson & Vehtari (2020) on the unreliability of small
ELPD differences.
"""


def as_dataset(node: Any) -> xr.Dataset:
    """Normalise an ArviZ DataTree node / Dataset to a plain ``xr.Dataset``.

    ArviZ 1.x returns DataTree-backed groups from ``az.from_netcdf``; older
    call sites hold plain Datasets. Both repos spelled this normalisation four
    different ways (with opposite predicates); this is the one way.
    """
    if isinstance(node, xr.Dataset):
        return node
    return node.to_dataset() if hasattr(node, "to_dataset") else node


def group_names(idata: Any) -> set[str]:
    """The trace's group names, normalised across ArviZ 0.x / 1.x containers.

    DataTree ``groups`` are slash-prefixed paths (``"/posterior"``); this
    returns bare names (``"posterior"``) so membership tests read the same for
    both container generations.
    """
    groups = getattr(idata, "groups", None)
    if groups is None:
        return set()
    return {str(g).rstrip("/").rsplit("/", 1)[-1] for g in groups if str(g) not in ("", "/")}


def has_group(idata: Any, name: str) -> bool:
    """True when ``idata`` carries a group called ``name`` (0.x/1.x tolerant)."""
    return name in group_names(idata)


def sampled_parameter_names(
    trace: Any,
    *,
    names: Sequence[str] | None = None,
    attr_reader: Callable[[Any], list[str] | None] | None = None,
) -> list[str]:
    """The sampled parameters of ``trace``: ``names`` if given, else via ``attr_reader``.

    ``attr_reader`` is a repo-supplied callable that reads the recorded
    sampled-parameter names off the trace (the repositories record them as a
    trace attribute at sampling time). Raises ``LookupError`` when neither is
    available — a trace from before the attribute existed, read without a model
    to name the parameters.
    """
    if names is not None:
        return list(names)
    recorded = attr_reader(trace) if attr_reader is not None else None
    if recorded is None:
        raise LookupError(
            "The trace does not record its sampled parameters and none were "
            "supplied; pass names=[rv.name for rv in model.free_RVs] from a "
            "rebuilt model, or accept ArviZ's default."
        )
    return recorded


def sampled_parameter_reff(
    trace: Any,
    *,
    names: Sequence[str] | None = None,
    attr_reader: Callable[[Any], list[str] | None] | None = None,
) -> float:
    """Mean ESS over the sampled parameters divided by the number of draws.

    The same quantity ``arviz_stats`` computes by default, restricted to the
    model's free random variables: 1.0 for a single chain (as ArviZ), else
    ``mean(ess_mean over every element of every named variable) / n_samples``.
    Every named variable must be in the posterior; a compacted trace keeps the
    free random variables, so this holds for every storage tier.
    """
    import arviz_stats  # noqa: F401  (registers the ``azstats`` accessor)

    posterior = as_dataset(trace if isinstance(trace, xr.Dataset) else trace["posterior"])
    wanted = sampled_parameter_names(trace, names=names, attr_reader=attr_reader)
    missing = [name for name in wanted if name not in posterior.data_vars]
    if missing:
        raise KeyError(f"Sampled parameters absent from the posterior: {missing}")
    if posterior.sizes.get("chain", 1) == 1:
        return 1.0
    n_samples = int(posterior.sizes["chain"] * posterior.sizes["draw"])
    ess = posterior[wanted].azstats.ess(method="mean")
    values = np.hstack([ess[name].values.ravel() for name in ess.data_vars])
    return float(values.mean() / n_samples)


def reff_or_default(
    trace: Any,
    *,
    names: Sequence[str] | None = None,
    attr_reader: Callable[[Any], list[str] | None] | None = None,
    label: str = "",
    warn: Callable[[str], Any] = print,
) -> float | None:
    """``sampled_parameter_reff`` where it can be pinned, else ``None`` and a notice.

    ``None`` hands ``az.loo`` its default — the posterior-wide average — which is
    the only option for a trace that neither records its sampled parameters nor
    comes with a model to name them. The notice is printed rather than swallowed
    because a comparison that mixes the two conventions should say so.
    """
    try:
        return sampled_parameter_reff(trace, names=names, attr_reader=attr_reader)
    except LookupError:
        warn(
            f"  {label + ': ' if label else ''}reff left at ArviZ's posterior-wide "
            "default — the trace predates the sampled-parameters record and no "
            "model was supplied to pin it."
        )
        return None


def pareto_k_values(loo: Any) -> np.ndarray:
    """The pointwise Pareto-k array off an ``ELPDData``, 0.x/1.x tolerant.

    ArviZ 0.x exposed ``loo.pareto_k``; the 1.x split renamed it
    ``loo.diagnostics``. Both repos carried this shim independently.
    """
    if hasattr(loo, "pareto_k"):
        return np.asarray(loo.pareto_k.values)
    return np.asarray(loo.diagnostics.values)


def pareto_k_reliability(pareto_k: np.ndarray | Sequence[float], good_k: float) -> dict[str, Any]:
    """Reliability reductions of a pointwise Pareto-k array against ``good_k``.

    ``good_k`` should normally be the fit's own ``loo.good_k`` (ArviZ's
    sample-size-dependent threshold); hard-coding 0.7 counts a k the fit's own
    threshold rejects as fine.

    Returns
    -------
    dict
        ``max`` (largest k), ``n_above`` (count above ``good_k``),
        ``share_above`` (their share of the observations), ``reliable``
        (no k above ``good_k``), and ``good_k`` (the threshold used).
    """
    values = np.asarray(pareto_k, dtype=float)
    n_above = int((values > good_k).sum())
    return {
        "max": float(np.max(values)) if values.size else float("nan"),
        "n_above": n_above,
        "share_above": (n_above / values.size) if values.size else float("nan"),
        "reliable": bool(values.size) and n_above == 0,
        "good_k": float(good_k),
    }


def pareto_k_bands(pareto_k: np.ndarray | Sequence[float], good_k: float) -> dict[str, int]:
    """Count Pareto-k values in the good / bad / very bad bands.

    ``good``: ``k <= good_k``; ``bad``: ``good_k < k <= 1``; ``very_bad``:
    ``k > 1`` (Vehtari, Simpson, Gelman, Yao & Gabry 2024 banding, with the
    good/bad boundary at the fit's own ``good_k``).
    """
    values = np.asarray(pareto_k, dtype=float)
    return {
        "good": int((values <= float(good_k)).sum()),
        "bad": int(((values > float(good_k)) & (values <= 1.0)).sum()),
        "very_bad": int((values > 1.0).sum()),
    }


def loo_summary_row(
    loo: Any,
    *,
    label: str,
    reff: float | None = None,
    unit_name: str = "n_observations",
    k_threshold: float | None = None,
    include_looic: bool = False,
) -> dict[str, Any]:
    """One canonical LOO summary-table row for an ``ELPDData``.

    Parameters
    ----------
    loo
        An ArviZ ``ELPDData`` (0.x or 1.x attribute conventions both work).
    label : str
        Row label (the model / outcome name).
    reff : float, optional
        The relative efficiency the LOO was computed with, recorded in the row.
    unit_name : str, optional
        Name for the observation-count column (``"n_observations"`` by default;
        pass e.g. ``"n_subjects"`` for a subject-level LOO).
    k_threshold : float, optional
        Pareto-k threshold for the ``pareto_k_above`` count. ``None`` (default)
        uses the fit's own ``loo.good_k`` when present, else 0.7.
    include_looic : bool, optional
        Also emit ``looic`` / ``looic_se`` (the deviance-scale view).

    Returns
    -------
    dict
        ``label``, ``elpd_loo``, ``se``, ``p_loo``, ``reff``,
        ``pareto_k_above``, ``k_threshold``, the ``unit_name`` count, and
        optionally ``looic`` / ``looic_se``.
    """
    k = pareto_k_values(loo)
    if k_threshold is None:
        k_threshold = float(getattr(loo, "good_k", 0.7) or 0.7)
    row: dict[str, Any] = {
        "label": label,
        "elpd_loo": float(loo.elpd),
        "se": float(loo.se),
        "p_loo": float(loo.p),
        "reff": None if reff is None else float(reff),
    }
    if include_looic:
        row["looic"] = float(-2.0 * loo.elpd)
        row["looic_se"] = float(2.0 * loo.se)
    row["pareto_k_above"] = int((k > k_threshold).sum())
    row["k_threshold"] = float(k_threshold)
    row[unit_name] = int(k.size)
    return row


def elpd_total_se(pointwise: np.ndarray | Sequence[float]) -> float:
    """Standard error of a total ELPD from its pointwise contributions.

    ``sqrt(n) * sd(pointwise, ddof=1)`` (Vehtari, Gelman & Gabry 2017, eq. 23).
    """
    values = np.asarray(pointwise, dtype=float)
    return float(np.sqrt(values.size) * np.std(values, ddof=1))


def elpd_diff_paired(
    pointwise_a: np.ndarray | Sequence[float],
    pointwise_b: np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Paired ELPD difference (a - b) with its paired standard error.

    The paired SE uses the pointwise differences, which is what makes the
    comparison honest for two models scored on the same observations
    (Vehtari, Gelman & Gabry 2017, eq. 24).

    Returns
    -------
    dict
        ``diff`` (total a - b), ``dse`` (paired SE), ``diff_over_dse``, ``n``.
    """
    a = np.asarray(pointwise_a, dtype=float)
    b = np.asarray(pointwise_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired ELPD requires equal shapes, got {a.shape} and {b.shape}")
    diff = a - b
    total = float(diff.sum())
    dse = elpd_total_se(diff)
    return {
        "diff": total,
        "dse": dse,
        "diff_over_dse": total / dse if dse > 0 else float("nan"),
        "n": float(diff.size),
    }


def elpd_verdict(elpd_diff: float | None, *, threshold: float = ELPD_DIFF_INCONCLUSIVE) -> str:
    """Classify an ELPD difference against the house inconclusive threshold.

    Returns ``"unavailable"`` for ``None``/non-finite, ``"inconclusive"`` below
    ``threshold`` in absolute value, else ``"discriminating"``.
    """
    if elpd_diff is None or not np.isfinite(elpd_diff):
        return "unavailable"
    if abs(float(elpd_diff)) < threshold:
        return f"inconclusive (|elpd_diff| < {threshold:g})"
    return f"discriminating (|elpd_diff| >= {threshold:g})"
