# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Map a posterior probability to a reporting label or odds.

The probability must refer to a named claim, such as ``P(effect >= delta)``.
Labels describe the probability of that claim, not the effect size. They
remain conditional on the model and its data.
"""

_EVIDENCE_LADDER: tuple[tuple[float, str], ...] = (
    (0.75, "inconclusive"),
    (0.91, "suggestive"),
    (0.97, "moderate"),
    (0.99, "strong"),
)


def evidence_label(prob: float) -> str:
    """Label the posterior probability of a named claim.

    Thresholds are 0.75, 0.91, 0.97 and 0.99; equality enters the higher band.
    These approximate odds of 3:1, 10:1, 30:1 and 100:1. Orient ``prob`` toward
    the claim before calling. The label does not describe effect size.

    Parameters
    ----------
    prob : float
        Posterior probability of the claim, in [0, 1].

    Returns
    -------
    str
        One of ``"inconclusive"``, ``"suggestive"``, ``"moderate"``, ``"strong"``,
        or ``"very strong"``.
    """
    if not 0.0 <= float(prob) <= 1.0:
        raise ValueError(f"prob must be a probability in [0, 1], got {prob!r}")
    for threshold, label in _EVIDENCE_LADDER:
        if prob < threshold:
            return label
    return "very strong"


def odds_string(prob: float) -> str:
    """A posterior probability as approximate whole-number odds, e.g. ``"19:1"``.

    Parameters
    ----------
    prob : float
        Posterior probability, clamped to (0, 1) to keep the odds finite.

    Returns
    -------
    str
        The odds as ``"o:1"`` when ``prob >= 0.5`` and ``"1:o"`` otherwise.
    """
    p = min(max(float(prob), 1e-9), 1 - 1e-9)
    o = p / (1 - p)
    return f"{o:.0f}:1" if o >= 1 else f"1:{1 / o:.0f}"


def favoured_direction(prob_positive: float) -> dict[str, float | str]:
    """Label the more probable side of a signed effect.

    Parameters
    ----------
    prob_positive : float
        ``P(effect > 0)`` in [0, 1]. The complement is treated as the negative
        direction, which assumes no probability mass at exactly zero. If zero
        has positive mass, the complement instead describes a non-positive effect.

    Returns
    -------
    dict
        ``favoured_direction`` is positive when the supplied probability is
        at least 0.5, otherwise negative. ``favoured_direction_prob`` is
        ``max(prob_positive, 1 - prob_positive)``. ``favoured_direction_label``
        applies :func:`evidence_label` to that probability. Report the raw
        positive-direction probability separately when the claim requires it.
    """
    p_pos = float(prob_positive)
    if not 0.0 <= p_pos <= 1.0:
        raise ValueError(f"prob_positive must be a probability in [0, 1], got {prob_positive!r}")
    prob = max(p_pos, 1.0 - p_pos)
    return {
        "favoured_direction": "positive" if p_pos >= 0.5 else "negative",
        "favoured_direction_prob": prob,
        "favoured_direction_label": evidence_label(prob),
    }
