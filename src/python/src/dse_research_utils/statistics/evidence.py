# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evidence-strength helpers: turn a posterior probability into a verbal label or odds.

These are deliberately small, pure helpers shared across DSE research reports so the
*evidence* attached to a claim is described the same way everywhere. They operate on a
posterior probability that is already oriented toward a *named claim* (e.g. a
probability of direction, or ``P(effect > delta)``); the label qualifies the strength
of the evidence, never the size of the effect.
"""

_EVIDENCE_LADDER: tuple[tuple[float, str], ...] = (
    (0.75, "inconclusive"),
    (0.91, "suggestive"),
    (0.97, "moderate"),
    (0.99, "strong"),
)


def evidence_label(prob: float) -> str:
    """Round-odds evidence label for a posterior probability of a *named claim*.

    The boundaries are round odds (3:1 / 10:1 / 30:1 / 100:1), deliberately *not*
    the ``p = 0.05 / 0.025 / 0.01`` images, so they do not smuggle the significance
    grid back in. ``prob`` must already be oriented toward the claim (e.g. the
    probability of direction for a sign claim, or ``P(effect >= delta)`` for a
    magnitude claim); the label qualifies the *evidence*, never the effect size.

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
