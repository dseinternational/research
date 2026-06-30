# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cross-validation scoring helpers shared across DSE regression projects.

The sign logic keeps error scorers (``neg_*``) positive for display while leaving
signed metrics such as ``r2`` untouched, so a model that is worse than the mean
(negative R²) is reported as negative rather than ``abs()``'d into a deceptively
positive value.
"""

from __future__ import annotations

import numpy as np

# Shared defaults for regression hyperparameter search / scoring.
DEFAULT_REGRESSION_SCORING = "neg_mean_absolute_error"

DEFAULT_REGRESSION_CRITERION = "squared_error"

DEFAULT_REGRESSION_SCORERS = {
    "Mean Absolute Error (MAE)": "neg_mean_absolute_error",
    "Root Mean Squared Error (RMSE)": "neg_root_mean_squared_error",
    "R-squared (R2)": "r2",
    "Median Absolute Error (MedAE)": "neg_median_absolute_error",
}

DEFAULT_REGRESSION_SEARCH_ITERATIONS = 60

DEFAULT_REGRESSION_PERM_IMPORTANCE_REPEATS = 30

# Metric labels (the suffix after ``test_``) whose sklearn scores are stored
# negated (``neg_*`` scorers). Their fold means are flipped back to positive
# for display. Anything not listed here — notably ``r2`` — is reported with
# its true sign, so a model that is worse than the mean (negative R²) is not
# disguised as a positive score.
_NEG_SCORER_LABELS = frozenset(
    {
        "mae",
        "rmse",
        "medae",
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
        "neg_median_absolute_error",
        "neg_mean_squared_error",
    }
)


def cross_validation_score_rows(
    scores: dict[str, np.ndarray],
) -> list[dict[str, str | float]]:
    """Build the per-metric summary rows from a ``cross_validate`` result.

    Only ``test_*`` entries are summarised. ``neg_*`` error scorers (see
    :data:`_NEG_SCORER_LABELS`) are flipped back to positive; every other
    metric — notably ``r2`` — keeps its true sign, so a model that is worse
    than the mean (negative R²) is reported as negative rather than being
    ``abs()``'d into a deceptively positive value.

    Separated from :func:`report_cross_validation_scores` so the sign logic
    can be unit-tested without the console-rendering dependency.
    """
    rows: list[dict[str, str | float]] = []
    for key, score in scores.items():
        if not key.startswith("test_"):
            continue
        label = key[5:]
        mean = float(np.mean(score))
        if label in _NEG_SCORER_LABELS:
            mean = -mean
        rows.append({"metric": label, "mean": mean, "sd": float(np.std(score))})
    return rows


def report_cross_validation_scores(model_name: str, scores: dict[str, np.ndarray]):
    """Print a cross-validation score table via the shared console layer."""
    from dse_research_utils.console.console import print_table
    from dse_research_utils.console.sections import section_header
    from dse_research_utils.console.tables import metrics_table

    section_header(f"{model_name}: Cross-validation scores")
    rows = cross_validation_score_rows(scores)
    print_table(metrics_table(rows, columns=["metric", "mean", "sd"]))
