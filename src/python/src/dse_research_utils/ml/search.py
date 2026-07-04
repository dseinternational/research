# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generic hyperparameter-search scaffolding shared across DSE projects."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV


def hyperparam_search_randomized(
    X: Any,
    y: Any,
    groups: Any,
    estimator: Any,
    param_distributions: dict[str, Any],
    n_iter: int = 10,
    scoring: str | Mapping[str, Any] | None = None,
    n_jobs: int | None = None,
    cv: Any | None = None,
    verbose: int = 0,
    random_state: int | None = None,
    error_score: float | str = np.nan,
    output_csv: str | Path | None = None,
) -> tuple[RandomizedSearchCV, pd.DataFrame, dict[str, Any]]:
    """Fit a ``RandomizedSearchCV`` (group-aware) and return the search + results.

    Returns ``(search, cv_results_dataframe, best_params_)``. ``best_params_`` is
    sklearn's own tie-broken best configuration; deriving it from ``cv_results_``
    by hand was fragile under ties in ``rank_test_score``.
    """
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        scoring=scoring,
        n_jobs=n_jobs,
        n_iter=n_iter,
        cv=cv,
        verbose=verbose,
        random_state=random_state,
        error_score=error_score,
        return_train_score=True,
    )

    search.fit(X, y, groups=groups)

    results = pd.DataFrame(search.cv_results_)

    if output_csv is not None:
        results.to_csv(f"{output_csv}", index=False)

    best_params = search.best_params_

    return search, results, best_params
