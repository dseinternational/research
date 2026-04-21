# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import statsmodels.api as sm


def linear_regression_ols(x, y):
    """
    Perform ordinary least squares linear regression.

    Parameters
    ----------
    x : array-like
        Independent variable (predictor).
    y : array-like
        Dependent variable (response).

    Returns
    -------
    dict
        A dictionary containing the regression results, including:
        - 'intercept': The estimated intercept of the regression line.
        - 'slope': The estimated slope of the regression line.
        - 'fitted': The fitted values (predicted y values).
        - 'residuals': The residuals (differences between observed and fitted values).
        - 'r_squared': The coefficient of determination (R²) — proportion of variance explained by the model.
        - 'residual_standard_error': The standard error of the residuals.
        - 'model': The statsmodels OLS model object.
        - 'results': The statsmodels OLS results object containing detailed regression output.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError("need at least two observations")

    X = sm.add_constant(x)  # adds intercept column
    model = sm.OLS(y, X)
    results = model.fit()

    fitted = results.fittedvalues
    residuals = results.resid

    return {
        "intercept": results.params[0],
        "slope": results.params[1],
        "fitted": fitted,
        "residuals": residuals,
        "r_squared": results.rsquared,
        "residual_standard_error": np.sqrt(results.mse_resid),
        "model": model,
        "results": results,
    }
