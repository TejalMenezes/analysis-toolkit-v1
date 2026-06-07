
import numpy as np
import pandas as pd

import statsmodels.api as sm


def linear_regression(df, y_var, x_vars):
    """Multiple OLS regression — returns the fitted statsmodels model."""

    X = df[x_vars]

    y = df[y_var]

    X = sm.add_constant(X)

    model = sm.OLS(
        y,
        X
    ).fit()

    return model


def simple_linear_regression(df, x_var, y_var):
    """Simple Y = a + bX fit with the headline numbers shown in the
    HTML toolkit: intercept, slope, R-squared, slope p-value and the
    fitted line for plotting."""

    data = df[[x_var, y_var]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(data) < 3:
        return None

    x = data[x_var].values
    y = data[y_var].values

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    intercept, slope = model.params
    p_slope = model.pvalues[1]
    r2 = model.rsquared

    x_line = np.array([x.min(), x.max()])
    y_line = intercept + slope * x_line

    return {
        "n": len(data),
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(r2),
        "p_slope": float(p_slope),
        "x": x,
        "y": y,
        "x_line": x_line,
        "y_line": y_line,
        "equation": f"Y = {intercept:.3f} + {slope:.3f} x {x_var}",
        "model": model,
    }
