
import numpy as np
import pandas as pd

from scipy import stats


def qq_data(series):
    """Return theoretical (normal) quantiles vs. ordered sample quantiles
    plus a reference line through Q1 / Q3 — mirrors the HTML Q-Q plot."""

    s = pd.to_numeric(series, errors="coerce").dropna()
    d = np.sort(s.values)
    n = len(d)

    if n < 3:
        return None

    # Blom plotting positions: (i - 3/8) / (n + 1/4)
    i = np.arange(1, n + 1)
    p = (i - 0.375) / (n + 0.25)
    theo = stats.norm.ppf(p)

    # Reference line through the first and third quartiles
    q1, q3 = np.percentile(d, [25, 75])
    tq1, tq3 = stats.norm.ppf([0.25, 0.75])
    slope = (q3 - q1) / (tq3 - tq1)
    intercept = q1 - slope * tq1

    x_min, x_max = theo[0] - 0.5, theo[-1] + 0.5

    return {
        "theoretical": theo,
        "sample": d,
        "ref_x": np.array([x_min, x_max]),
        "ref_y": intercept + slope * np.array([x_min, x_max]),
    }


def normality_report(series):
    """Shapiro-Wilk test + shape descriptors with a plain-language verdict."""

    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)

    if n < 3:
        return {
            "n": n,
            "W": None,
            "p": None,
            "normal": None,
            "skew": None,
            "kurtosis": None,
            "label": "Need at least 3 data points",
        }

    # scipy caps Shapiro-Wilk at 5000 samples
    sample = s if n <= 5000 else s.sample(5000, random_state=0)
    W, p = stats.shapiro(sample)

    skew = float(stats.skew(s))
    kurt = float(stats.kurtosis(s))  # excess kurtosis (Fisher)

    normal = p > 0.05

    if normal:
        label = "Points follow the line — approximately normal (fail to reject normality)."
    else:
        label = "Significant departure from normality (p < 0.05) — consider a transformation or non-parametric test."

    return {
        "n": n,
        "W": round(float(W), 4),
        "p": float(p),
        "normal": normal,
        "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "label": label,
    }
