
import numpy as np
import pandas as pd


def categorical_frequency(series):
    """Absolute, relative and cumulative frequency for a categorical column."""

    s = series.dropna().astype(str)
    s = s[s.str.strip() != ""]

    counts = s.value_counts()
    total = int(counts.sum())

    if total == 0:
        return pd.DataFrame(
            columns=["Value", "Abs. Frequency", "Rel. Frequency %", "Cumulative %"]
        )

    rel = counts / total * 100
    cum = rel.cumsum()

    return pd.DataFrame({
        "Value": counts.index.astype(str),
        "Abs. Frequency": counts.values,
        "Rel. Frequency %": rel.round(2).values,
        "Cumulative %": cum.round(2).values,
    })


def grouped_frequency(series, bins=5):
    """Binned (grouped) frequency table for a metric column."""

    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return pd.DataFrame(
            columns=["Interval", "Abs. Frequency", "Rel. Frequency %", "Cumulative %"]
        )

    lo, hi = float(s.min()), float(s.max())

    if lo == hi:
        edges = np.array([lo, lo + 1])
    else:
        edges = np.linspace(lo, hi, bins + 1)

    # right-closed-exclusive bins, last bin includes the max value
    cats = pd.cut(s, bins=edges, include_lowest=True, right=False)
    counts = cats.value_counts().sort_index()

    total = int(counts.sum())
    rel = counts / total * 100
    cum = rel.cumsum()

    intervals = [
        f"[{iv.left:.2f}, {iv.right:.2f})" for iv in counts.index
    ]

    return pd.DataFrame({
        "Interval": intervals,
        "Abs. Frequency": counts.values,
        "Rel. Frequency %": rel.round(2).values,
        "Cumulative %": cum.round(2).values,
    })
