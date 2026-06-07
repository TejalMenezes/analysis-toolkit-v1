
import pandas as pd


def descriptive_stats(series):

    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return {"N": 0}

    q1 = s.quantile(.25)
    q3 = s.quantile(.75)

    counts = s.value_counts()
    # a mode is only meaningful if some value repeats
    if len(counts) and counts.iloc[0] > 1:
        top = counts[counts == counts.iloc[0]].index
        mode_str = ", ".join(str(round(v, 4)) for v in top.tolist())
    else:
        mode_str = "—"

    return {

        "N": int(s.count()),

        "Mean": round(s.mean(), 4),

        "Median": round(s.median(), 4),

        "Mode": mode_str,

        "Variance": round(s.var(), 4),

        "Std": round(s.std(), 4),

        "Min": round(s.min(), 4),

        "Max": round(s.max(), 4),

        "Range": round(s.max() - s.min(), 4),

        "Q1": round(q1, 4),

        "Q3": round(q3, 4),

        "IQR": round(q3 - q1, 4),

        "Skewness": round(s.skew(), 4),

        "Kurtosis": round(s.kurt(), 4),

        "Missing": int(series.isna().sum())

    }


# One-line explanation for each statistic, surfaced as tooltips / help text
STAT_HELP = {
    "N": "Total number of valid observations",
    "Mean": "Arithmetic average: sum of all values divided by N",
    "Median": "Middle value when sorted; robust to outliers",
    "Mode": "Most frequently occurring value(s)",
    "Variance": "Average squared deviation from the mean (sample, N-1)",
    "Std": "Typical distance of values from the mean",
    "Min": "Smallest observed value",
    "Max": "Largest observed value",
    "Range": "Difference between maximum and minimum",
    "Q1": "First quartile: 25% of values fall below this",
    "Q3": "Third quartile: 75% of values fall below this",
    "IQR": "Q3 - Q1; contains the middle 50% of the data",
    "Skewness": "Asymmetry: 0=symmetric, >0=right tail, <0=left tail",
    "Kurtosis": "Excess kurtosis vs. normal; 0=normal-like tails",
    "Missing": "Number of missing / non-numeric values",
}


def describe_all(df, columns):
    """Stacked summary table: one column of stats per numeric variable."""

    out = {}

    for col in columns:
        out[col] = descriptive_stats(df[col])

    # cells mix numbers and strings (e.g. Mode "—"); render as strings so the
    # table serialises cleanly in Streamlit / Arrow and exports consistently.
    return pd.DataFrame(out).astype(str)
