
"""Automatically analyse a dataset and populate a full report.

Runs a standard battery — overview, descriptive stats, target distribution,
correlation, best-predictor regression, a categorical breakdown, a group
comparison test and a normality check — and turns each into a report item with
an editable inference sentence.
"""

import numpy as np
import pandas as pd

from modules import report as R
from modules.descriptive import describe_all, descriptive_stats
from modules.frequency import categorical_frequency
from modules.normality import qq_data, normality_report
from modules.regression import simple_linear_regression
from modules.tests import independent_ttest, effect_label_d


def _numeric(df):
    return list(df.select_dtypes(include="number").columns)


def _categorical(df):
    return list(df.select_dtypes(include=["object", "category", "bool"]).columns)


def _pick_target(df, numeric):
    for cand in ("exam_score", "score", "target", "result"):
        for c in numeric:
            if c.lower() == cand:
                return c
    # else the metric column with the most distinct values
    return max(numeric, key=lambda c: df[c].nunique()) if numeric else None


def build_default_report(df, dataset_name="Dataset"):
    """Reset the report and fill it with a complete analysis of ``df``."""

    R.reset_report()
    rep = R.get_report()
    rep["cover"]["subtitle"] = dataset_name

    numeric = _numeric(df)
    categorical = _categorical(df)
    target = _pick_target(df, numeric)

    # 1 ── Overview
    overview = pd.DataFrame(
        {"Value": [len(df), df.shape[1], int(df.isna().sum().sum()),
                   int(df.duplicated().sum()), len(numeric), len(categorical)]},
        index=["Rows", "Columns", "Missing values", "Duplicate rows",
               "Metric variables", "Categorical variables"],
    )
    R.add_item(
        "Dataset Overview",
        inference=(
            f"The {dataset_name} dataset contains {len(df):,} records across "
            f"{df.shape[1]} variables ({len(numeric)} metric, {len(categorical)} categorical) "
            f"with {int(df.isna().sum().sum())} missing values. "
            "Data quality is sufficient for the analyses that follow."
        ),
        table=overview,
    )

    # 2 ── Descriptive statistics
    if numeric:
        desc = describe_all(df, numeric)
        R.add_item(
            "Descriptive Statistics",
            inference=(R.describe_inference(target, df[target]) if target else
                       "Summary statistics for all metric variables are shown above."),
            table=desc,
        )

    # 3 ── Target distribution (histogram)
    if target:
        R.add_item(
            f"Distribution of {target}",
            inference=R.describe_inference(target, df[target]),
            image=R.hist_png(df[target], title=f"Distribution of {target}"),
        )
        # 3b ── Box plot for spread / outliers
        s = pd.to_numeric(df[target], errors="coerce").dropna()
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        R.add_item(
            f"Spread & Outliers — {target}",
            inference=(
                f"The middle 50% of {target} lies between {q1:.2f} and {q3:.2f} "
                f"(IQR {iqr:.2f}). The box plot flags {outliers} potential outlier(s) "
                "beyond 1.5×IQR."
            ),
            image=R.box_png(df[target], title=f"{target} — box plot"),
        )

    # 4 ── Correlation
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        R.add_item(
            "Correlation Matrix",
            inference=R.correlation_inference(corr),
            image=R.heatmap_png(corr),
        )

    # 5 ── Best-predictor regression
    if target and len(numeric) >= 2:
        corr_t = df[numeric].corr()[target].drop(target).abs()
        if len(corr_t):
            predictor = corr_t.idxmax()
            r = simple_linear_regression(df, predictor, target)
            if r:
                R.add_item(
                    f"Regression — {target} vs {predictor}",
                    inference=R.regression_inference(predictor, target, r),
                    image=R.regression_png(
                        r["x"], r["y"], r["x_line"], r["y_line"],
                        xlabel=predictor, ylabel=target,
                        title=f"{target} vs {predictor}",
                    ),
                )

    # 6 ── Categorical breakdown
    if categorical:
        cat = categorical[0]
        freq = categorical_frequency(df[cat])
        top = freq.iloc[0]
        R.add_item(
            f"Breakdown by {cat}",
            inference=(
                f"{cat} has {len(freq)} categories. The most common is "
                f"\"{top['Value']}\" ({top['Abs. Frequency']} records, "
                f"{top['Rel. Frequency %']}% of the data)."
            ),
            image=R.bar_png(freq["Value"], freq["Abs. Frequency"],
                            title=f"{cat} distribution", xlabel=cat),
        )

        # 7 ── Group comparison (independent t-test) if exactly 2 groups
        groups = list(df[cat].dropna().unique())
        if target and len(groups) == 2:
            g1 = df[df[cat] == groups[0]][target]
            g2 = df[df[cat] == groups[1]][target]
            res = independent_ttest(g1, g2)
            sig = res["p"] < 0.05
            p_str = "< 0.001" if res["p"] < 0.001 else f"= {res['p']:.3f}"
            comp = pd.DataFrame(
                {"Mean": [res["mean1"], res["mean2"]],
                 "SD": [res["sd1"], res["sd2"]],
                 "N": [res["n1"], res["n2"]]},
                index=[str(groups[0]), str(groups[1])],
            )
            R.add_item(
                f"{target} by {cat} (t-test)",
                inference=R.test_inference(
                    sig,
                    f"{target} differs significantly between \"{groups[0]}\" and "
                    f"\"{groups[1]}\" (t = {res['stat']:.2f}, p {p_str}, "
                    f"Cohen's d = {res['cohen_d']:.2f}, {effect_label_d(res['cohen_d'])} effect).",
                    f"there is no significant difference in {target} between "
                    f"\"{groups[0]}\" and \"{groups[1]}\" (p {p_str}).",
                ),
                table=comp,
            )

    # 8 ── Normality check
    if target:
        rep_norm = normality_report(df[target])
        if rep_norm["W"] is not None:
            qq = qq_data(df[target])
            verdict = ("approximately normal" if rep_norm["normal"]
                       else "a significant departure from normality")
            R.add_item(
                f"Normality of {target} (Q-Q)",
                inference=(
                    f"A Shapiro-Wilk test gives W = {rep_norm['W']}, "
                    f"p {'< 0.001' if rep_norm['p'] < 0.001 else f'= {rep_norm['p']:.3f}'}, "
                    f"indicating {verdict}. Skewness is {rep_norm['skew']} and excess "
                    f"kurtosis {rep_norm['kurtosis']}."
                ),
                image=R.qq_png(qq["theoretical"], qq["sample"], qq["ref_x"], qq["ref_y"]),
            )

    return rep
