
import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    f_oneway,
    chi2_contingency,
    levene,
    norm,
)


def one_sample_ttest(data, pop_mean):

    d = data.dropna()
    stat, p = ttest_1samp(d, pop_mean)

    n = len(d)
    mean = d.mean()
    sd = d.std()
    ci = 1.96 * sd / np.sqrt(n)

    return {
        "stat": float(stat),
        "p": float(p),
        "n": n,
        "mean": float(mean),
        "sd": float(sd),
        "df": n - 1,
        "ci": (float(mean - ci), float(mean + ci)),
    }


def independent_ttest(group1, group2):
    """Welch's t-test (does not assume equal variances) + Cohen's d."""

    g1 = group1.dropna()
    g2 = group2.dropna()

    stat, p = ttest_ind(g1, g2, equal_var=False)

    n1, n2 = len(g1), len(g2)
    m1, m2 = g1.mean(), g2.mean()
    v1, v2 = g1.var(), g2.var()

    # pooled SD for Cohen's d
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    cohen_d = abs(m1 - m2) / pooled if pooled else float("nan")

    return {
        "stat": float(stat),
        "p": float(p),
        "n1": n1, "n2": n2,
        "mean1": float(m1), "mean2": float(m2),
        "sd1": float(np.sqrt(v1)), "sd2": float(np.sqrt(v2)),
        "cohen_d": float(cohen_d),
    }


def paired_ttest(before, after):

    df = pd.concat([before, after], axis=1).dropna()
    b = df.iloc[:, 0]
    a = df.iloc[:, 1]

    stat, p = ttest_rel(b, a)

    diffs = b - a
    n = len(diffs)
    md = diffs.mean()
    sd = diffs.std()
    ci = 1.96 * sd / np.sqrt(n)
    cohen_dz = abs(md / sd) if sd else float("nan")

    return {
        "stat": float(stat),
        "p": float(p),
        "n": n,
        "mean_diff": float(md),
        "sd_diff": float(sd),
        "df": n - 1,
        "ci": (float(md - ci), float(md + ci)),
        "cohen_dz": float(cohen_dz),
        "diffs": diffs,
    }


def anova(groups):

    stat, p = f_oneway(*groups)

    return float(stat), float(p)


def variance_test(group1, group2):

    stat, p = levene(
        group1.dropna(),
        group2.dropna()
    )

    return float(stat), float(p)


def chi_square(df, col1, col2):

    table = pd.crosstab(
        df[col1],
        df[col2]
    )

    chi2, p, dof, expected = chi2_contingency(table)

    n = int(table.values.sum())
    k = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if n and k else float("nan")
    low_expected = int((expected < 5).sum())

    return {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "table": table,
        "expected": expected,
        "n": n,
        "cramers_v": float(cramers_v),
        "low_expected": low_expected,
    }


def one_sample_ztest(data, pop_mean, sigma):
    """Z = (x̄ - μ₀) / (σ/√n) with known population sigma."""

    d = data.dropna()
    n = len(d)
    xbar = d.mean()

    se = sigma / np.sqrt(n)
    z = (xbar - pop_mean) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    ci = 1.96 * se

    return {
        "z": float(z),
        "p": float(p),
        "n": n,
        "mean": float(xbar),
        "se": float(se),
        "ci": (float(xbar - ci), float(xbar + ci)),
    }


def two_sample_ztest(group1, group2, sigma1, sigma2):
    """Two-sample Z-test with known population sigmas."""

    g1 = group1.dropna()
    g2 = group2.dropna()
    n1, n2 = len(g1), len(g2)
    m1, m2 = g1.mean(), g2.mean()

    se = np.sqrt(sigma1 ** 2 / n1 + sigma2 ** 2 / n2)
    z = (m1 - m2) / se
    p = 2 * (1 - norm.cdf(abs(z)))

    return {
        "z": float(z),
        "p": float(p),
        "n1": n1, "n2": n2,
        "mean1": float(m1), "mean2": float(m2),
        "se": float(se),
    }


# ── interpretation helpers (shared by the pages) ──

def effect_label_d(d):
    d = abs(d)
    if d < 0.2:
        return "trivial"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def effect_label_v(v):
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "moderate"
    return "strong"
