
import numpy as np
import pandas as pd

from scipy import stats


def _shapiro(data):
    d = pd.to_numeric(pd.Series(data), errors="coerce").dropna()
    n = len(d)
    if n < 3:
        return None, None, n
    sample = d if n <= 5000 else d.sample(5000, random_state=0)
    W, p = stats.shapiro(sample)
    return float(W), float(p), n


def _normality_check(data, label):
    W, p, n = _shapiro(data)
    large = n >= 30

    if W is None:
        return {"label": label, "ok": large,
                "detail": f"n={n}. Too few values for a normality test." +
                          (" Large sample (n>=30) — CLT applies." if large else "")}

    normal = p > 0.05
    ok = normal or large

    if large:
        tail = "Large sample (n>=30) — CLT applies."
    elif normal:
        tail = "Normally distributed (Shapiro-Wilk p>0.05)."
    else:
        tail = "Non-normal & small sample — consider a non-parametric test."

    return {
        "label": label,
        "ok": ok,
        "detail": f"W={W:.3f}, p={p:.3f}, n={n}. {tail}",
    }


def check_one_sample_t(data):
    checks = [_normality_check(data, "Normality (variable)")]
    checks.append({
        "label": "Independence of observations",
        "ok": True,
        "detail": "Assumed by design — each observation must be independent of the others.",
    })
    return checks


def check_independent_t(group1, group2):
    checks = [
        _normality_check(group1, "Normality (group 1)"),
        _normality_check(group2, "Normality (group 2)"),
    ]

    # Levene's test for equality of variances
    g1 = pd.to_numeric(pd.Series(group1), errors="coerce").dropna()
    g2 = pd.to_numeric(pd.Series(group2), errors="coerce").dropna()
    if len(g1) >= 2 and len(g2) >= 2:
        stat, p = stats.levene(g1, g2)
        equal = p > 0.05
        checks.append({
            "label": "Equal variances (Levene)",
            "ok": True,  # Welch's t-test handles unequal variances regardless
            "detail": f"Levene p={p:.3f}. " +
                      ("Variances appear equal." if equal
                       else "Variances differ — Welch's correction (used here) handles this."),
        })

    checks.append({
        "label": "Independence of groups",
        "ok": True,
        "detail": "Assumed by design — observations in each group must be independent.",
    })
    return checks


def check_paired_t(diffs, n_pairs):
    checks = [_normality_check(diffs, "Normality of differences")]
    checks.append({
        "label": "Matched pairs",
        "ok": n_pairs >= 2,
        "detail": f"{n_pairs} matched pairs. Each pair must come from the same subject / unit measured twice.",
    })
    return checks


def check_chi_square(low_expected, n):
    return [
        {"label": "Categorical variables", "ok": True,
         "detail": "Chi-square requires nominal or ordinal data — confirmed."},
        {"label": "Expected cell frequencies >= 5", "ok": low_expected == 0,
         "detail": ("All expected cell frequencies >= 5."
                    if low_expected == 0
                    else f"{low_expected} cell(s) have expected frequency < 5 — "
                         "results may be unreliable; consider Fisher's exact test.")},
        {"label": "Independence of observations", "ok": True,
         "detail": "Each subject must contribute to only one cell of the table."},
    ]


def check_z_test(data1, data2=None):
    n1 = len(pd.to_numeric(pd.Series(data1), errors="coerce").dropna())
    checks = [{
        "label": "Known sigma or large sample",
        "ok": n1 >= 30,
        "detail": f"n={n1}. " +
                  ("Sample size >= 30 — Z-test is appropriate." if n1 >= 30
                   else "Small sample (n<30). Valid only if sigma is truly known; otherwise prefer a t-test."),
    }]
    if data2 is not None:
        n2 = len(pd.to_numeric(pd.Series(data2), errors="coerce").dropna())
        checks.append({
            "label": "Second group sample size",
            "ok": n2 >= 30,
            "detail": f"n={n2}. " +
                      ("Large enough." if n2 >= 30 else "Small sample — consider a t-test if sigma not known."),
        })
    return checks
