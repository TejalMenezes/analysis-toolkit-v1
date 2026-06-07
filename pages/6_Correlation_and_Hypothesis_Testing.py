
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hypothesis Testing · Smart Analysis Reporter",
                   page_icon="🧪", layout="wide")

from modules import ui
from modules.datasets import ensure_dataset_loaded
from modules.tests import (
    one_sample_ttest, independent_ttest, paired_ttest, anova, chi_square,
    one_sample_ztest, two_sample_ztest, effect_label_d, effect_label_v,
)
from modules.assumptions import (
    check_one_sample_t, check_independent_t, check_paired_t,
    check_chi_square, check_z_test,
)
from modules.correlation import pearson_corr, spearman_corr, kendall_corr
from modules import report as R

ui.setup()
ensure_dataset_loaded()
ui.header("Correlation & Hypothesis Testing", "Relationships and significance tests at α = 0.05.", icon="🧪")

ALPHA = 0.05
df = st.session_state["df"]
numeric_cols = list(df.select_dtypes(include="number").columns)
categorical_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
all_cols = list(df.columns)


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def show_assumptions(checks):
    with st.container(border=True):
        st.markdown("**Assumption checks**")
        for c in checks:
            st.markdown(f"{'✅' if c['ok'] else '⚠️'} **{c['label']}** — {c['detail']}")


def stash(title, inference, table=None, slot="tests"):
    """Persist a test result so it survives reruns and can be added to the report."""
    st.session_state[f"last_test_{slot}"] = {"title": title, "inference": inference, "table": table}


def render_last_test(slot="tests"):
    lt = st.session_state.get(f"last_test_{slot}")
    if not lt:
        return
    st.divider()
    st.markdown(f"#### Result · {lt['title']}")
    if "Reject H₀" in lt["inference"] and "Fail" not in lt["inference"]:
        st.success(lt["inference"])
    else:
        st.info(lt["inference"])
    if lt["table"] is not None:
        st.dataframe(lt["table"], use_container_width=True)
    R.add_to_report_button(lt["title"], table=lt["table"], inference=lt["inference"],
                           key=f"add_test_{lt['title']}")


tab_corr, tab_tests, tab_z = st.tabs(["Correlation", "Statistical Tests", "Z-Test"])

# ── CORRELATION ──
with tab_corr:
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns.")
    else:
        method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
        num_df = df[numeric_cols]
        corr = {"pearson": pearson_corr, "spearman": spearman_corr,
                "kendall": kendall_corr}[method](num_df)
        st.dataframe(corr.round(3), use_container_width=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Oranges",
                        zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        R.add_to_report_button(
            "Correlation Matrix", image=R.heatmap_png(corr),
            inference=R.correlation_inference(corr), key="add_corr",
        )

# ── STATISTICAL TESTS ──
with tab_tests:
    test = st.selectbox("Choose Test", [
        "One Sample T Test", "Independent T Test (Welch)", "Paired T Test",
        "ANOVA", "Chi Square",
    ])

    if test == "One Sample T Test":
        col = st.selectbox("Variable", numeric_cols)
        mu0 = st.number_input("Population Mean (μ₀)", value=0.0)
        if col:
            show_assumptions(check_one_sample_t(df[col]))
        if st.button("Run T Test", type="primary"):
            r = one_sample_ttest(df[col], mu0)
            inf = R.test_inference(
                r["p"] < ALPHA,
                f"the mean of {col} (M={r['mean']:.2f}) differs significantly from {mu0} "
                f"(t({r['df']})={r['stat']:.2f}, {fmt_p(r['p'])}).",
                f"no significant evidence the mean of {col} differs from {mu0} ({fmt_p(r['p'])}).")
            tbl = pd.DataFrame({"Value": [r["mean"], r["sd"], r["n"], r["stat"], r["p"]]},
                               index=["Mean", "SD", "N", "t", "p"])
            stash(f"One-sample t-test — {col}", inf, tbl)

    elif test == "Independent T Test (Welch)":
        num = st.selectbox("Numeric Variable", numeric_cols)
        cat = st.selectbox("Grouping Variable", all_cols)
        groups = list(df[cat].dropna().unique())
        if len(groups) != 2:
            st.warning(f"Grouping variable must have exactly 2 groups (found {len(groups)}).")
        else:
            g1 = df[df[cat] == groups[0]][num]
            g2 = df[df[cat] == groups[1]][num]
            show_assumptions(check_independent_t(g1, g2))
            if st.button("Run Test", type="primary"):
                r = independent_ttest(g1, g2)
                inf = R.test_inference(
                    r["p"] < ALPHA,
                    f"{num} differs significantly between \"{groups[0]}\" and \"{groups[1]}\" "
                    f"(t={r['stat']:.2f}, {fmt_p(r['p'])}, Cohen's d={r['cohen_d']:.2f}, "
                    f"{effect_label_d(r['cohen_d'])} effect).",
                    f"no significant difference in {num} between the groups ({fmt_p(r['p'])}).")
                tbl = pd.DataFrame({"Mean": [r["mean1"], r["mean2"]], "SD": [r["sd1"], r["sd2"]],
                                    "N": [r["n1"], r["n2"]]}, index=[str(groups[0]), str(groups[1])])
                stash(f"{num} by {cat} (Welch t-test)", inf, tbl)

    elif test == "Paired T Test":
        v1 = st.selectbox("Variable 1 (A)", numeric_cols, key="p_v1")
        v2 = st.selectbox("Variable 2 (B)", numeric_cols, key="p_v2")
        if v1 == v2:
            st.warning("Please select two different columns.")
        else:
            paired = pd.concat([df[v1], df[v2]], axis=1).dropna()
            diffs = paired.iloc[:, 0] - paired.iloc[:, 1]
            show_assumptions(check_paired_t(diffs, len(paired)))
            if st.button("Run Paired Test", type="primary"):
                r = paired_ttest(df[v1], df[v2])
                inf = R.test_inference(
                    r["p"] < ALPHA,
                    f"a significant mean difference between {v1} and {v2} "
                    f"(mean diff={r['mean_diff']:.2f}, {fmt_p(r['p'])}, d_z={r['cohen_dz']:.2f}).",
                    f"no significant mean difference between {v1} and {v2} ({fmt_p(r['p'])}).")
                stash(f"Paired t-test — {v1} vs {v2}", inf)

    elif test == "ANOVA":
        num = st.selectbox("Numeric", numeric_cols)
        cat = st.selectbox("Category", categorical_cols if categorical_cols else all_cols)
        if st.button("Run ANOVA", type="primary"):
            grps = [df[df[cat] == g][num].dropna() for g in df[cat].dropna().unique()]
            stat, p = anova(grps)
            inf = R.test_inference(
                p < ALPHA,
                f"at least one group mean of {num} differs across {cat} (F={stat:.2f}, {fmt_p(p)}).",
                f"no significant difference in {num} across {cat} groups ({fmt_p(p)}).")
            stash(f"ANOVA — {num} by {cat}", inf)

    elif test == "Chi Square":
        chi_cols = categorical_cols if len(categorical_cols) >= 2 else all_cols
        col1 = st.selectbox("Variable 1", chi_cols, key="chi1")
        col2 = st.selectbox("Variable 2", chi_cols, index=min(1, len(chi_cols) - 1), key="chi2")
        if col1 == col2:
            st.warning("Select two different variables.")
        elif st.button("Run Chi Square", type="primary"):
            r = chi_square(df, col1, col2)
            show_assumptions(check_chi_square(r["low_expected"], r["n"]))
            inf = R.test_inference(
                r["p"] < ALPHA,
                f"a significant association between {col1} and {col2} "
                f"(χ²({r['dof']})={r['chi2']:.2f}, {fmt_p(r['p'])}, Cramér's V={r['cramers_v']:.2f}, "
                f"{effect_label_v(r['cramers_v'])}).",
                f"no significant association between {col1} and {col2} ({fmt_p(r['p'])}).")
            stash(f"Chi-square — {col1} × {col2}", inf, r["table"])

    render_last_test()

# ── Z-TEST ──
with tab_z:
    st.caption("Use when population σ is known, or n ≥ 30.")
    ztest = st.selectbox("Z-test type", ["One-sample Z-test", "Two-sample Z-test"])

    if ztest == "One-sample Z-test":
        col = st.selectbox("Variable", numeric_cols, key="z1col")
        mu0 = st.number_input("Population mean (μ₀)", value=0.0, key="z1mu")
        sigma = st.number_input("Population std dev (σ)", value=1.0, min_value=0.0001, key="z1sig")
        if col:
            show_assumptions(check_z_test(df[col]))
        if st.button("Run Z-Test", type="primary", key="btn_z1"):
            r = one_sample_ztest(df[col], mu0, sigma)
            inf = R.test_inference(
                r["p"] < ALPHA,
                f"the mean of {col} (x̄={r['mean']:.2f}) significantly differs from μ₀={mu0} "
                f"(Z={r['z']:.2f}, {fmt_p(r['p'])}).",
                f"no significant evidence the mean of {col} differs from {mu0} ({fmt_p(r['p'])}).")
            stash(f"One-sample Z-test — {col}", inf, slot="z")

    else:
        num = st.selectbox("Metric variable", numeric_cols, key="z2col")
        grp = st.selectbox("Grouping variable", all_cols, key="z2grp")
        c1, c2 = st.columns(2)
        s1 = c1.number_input("σ₁", value=1.0, min_value=0.0001, key="z2s1")
        s2 = c2.number_input("σ₂", value=1.0, min_value=0.0001, key="z2s2")
        groups = list(df[grp].dropna().unique())
        if len(groups) != 2:
            st.warning(f"Grouping variable must have exactly 2 groups (found {len(groups)}).")
        else:
            g1 = df[df[grp] == groups[0]][num]
            g2 = df[df[grp] == groups[1]][num]
            show_assumptions(check_z_test(g1, g2))
            if st.button("Run Z-Test", type="primary", key="btn_z2"):
                r = two_sample_ztest(g1, g2, s1, s2)
                inf = R.test_inference(
                    r["p"] < ALPHA,
                    f"a significant difference in {num} between the two groups "
                    f"(Z={r['z']:.2f}, {fmt_p(r['p'])}).",
                    f"no significant difference in {num} between the groups ({fmt_p(r['p'])}).")
                stash(f"Two-sample Z-test — {num} by {grp}", inf, slot="z")

    render_last_test(slot="z")
