
"""Generate two deliverable PDFs from the dataset:

* Analysis Documentation — problem-statement driven (descriptive, categorical
  testing, correlation & regression, predictive trends, synthesis).
* System Documentation — abstract, system design, tech stack, dataset intro,
  analysis, conclusions — with charts and an architecture diagram.

Both reuse the themed chart renderers and inference builders in ``report.py``.
"""

import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from modules import report as R
from modules.ui import (ORANGE, ORANGE_DARK, ORANGE_SOFT, ORANGE_LINE, INK, MUTED,
                        APP_NAME, AUTHOR_NAME, AUTHOR_ID)
from modules.descriptive import describe_all
from modules.frequency import categorical_frequency
from modules.normality import normality_report
from modules.regression import simple_linear_regression
from modules.tests import independent_ttest, effect_label_d


# ───────────────── helpers to read the dataset's structure ─────────────────

def _numeric(df):
    return list(df.select_dtypes(include="number").columns)


def _categorical(df):
    return list(df.select_dtypes(include=["object", "category", "bool"]).columns)


def _pick_target(df, numeric):
    for cand in ("exam_score", "score", "target", "result"):
        for c in numeric:
            if c.lower() == cand:
                return c
    return max(numeric, key=lambda c: df[c].nunique()) if numeric else None


# ───────────────────────── architecture diagram ─────────────────────────

def architecture_png():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def box(x, y, w, h, label, fill):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.4, edgecolor=ORANGE_DARK, facecolor=fill))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8.5, color=INK, weight="bold")

    box(0.2, 1.4, 1.7, 1.2, "Dataset\n(CSV cache)", "#FFFFFF")
    box(2.4, 1.4, 1.9, 1.2, "Analysis modules\n(stats / tests /\nregression)", ORANGE_SOFT)
    box(4.8, 2.3, 2.0, 1.0, "Streamlit pages\n(interactive UI)", "#FFFFFF")
    box(4.8, 0.5, 2.0, 1.0, "Auto-report\nbuilder", ORANGE_SOFT)
    box(7.3, 1.4, 2.4, 1.2, "Report engine →\nPDF · DOCX · HTML", ORANGE)

    ax.text(8.5, 2.05, "exports", ha="center", fontsize=7, color="#fff")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))

    arrow(1.9, 2.0, 2.4, 2.0)
    arrow(4.3, 2.2, 4.8, 2.6)
    arrow(4.3, 1.8, 4.8, 1.0)
    arrow(6.8, 2.7, 7.4, 2.2)
    arrow(6.8, 1.0, 7.4, 1.7)
    ax.set_title("System architecture & data flow", color=INK, fontsize=11,
                 weight="bold", pad=8)
    return R._png(fig)


# ───────────────────────── ANALYSIS DOCUMENTATION ─────────────────────────

def build_analysis_report(df, name):
    """Assemble a problem-statement-driven report dict for the analysis doc."""
    numeric = _numeric(df)
    categorical = _categorical(df)
    target = _pick_target(df, numeric)

    R.reset_report()
    rep = R.get_report()
    rep["cover"].update({
        "title": "Analysis Documentation",
        "subtitle": f"{name} — a guided statistical study",
        "summary": (
            f"This document investigates the {name} dataset ({len(df):,} records, "
            f"{df.shape[1]} variables) through five guided questions — describing the "
            "data, testing group differences, measuring relationships, building a "
            "predictive model, and synthesising what it all means for understanding "
            "student performance."
        ),
    })

    # a. Descriptive ------------------------------------------------------
    desc = describe_all(df, numeric)
    R.add_item(
        "A. Descriptive Statistics — What do typical students look like?",
        inference=(
            "Problem statement: What are the central tendencies and spread of the "
            "measured study habits and outcomes? "
            + R.describe_inference(target, df[target]) +
            " These baselines frame every comparison and model that follows."
        ),
        table=desc,
    )
    R.add_item(
        f"A. Distribution of {target}",
        inference=R.describe_inference(target, df[target]),
        image=R.hist_png(df[target], title=f"Distribution of {target}"),
    )

    # b. Categorical testing ---------------------------------------------
    if categorical:
        cat = categorical[0]
        freq = categorical_frequency(df[cat])
        top = freq.iloc[0]
        R.add_item(
            f"B. Categorical Testing — Does {target} depend on {cat}?",
            inference=(
                f"Problem statement: Are outcomes distributed evenly across {cat}, "
                f"and does {target} differ between its groups? {cat} is dominated by "
                f"\"{top['Value']}\" ({top['Rel. Frequency %']}% of records)."
            ),
            image=R.bar_png(freq["Value"], freq["Abs. Frequency"],
                            title=f"{cat} distribution", xlabel=cat),
        )
        groups = list(df[cat].dropna().unique())
        if target and len(groups) == 2:
            g1, g2 = df[df[cat] == groups[0]][target], df[df[cat] == groups[1]][target]
            r = independent_ttest(g1, g2)
            sig = r["p"] < 0.05
            p_str = "< 0.001" if r["p"] < 0.001 else f"= {r['p']:.3f}"
            comp = pd.DataFrame({"Mean": [r["mean1"], r["mean2"]], "SD": [r["sd1"], r["sd2"]],
                                 "N": [r["n1"], r["n2"]]}, index=[str(groups[0]), str(groups[1])])
            R.add_item(
                f"B. {target} by {cat} — independent t-test",
                inference=R.test_inference(
                    sig,
                    f"{target} differs significantly between \"{groups[0]}\" and "
                    f"\"{groups[1]}\" (t = {r['stat']:.2f}, p {p_str}, Cohen's d = "
                    f"{r['cohen_d']:.2f}, {effect_label_d(r['cohen_d'])} effect), so {cat} "
                    "is a meaningful lens on performance.",
                    f"{target} does not differ significantly between \"{groups[0]}\" and "
                    f"\"{groups[1]}\" (p {p_str}); {cat} alone does not separate outcomes."),
                table=comp,
            )

    # c. Correlation & regression ----------------------------------------
    predictor = None
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        R.add_item(
            "C. Correlation — Which factors move together?",
            inference=("Problem statement: Which variables are most strongly related, "
                       "and could therefore drive performance? " + R.correlation_inference(corr)),
            image=R.heatmap_png(corr),
        )
        if target:
            corr_t = corr[target].drop(target).abs()
            predictor = corr_t.idxmax() if len(corr_t) else None

    # d. Predictive trend / linear regression ----------------------------
    if target and predictor:
        r = simple_linear_regression(df, predictor, target)
        if r:
            xnew = float(df[predictor].mean() + df[predictor].std())
            pred = r["intercept"] + r["slope"] * xnew
            R.add_item(
                f"D. Predictive Trend — Modelling {target} from {predictor}",
                inference=(
                    f"Problem statement: Can {target} be predicted from {predictor}? "
                    + R.regression_inference(predictor, target, r) +
                    f" For example, a student one standard deviation above the mean "
                    f"{predictor} ({xnew:.1f}) is predicted to score {pred:.1f}."
                ),
                image=R.regression_png(r["x"], r["y"], r["x_line"], r["y_line"],
                                       xlabel=predictor, ylabel=target,
                                       title=f"{target} vs {predictor}"),
            )

    # e. Synthesis --------------------------------------------------------
    norm = normality_report(df[target]) if target else {"normal": None}
    R.add_item(
        "E. Synthesis — Putting the picture together",
        inference=(
            "Tying the threads together: the descriptive profile sets expectations, "
            "categorical testing shows which groups differ, correlation highlights the "
            "factors that move with performance, and the regression turns the strongest "
            f"of those into a predictive rule. Together they describe how "
            f"{predictor or 'study-related factors'} relate to {target}, give a baseline "
            "model for prediction, and flag where richer multivariate models could "
            "improve on a single predictor."
        ),
    )
    return rep


def build_analysis_doc(df, name):
    rep = build_analysis_report(df, name)
    return R.build_pdf(rep)


# ───────────────────────── SYSTEM DOCUMENTATION ─────────────────────────

def _key_findings(df):
    numeric, categorical = _numeric(df), _categorical(df)
    target = _pick_target(df, numeric)
    out = []
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i, j])
                 for i in range(len(corr.columns)) for j in range(i + 1, len(corr.columns))]
        a, b, rv = max(pairs, key=lambda p: abs(p[2]))
        out.append(f"The strongest linear relationship is {a}–{b} (r = {rv:.2f}).")
        if target:
            ct = corr[target].drop(target).abs()
            if len(ct):
                p = ct.idxmax()
                r = simple_linear_regression(df, p, target)
                if r:
                    out.append(f"A simple model of {target} on {p} explains "
                               f"{r['r2']*100:.1f}% of the variance (R² = {r['r2']:.2f}).")
    if categorical and target:
        cat = categorical[0]
        g = list(df[cat].dropna().unique())
        if len(g) == 2:
            r = independent_ttest(df[df[cat] == g[0]][target], df[df[cat] == g[1]][target])
            verdict = "a significant" if r["p"] < 0.05 else "no significant"
            out.append(f"There is {verdict} difference in {target} between {cat} groups "
                       f"(p {'< 0.001' if r['p'] < 0.001 else f'= {r['p']:.3f}'}).")
    if target:
        nr = normality_report(df[target])
        if nr["W"] is not None:
            out.append(f"{target} is "
                       f"{'approximately normal' if nr['normal'] else 'non-normal'} "
                       f"(Shapiro-Wilk W = {nr['W']}).")
    return out


def build_system_doc(df, name):
    import io as _io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, ListFlowable, ListItem, PageBreak)

    numeric, categorical = _numeric(df), _categorical(df)
    target = _pick_target(df, numeric)

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=HexColor(ORANGE_DARK), fontSize=16, spaceBefore=16, spaceAfter=6)
    BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=10.5, leading=15.5, textColor=HexColor("#2A2A2A"), spaceAfter=6)
    CAP = ParagraphStyle("CAP", parent=ss["BodyText"], fontSize=8.5, leading=11, textColor=HexColor(MUTED), spaceAfter=10)

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.4*cm, bottomMargin=1.4*cm,
                            leftMargin=1.7*cm, rightMargin=1.7*cm, title="System Documentation")
    W = doc.width
    S = R._rl_styles()
    story = []

    # cover
    cover_rows = [
        [Paragraph(APP_NAME.upper(), S["cover_app"])],
        [Paragraph("System Documentation", S["cover_title"])],
        [Paragraph(f"{name}", S["cover_sub"])],
        [Paragraph(f"Prepared by <b>{AUTHOR_NAME}</b> &nbsp;·&nbsp; ID {AUTHOR_ID} "
                   f"&nbsp;·&nbsp; {date.today().isoformat()}", S["cover_meta"])],
    ]
    band = Table(cover_rows, colWidths=[W])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), HexColor(ORANGE)),
                              ("LEFTPADDING", (0, 0), (-1, -1), 22), ("RIGHTPADDING", (0, 0), (-1, -1), 22),
                              ("TOPPADDING", (0, 0), (0, 0), 26), ("BOTTOMPADDING", (-1, -1), (-1, -1), 24)]))
    story += [band, Spacer(1, 16)]

    # 1 Abstract
    story += [Paragraph("1. Abstract", H1), Paragraph(
        f"{APP_NAME} is a web-based statistical analysis and reporting toolkit. It loads a "
        f"dataset, runs a battery of descriptive and inferential analyses, and assembles an "
        f"editable, exportable analytics report. This document describes the system behind the "
        f"toolkit and demonstrates it on the {name} dataset ({len(df):,} rows × {df.shape[1]} "
        f"columns), whose goal is to understand and predict student exam performance.", BODY)]

    # 2 System design
    story += [Paragraph("2. System Design", H1), Paragraph(
        "The application follows a layered, modular design. A thin data layer caches the "
        "dataset; a set of pure-Python analysis modules implement the statistics; a Streamlit "
        "multipage UI exposes them interactively; and a report engine turns any analysis into "
        "embeddable charts, tables and narrative that export to PDF, Word and HTML. An "
        "auto-report builder chains the modules to produce a full report in one click.", BODY),
        R._rl_image(architecture_png(), W),
        Paragraph("Figure 1 — System architecture and data flow.", CAP)]

    # 3 Tech stack
    story += [Paragraph("3. Technology Stack", H1)]
    stack = [["Layer", "Technology", "Purpose"],
             ["UI", "Streamlit", "Multipage interactive web app & widgets"],
             ["Computation", "pandas, NumPy", "Data handling and vectorised maths"],
             ["Statistics", "SciPy, statsmodels", "Tests, distributions, OLS regression"],
             ["Interactive charts", "Plotly", "On-screen exploration"],
             ["Report charts", "Matplotlib", "Themed images embedded in exports"],
             ["Export", "ReportLab, python-docx", "PDF and Word generation"],
             ["Data source", "kagglehub", "Fetching the demonstration dataset"]]
    t = Table(stack, colWidths=[W*0.22, W*0.28, W*0.50])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HexColor(ORANGE)),
                           ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor(ORANGE_SOFT)]),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.4, HexColor(ORANGE_LINE)),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story += [t, Spacer(1, 6)]

    # 4 Dataset introduction
    story += [PageBreak(), Paragraph("4. Dataset Introduction", H1), Paragraph(
        f"The {name} dataset contains {len(df):,} student records with {df.shape[1]} variables "
        f"({len(numeric)} metric, {len(categorical)} categorical) and "
        f"{int(df.isna().sum().sum())} missing values. The columns are:", BODY)]
    cols_tbl = [["Column", "Type", "Example"]]
    for c in df.columns:
        kind = "Categorical" if c in categorical else "Metric"
        cols_tbl.append([c, kind, str(df[c].iloc[0])])
    t2 = Table(cols_tbl, colWidths=[W*0.4, W*0.25, W*0.35])
    t2.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HexColor(ORANGE_SOFT)),
                            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor(ORANGE_DARK)),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#FBFBFB")]),
                            ("LINEBELOW", (0, 0), (-1, -1), 0.4, HexColor("#EEEEEE")),
                            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story += [t2, Spacer(1, 8)]
    if target:
        story += [R._rl_image(R.hist_png(df[target], title=f"Distribution of {target}"), W*0.8),
                  Paragraph(f"Figure 2 — Distribution of the target variable, {target}.", CAP)]

    # 5 Analysis
    story += [Paragraph("5. Analysis", H1), Paragraph(
        "The toolkit applies descriptive statistics, frequency analysis, normality checks "
        "(Q-Q + Shapiro-Wilk), correlation, hypothesis tests (t, ANOVA, chi-square, Z) with "
        "live assumption checks, and linear regression. Two representative outputs are shown "
        "below.", BODY)]
    if len(numeric) >= 2:
        story += [R._rl_image(R.heatmap_png(df[numeric].corr()), W*0.8),
                  Paragraph("Figure 3 — Correlation matrix of the metric variables.", CAP)]
        if target:
            ct = df[numeric].corr()[target].drop(target).abs()
            if len(ct):
                p = ct.idxmax()
                r = simple_linear_regression(df, p, target)
                if r:
                    story += [R._rl_image(R.regression_png(
                        r["x"], r["y"], r["x_line"], r["y_line"], xlabel=p, ylabel=target,
                        title=f"{target} vs {p}"), W*0.8),
                        Paragraph(f"Figure 4 — Linear regression of {target} on {p}.", CAP)]

    # 6 Conclusion of analysis
    findings = _key_findings(df)
    story += [Paragraph("6. Conclusion of Analysis", H1),
              ListFlowable([ListItem(Paragraph(f, BODY), bulletColor=HexColor(ORANGE))
                            for f in findings], bulletType="bullet", start="•")]

    # 7 Project conclusion
    story += [Paragraph("7. Project Conclusion", H1), Paragraph(
        f"{APP_NAME} demonstrates an end-to-end analytics workflow: from raw data to an "
        "interactive toolkit to a polished, editable report — without writing code for each "
        "study. Its modular structure means new analyses or datasets can be added with minimal "
        "change, and its one-click reporting makes findings immediately shareable. The "
        f"{name} case study shows the toolkit producing a complete, defensible statistical "
        "narrative automatically.", BODY)]

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
