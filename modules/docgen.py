
"""Professional documentation generators (PDF + Word).

Each document is defined once as a content *model* — a list of sections, each a
title plus typed blocks (heading, paragraph, bullets, problem-statement box,
figure, caption, table). The same model is rendered to:

* PDF  (ReportLab)      — cover page, table of contents, page headers/footers
                          with page numbers, each section on its own page.
* DOCX (python-docx)    — matching cover, Word TOC field, heading styles, page
                          breaks, footer page numbers.

Two documents are produced from the dataset:
  * Analysis Documentation — a guided statistical study.
  * System Documentation   — design, tech stack, dataset, analysis, conclusions.
"""

import io
import re
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


# ───────────────── content model (block constructors) ─────────────────

def H2(text):       return ("h2", text)
def P(text):        return ("para", text)
def BUL(items):     return ("bullets", items)
def PROB(text):     return ("problem", text)
def FIG(png, frac=0.82): return ("figure", png, frac)
def CAP(text):      return ("caption", text)
def TBL(df):        return ("table", df)


def SEC(title, *blocks):
    return {"title": title, "blocks": [b for b in blocks if b is not None]}


# ───────────────── dataset structure helpers ─────────────────

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


# ───────────────── architecture diagram ─────────────────

def architecture_png():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)

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

    arrow(1.9, 2.0, 2.4, 2.0); arrow(4.3, 2.2, 4.8, 2.6); arrow(4.3, 1.8, 4.8, 1.0)
    arrow(6.8, 2.7, 7.4, 2.2); arrow(6.8, 1.0, 7.4, 1.7)
    ax.set_title("System architecture & data flow", color=INK, fontsize=11, weight="bold", pad=8)
    return R._png(fig)


def _key_findings(df):
    numeric, categorical = _numeric(df), _categorical(df)
    target = _pick_target(df, numeric)
    out = []
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i, j])
                 for i in range(len(corr.columns)) for j in range(i + 1, len(corr.columns))]
        a, b, rv = max(pairs, key=lambda p: abs(p[2]))
        out.append(f"The strongest linear relationship is between <b>{a}</b> and <b>{b}</b> (r = {rv:.2f}).")
        if target:
            ct = corr[target].drop(target).abs()
            if len(ct):
                p = ct.idxmax()
                r = simple_linear_regression(df, p, target)
                if r:
                    out.append(f"A simple model of <b>{target}</b> on <b>{p}</b> explains "
                               f"{r['r2']*100:.1f}% of its variance (R² = {r['r2']:.2f}).")
    if categorical and target:
        cat = categorical[0]
        g = list(df[cat].dropna().unique())
        if len(g) == 2:
            r = independent_ttest(df[df[cat] == g[0]][target], df[df[cat] == g[1]][target])
            verdict = "a statistically significant" if r["p"] < 0.05 else "no significant"
            out.append(f"There is {verdict} difference in <b>{target}</b> between <b>{cat}</b> "
                       f"groups (p {'< 0.001' if r['p'] < 0.001 else f'= {r['p']:.3f}'}).")
    if target:
        nr = normality_report(df[target])
        if nr["W"] is not None:
            out.append(f"<b>{target}</b> is "
                       f"{'approximately normally distributed' if nr['normal'] else 'not normally distributed'} "
                       f"(Shapiro-Wilk W = {nr['W']}).")
    return out


# ═══════════════════ DOCUMENT MODELS ═══════════════════

def build_analysis_model(df, name):
    numeric, categorical = _numeric(df), _categorical(df)
    target = _pick_target(df, numeric)
    sections = []

    sections.append(SEC(
        "1. Introduction",
        H2("1.1  Purpose of this document"),
        P("This document presents a structured statistical analysis of the "
          f"<b>{name}</b> dataset. It is written to be read top-to-bottom: each section poses "
          "a concrete question, states the method used to answer it, presents the evidence (a "
          "chart or table), and closes with an interpretation. The goal is a defensible, "
          "reproducible understanding of what drives student outcomes."),
        H2("1.2  Problem statement"),
        PROB("Educational institutions need to understand which study-related factors are "
             "associated with student exam performance, whether outcomes differ across student "
             "groups, and whether performance can be predicted from measurable behaviours. This "
             f"study interrogates the {name} dataset ({len(df):,} students, {df.shape[1]} "
             "variables) to answer those questions."),
        H2("1.3  Objectives"),
        BUL(["Describe the central tendency, spread and shape of the key variables.",
             "Test whether exam performance differs significantly across categorical groups.",
             "Measure which factors are most strongly correlated with performance.",
             "Build and evaluate a predictive linear-regression model.",
             "Synthesise the findings into an overall understanding of the dataset."]),
        H2("1.4  Methodology & tools"),
        P("Analyses were produced with the Smart Analysis Reporter toolkit (Python, pandas, "
          "SciPy, statsmodels). Descriptive statistics, frequency analysis, correlation, "
          "independent-samples t-tests and ordinary-least-squares regression were applied, each "
          "with a significance level of α = 0.05."),
    ))

    desc = describe_all(df, numeric)
    sections.append(SEC(
        "2. Descriptive Analysis",
        PROB("What are the typical values, variability and distribution shape of the study "
             "habits and performance measures in the dataset?"),
        H2("2.1  Method"),
        P("Summary statistics (mean, median, mode, variance, standard deviation, quartiles, "
          "skewness and kurtosis) were computed for every metric variable."),
        H2("2.2  Results"),
        TBL(desc),
        FIG(R.hist_png(df[target], title=f"Distribution of {target}")) if target else None,
        CAP(f"Figure 2.1 — Distribution of the target variable, {target}.") if target else None,
        H2("2.3  Interpretation"),
        P(R.describe_inference(target, df[target]) if target else
          "The table above establishes the baseline profile of every metric variable."),
    ))

    cat_blocks = [PROB("Do students differ in exam performance across categorical groups — "
                       "for example, between placement outcomes?")]
    if categorical:
        cat = categorical[0]
        freq = categorical_frequency(df[cat])
        top = freq.iloc[0]
        cat_blocks += [
            H2("3.1  Group composition"),
            FIG(R.bar_png(freq["Value"], freq["Abs. Frequency"],
                          title=f"{cat} distribution", xlabel=cat)),
            CAP(f"Figure 3.1 — Distribution of {cat}."),
            P(f"{cat} has {len(freq)} categories; the most common is \"{top['Value']}\" "
              f"({top['Rel. Frequency %']}% of records)."),
        ]
        groups = list(df[cat].dropna().unique())
        if target and len(groups) == 2:
            g1, g2 = df[df[cat] == groups[0]][target], df[df[cat] == groups[1]][target]
            r = independent_ttest(g1, g2)
            sig = r["p"] < 0.05
            p_str = "< 0.001" if r["p"] < 0.001 else f"= {r['p']:.3f}"
            comp = pd.DataFrame({"Mean": [r["mean1"], r["mean2"]], "SD": [r["sd1"], r["sd2"]],
                                 "N": [r["n1"], r["n2"]]}, index=[str(groups[0]), str(groups[1])])
            comp.index.name = str(cat)
            cat_blocks += [
                H2("3.2  Hypothesis test (independent-samples t-test)"),
                P(f"H₀: the mean {target} is equal across {cat} groups. "
                  f"H₁: the means differ."),
                TBL(comp),
                H2("3.3  Interpretation"),
                P(R.test_inference(
                    sig,
                    f"{target} differs significantly between \"{groups[0]}\" and \"{groups[1]}\" "
                    f"(t = {r['stat']:.2f}, p {p_str}, Cohen's d = {r['cohen_d']:.2f}, "
                    f"{effect_label_d(r['cohen_d'])} effect).",
                    f"{target} does not differ significantly between \"{groups[0]}\" and "
                    f"\"{groups[1]}\" (p {p_str}).")),
            ]
    sections.append(SEC("3. Categorical Testing", *cat_blocks))

    predictor = None
    corr_blocks = [PROB("Which variables are most strongly associated with exam performance, "
                        "and could therefore serve as predictors?")]
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        if target:
            ct = corr[target].drop(target).abs()
            predictor = ct.idxmax() if len(ct) else None
        corr_blocks += [
            H2("4.1  Method"),
            P("Pearson correlation coefficients were computed for all metric variable pairs "
              "(range −1 to +1)."),
            H2("4.2  Results"),
            FIG(R.heatmap_png(corr)),
            CAP("Figure 4.1 — Correlation matrix of the metric variables."),
            H2("4.3  Interpretation"),
            P(R.correlation_inference(corr)),
        ]
    sections.append(SEC("4. Correlation & Relationships", *corr_blocks))

    reg_blocks = [PROB("Can exam performance be predicted from its strongest single factor, and "
                       "how much of the variation does that factor explain?")]
    if target and predictor:
        r = simple_linear_regression(df, predictor, target)
        if r:
            xnew = float(df[predictor].mean() + df[predictor].std())
            pred = r["intercept"] + r["slope"] * xnew
            reg_blocks += [
                H2("5.1  Method"),
                P(f"A simple ordinary-least-squares linear regression was fitted, modelling "
                  f"{target} as a function of {predictor}."),
                H2("5.2  Results"),
                FIG(R.regression_png(r["x"], r["y"], r["x_line"], r["y_line"],
                                     xlabel=predictor, ylabel=target,
                                     title=f"{target} vs {predictor}")),
                CAP(f"Figure 5.1 — Regression of {target} on {predictor}."),
                H2("5.3  Interpretation"),
                P(R.regression_inference(predictor, target, r)),
                P(f"As a worked example, a student one standard deviation above the mean "
                  f"{predictor} ({xnew:.1f}) is predicted to score <b>{pred:.1f}</b> on {target}."),
            ]
    sections.append(SEC("5. Predictive Trends — Linear Regression", *reg_blocks))

    sections.append(SEC(
        "6. Synthesis & Conclusions",
        H2("6.1  Bringing the analysis together"),
        P("Each section answered one question, and together they form a coherent picture. The "
          "descriptive profile set expectations; categorical testing showed where groups "
          "genuinely differ; correlation identified the factors that move with performance; and "
          "the regression turned the strongest of those into a predictive rule."),
        H2("6.2  Key findings"),
        BUL(_key_findings(df)),
        H2("6.3  Limitations & next steps"),
        BUL(["A single-predictor model is a baseline — a multiple-regression model using "
             "several factors together would likely explain more variance.",
             "Correlation does not imply causation; a controlled study is needed to confirm drivers.",
             "Findings describe this sample; external validation on new cohorts is recommended."]),
    ))

    meta = {"title": "Analysis Documentation", "doc_kind": "Statistical Analysis Report",
            "subtitle": name}
    return meta, sections


def build_system_model(df, name):
    numeric, categorical = _numeric(df), _categorical(df)
    target = _pick_target(df, numeric)
    sections = []

    sections.append(SEC(
        "1. Introduction & Purpose",
        H2("1.1  About this document"),
        P("This document describes the design and implementation of <b>Smart Analysis "
          "Reporter</b>, a web-based statistical analysis and reporting toolkit. It is intended "
          "for evaluators and future maintainers, and explains what the system does, how it is "
          "built, the technologies it uses, and the results it produces on its demonstration "
          "dataset."),
        H2("1.2  Problem the system solves"),
        PROB("Producing a statistical report normally means repeating the same manual steps — "
             "loading data, running tests, making charts, writing up findings — for every "
             "dataset. Smart Analysis Reporter automates that pipeline: it analyses a dataset "
             "and assembles an editable, exportable analytics report with minimal effort."),
        H2("1.3  Scope"),
        BUL(["Interactive exploration: profiling, descriptive statistics, frequency tables, "
             "Q-Q plots, correlation, hypothesis testing and regression.",
             "Automated reporting: one-click report generation with editable narrative.",
             "Export: PDF, Word and HTML deliverables."]),
    ))

    sections.append(SEC(
        "2. Abstract",
        P(f"Smart Analysis Reporter loads a dataset, runs a battery of descriptive and "
          f"inferential analyses, and assembles an editable analytics report that exports to "
          f"PDF, Word and HTML. This document demonstrates the system on the <b>{name}</b> "
          f"dataset ({len(df):,} rows × {df.shape[1]} columns), whose objective is to "
          f"understand and predict student exam performance. The toolkit is organised as a "
          f"layered, modular Python application with a Streamlit front end and a dedicated "
          f"report engine."),
    ))

    sections.append(SEC(
        "3. System Design",
        H2("3.1  Architecture"),
        P("The application follows a layered, modular design with a clear separation of "
          "concerns. A thin data layer caches the dataset; pure-Python analysis modules "
          "implement the statistics independently of the UI; a Streamlit multipage front end "
          "exposes them interactively; and a report engine converts any analysis into "
          "embeddable charts, tables and narrative."),
        FIG(architecture_png()),
        CAP("Figure 3.1 — System architecture and data flow."),
        H2("3.2  Data flow"),
        BUL(["The dataset is loaded once into session state and shared across all pages.",
             "Each analysis page calls the relevant module and renders interactive Plotly charts.",
             "An “Add to report” action captures any chart/table plus an auto-written inference.",
             "The auto-report builder chains the modules to assemble a full report in one click.",
             "The report engine renders themed Matplotlib images and exports PDF / DOCX / HTML."]),
        H2("3.3  Design principles"),
        BUL(["Modularity — analysis logic lives in importable modules, not in the UI.",
             "Reusability — the same chart and inference helpers power the app and these documents.",
             "Portability — the dataset is cached in-repo so the deployed app needs no credentials."]),
    ))

    stack = pd.DataFrame({
        "Technology": ["Streamlit", "pandas, NumPy", "SciPy, statsmodels", "Plotly",
                       "Matplotlib", "ReportLab, python-docx", "kagglehub"],
        "Purpose": ["Multipage interactive web app & widgets",
                    "Data handling and vectorised computation",
                    "Statistical tests, distributions, OLS regression",
                    "Interactive on-screen charts",
                    "Themed images embedded in exported documents",
                    "PDF and Word document generation",
                    "Fetching the demonstration dataset"],
    }, index=["UI", "Computation", "Statistics", "Interactive charts",
              "Report charts", "Export", "Data source"])
    stack.index.name = "Layer"
    sections.append(SEC(
        "4. Technology Stack",
        P("The toolkit is built entirely in Python with the following components:"),
        TBL(stack),
    ))

    cols = pd.DataFrame({
        "Type": ["Categorical" if c in categorical else "Metric" for c in df.columns],
        "Example": [str(df[c].iloc[0]) for c in df.columns],
        "Unique": [int(df[c].nunique()) for c in df.columns],
    }, index=list(df.columns))
    cols.index.name = "Column"
    ds_blocks = [
        P(f"The <b>{name}</b> dataset contains {len(df):,} student records with {df.shape[1]} "
          f"variables ({len(numeric)} metric, {len(categorical)} categorical) and "
          f"{int(df.isna().sum().sum())} missing values. The target variable is <b>{target}</b>. "
          "The full schema is:"),
        TBL(cols),
    ]
    if target:
        ds_blocks += [FIG(R.hist_png(df[target], title=f"Distribution of {target}"), 0.7),
                      CAP(f"Figure 5.1 — Distribution of the target variable, {target}.")]
    sections.append(SEC("5. Dataset Introduction", *ds_blocks))

    an_blocks = [P("The toolkit applies descriptive statistics, frequency analysis, normality "
                   "checks (Q-Q plot + Shapiro-Wilk), correlation, hypothesis tests (t-tests, "
                   "ANOVA, chi-square, Z-tests) with live assumption checks, and linear "
                   "regression. Two representative outputs are shown below.")]
    if len(numeric) >= 2:
        an_blocks += [FIG(R.heatmap_png(df[numeric].corr())),
                      CAP("Figure 6.1 — Correlation matrix of the metric variables.")]
        if target:
            ct = df[numeric].corr()[target].drop(target).abs()
            if len(ct):
                p = ct.idxmax()
                r = simple_linear_regression(df, p, target)
                if r:
                    an_blocks += [FIG(R.regression_png(r["x"], r["y"], r["x_line"], r["y_line"],
                                                       xlabel=p, ylabel=target,
                                                       title=f"{target} vs {p}")),
                                  CAP(f"Figure 6.2 — Linear regression of {target} on {p}.")]
    sections.append(SEC("6. Analysis", *an_blocks))

    sections.append(SEC(
        "7. Conclusion of Analysis",
        P("Applied to the demonstration dataset, the toolkit produced the following "
          "evidence-based findings:"),
        BUL(_key_findings(df)),
    ))

    sections.append(SEC(
        "8. Project Conclusion",
        P(f"Smart Analysis Reporter demonstrates an end-to-end analytics workflow — from raw "
          f"data, through an interactive toolkit, to a polished and editable report — without "
          f"hand-coding each study. Its modular structure means new analyses or datasets can be "
          f"added with minimal change, and its one-click reporting makes findings immediately "
          f"shareable. The {name} case study shows the toolkit producing a complete, defensible "
          f"statistical narrative automatically."),
        H2("8.1  Future work"),
        BUL(["Multiple-regression and classification models for richer prediction.",
             "Automated outlier and data-quality reporting.",
             "User accounts and saved report templates."]),
    ))

    meta = {"title": "System Documentation", "doc_kind": "Technical & System Report",
            "subtitle": name}
    return meta, sections


# ═══════════════════ PDF RENDERER ═══════════════════

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Table, TableStyle, PageBreak, NextPageTemplate,
                                ListFlowable, ListItem)
from reportlab.platypus.tableofcontents import TableOfContents

_ss = getSampleStyleSheet()
_PS = {
    "h1": ParagraphStyle("SecH1", parent=_ss["Heading1"], textColor=HexColor(ORANGE_DARK),
                         fontSize=17, leading=21, spaceBefore=2, spaceAfter=10),
    "h2": ParagraphStyle("SecH2", parent=_ss["Heading2"], textColor=HexColor(INK),
                         fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=4),
    "body": ParagraphStyle("Body", parent=_ss["BodyText"], fontSize=10.5, leading=15.5,
                           textColor=HexColor("#2A2A2A"), spaceAfter=8, alignment=TA_LEFT),
    "bullet": ParagraphStyle("Bullet", parent=_ss["BodyText"], fontSize=10.5, leading=15,
                             textColor=HexColor("#2A2A2A")),
    "cap": ParagraphStyle("Cap", parent=_ss["BodyText"], fontSize=8.5, leading=11,
                          textColor=HexColor(MUTED), spaceBefore=3, spaceAfter=12),
    "tocHead": ParagraphStyle("TocHead", parent=_ss["Heading1"], textColor=HexColor(INK),
                              fontSize=16, spaceAfter=14),
    "pbLabel": ParagraphStyle("PbLabel", parent=_ss["BodyText"], fontSize=8.5,
                              textColor=HexColor(ORANGE_DARK), spaceAfter=2),
    "pbText": ParagraphStyle("PbText", parent=_ss["BodyText"], fontSize=10.5, leading=15,
                             textColor=HexColor(INK)),
}
_AVAIL_W = A4[0] - 4 * cm


def _pb_box(text):
    inner = [[Paragraph("PROBLEM STATEMENT", _PS["pbLabel"])], [Paragraph(text, _PS["pbText"])]]
    t = Table(inner, colWidths=[_AVAIL_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor(ORANGE_SOFT)),
        ("LINEBEFORE", (0, 0), (0, -1), 3, HexColor(ORANGE)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (0, 0), 9), ("BOTTOMPADDING", (-1, -1), (-1, -1), 10),
        ("TOPPADDING", (0, 1), (0, 1), 0),
    ]))
    return t


def _pdf_block(b):
    kind = b[0]
    if kind == "h2":      return _PS_para(b[1], "h2")
    if kind == "para":    return _PS_para(b[1], "body")
    if kind == "caption": return _PS_para(b[1], "cap")
    if kind == "problem": return _pb_box(b[1])
    if kind == "figure":  return R._rl_image(b[1], _AVAIL_W * b[2])
    if kind == "table":   return R._rl_table(b[1], _AVAIL_W)
    if kind == "bullets":
        return ListFlowable([ListItem(Paragraph(t, _PS["bullet"]), bulletColor=HexColor(ORANGE),
                                      value="•") for t in b[1]],
                            bulletType="bullet", start="•", leftIndent=14)
    return Spacer(1, 1)


def _PS_para(text, style):
    return Paragraph(text, _PS[style])


class _DocTemplate(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "SecH1":
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))


def render_pdf(meta, sections):
    buf = io.BytesIO()
    W, H = A4

    def draw_cover(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(HexColor(ORANGE)); canvas.rect(0, H - 10.5 * cm, W, 10.5 * cm, fill=1, stroke=0)
        canvas.setFillColor(HexColor(ORANGE_DARK)); canvas.rect(0, H - 10.9 * cm, W, 0.4 * cm, fill=1, stroke=0)
        canvas.setFillColor(white)
        canvas.setFont("Helvetica-Bold", 12); canvas.drawString(2 * cm, H - 2.3 * cm, APP_NAME.upper())
        canvas.setFont("Helvetica", 10); canvas.drawString(2 * cm, H - 2.9 * cm, meta["doc_kind"])
        canvas.setFont("Helvetica-Bold", 30); canvas.drawString(2 * cm, H - 5.2 * cm, meta["title"])
        canvas.setFont("Helvetica", 14); canvas.drawString(2 * cm, H - 6.3 * cm, meta["subtitle"])
        canvas.setFont("Helvetica", 11)
        canvas.drawString(2 * cm, H - 8.6 * cm, f"Prepared by {AUTHOR_NAME}")
        canvas.drawString(2 * cm, H - 9.2 * cm, f"Student ID: {AUTHOR_ID}")
        canvas.drawString(2 * cm, H - 9.8 * cm, f"Date: {date.today().isoformat()}")
        canvas.setFillColor(HexColor(MUTED)); canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, 1.4 * cm, f"{APP_NAME} — Tools & Methods of Data Analysis")
        canvas.restoreState()

    def draw_body(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(HexColor(ORANGE_LINE)); canvas.setLineWidth(1)
        canvas.line(2 * cm, H - 1.55 * cm, W - 2 * cm, H - 1.55 * cm)
        canvas.setFillColor(HexColor(MUTED)); canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, H - 1.4 * cm, meta["title"])
        canvas.drawRightString(W - 2 * cm, H - 1.4 * cm, APP_NAME)
        canvas.line(2 * cm, 1.5 * cm, W - 2 * cm, 1.5 * cm)
        canvas.drawString(2 * cm, 1.1 * cm, f"{AUTHOR_NAME} · ID {AUTHOR_ID}")
        canvas.drawRightString(W - 2 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = _DocTemplate(buf, pagesize=A4, title=meta["title"], author=AUTHOR_NAME,
                       leftMargin=2 * cm, rightMargin=2 * cm)
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[Frame(2 * cm, 2 * cm, W - 4 * cm, H - 4 * cm)], onPage=draw_cover),
        PageTemplate(id="body", frames=[Frame(2 * cm, 1.8 * cm, W - 4 * cm, H - 3.8 * cm)], onPage=draw_body),
    ])

    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc0", fontSize=11, leading=20, textColor=HexColor(INK))]

    story = [Spacer(1, 1), NextPageTemplate("body"), PageBreak(),
             Paragraph("Table of Contents", _PS["tocHead"]), toc, PageBreak()]

    for sec in sections:
        story.append(Paragraph(sec["title"], _PS["h1"]))
        for b in sec["blocks"]:
            story.append(_pdf_block(b))
        story.append(PageBreak())

    doc.multiBuild(story)
    buf.seek(0)
    return buf.getvalue()


# ═══════════════════ DOCX RENDERER ═══════════════════

def _docx_runs(paragraph, text):
    bold = False
    for tok in re.split(r"(<b>|</b>)", text):
        if tok == "<b>":
            bold = True
        elif tok == "</b>":
            bold = False
        elif tok:
            run = paragraph.add_run(tok)
            run.bold = bold


def _shade(cell_or_para_pr, hex_color):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_color)
    cell_or_para_pr.append(shd)


def render_docx(meta, sections):
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    O = RGBColor(0xF5, 0x7C, 0x00)
    OD = RGBColor(0xE6, 0x51, 0x00)
    GREY = RGBColor(0x7A, 0x82, 0x90)

    doc = Document()
    # base styling
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)

    sec0 = doc.sections[0]
    sec0.top_margin = Cm(2.2); sec0.bottom_margin = Cm(2)
    sec0.left_margin = Cm(2.2); sec0.right_margin = Cm(2.2)

    # footer with page number
    footer = sec0.footer
    fp = footer.paragraphs[0]
    fp.text = f"{AUTHOR_NAME} · ID {AUTHOR_ID}        "
    fp.runs[0].font.size = Pt(8); fp.runs[0].font.color.rgb = GREY
    run = fp.add_run("Page ")
    run.font.size = Pt(8); run.font.color.rgb = GREY
    # PAGE field
    fld1 = OxmlElement("w:fldSimple"); fld1.set(qn("w:instr"), "PAGE")
    fp._p.append(fld1)

    # header with doc title
    hp = sec0.header.paragraphs[0]
    hp.text = meta["title"]
    hp.runs[0].font.size = Pt(8); hp.runs[0].font.color.rgb = GREY

    # ── cover ──
    band = doc.add_paragraph()
    _shade(band._p.get_or_add_pPr(), "F57C00")
    band.paragraph_format.space_after = Pt(2)
    r = band.add_run(APP_NAME.upper()); r.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.font.size = Pt(12)
    k = doc.add_paragraph(); _shade(k._p.get_or_add_pPr(), "F57C00")
    rk = k.add_run(meta["doc_kind"]); rk.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); rk.font.size = Pt(10)
    tt = doc.add_paragraph(); _shade(tt._p.get_or_add_pPr(), "F57C00")
    rt = tt.add_run(meta["title"]); rt.bold = True; rt.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); rt.font.size = Pt(28)
    sb = doc.add_paragraph(); _shade(sb._p.get_or_add_pPr(), "F57C00")
    rs = sb.add_run(meta["subtitle"]); rs.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); rs.font.size = Pt(14)
    sb.paragraph_format.space_after = Pt(16)

    for line in [f"Prepared by {AUTHOR_NAME}", f"Student ID: {AUTHOR_ID}", f"Date: {date.today().isoformat()}"]:
        p = doc.add_paragraph(); rr = p.add_run(line); rr.font.size = Pt(11)

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ── table of contents (Word field; updates on open) ──
    th = doc.add_paragraph(); rth = th.add_run("Table of Contents"); rth.bold = True; rth.font.size = Pt(16)
    tocp = doc.add_paragraph()
    fb = OxmlElement("w:r"); fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), "begin"); fb.append(fc); tocp._p.append(fb)
    it = OxmlElement("w:r"); itx = OxmlElement("w:instrText"); itx.set(qn("xml:space"), "preserve")
    itx.text = 'TOC \\o "1-1" \\h \\z \\u'; it.append(itx); tocp._p.append(it)
    fs = OxmlElement("w:r"); fsc = OxmlElement("w:fldChar"); fsc.set(qn("w:fldCharType"), "separate"); fs.append(fsc); tocp._p.append(fs)
    pl = OxmlElement("w:r"); pt = OxmlElement("w:t"); pt.text = "Right-click and ‘Update Field’ to build the table of contents."
    pl.append(pt); tocp._p.append(pl)
    fe = OxmlElement("w:r"); fec = OxmlElement("w:fldChar"); fec.set(qn("w:fldCharType"), "end"); fe.append(fec); tocp._p.append(fe)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ── sections ──
    for si, sec in enumerate(sections):
        if si > 0:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        hh = doc.add_heading(level=1)
        rhh = hh.add_run(sec["title"]); rhh.font.color.rgb = OD
        for b in sec["blocks"]:
            kind = b[0]
            if kind == "h2":
                h = doc.add_heading(level=2); rh = h.add_run(b[1]); rh.font.color.rgb = RGBColor(0x1F, 0x24, 0x30)
            elif kind == "para":
                _docx_runs(doc.add_paragraph(), b[1])
            elif kind == "caption":
                p = doc.add_paragraph(); rr = p.add_run(b[1]); rr.italic = True
                rr.font.size = Pt(8.5); rr.font.color.rgb = GREY
            elif kind == "bullets":
                for item in b[1]:
                    _docx_runs(doc.add_paragraph(style="List Bullet"), item)
            elif kind == "problem":
                lab = doc.add_paragraph(); _shade(lab._p.get_or_add_pPr(), "FFF6EE")
                rl = lab.add_run("PROBLEM STATEMENT"); rl.bold = True; rl.font.size = Pt(8.5); rl.font.color.rgb = OD
                lab.paragraph_format.space_after = Pt(0)
                body = doc.add_paragraph(); _shade(body._p.get_or_add_pPr(), "FFF6EE")
                _docx_runs(body, b[1])
            elif kind == "figure":
                doc.add_picture(io.BytesIO(b[1]), width=Inches(6.2 * b[2]))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif kind == "table":
                _docx_table(doc, b[1], O)

    buf = io.BytesIO()
    doc.save(buf); buf.seek(0)
    return buf.getvalue()


def _docx_table(doc, df, header_rgb):
    from docx.shared import Pt, RGBColor
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    t = doc.add_table(rows=1, cols=len(df.columns) + 1)
    t.style = "Light Grid Accent 6"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    hdr[0].text = str(df.index.name or "")
    for j, c in enumerate(df.columns):
        hdr[j + 1].text = str(c)
    for cell in hdr:
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
    for idx, row in df.iterrows():
        cells = t.add_row().cells
        cells[0].text = str(idx)
        for j, c in enumerate(df.columns):
            v = row[c]
            cells[j + 1].text = (f"{v:.3f}" if isinstance(v, (int, float, np.floating))
                                 and not isinstance(v, bool) else str(v))
        for cell in cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(8.5)


# ═══════════════════ public builders ═══════════════════

def build_analysis_doc(df, name):
    meta, sections = build_analysis_model(df, name)
    return render_pdf(meta, sections)


def build_analysis_docx(df, name):
    meta, sections = build_analysis_model(df, name)
    return render_docx(meta, sections)


def build_system_doc(df, name):
    meta, sections = build_system_model(df, name)
    return render_pdf(meta, sections)


def build_system_docx(df, name):
    meta, sections = build_system_model(df, name)
    return render_docx(meta, sections)
