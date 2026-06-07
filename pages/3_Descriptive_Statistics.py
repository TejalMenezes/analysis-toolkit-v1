
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Descriptive Statistics · Smart Analysis Reporter",
                   page_icon="📐", layout="wide")

from modules import ui
from modules.datasets import ensure_dataset_loaded
from modules.descriptive import descriptive_stats, describe_all, STAT_HELP
from modules.data_loader import get_numeric_columns
from modules import report as R

ui.setup()
ensure_dataset_loaded()
ui.header("Descriptive Statistics", "Central tendency, spread and shape of each metric variable.", icon="📐")

df = st.session_state["df"]
numeric_cols = get_numeric_columns(df)

if not numeric_cols:
    st.info("No metric (numeric) columns found in this dataset.")
else:
    st.subheader("Summary — all metric variables")
    summary = describe_all(df, numeric_cols)
    st.dataframe(summary, use_container_width=True)

    R.add_to_report_button(
        "Descriptive Statistics", table=summary,
        inference="Summary statistics (mean, median, spread and shape) for all metric variables.",
        key="add_desc_table",
    )

    with st.expander("What do these statistics mean?"):
        for k, v in STAT_HELP.items():
            st.markdown(f"**{k}** — {v}")

    st.divider()

    selected_col = st.selectbox("Inspect a variable", numeric_cols)
    series = pd.to_numeric(df[selected_col], errors="coerce").dropna()
    sd = descriptive_stats(df[selected_col])

    ui.kpi_cards([
        (sd.get("N", 0), "N"), (sd.get("Mean", "—"), "Mean"),
        (sd.get("Std", "—"), "Std Dev"), (sd.get("Median", "—"), "Median"),
        (sd.get("IQR", "—"), "IQR"), (sd.get("Skewness", "—"), "Skewness"),
    ])

    left, right = st.columns(2)

    with left:
        st.markdown("**Histogram + Normal curve**")
        if len(series) >= 2:
            fig = px.histogram(series, nbins=max(5, int(np.sqrt(len(series)))),
                               histnorm="probability density", opacity=0.85)
            m, s = series.mean(), series.std()
            xs = np.linspace(series.min(), series.max(), 200)
            fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, m, s), mode="lines",
                                     name="Normal", line=dict(color=ui.ORANGE_DARK, width=2)))
            fig.update_layout(showlegend=False, bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
            R.add_to_report_button(
                f"Distribution of {selected_col}",
                image=R.hist_png(series, title=f"Distribution of {selected_col}"),
                inference=R.describe_inference(selected_col, series),
                key="add_hist",
            )

    with right:
        st.markdown("**Box Plot**")
        fig2 = px.box(df, y=selected_col, points="outliers")
        st.plotly_chart(fig2, use_container_width=True)
        R.add_to_report_button(
            f"Spread & Outliers — {selected_col}",
            image=R.box_png(series, title=f"{selected_col} — box plot"),
            inference=(f"{selected_col}: median {series.median():.2f}, "
                       f"IQR {series.quantile(.75) - series.quantile(.25):.2f}."),
            key="add_box",
        )
