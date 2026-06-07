
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Q-Q Plot · Smart Analysis Reporter",
                   page_icon="📈", layout="wide")

from modules import ui
from modules.datasets import ensure_dataset_loaded
from modules.normality import qq_data, normality_report
from modules.data_loader import get_numeric_columns
from modules import report as R

ui.setup()
ensure_dataset_loaded()
ui.header("Q-Q Plot (Normality)", "Compare a variable's distribution against the normal.", icon="📈")

df = st.session_state["df"]
numeric_cols = get_numeric_columns(df)

if not numeric_cols:
    st.info("No metric columns found.")
else:
    col = st.selectbox("Variable", numeric_cols)
    rep = normality_report(df[col])

    if rep["W"] is None:
        st.warning(rep["label"])
    else:
        ui.kpi_cards([
            (rep["n"], "N"), (rep["W"], "Shapiro-Wilk W"),
            (rep["skew"], "Skewness"), (rep["kurtosis"], "Excess Kurtosis"),
        ])

        p = rep["p"]
        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        (st.success if rep["normal"] else st.warning)(f"{p_str} — {rep['label']}")

        qq = qq_data(df[col])
        left, right = st.columns(2)

        with left:
            st.markdown("**Q-Q Plot**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=qq["theoretical"], y=qq["sample"], mode="markers",
                                     marker=dict(color=ui.ORANGE, size=7), name="Data"))
            fig.add_trace(go.Scatter(x=qq["ref_x"], y=qq["ref_y"], mode="lines",
                                     line=dict(color=ui.ORANGE_DARK, width=2), name="Reference"))
            fig.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles",
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("**Histogram + Normal curve**")
            s = qq["sample"]
            m, sd = s.mean(), s.std()
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=s, nbinsx=max(5, int(np.sqrt(len(s)))),
                                        histnorm="probability density",
                                        marker=dict(color=ui.ORANGE), name="Observed"))
            xs = np.linspace(s.min(), s.max(), 200)
            fig2.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, m, sd), mode="lines",
                                      line=dict(color=ui.ORANGE_DARK, width=2), name="Normal"))
            fig2.update_layout(showlegend=False, bargap=0.05)
            st.plotly_chart(fig2, use_container_width=True)

        R.add_to_report_button(
            f"Normality of {col} (Q-Q)",
            image=R.qq_png(qq["theoretical"], qq["sample"], qq["ref_x"], qq["ref_y"]),
            inference=(f"Shapiro-Wilk W = {rep['W']}, {p_str}. {rep['label']} "
                       f"Skewness {rep['skew']}, excess kurtosis {rep['kurtosis']}."),
            key="add_qq",
        )
