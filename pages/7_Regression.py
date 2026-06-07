
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Regression · Smart Analysis Reporter",
                   page_icon="📉", layout="wide")

from modules import ui
from modules.datasets import ensure_dataset_loaded
from modules.regression import simple_linear_regression, linear_regression
from modules import report as R

ui.setup()
ensure_dataset_loaded()
ui.header("Regression Modeling", "Quantify how predictors relate to an outcome.", icon="📉")

df = st.session_state["df"]
numeric_cols = list(df.select_dtypes(include="number").columns)

tab_simple, tab_multiple = st.tabs(["Simple Linear", "Multiple OLS"])

with tab_simple:
    if len(numeric_cols) < 2:
        st.info("Need at least 2 metric columns.")
    else:
        c1, c2 = st.columns(2)
        x_var = c1.selectbox("X (Independent)", numeric_cols, key="reg_x")
        y_var = c2.selectbox("Y (Dependent)", numeric_cols,
                             index=min(1, len(numeric_cols) - 1), key="reg_y")
        if x_var == y_var:
            st.warning("Pick two different variables.")
        elif st.button("Run Regression", type="primary"):
            r = simple_linear_regression(df, x_var, y_var)
            if r is None:
                st.warning("Not enough complete (X, Y) pairs.")
            else:
                st.session_state["last_reg"] = {"x": x_var, "y": y_var, "r": r}

        lr = st.session_state.get("last_reg")
        if lr and lr["x"] in numeric_cols and lr["y"] in numeric_cols:
            r, x_var, y_var = lr["r"], lr["x"], lr["y"]
            st.markdown(f"### `{r['equation']}`")
            ui.kpi_cards([
                (round(r["intercept"], 3), "Intercept (a)"),
                (round(r["slope"], 3), "Slope (b)"),
                (round(r["r2"], 3), "R²"),
                ("p < 0.001" if r["p_slope"] < 0.001 else f"p = {r['p_slope']:.3f}", "p-value (slope)"),
            ])
            st.progress(max(0.0, min(1.0, r["r2"])),
                        text=f"R² = {r['r2'] * 100:.1f}% of variance in {y_var} explained")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=r["x"], y=r["y"], mode="markers", name="Data",
                                     marker=dict(color=ui.ORANGE, size=8, opacity=0.5)))
            fig.add_trace(go.Scatter(x=r["x_line"], y=r["y_line"], mode="lines", name="Fit",
                                     line=dict(color=ui.ORANGE_DARK, width=2)))
            fig.update_layout(xaxis_title=x_var, yaxis_title=y_var, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            R.add_to_report_button(
                f"Regression — {y_var} vs {x_var}",
                image=R.regression_png(r["x"], r["y"], r["x_line"], r["y_line"],
                                       xlabel=x_var, ylabel=y_var,
                                       title=f"{y_var} vs {x_var}"),
                inference=R.regression_inference(x_var, y_var, r),
                key="add_reg",
            )

with tab_multiple:
    y_var = st.selectbox("Dependent Variable", numeric_cols, key="mreg_y")
    x_vars = st.multiselect("Independent Variables",
                            [c for c in numeric_cols if c != y_var], key="mreg_x")
    if st.button("Run OLS", type="primary"):
        if x_vars:
            model = linear_regression(df, y_var, x_vars)
            st.text(model.summary().as_text())
        else:
            st.warning("Select at least one independent variable.")
