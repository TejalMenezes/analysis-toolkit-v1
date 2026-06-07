
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Time Series · Smart Analysis Reporter",
                   page_icon="⏱️", layout="wide")

from modules import ui
from modules.datasets import ensure_dataset_loaded
from modules.timeseries import prepare_timeseries, forecast_series

ui.setup()
ensure_dataset_loaded()
ui.header("Time Series Analysis", "Trend and a short-horizon forecast.", icon="⏱️")

df = st.session_state["df"]

date_cols = list(df.select_dtypes(include=["object", "datetime"]).columns)
numeric_cols = list(df.select_dtypes(include="number").columns)

if not date_cols:
    st.info("This dataset has no date-like columns, so time-series analysis does not apply. "
            "Upload a dataset with a date column to use this page.")
else:
    date_col = st.selectbox("Date Column", date_cols)
    value_col = st.selectbox("Value Column", numeric_cols)

    if st.button("Run Analysis", type="primary"):
        ts = prepare_timeseries(df, date_col, value_col)
        fig = px.line(ts, y=value_col)
        st.plotly_chart(fig, use_container_width=True)

        forecast = forecast_series(ts[value_col])
        st.subheader("30 Step Forecast")
        st.dataframe(forecast, use_container_width=True)
