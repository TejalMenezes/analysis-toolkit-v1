
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Frequency Tables · Smart Analysis Reporter",
                   page_icon="🔢", layout="wide")

from modules import ui
from modules.datasets import ensure_dataset_loaded
from modules.frequency import categorical_frequency, grouped_frequency
from modules.data_loader import get_numeric_columns, get_categorical_columns
from modules import report as R

ui.setup()
ensure_dataset_loaded()
ui.header("Frequency Tables", "How often each value or binned range occurs.", icon="🔢")

df = st.session_state["df"]
cat_cols = get_categorical_columns(df)
num_cols = get_numeric_columns(df)

tab_cat, tab_num = st.tabs(["Categorical", "Grouped (metric)"])

with tab_cat:
    if not cat_cols:
        st.info("No categorical columns found.")
    else:
        col = st.selectbox("Categorical variable", cat_cols, key="freq_cat")
        table = categorical_frequency(df[col])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(f"n = {int(table['Abs. Frequency'].sum())} valid observations")
        if not table.empty:
            fig = px.bar(table, x="Value", y="Abs. Frequency", title=f"{col} — distribution")
            st.plotly_chart(fig, use_container_width=True)
            top = table.iloc[0]
            R.add_to_report_button(
                f"Breakdown by {col}",
                image=R.bar_png(table["Value"], table["Abs. Frequency"],
                                title=f"{col} distribution", xlabel=col),
                inference=(f"{col} has {len(table)} categories; the most common is "
                           f"\"{top['Value']}\" ({top['Rel. Frequency %']}% of records)."),
                key="add_cat_freq",
            )

with tab_num:
    if not num_cols:
        st.info("No metric columns found.")
    else:
        col = st.selectbox("Metric variable", num_cols, key="freq_num")
        bins = st.slider("Number of bins", 3, 20, 5)
        table = grouped_frequency(df[col], bins=bins)
        st.dataframe(table, use_container_width=True, hide_index=True)
        if not table.empty:
            fig = px.bar(table, x="Interval", y="Abs. Frequency", title=f"{col} — grouped frequency")
            st.plotly_chart(fig, use_container_width=True)
            R.add_to_report_button(
                f"Grouped frequency of {col}",
                image=R.bar_png(table["Interval"], table["Abs. Frequency"],
                                title=f"{col} — grouped frequency", xlabel=col),
                inference=f"{col} grouped into {bins} intervals across its observed range.",
                key="add_num_freq",
            )
