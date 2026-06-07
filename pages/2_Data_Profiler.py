
import streamlit as st

st.set_page_config(page_title="Data Profiler · Smart Analysis Reporter",
                   page_icon="🔍", layout="wide")

from modules import ui
from modules.datasets import ensure_dataset_loaded
from modules.profiling import dataset_summary, column_profile
from modules.data_loader import classify_columns
from modules import report as R

ui.setup()
ensure_dataset_loaded()
ui.header("Data Profiler", "Shape, quality and measurement levels of your dataset.", icon="🔍")

df = st.session_state["df"]
summary = dataset_summary(df)

ui.kpi_cards([
    (f"{summary['Rows']:,}", "Rows"),
    (summary["Columns"], "Columns"),
    (summary["Missing Values"], "Missing"),
    (summary["Duplicate Rows"], "Duplicates"),
])

st.subheader("Column Profile")

profile = column_profile(df)
types = classify_columns(df)
merged = profile.merge(types, on="Column", how="left")
merged = merged.rename(columns={"Type_x": "Dtype", "Type_y": "Measurement"})

st.dataframe(merged, use_container_width=True, hide_index=True)
st.caption("Measurement level — Metric (numbers), Ordinal (ranked / low-cardinality "
           "integers), Nominal (unordered categories).")

R.add_to_report_button(
    "Column Profile",
    table=merged.set_index("Column"),
    inference=(f"The dataset has {summary['Columns']} columns and {summary['Rows']:,} rows "
               f"with {summary['Missing Values']} missing values and "
               f"{summary['Duplicate Rows']} duplicate rows."),
    key="add_profile",
)

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)
