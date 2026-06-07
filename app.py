
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Smart Analysis Reporter",
    page_icon="📊",
    layout="wide",
)

from modules import ui
from modules.data_loader import load_file, classify_columns
from modules.datasets import ensure_dataset_loaded, DEFAULT_NAME
from modules.autoreport import build_default_report
from modules.report import get_report

ui.setup()

# auto-load the default student-performance dataset on first run
ensure_dataset_loaded()
get_report()  # ensure report state exists

ui.header(
    "Smart Analysis Reporter",
    "Analyse your data and generate a clean, editable analytics report — automatically.",
    icon="📊",
)
ui.author_chip()

st.write("")

df = st.session_state["df"]
name = st.session_state.get("dataset_name", DEFAULT_NAME)

# ── dataset source ──
with st.expander("📁 Dataset source — using a default, or upload your own", expanded=False):

    c1, c2 = st.columns([3, 2])

    with c1:
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if uploaded_file:
            st.session_state["df"] = load_file(uploaded_file)
            st.session_state["dataset_name"] = uploaded_file.name.rsplit(".", 1)[0]
            st.success("Dataset loaded — analysis below will use it.")
            st.rerun()

    with c2:
        st.caption(f"**Current dataset:** {name}")
        if st.button("↺ Reset to default dataset"):
            for k in ("df", "dataset_name"):
                st.session_state.pop(k, None)
            ensure_dataset_loaded()
            st.rerun()

df = st.session_state["df"]
name = st.session_state.get("dataset_name", DEFAULT_NAME)

# ── overview ──
st.subheader(f"Overview — {name}")

ui.kpi_cards([
    (f"{len(df):,}", "Rows"),
    (df.shape[1], "Columns"),
    (int(df.isna().sum().sum()), "Missing"),
    (int(df.duplicated().sum()), "Duplicates"),
])

c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("**Preview**")
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)
with c2:
    st.markdown("**Column types**")
    st.dataframe(classify_columns(df), use_container_width=True, hide_index=True)

st.divider()

# ── one-click report ──
st.subheader("🪄 Generate a report")

st.markdown(
    "Build a complete analytics report in one click — distributions, correlation, "
    "regression, group tests and normality — each with an editable inference. "
    "Then fine-tune everything on the **Report Builder** page."
)

cta1, cta2 = st.columns([1, 3])
with cta1:
    if st.button("✨ Auto-generate report", type="primary", use_container_width=True):
        build_default_report(df, name)
        st.session_state["_just_built"] = True
        st.rerun()

with cta2:
    n_items = len(get_report()["items"])
    if st.session_state.pop("_just_built", False):
        st.success(
            f"Report built with {n_items} sections. "
            "Open **Report Builder** in the sidebar to edit and download it."
        )
    elif n_items:
        st.info(f"Your report currently has {n_items} section(s). Open **Report Builder** to edit/export.")
    else:
        st.caption("No report yet — click *Auto-generate report*, or add items from the analysis pages.")

st.divider()
st.caption("Use the sidebar to explore individual analyses — each has a **➕ Add to report** button.")
