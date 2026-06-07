
"""Default dataset loading for Smart Analysis Reporter.

The student-performance dataset is cached in ``data/student_dataset.csv`` so the
app analyses it out of the box with no network call. If the cached file is
missing we fall back to downloading it via kagglehub.
"""

import os
import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHED = os.path.join(_HERE, "data", "student_dataset.csv")

DEFAULT_NAME = "Student Performance Prediction"
KAGGLE_REF = "shambhurajejagadale/student-performance-prediction-dataset"


@st.cache_data(show_spinner=False)
def load_default_dataset():
    if os.path.exists(_CACHED):
        return pd.read_csv(_CACHED)

    # fallback: pull from Kaggle and cache locally
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_REF)
    csv = os.path.join(path, os.listdir(path)[0])
    df = pd.read_csv(csv)
    os.makedirs(os.path.dirname(_CACHED), exist_ok=True)
    df.to_csv(_CACHED, index=False)
    return df


def ensure_dataset_loaded():
    """Load the default dataset into session_state on first run."""
    if "df" not in st.session_state:
        st.session_state["df"] = load_default_dataset()
        st.session_state["dataset_name"] = DEFAULT_NAME
