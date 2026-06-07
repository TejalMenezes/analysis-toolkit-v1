
import pandas as pd


def load_file(uploaded_file):

    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError("Unsupported file format")


def get_numeric_columns(df):

    return list(
        df.select_dtypes(
            include=["number"]
        ).columns
    )


def get_categorical_columns(df):

    return list(
        df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )


def get_datetime_columns(df):

    return list(
        df.select_dtypes(
            include=["datetime", "datetimetz"]
        ).columns
    )


def classify_columns(df):
    """Label each column Metric / Ordinal / Nominal, mirroring the column
    types in the HTML toolkit. Low-cardinality integer columns are treated
    as ordinal (e.g. Likert-style ratings)."""

    out = []

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            nunique = s.nunique(dropna=True)
            is_int = pd.api.types.is_integer_dtype(s) or (
                s.dropna() % 1 == 0
            ).all()

            if is_int and nunique <= 7:
                kind = "Ordinal"
            else:
                kind = "Metric"
        else:
            kind = "Nominal"

        out.append({"Column": col, "Type": kind})

    return pd.DataFrame(out)
