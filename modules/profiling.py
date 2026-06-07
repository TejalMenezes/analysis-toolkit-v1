
import pandas as pd


def dataset_summary(df):

    return {

        "Rows": len(df),

        "Columns": len(df.columns),

        "Missing Values":
            int(df.isna().sum().sum()),

        "Duplicate Rows":
            int(df.duplicated().sum())

    }


def column_profile(df):

    profile = []

    for col in df.columns:

        profile.append({

            "Column": col,

            "Type": str(df[col].dtype),

            "Missing %":
                round(
                    df[col].isna().mean()*100,
                    2
                ),

            "Unique Values":
                int(df[col].nunique())

        })

    return pd.DataFrame(profile)

