
import pandas as pd


def pearson_corr(df):

    return df.corr(method="pearson")


def spearman_corr(df):

    return df.corr(method="spearman")


def kendall_corr(df):

    return df.corr(method="kendall")

