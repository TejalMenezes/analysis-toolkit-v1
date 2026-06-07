
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima.model import ARIMA


def prepare_timeseries(df,date_col,value_col):

    data = df[[date_col,value_col]].copy()

    data[date_col] = pd.to_datetime(
        data[date_col]
    )

    data = data.sort_values(
        date_col
    )

    data = data.set_index(
        date_col
    )

    return data


def forecast_series(series):

    model = ARIMA(
        series,
        order=(1,1,1)
    )

    fitted = model.fit()

    return fitted.forecast(
        steps=30
    )

