import pandas as pd
from collections import Counter
from pandas.api.types import is_list_like
from functools import reduce


def select(df: pd.DataFrame, **kwargs):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df should be a Pandas DataFrame")

    keys_columns = list(df)
    keys_index_names = list(df.index.names)
    keys_all = keys_columns + keys_index_names
    keys_dup = [k for k, c in Counter(keys_all).items() if c > 1]

    conditions = []
    for key, value in kwargs.items():
        if key in keys_dup:
            raise ValueError(f"Duplicated keys in the dataframe: {key}")
        if key not in keys_all:
            raise KeyError(key)

        if key in keys_columns:
            series = df[key]
        elif key in df.index.names:
            series = df.index.get_level_values(key).to_series()

        if is_list_like(value):
            condition = series.isin(value).values
        elif callable(value):
            condition = value(series).values
        else:
            condition = series.eq(value).values

        conditions.append(condition)

    condition_reduce = reduce(lambda x, y: x & y, conditions)
    return df[condition_reduce]