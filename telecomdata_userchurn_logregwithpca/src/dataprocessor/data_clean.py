import pandas as pd


def dropMultipleCols(df, cols):
    df = df.drop(cols, axis=1)
    return df
