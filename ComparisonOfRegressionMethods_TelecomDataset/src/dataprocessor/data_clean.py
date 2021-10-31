import pandas as pd


def dropMultipleCols(df, cols):
    df = df.drop(cols, axis=1)
    return df


def getColsWithOnlyOneUniqueValue(data):
    cols_to_drop = []
    for i in data.columns:
        if len(data[i].dropna().unique()) == 1:
            cols_to_drop.append(i)
    return cols_to_drop


def computeMissingValuesPercent(data):
    print(round(data.isnull().sum(axis=0) / len(data) * 100, 2))


def meaningfulImpute(data, cols, method="mode"):
    if method == "mode":
        for i in cols:
            data[i] = data[i].fillna(data[i].mode()[0])
    elif method == "zero":
        for i in cols:
            data[i] = data[i].fillna(0)
    return data


def dropColsWithMissingValues(data,percent=70):
    for i in data.columns:
        if (data[i].isnull().sum() / len(data)) * 100 > percent:
            data.drop(i, inplace=True, axis=1)
    return data


def getColsWithMissingValues(data):
    cols = []
    for i in data.columns:
        if data[i].isnull().sum() / len(data) > 0.00:
            cols.append(i)
    return cols