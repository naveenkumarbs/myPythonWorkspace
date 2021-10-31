import pandas as pd
import numpy as np

from src.dataprocessor.data_clean import dropMultipleCols
from src.compareregressions.loggers import Loggers


def convertToBinary(df, cols):
    for attr in cols:
        df[attr] = df[attr].map({'Yes': 1, 'No': 0})

    return df


def createDummyVars(df, cols, dropfirst=False, addPrefix=True):
    logger = Loggers.__call__().get_logger()
    logger.info("dummy vars creation for cols: ")
    logger.info(cols)
    if addPrefix:
        for attr in cols:
            df_temp = pd.get_dummies(df[attr], prefix=attr, drop_first=dropfirst)
            df = pd.concat([df, df_temp], axis=1)
        df = dropMultipleCols(df, cols)
    else:
        for attr in cols:
            df_temp = pd.get_dummies(df[attr], drop_first=dropfirst)
            df = pd.concat([df, df_temp], axis=1)
        df = dropMultipleCols(df, cols)
    return df


def convertColToNumeric(df, col):
    df[col] = df[col].astype(str).astype(float)
    return df


def standardizeData(df, cols):
    df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()
    return df


def dropMissingValuesInRow(df, col):
    df = df[~np.isnan(df[col])]
    return df


def normalizeData(df, cols):
    normalized_df = (df[cols] - df[cols].mean()) / df[cols].std()
    df = df.drop(cols, 1)
    df = pd.concat([df, normalized_df], axis=1)
    return df


def createBins(df, col, col_labels):
    return pd.qcut(df[col], q=[0, .33, .66, 1], labels=col_labels)


def interQuartileRange(df, cols):
    for i in enumerate(cols):
        Q1 = df[i[1]].quantile(0.010)
        Q3 = df[i[1]].quantile(0.99)
        IQR = Q3 - Q1
        lower = Q1 - IQR * 1.5
        upper = Q3 + IQR * 1.5
        df = df[(df[i[1]] >= lower) & (df[i[1]] <= upper)]
    return df
