import pandas as pd
import numpy as np

from src.dataprocessor.data_clean import dropMultipleCols
from src.logregpca.loggers import Loggers


def convertToBinary(df, cols):
    for attr in cols:
        df[attr] = df[attr].map({'Yes': 1, 'No': 0})

    return df


def createDummyVars(df, cols, dropfirst=False):
    logger = Loggers.__call__().get_logger()
    logger.info("dummy vars creation for cols: ")
    logger.info(cols)
    for attr in cols:
        df_temp = pd.get_dummies(df[attr], prefix=attr, drop_first=dropfirst)
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
