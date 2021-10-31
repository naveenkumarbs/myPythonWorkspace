import pandas as pd
from sklearn.model_selection import train_test_split

from src.linreg.loggers import Loggers

logger = Loggers.__call__().get_logger()


def split_train_test(df, targetVar, div=0.7):
    # Putting feature variable to X
    X = df.drop(targetVar, axis=1)

    # Putting response variable to y
    y = df[targetVar]
    if div >= 1 or div <= 0:
        logger.error("div should be between 0 and 1")
        return []

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=div, test_size=(1 - div), random_state=100)
    logger.info("Train Test Split Success")
    return X_train, X_test, y_train, y_test
