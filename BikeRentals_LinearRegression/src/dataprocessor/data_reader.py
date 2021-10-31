import pandas as pd

from src.linreg.loggers import Loggers


def import_data(data_dir_external, filename):
    logger = Loggers.__call__().get_logger()
    logger.info("importing data from " + filename)
    df = pd.read_csv(data_dir_external/filename)
    logger.debug("file import success")
    return df
