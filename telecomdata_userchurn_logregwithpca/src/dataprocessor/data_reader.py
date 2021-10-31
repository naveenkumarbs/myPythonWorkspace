import pandas as pd

from src.logregpca.loggers import Loggers


def import_data(data_dir_external, filename):
    logger = Loggers.__call__().get_logger()
    logger.info("importing data from " + filename)
    filesList = filename.split(',')
    df = pd.read_csv(data_dir_external/filesList[0])
    for i in range(1,len(filesList)):
        df = pd.merge(df, pd.read_csv(data_dir_external/filesList[i]), how='inner', on='customerID')

    logger.debug("file import success")
    return df
