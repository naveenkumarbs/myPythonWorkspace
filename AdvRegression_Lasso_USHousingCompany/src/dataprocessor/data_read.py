import pandas as pd
from src.ushousingcompany.loggers import Loggers
#get filename from notebook

def read_data(filename):
    logger = Loggers.__call__().get_logger()
    logger.info("Reading raw data")
    pd.set_option('max_columns', None)
    pd.set_option('max_rows', None)
    housing = pd.read_csv(filename)
    logger.info("Reading raw data completed successfully")
    return housing