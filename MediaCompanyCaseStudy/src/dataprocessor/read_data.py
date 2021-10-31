import pandas as pd
from src.mediacompanycasestudy.loggers import Loggers
from datetime import datetime
#get filename from notebook

def data_read(filename):
    logger = Loggers.__call__().get_logger()
    logger.info("Reading raw data")
    pd.set_option('max_columns', None)
    pd.set_option('max_rows', None)
    media = pd.read_csv(filename)
    logger.info("Reading raw data completed successfully")
    return media

def day_calculation(media):
    basedate = pd.Timestamp('2017-02-28')
    media['day'] = media['Date'].apply(lambda x: (x - basedate).days)
    return media