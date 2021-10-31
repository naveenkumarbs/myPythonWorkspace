import gzip
import pickle

from src.digitsrecognizerann.loggers import Loggers


def import_data(filename):
    logger = Loggers.__call__().get_logger()
    logger.info("importing data from " + filename)
    f = gzip.open(filename, 'rb')
    f.seek(0)
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    logger.debug("file import success")
    return training_data, validation_data, test_data
