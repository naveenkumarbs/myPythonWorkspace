import pickle

from src.cnn.config import serialization_path, serialization_filename
from src.cnn.loggers import Loggers

filename = serialization_path/serialization_filename


def serializeModel(obj):
    logger = Loggers.__call__().get_logger()
    logger.info("model serializer started")
    pickle_out = open(filename, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()
    logger.info("model serializer completed successfully")


def deserializeModel():
    logger = Loggers.__call__().get_logger()
    logger.info("model deserializer started")
    pickle_in = open(filename, "rb")
    obj = pickle.load(pickle_in)
    pickle_in.close()
    logger.info("model deserializer completed successfully")
    return obj


