import pickle

from src.logregpca.config import serialization_path, serialization_filename, serialization_pca_filename
from src.logregpca.loggers import Loggers

filename = serialization_path/serialization_filename
filename_pca = serialization_path/serialization_pca_filename


def serializeModel(obj):
    logger = Loggers.__call__().get_logger()
    logger.info("model serializer started")
    pickle_out = open(filename, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()
    logger.info("model serializer completed successfully")


def serializePCA(obj):
    logger = Loggers.__call__().get_logger()
    logger.info("model serializer started")
    pickle_out = open(filename_pca, "wb")
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


def deserializePCA():
    logger = Loggers.__call__().get_logger()
    logger.info("model deserializer started")
    pickle_in = open(filename_pca, "rb")
    obj = pickle.load(pickle_in)
    pickle_in.close()
    logger.info("model deserializer completed successfully")
    return obj


