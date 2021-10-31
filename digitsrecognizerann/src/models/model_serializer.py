import pickle
from src.digitsrecognizerann.config import serialization_path
from src.digitsrecognizerann.config import serialization_filename
from src.digitsrecognizerann.loggers import Loggers

filename = serialization_path/serialization_filename


def serializeModel(modelparams):
    logger = Loggers.__call__().get_logger()
    logger.info("model serializer started")
    pickle_out = open(filename, "wb")
    pickle.dump(modelparams, pickle_out)
    pickle_out.close()
    logger.info("model serializer completed successfully")


def deserializeModel():
    logger = Loggers.__call__().get_logger()
    logger.info("model deserializer started")
    pickle_in = open(filename, "rb")
    modelparams = pickle.load(pickle_in)
    pickle_in.close()
    logger.info("model deserializer completed successfully")
    return modelparams


