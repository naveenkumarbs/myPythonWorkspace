import pickle
from src.mediacompanycasestudy.config import serialization_path
from src.mediacompanycasestudy.config import serialization_dataframe
from src.mediacompanycasestudy.config import serialization_model1
from src.mediacompanycasestudy.config import serialization_model2
from src.mediacompanycasestudy.loggers import Loggers

frame = serialization_path/serialization_dataframe
model1 = serialization_path/serialization_model1
model2 = serialization_path/serialization_model2


def serializeModel(modelparams, premodel1, premodel2):
    logger = Loggers.__call__().get_logger()
    logger.info("model serializer started")
    pickle_out = open(frame, "wb")
    pickle.dump(modelparams, pickle_out)
    pickle_out.close()

    pickle_out = open(model1, "wb")
    pickle.dump(premodel1, pickle_out)
    pickle_out.close()

    pickle_out = open(model2, "wb")
    pickle.dump(premodel2, pickle_out)
    pickle_out.close()

    logger.info("model serializer completed successfully")


def deserializeModel():
    logger = Loggers.__call__().get_logger()
    logger.info("model deserializer started")
    pickle_in = open(frame, "rb")
    modelparams = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(model1, "rb")
    premodel1 = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(model2, "rb")
    premodel2 = pickle.load(pickle_in)
    pickle_in.close()

    logger.info("model deserializer completed successfully")
    return modelparams, premodel1, premodel2


