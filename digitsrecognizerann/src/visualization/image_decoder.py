from src.digitsrecognizerann.loggers import Loggers


def decodeImageFromArray(k, h, w):
    logger = Loggers.__call__().get_logger()
    logger.info("Decoding image from given array started")
    k = k.reshape((h, w))
    logger.info("Decoding image from given array completed successfully")
    return k
