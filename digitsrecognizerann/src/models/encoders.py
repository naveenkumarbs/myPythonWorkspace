import numpy as np


# The target variable is converted to a one hot matrix.
# We use the function one_hot to convert the target dataset to one hot encoding.
from src.digitsrecognizerann.loggers import Loggers


def one_hot(j):
    # input is the target dataset of shape (m,) where m is the number of data points
    # returns a 2 dimensional array of shape (10, m) where each target value is converted to a one hot encoding
    # Look at the next block of code for a better understanding of one hot encoding
    logger = Loggers.__call__().get_logger()
    logger.info("creating one hot encoding for target param")
    n = j.shape[0]
    new_array = np.zeros((10, n))
    index = 0
    for res in j:
        new_array[res][index] = 1.0
        index = index + 1
    return new_array


# data_wrapper() will convert the dataset into the desired shape
# and also convert the ground truth labels to one_hot matrix.

def data_wrapper(tr_d, va_d, te_d):
    logger = Loggers.__call__().get_logger()
    logger.info("creating data wrapper to convert the dataset into desired shape")
    training_inputs = np.array(tr_d[0][:]).T
    training_results = np.array(tr_d[1][:])
    train_set_y = one_hot(training_results)

    validation_inputs = np.array(va_d[0][:]).T
    validation_results = np.array(va_d[1][:])
    validation_set_y = one_hot(validation_results)

    test_inputs = np.array(te_d[0][:]).T
    test_results = np.array(te_d[1][:])
    test_set_y = one_hot(test_results)

    return training_inputs, train_set_y, test_inputs, test_set_y
