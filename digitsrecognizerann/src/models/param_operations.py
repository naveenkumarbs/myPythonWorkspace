import numpy as np

from src.digitsrecognizerann.loggers import Loggers


def initialize_parameters(dimensions):
    # dimensions is a list containing the number of neuron in each layer in the network
    # It returns parameters which is a python dictionary containing the parameters "W1", "b1", ..., "WL", "bL":
    logger = Loggers.__call__().get_logger()
    logger.info("initializing random values to weights and biases started")
    np.random.seed(2)
    parameters = {}
    L = len(dimensions)  # number of layers in the network + 1

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l - 1]) * 0.1
        parameters['b' + str(l)] = np.zeros((dimensions[l], 1))

        assert (parameters['W' + str(l)].shape == (dimensions[l], dimensions[l - 1]))
        assert (parameters['b' + str(l)].shape == (dimensions[l], 1))

    logger.info("initializing weights and biases completed successfully")
    return parameters


def update_parameters(parameters, gradients, learning_rate):
    # parameters is the python dictionary containing the parameters W and b for all the layers
    # gradients is the python dictionary containing your gradients, output of L_model_backward

    # returns updated weights after applying the gradient descent update

    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (learning_rate * gradients["dW" + str(l + 1)])
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (learning_rate * gradients["db" + str(l + 1)])

    return parameters
