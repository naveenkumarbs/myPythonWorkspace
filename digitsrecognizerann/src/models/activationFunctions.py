import numpy as np

from src.digitsrecognizerann.loggers import Loggers


def sigmoid(z):
    # Z is numpy array of shape (n, m) where n is number of neurons in the layer and m is the number of samples
    # sigmoid_memory is stored as it is used later on in backpropagation
    H = 1 / (1 + np.exp(-z))
    sigmoid_memory = z

    return H, sigmoid_memory


def relu(z):
    # Z is numpy array of shape (n, m) where n is number of neurons in the layer and m is the number of samples
    # relu_memory is stored as it is used later on in backpropagation

    H = np.maximum(0, z)
    assert (H.shape == z.shape)
    relu_memory = z

    return H, relu_memory


def softmax(z):
    # Z is numpy array of shape (n, m) where n is number of neurons in the layer and m is the number of samples
    # softmax_memory is stored as it is used later on in backpropagation

    Z_exp = np.exp(z)
    Z_sum = np.sum(Z_exp, axis=0, keepdims=True)
    H = Z_exp / Z_sum  # normalising step
    softmax_memory = z

    return H, softmax_memory


def sigmoid_backward(dH, sigmoid_memory):
    # Implement the backpropagation of a sigmoid function
    # dH is gradient of the sigmoid activated activation of shape same as H or Z in the same layer
    # sigmoid_memory is the memory stored in the sigmoid(Z) calculation

    Z = sigmoid_memory

    H = 1 / (1 + np.exp(-Z))
    dZ = dH * H * (1 - H)

    assert (dZ.shape == Z.shape)

    return dZ


def relu_backward(dH, relu_memory):
    # Implement the backpropagation of a relu function
    # dH is gradient of the relu activated activation of shape same as H or Z in the same layer
    # relu_memory is the memory stored in the sigmoid(Z) calculation

    Z = relu_memory
    dZ = np.array(dH, copy=True)  # dZ will be the same as dA wherever the elements of A weren't 0

    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ
