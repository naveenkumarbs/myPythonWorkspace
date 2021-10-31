import numpy as np

from src.models.activationFunctions import sigmoid
from src.models.activationFunctions import relu
from src.models.activationFunctions import softmax


def layer_forward(H_prev, W, b, activation='relu'):
    # H_prev is of shape (size of previous layer, number of examples)
    # W is weights matrix of shape (size of current layer, size of previous layer)
    # b is bias vector of shape (size of the current layer, 1)
    # activation is the activation to be used for forward propagation : "softmax", "relu", "sigmoid"

    # H is the output of the activation function
    # memory is a python dictionary containing "linear_memory" and "activation_memory"

    if activation == "sigmoid":
        Z = np.matmul(W, H_prev) + b
        linear_memory = (H_prev, W, b)
        H, activation_memory = sigmoid(Z)

    elif activation == "softmax":
        Z = np.matmul(W, H_prev) + b
        linear_memory = (H_prev, W, b)
        H, activation_memory = softmax(Z)

    elif activation == "relu":
        Z = np.matmul(W, H_prev) + b
        linear_memory = (H_prev, W, b)
        H, activation_memory = relu(Z)

    assert (H.shape == (W.shape[0], H_prev.shape[1]))
    memory = (linear_memory, activation_memory)

    return H, memory


def L_layer_forward(X, parameters):
    # X is input data of shape (input size, number of examples)
    # parameters is output of initialize_parameters()

    # HL is the last layer's post-activation value
    # memories is the list of memory containing (for a relu activation, for example):
    # - every memory of relu forward (there are L-1 of them, indexed from 1 to L-1),
    # - the memory of softmax forward (there is one, indexed L)

    memories = []
    H = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement relu layer (L-1) times as the Lth layer is the softmax layer
    for l in range(1, L):
        H_prev = H

        H, memory = layer_forward(H_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation="relu")

        memories.append(memory)

    # Implement the final softmax layer
    # HL here is the final prediction P
    HL, memory = layer_forward(H, parameters["W" + str(L)], parameters["b" + str(L)], activation='softmax')

    memories.append(memory)

    assert (HL.shape == (10, X.shape[1]))

    return HL, memories


def predict(X, y, parameters):
    # Performs forward propogation using the trained parameters and calculates the accuracy

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network

    # Forward propagation
    probas, caches = L_layer_forward(X, parameters)

    p = np.argmax(probas, axis=0)
    act = np.argmax(y, axis=0)

    print("Accuracy: " + str(np.sum((p == act) / m)))

    return p
