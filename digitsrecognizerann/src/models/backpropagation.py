import numpy as np

from src.models.activationFunctions import relu_backward
from src.models.activationFunctions import sigmoid_backward


def layer_backward(dH, memory, activation='relu'):
    # takes dH and the memory calculated in layer_forward and activation as input to calculate the dH_prev, dW, db
    # performs the backprop depending upon the activation function

    linear_memory, activation_memory = memory

    if activation == "relu":
        dZ = relu_backward(dH, activation_memory)  # write your code here
        H_prev, W, b = linear_memory
        m = H_prev.shape[1]
        dW = (1. / m) * np.dot(dZ, H_prev.T)
        # write your code here, use (1./m) and not (1/m)
        db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
        # write your code here, use (1./m) and not (1/m)
        dH_prev = np.dot(W.T, dZ)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dH, memory)
        H_prev, W, b = linear_memory
        m = H_prev.shape[1]
        dW = (1. / m) * np.dot(dZ, H_prev.T)  # write your code here, use (1./m) and not (1/m)
        db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)  # (1./m)* dZ #write your code here, use (1./m) and not (1/m)
        dH_prev = np.dot(W.T, dZ)

    return dH_prev, dW, db


def L_layer_backward(HL, Y, memories):
    # Takes the predicted value HL and the true target value Y and the
    # memories calculated by L_layer_forward as input

    # returns the gradients calulated for all the layers as a dict

    gradients = {}
    L = len(memories)  # the number of layers
    m = HL.shape[1]
    Y = Y.reshape(HL.shape)  # after this line, Y is the same shape as AL

    # Perform the backprop for the last layer that is the softmax layer
    current_memory = memories[-1]
    linear_memory, activation_memory = current_memory
    dZ = HL - Y
    H_prev, W, b = linear_memory
    # Use the expressions you have used in 'layer_backward'
    gradients["dH" + str(L - 1)] = np.dot(W.T, dZ)
    gradients["dW" + str(L)] = (1. / m) * np.dot(dZ, H_prev.T)  # use (1./m) and not (1/m)
    gradients["db" + str(L)] = (1. / m) * np.sum(dZ, axis=1, keepdims=True)  # use (1./m) and not (1/m)

    # Perform the backpropagation l-1 times
    for l in reversed(range(L - 1)):
        # Lth layer gradients: "gradients["dH" + str(l + 1)] ", gradients["dW" + str(l + 2)] ,
        # gradients["db" + str(l + 2)]
        current_memory = memories[l]

        dH_prev_temp, dW_temp, db_temp = layer_backward(gradients["dH" + str(l + 1)], current_memory,
                                                        'relu')  # write your code here
        gradients["dH" + str(l)] = dH_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp

    return gradients

