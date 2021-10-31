import numpy as np


def compute_loss(HL, Y):
    # HL is probability matrix of shape (10, number of examples)
    # Y is true "label" vector shape (10, number of examples)

    # loss is the cross-entropy loss

    A = HL

    m = Y.shape[1]

    loss = (-1. / m) * np.sum((np.multiply(Y, np.log(HL))))

    loss = np.squeeze(loss)  # To make sure that the loss's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (loss.shape == ())

    return loss
