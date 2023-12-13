import numpy as np

def sigmoid(value) -> float:
    sig = 1 / (1 + np.exp(-value))
    return sig


def ReLu(value) -> float:
    return value


def Tanh(value) -> float:
    tanh = 1 - (2 / (1 + np.exp(2 * value)))
    return tanh




