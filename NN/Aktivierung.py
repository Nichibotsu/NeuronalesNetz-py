import numpy as np

def sigmoid(value) -> float:
    """
        Berechnet die Sigmoid-Aktivierungsfunktion für einen gegebenen Wert.

        :param value: Der Eingabewert für die Sigmoid-Funktion.
        :return: Der berechnete Sigmoid-Wert.
    """
    sig = 1 / (1 + np.exp(-value))
    return sig


def ReLu(value) -> float:
    """
       Berechnet die Rectified Linear Unit (ReLU)-Aktivierungsfunktion für einen gegebenen Wert.

       :param value: Der Eingabewert für die ReLU-Funktion.
       :return: Der berechnete ReLU-Wert.
    """
    return value if value > 0 else 0


def Tanh(value) -> float:
    """
        Berechnet die Tangens hyperbolicus (Tanh)-Aktivierungsfunktion für einen gegebenen Wert.

        :param value: Der Eingabewert für die Tanh-Funktion.
        :return: Der berechnete Tanh-Wert.
    """
    tanh = 1 - (2 / (1 + np.exp(2 * value)))
    return tanh




