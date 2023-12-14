from NN.Neuron import neuron
from NN.Connections import Connection
from NN.Aktivierung import *


class hiddenneuron(neuron):
    """
        Klasse für ein verstecktes Neuron in einem neuronalen Netzwerk.

        Vererbung: Die Klasse erbt von der allgemeinen Neuron-Klasse.

        :ivar Connections: Eine Liste von Verbindungen zu anderen Neuronen.
        :ivar value: Der aktuelle Wert des Neurons nach der Aktivierungsfunktion.
    """

    def __init__(self):
        """
            Initialisiert ein verstecktes Neuron mit einer leeren Liste von Verbindungen und einem Wert von 0.
        """
        self.Connections: list[Connection] = []
        self.value: float = 0

    def get_Value(self) -> float:
        """
            Berechnet den Wert des Neurons nach der Aktivierungsfunktion (hier ReLU).

            :return: Der aktuelle Wert des Neurons.
        """
        value_sum = 0
        for c in self.Connections:
            value_sum += c.getValue()

        self.value = ReLu(value_sum)
        return self.value

    def addConnection(self, c: Connection) -> None:
        """
            Fügt eine Verbindung zu anderen Neuronen hinzu.

            :param c: Die hinzuzufügende Verbindung.
        """
        self.Connections.append(c)
        return None
