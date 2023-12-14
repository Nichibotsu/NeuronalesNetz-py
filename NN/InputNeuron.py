from NN.Neuron import neuron


class inputneuron(neuron):
    """
        Klasse für ein Input-Neuron in einem neuronalen Netzwerk.

        Vererbung: Die Klasse erbt von der allgemeinen Neuron-Klasse.

        :ivar value: Der aktuelle Wert des Input-Neurons.
    """
    def __init__(self):
        """
            Initialisiert ein Input-Neuron mit einem Wert von 0.
        """
        self.value = 0

    def get_Value(self) -> float:
        """
            Gibt den aktuellen Wert des Input-Neurons zurück.

            :return: Der aktuelle Wert des Input-Neurons.
        """
        return self.value

    def set_Value(self, value) -> None:
        """
            Setzt den Wert des Input-Neurons auf den angegebenen Wert.
            :param value: Der Wert, auf den das Input-Neuron gesetzt werden soll.
        """
        self.value = value
        return None
