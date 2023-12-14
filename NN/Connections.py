from NN.Neuron import neuron


class Connection:
    """
        Eine Verbindung zwischen zwei Neuronen mit einem bestimmten Gewicht.

        :param neuron1: Das verknüpfte Neuron.
        :param gewicht: Das Gewicht der Verbindung.
    """

    def __init__(self, neuron1: "neuron", gewicht: float):
        """
                Initialisiert eine Verbindung mit einem Neuron und einem Gewicht.

                :param neuron1: Das verknüpfte Neuron.
                :param gewicht: Das Gewicht der Verbindung.
        """

        self.gewicht = gewicht
        self.neuron = neuron1

    def getValue(self) -> float:
        """
                Berechnet den Wert der Verbindung, indem der Wert des verknüpften Neurons mit dem Gewicht multipliziert wird.

                :return: Der berechnete Wert der Verbindung.
        """
        return self.neuron.get_Value() * self.gewicht
