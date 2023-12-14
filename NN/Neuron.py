
class neuron:
    """
        Eine abstrakte Basisklasse für Neuronen in einem neuronalen Netzwerk.

        Alle Unterklassen müssen die Methode 'get_Value' implementieren.

        meth get_Value: Eine abstrakte Methode, die den Wert des Neurons zurückgeben soll.
    """

    def get_Value(self) -> float:
        pass
