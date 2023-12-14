from NN.Neuron import neuron


class Connection:

    def __init__(self, neuron1: "neuron", gewicht: float):
        self.gewicht = gewicht
        self.neuron = neuron1

    def getValue(self) -> float:
        return self.neuron.get_Value() * self.gewicht
