from NN.Neuron import neuron


class inputneuron(neuron):

    def __init__(self):
        self.value = 0

    def get_Value(self) -> float:
        return self.value

    def set_Value(self, value) -> None:
        self.value = value
        return None
