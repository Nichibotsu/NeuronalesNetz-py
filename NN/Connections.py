from NN.Neuron import neuron

class Connection():

    def __init__(self,neuron:"neuron",gewicht:float):
        self.gewicht=gewicht
        self.neuron=neuron

    def getValue(self)->float:
        return self.neuron.get_Value() * self.gewicht
