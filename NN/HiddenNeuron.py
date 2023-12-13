from NN.Neuron import neuron
from NN.Connections import Connection
from NN.Aktivierung import *

class hiddenneuron(neuron):


    def __init__(self):
        self.Connections:list[Connection]=[]
        self.value:float=0

    def get_Value(self) ->float:
        value_sum=0
        for c in self.Connections:
            value_sum += c.getValue()

        self.value=ReLu(value_sum)
        return self.value

    def addConnection(self,c:Connection) -> None:
        self.Connections.append(c)
        return None
