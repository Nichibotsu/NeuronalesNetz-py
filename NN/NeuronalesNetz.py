from NN.HiddenNeuron import hiddenneuron
from NN.InputNeuron import inputneuron
from NN.Connections import Connection


class neuronalesNetz:
    inputNeuronen: list[inputneuron] = []
    hiddenneuronen: list[list[hiddenneuron]] = []
    outputneuronen: list[hiddenneuron] = []
    gewichtzahl: int

    def __init__(self):
        pass

    def createInputNeuron(self) -> "inputneuron":
        i1 = inputneuron()
        self.inputNeuronen.append(i1)
        return i1

    def createOutputNeuron(self) -> "hiddenneuron":
        o1 = hiddenneuron()
        self.outputneuronen.append(o1)
        return o1

    def createHiddenNeuron(self, Neuronenanzahl: int, layer: int = 1) -> None:
        for i in range(layer):
            x: list[hiddenneuron] = []

            for n in range(Neuronenanzahl):
                h1 = hiddenneuron()
                x.append(h1)

            self.hiddenneuronen.append(x)
        return None

    def calcGewichtAnzahl(self) -> int:
        calcgewichte = 0
        gsize = [len(self.inputNeuronen)]

        for c in self.hiddenneuronen:
            gsize.append(len(c))
        gsize.append(len(self.outputneuronen))

        for i in range(len(gsize)):
            if i+1 == len(gsize):
                break
            calcgewichte += gsize[i]*gsize[i+1]

        self.gewichtzahl = calcgewichte
        return self.gewichtzahl

    def createFullMesh(self, gewichte: list[float]) -> None:

        if len(self.inputNeuronen) == 0 or len(self.hiddenneuronen) == 0 or len(self.outputneuronen) == 0:
            raise NotImplemented

        if self.calcGewichtAnzahl() != len(gewichte):
            raise NotImplemented

        # Verkettetung

        index = 0
        for o1 in self.outputneuronen:
            for hlast in self.hiddenneuronen[len(self.hiddenneuronen)-1]:
                o1.addConnection(Connection(hlast, gewichte[index]))
                index += 1

        if len(self.hiddenneuronen)>1:
            index = self.verkettungHiddens(gewichte, index, len(self.hiddenneuronen)-1, len(self.hiddenneuronen)-2)

        for h1 in self.hiddenneuronen[0]:
            for i1 in self.inputNeuronen:
                h1.addConnection(Connection(i1, gewichte[index]))
                index += 1

    def verkettungHiddens(self, gewichte: list[float], index: int, index1: int, index2: int) -> int:

        if index1 == 0:
            return index

        for h2 in self.hiddenneuronen[index1]:
            for h1 in self.hiddenneuronen[index2]:
                h2.addConnection(Connection(h1, gewichte[index]))
                index += 1

        return self.verkettungHiddens(gewichte, index, index1-1, index2-1)
