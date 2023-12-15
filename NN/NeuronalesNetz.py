from NN.HiddenNeuron import hiddenneuron
from NN.InputNeuron import inputneuron
from NN.Connections import Connection


class neuronalesNetz:
    """
        Eine Klasse, die ein neuronales Netzwerk repräsentiert.

        :ivar inputNeuronen: Eine Liste von Input-Neuronen.
        :ivar hiddenneuronen: Eine Liste von Listen, die die versteckten Neuronen für jede Schicht enthält.
        :ivar outputneuronen: Eine Liste von Output-Neuronen.
        :ivar gewichtzahl: Die Gesamtanzahl der Gewichte im Netzwerk.
    """
    inputNeuronen: list[inputneuron] = []
    hiddenneuronen: list[list[hiddenneuron]] = []
    outputneuronen: list[hiddenneuron] = []
    gewichtzahl: int

    def __init__(self):
        """
            Initialisiert ein neuronales Netzwerk.
        """
        pass

    def createInputNeuron(self) -> "inputneuron":
        """
            Erstellt ein Input-Neuron und fügt es zur Liste der Input-Neuronen hinzu.

            :return: Das erstellte Input-Neuron.
        """
        i1 = inputneuron()
        self.inputNeuronen.append(i1)
        return i1

    def createOutputNeuron(self) -> "hiddenneuron":
        """
            Erstellt ein Output-Neuron und fügt es zur Liste der Output-Neuronen hinzu.

            :return: Das erstellte Output-Neuron.
        """
        o1 = hiddenneuron()
        self.outputneuronen.append(o1)
        return o1

    def createHiddenNeuron(self, Neuronenanzahl: int, layer: int = 1) -> None:
        """
            Erstellt eine Schicht von versteckten Neuronen und fügt sie zur Liste der versteckten Neuronen hinzu.

            :param Neuronenanzahl: Die Anzahl der versteckten Neuronen in der Schicht.
            :param layer: Die Schichtanzahl. Standardmäßig ist die Schichtanzahl gleich 1 (layer=1).
        """
        for i in range(layer):
            x: list[hiddenneuron] = []

            for n in range(Neuronenanzahl):
                h1 = hiddenneuron()
                x.append(h1)

            self.hiddenneuronen.append(x)
        return None

    def calcGewichtAnzahl(self) -> int:
        """
            Berechnet die Gesamtanzahl der Gewichte im neuronalen Netzwerk.

            :return: Die Gesamtanzahl der Gewichte im Netzwerk.
        """
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
        """
            Erstellt ein vollständiges Netzwerk mit den gegebenen Gewichten.

            :param gewichte: Eine Liste von Gewichten für alle Verbindungen im Netzwerk.
        """

        if len(self.inputNeuronen) == 0 or len(self.hiddenneuronen) == 0 or len(self.outputneuronen) == 0:
            raise NotImplemented("Das Netzwerk ist nicht vollständig initialisiert.")

        if self.calcGewichtAnzahl() != len(gewichte):
            raise NotImplemented("Die Anzahl der Gewichte stimmt nicht mit der erwarteten Anzahl überein.")


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
        """
            Verkettet die versteckten Neuronen in den Schichten.

            :param gewichte: Eine Liste von Gewichten für alle Verbindungen im Netzwerk.
            :param index: Der aktuelle Index in der Liste der Gewichte.
            :param index1: Der Index der ersten Schicht von Neuronen.
            :param index2: Der Index der zweiten Schicht von Neuronen.
            :return: Der aktualisierte Index in der Liste der Gewichte.
        """
        if index1 == 0:
            return index

        for h2 in self.hiddenneuronen[index1]:
            for h1 in self.hiddenneuronen[index2]:
                h2.addConnection(Connection(h1, gewichte[index]))
                index += 1

        return self.verkettungHiddens(gewichte, index, index1-1, index2-1)
