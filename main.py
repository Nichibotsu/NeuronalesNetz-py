from NN.NeuronalesNetz import neuronalesNetz


if __name__ == '__main__':
    #Initialisiere ein neuronales Netzwerk
    nn=neuronalesNetz()

    #Erstelle Input-Neuronen
    i1=nn.createInputNeuron()
    i2=nn.createInputNeuron()

    """
    Erstelle versteckte Neuronen
    :param Neuronenanzahl pro Layer
    :param Anzahl der Layer 
    Default: Layer = 1
    """
    nn.createHiddenNeuron(2,2)

    #Erstellt ein Output-Neuron
    o1 = nn.createOutputNeuron()

    #Setze Werte für die Input-Neuronen
    i1.set_Value(2)
    i2.set_Value(3)

    #Erstelle eine vollständige Verbindung zwischen allen Neuronen
    nn.createFullMesh([1,1,1,1,1,1,1,1,1,1])

    #Druckt den Wert des Output-Neurons nach der Berechnung
    print(o1.get_Value())








