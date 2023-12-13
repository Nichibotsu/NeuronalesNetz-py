from NN.NeuronalesNetz import neuronalesNetz


if __name__ == '__main__':
    nn=neuronalesNetz()
    i1=nn.createInputNeuron()
    i2=nn.createInputNeuron()
    nn.createHiddenNeuron(2,3)
    o1 = nn.createOutputNeuron()

    i1.set_Value(2)
    i2.set_Value(3)

    nn.createFullMesh([1,1,1,1,1,1,1,1,1,1,1,1,1,1])


    print(o1.get_Value())








