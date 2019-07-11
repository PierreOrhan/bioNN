"""
    The method here can be called after some tensorflow training, or separately given the directory for the weight.
    It simulates the network on a set of inputs
"""

import os
import numpy as np
import pandas

from simulOfBioNN.parseUtils.parser import generateTemplateNeuralNetwork
from simulOfBioNN.simulNN import simulator
from simulOfBioNN.odeUtils.systemEquation import fPythonSparse
from simulOfBioNN.odeUtils.utils import obtainOutputArray,obtainTemplateArray


def launch(inputsArray,y,resultArray,directory_name="weightDic",simulateMethod = "fixPoint",layerInit=10**(-13),enzymeInit=10**(-6),inhibInit=10**(-4),activInit=10**(-4),endoInit=None,
           chemicalModel="templateModel"):
    """
        Load and then simulate:
            either with the fixed point strategy at equilibrium
            or throught the solving of ODE
    :param inputsArray:
    :param y :the result
    :param resultArray: the answer for the neural network to these test
    :param directory_name: directory where the weight are stored
    :param simulateMethod: string, either "ODE" or "fixPoint".

    :param layerInit: float, value for initial concentration of intermediate nodes
    :param enzymeInit: float, value for initial concentration of polymerase
    :param inhibInit: float, value for initial concentration of inhibition template
    :param activInit: float, value for initial concentration of activation template
    :param endoInit: float, if given we use the the complicated endo model.
    :return:
    """
    assert inputsArray.shape[0] == y.shape[0]
    if chemicalModel=="templateModel":
        complexity = "simple"
    elif chemicalModel=="normalTemplateModel":
        assert simulateMethod=="ODE"
        complexity = "normal"
    elif chemicalModel=="fullTemplateModel":
        assert simulateMethod=="ODE"
        complexity = "full"
    useEndo = False
    if endoInit is not None:
        assert simulateMethod=="ODE"
        useEndo = True
    useProtectionOnActivator = False
    useEndoOnInputs = False
    useEndoOnOutputs = True

    directory_for_network = os.path.join(directory_name,"Simul")
    _,masks = load(directory_name, directory_for_network,useEndo=useEndo,complexity=complexity,useProtectionOnActivator=useProtectionOnActivator,
                   useEndoOnOutputs=useEndoOnOutputs,useEndoOnInputs=useEndoOnInputs)

    # We realised that the rescale factor should be proportionate to the number of edges in order to keep competition low compare to the inhibition.
    # Moreover we observe that rescaling the template rather than the activation is probably better.
    computedRescaleFactor = np.sum([np.sum(m>0)+np.sum(m<0) for m in masks])
    print("Computed rescale factor is "+str(computedRescaleFactor))
    # Finally we observe that rescaling the number of enzyme up can also help to diminish the competition compare to the inhibition as the enzyme appear with a squared power in the last one
    enzymeInit = enzymeInit*(computedRescaleFactor**0.5)

    initialization_dic={}
    for layer in range(0,len(masks)): ## The first layer need not to be initiliazed
        for node in range(masks[layer].shape[0]):
            initialization_dic["X_"+str(layer+1)+"_"+str(node)] = layerInit
    inhibTemplateNames = obtainTemplateArray(masks=masks,activ=False)
    for k in inhibTemplateNames:
        initialization_dic[k] = inhibInit
    activTemplateNames = obtainTemplateArray(masks=masks,activ=True)
    for k in activTemplateNames:
        initialization_dic[k] = activInit
    initialization_dic["E"] = enzymeInit
    if complexity!="simple":
        initialization_dic["E2"] = enzymeInit
    if complexity!=None and useEndo:
        initialization_dic["Endo"] = endoInit


    if simulateMethod =="ODE":
        modes=["outputEqui","verbose"]
        results = simulator.executeODESimulation(fPythonSparse, directory_for_network, inputsArray,
                                                 initializationDic=initialization_dic, outputList=None, leak=10 ** (-13), endTime=1000,
                                                 sparse=True, modes=modes, timeStep=0.1)
        outputArray = results[modes.index("outputEqui")]
    elif simulateMethod=="fixPoint":
        modes=["outputEqui","verbose","time"]
        results = simulator.executeFixPointSimulation(directory_for_network, inputsArray, masks,initializationDic=initialization_dic, outputList=None,
                                                        sparse=True, modes=modes,
                                                        initValue=10**(-13), rescaleFactor=None)
        outputArray = results[modes.index("outputEqui")]
    ##We now compute the accuracy obtained:
    acc = 0
    distanceToNN = 0
    for test in range(inputsArray.shape[0]):
        answer = np.argmax(outputArray[:,test])
        if(answer == y[test]):
            acc += 1
        if(np.argmax(resultArray[test,:]) == answer):
            distanceToNN += 1
    acc = float(acc/inputsArray.shape[0])
    distanceToNN = float(distanceToNN)/inputsArray.shape[0]
    print("reached acc is: "+str(acc)+" ,the distance from the neural network is "+str(distanceToNN)+" in percentage of same answer")


def load(directory_name="weightDir",directory_for_network="",useEndo=False,complexity="simple",useProtectionOnActivator=False,useEndoOnOutputs=True,useEndoOnInputs=False):
    """
        Generate the biochemical equations from the weights of a deep neural network.
        The weights should be stored in the format csv, with the following name:
            weight_LayerNb
        The weight dir should be a directory containing only weights files.
        Bias should not be given in this file as we ignore it in the current algorithm.
    """
    files = os.listdir(directory_name)
    files = np.sort(files)

    # Load the masks:
    # We use 0.2 to discriminate the clipping values, as was done during training and validation
    masks=[]
    for file in files:
        if(file.split("_")[0]=="weight"):
            df=pandas.read_csv(os.path.join(directory_name,file))
            W = np.transpose(df.values[:,1:])
            WZeros = np.zeros((W.shape))
            WPos = np.zeros((W.shape)) + 1
            WNeg = np.zeros((W.shape)) + -1
            masks+=[np.where(W<-0.2,WNeg,np.where(W>0.2,WPos,WZeros))]

    # We generate the chemical network
    # For constants we can either let the default value or give pre-defined values, for now use default value
    if useEndo:
        generateTemplateNeuralNetwork(directory_for_network,masks,complexity=complexity,useProtectionOnActivator=useProtectionOnActivator,
                                      useEndoOnOutputs=useEndoOnOutputs,useEndoOnInputs=useEndoOnInputs)
    else:
        generateTemplateNeuralNetwork(directory_for_network,masks,complexity=complexity,endoConstants=None,useProtectionOnActivator=useProtectionOnActivator,
                                      useEndoOnInputs=useEndoOnInputs,useEndoOnOutputs=useEndoOnOutputs)
    print("Generated network at "+str(directory_for_network))
    return directory_for_network,masks