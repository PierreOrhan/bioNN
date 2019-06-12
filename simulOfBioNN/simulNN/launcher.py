"""
    The method here can be called after some tensorflow training, or separately given the directory for the weight.
    It simulates the network on a set of inputs
"""

import os
import numpy as np
import pandas

from simulOfBioNN.parseUtils.parser import generateNeuralNetwork
from simulOfBioNN.simulNN import simulator
from simulOfBioNN.odeUtils.systemEquation import fPythonSparse
from simulOfBioNN.odeUtils.utils import obtainOutputDic


def launch(inputsArray,y,resultArray,directory_name="weightDic",layerInit=10**(-8),enzymeInit=10**(-6)):
    """
        Load and then simulate
    :param inputsArray:
    :param y :the result
    :param resultArray: the answer for the neural network to these test
    :param directory_name: directory where the weight are stored
    :return:
    """
    assert inputsArray.shape[0] == y.shape[0]

    directory_for_network = os.path.join(directory_name,"Simul")
    _,masks = load(directory_name, directory_for_network)

    initialization_dic={}
    outputList=[]
    for layer in range(1,len(masks)): ## the first layer need not to be initiliazed
        for node in range(masks[layer].shape[0]):
            initialization_dic["X_"+str(layer)+"_"+str(node)] = layerInit
            if(layer == len(masks)-1):
                outputList+=["X_"+str(layer)+"_"+str(node)]
    initialization_dic["E"] = enzymeInit
    initialization_dic["E2"] = enzymeInit


    modes=["outputEqui","verbose"]
    results = simulator.executeSimulation(fPythonSparse, directory_for_network, inputsArray,
                                          initializationDic=initialization_dic,outputList=outputList ,leak=10 ** (-13), endTime=1000,
                                          sparse=True,modes=modes,timeStep=0.1)
    outputArray = results[modes.index("outputEqui")]
    ##We now compute the accuracy obtained:
    acc = 0
    distanceToNN = 0
    for test in range(inputsArray.shape[0]):
        answer = np.argmax(outputArray[test,:])
        if(answer == y[test]):
            acc += 1
        if(resultArray == answer):
            distanceToNN += 1
    acc = float(acc/inputsArray.shape[0])
    distanceToNN = float(distanceToNN/inputsArray.shape[0])
    print("reached acc is: "+str(acc)+" ,the distance from the neural network is "+str(distanceToNN)+" in percentage of same answer")


def load(directory_name="weightDir",directory_for_network=""):
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
    generateNeuralNetwork(directory_for_network,masks)
    print("Generated network at "+str(directory_for_network))
    return directory_for_network,masks