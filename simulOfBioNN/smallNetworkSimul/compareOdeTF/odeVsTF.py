"""

    This script was used to produce a random first initial layer on which we compare the simulation results and the fit.
        Moreover: the value of inhibitors and activators are selected as random from a law with two gaussian peak, one around 10**(-4) and one around 10**(-8)

    After the simulation, we randomly select an output neuron.
    On this neuron we display the same diagram where activator (inhibitors) are computed as the sum of activations (inhibitions).

"""
import numpy as np
import os
from simulOfBioNN.parseUtils.parser import generateTemplateNeuralNetwork,read_file

from simulOfBioNN.odeUtils.utils import readAttribute,obtainTemplateArray,obtainOutputArray
from simulOfBioNN.plotUtils.adaptivePlotUtils import colorDiagram,neuronPlot,plotEvolution,fitComparePlot

from simulOfBioNN.odeUtils import utils as utilForODE
from simulOfBioNN.simulNN.simulator import executeODESimulation
from simulOfBioNN.odeUtils.systemEquation import fPythonSparse
from simulOfBioNN.smallNetworkSimul.compareTFvsPython.pythonBasicSolver import pythonSolver


from scipy.optimize import minimize,root,brentq
import sys
import time
import pandas




def sample_bimodal_distrib(nbrInputs,peak1=-4,peak2=-8,sigma1=1.,sigma2=1.,mixture = 0.5):
    """
        Sample from a bimodal distribution around peak 1 and peak 2 based on gaussian law.
    :param nbrInputs: int, indicate the number of inputs
    :param peak1: int, indicate the power of 10 around which the first peak should be centered
    :param peak2: int, indicate the power of 10 around which the second peak should be centered
    :param sigma1: float, indicate the width in power of 10 around the peak1.
    :param sigma2: float, indicate the width in power of 10 around the peak2.
    :param mixture: proba for the mixture of the two.
    :return:
    """
    mixChoice = np.random.random(nbrInputs)
    gaussian1 = np.random.normal(peak1,sigma1,nbrInputs)
    gaussian2 = np.random.normal(peak2,sigma2,nbrInputs)

    powers = np.where(mixChoice>mixture,gaussian1,gaussian2)
    inputs = [10**p for p in powers]
    return inputs,powers


def _sample_layer_architecture(nbrInputs,nbrOutputs,sparsity=0.5):
    """
        The sparsity indicate the percentage of empty connections.
    :param nbrInputs: number of nodes in the first layer
    :param nbrOutputs: number of nodes in the second layer
    :param sparsity: indicate the percentage of empty connections
    :return:
    """
    choiceEdges = np.random.random((nbrOutputs,nbrInputs)) # choose the number of edges that should be connected.
    choiceActiv = np.random.random((nbrOutputs,nbrInputs)) # choose the edges that are activators (found number below 0.5).
    masks = [np.where(choiceEdges<sparsity,np.zeros(choiceEdges.shape),np.where(choiceActiv>0.5,np.zeros(choiceEdges.shape)+1,np.zeros(choiceEdges.shape)-1))]
    return masks

def _generate_initialConcentration_firstlayer(activatorsOnObserved,inhibitorsOnObserved,masks,nbrInputs,
                                              minLogSpace=-8,maxLogSpace=-4,nbrValue=10):
    """
        Generate the concentrations for the first layer.

    :param minLogSpace: minimal power of 10 for the gaussian mean, default to -8
    :param maxLogSpace: maximal power of 10 for the gaussian mean, default to -4
    :param nbrValue: indicate the number of points that should be taken between [minLogSpace,maxLogspace]
    :return:
    """

    # Generate the first layer concentration:
    #For the activators from the observed node, we sample values in a gaussian disribution, with mean varying on a log scale.
    peaks = np.log(np.logspace( minLogSpace , maxLogSpace , nbrValue))/np.log(10)
    activPowers = [np.random.normal(p,1,len(activatorsOnObserved)) for p in peaks]
    inhibPowers = [np.random.normal(p,1,len(inhibitorsOnObserved)) for p in peaks]
    activInputs = []
    for acp in activPowers:
        activInputs+=[[10**p for p in acp]]
    inhibInputs = []
    for inp in inhibPowers:
        inhibInputs+=[[10**p for p in inp]]
    activInputs = np.array(activInputs)
    inhibInputs = np.array(inhibInputs)


    X1 = np.sum(activInputs,axis=1)
    argsortX1 = np.argsort(X1)
    X1 = X1[argsortX1]
    X2 = np.sum(inhibInputs,axis=1)
    argsortX2 = np.argsort(X2)
    X2 = X2[argsortX2]

    activInputs = activInputs[argsortX1,:]
    inhibInputs = inhibInputs[argsortX2,:]

    # For the remaining inputs we sample from a bimodal distribution:
    myOtherInputs = sample_bimodal_distrib(nbrInputs-len(activatorsOnObserved)-len(inhibitorsOnObserved),sigma1=0.5,sigma2=0.5)[0]
    otherInputs = np.array([myOtherInputs for p in peaks])

    # To compute the competition properly, we must add the concentrations of species for each of the activations or inhibitions
    otherActivInitialC = np.zeros((len(peaks),len(peaks)))
    otherInhibInitialC = np.zeros((len(peaks),len(peaks)))

    x_test=np.zeros((len(peaks),len(peaks),nbrInputs))
    for idxp0,p0 in enumerate(peaks):
        line=np.zeros((len(peaks),nbrInputs))
        for idxp,p in enumerate(peaks):
            for i,idx in enumerate(activatorsOnObserved):
                line[idxp,idx] = activInputs[idxp0,i] #We keep the activation constant for the second axis.
                otherActivInitialC[idxp0,idxp]+= line[idxp, idx] * (np.sum(masks[0][ : , idx] > 0) - 1)
                otherInhibInitialC[idxp0,idxp]+= line[idxp, idx] * np.sum(masks[0][ :, idx] < 0)
            for i,idx in enumerate(inhibitorsOnObserved):
                line[idxp,idx] = inhibInputs[idxp,i]
                otherActivInitialC[idxp0,idxp]+= line[idxp, idx] * np.sum(masks[0][ :, idx] > 0)
                otherInhibInitialC[idxp0,idxp]+= line[idxp, idx] * (np.sum(masks[0][ :, idx] < 0)-1)
            c=0
            for idx in range(nbrInputs):
                if idx not in activatorsOnObserved and idx not in inhibitorsOnObserved:
                    line[idxp,idx] = otherInputs[idxp,c]
                    c=c+1
                    otherActivInitialC[idxp0,idxp]+= line[idxp, idx] * np.sum(masks[0][ :, idx] > 0)
                    otherInhibInitialC[idxp0,idxp]+= line[idxp, idx] * np.sum(masks[0][ :, idx] < 0)
        x_test[idxp0]=line

    x_test = np.reshape(x_test,(len(peaks)*len(peaks),nbrInputs))

    return X1,X2,x_test,otherActivInitialC,otherInhibInitialC


if __name__ == '__main__':

    name = "compare100"
    endTime = 10000
    timeStep = 0.1

    doODEvsTF = True
    doODEvsPython = True
    doTFvsPython = True

    # masks=[np.array([[1,-1,0,0],[0,0,1,-1]]),np.array([[1,-1]])]
    # nbrInputs=masks[0].shape[1]
    #
    # activatorsOnObserved = [0]
    # inhibitorsOnObserved = [1]
    # nodeObserved = 0
    nbrInputs=[10,10]
    nbrOutputs=[10,5]
    activatorsOnObserved = []
    inhibitorsOnObserved = []
    while(len(activatorsOnObserved)==0 and len(inhibitorsOnObserved)==0):
        masks=[]
        for idx,nbi in enumerate(nbrInputs):
            masks+= _sample_layer_architecture(nbi,nbrOutputs[idx],sparsity=.5)
        nodeObserved = np.random.randint(0,nbrOutputs[0],1)[0]
        # We look at the activators and inhibitors on the observed node, and verify that they have a length greater than one (while loop).
        for idx,m in enumerate(masks[0][nodeObserved]):
            if m==1:
                activatorsOnObserved += [idx]
            elif m==-1:
                inhibitorsOnObserved += [idx]

    ## FOR SOlVING REASONS WE ADD A FIRST NON-LINEAR INITIAL LAYER:
    masksForTF = [np.transpose(m) for m in masks]

    # We add a non-linear layer, but as its node-to-node, we can keep the value we choosed for the inputs :)
    # The node observed will then be in the second layer, we just need to change this for the ode !

    masks = [np.identity(nbrInputs[0])] + masks
    print(masks)
    print("Observed nodes has "+str(len(activatorsOnObserved))+" activators and "+str(len(inhibitorsOnObserved))+" inhibitors")
    print(masks[0][nodeObserved])

    modes = ["verbose","outputEqui"]
    FULL = True
    outputMode = "all"
    #Here we need to choose X_2 this time
    outputList = ["X_2_"+str(nodeObserved)] #"all" or None or list of output species
    complexity="simple"
    useEndo = False  # if we want to use the complicated endo model
    useProtectionOnActivator = False
    useEndoOnInputs = False
    useEndoOnOutputs = True
    useDerivativeLeak = True

    leak = 10**(-10)

    layerInit = 10**(-13) #initial concentation value for species in layers
    initValue = 10**(-13) #initial concentration value for all species.
    enzymeInit = 5*10**(-7)
    endoInit = 10**(-5) #only used if useEndo == True
    activInit =  10**(-4)
    inhibInit =  10**(-4)

    if useEndo:
        generateTemplateNeuralNetwork(name,masks,complexity=complexity,useProtectionOnActivator=useProtectionOnActivator,
                                      useEndoOnOutputs=useEndoOnOutputs,useEndoOnInputs=useEndoOnInputs)
    else:
        generateTemplateNeuralNetwork(name,masks,complexity=complexity,endoConstants=None,useProtectionOnActivator=useProtectionOnActivator,
                                      useEndoOnInputs=useEndoOnInputs,useEndoOnOutputs=useEndoOnOutputs)

    #Here take mass[1:] to use the function withotu considerations of the first non-linear layer.
    X1,X2,x_test,otherActivInitialC,otherInhibInitialC = _generate_initialConcentration_firstlayer(activatorsOnObserved,inhibitorsOnObserved,masks[1:],nbrInputs[0],minLogSpace=-8,maxLogSpace=-4,nbrValue=5)

    # We realised that the rescale factor should be proportionate to the number of edges in order to keep competition low compare to the inhibition.
    # Moreover we observe that rescaling the template rather than the activation is probably better.

    if("outputEqui" in modes):
        experiment_path = name


        _,rescaleFactor = utilForODE.rescaleInputConcentration(networkMask=masks)

        if doODEvsPython or doODEvsTF:
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
            initialization_dic["E"] = enzymeInit*rescaleFactor**0.5
            if complexity!="simple":
                initialization_dic["E2"] = enzymeInit*rescaleFactor**0.5
            if complexity!=None and useEndo:
                initialization_dic["Endo"] = endoInit
            if useDerivativeLeak:
                results = executeODESimulation(fPythonSparse, name, x_test, initialization_dic, outputList= outputList,
                                               leak = leak, endTime=endTime, sparse=True, modes=modes,
                                               timeStep=timeStep, initValue= initValue, rescaleFactor=rescaleFactor)
            else:
                results = executeODESimulation(fPythonSparse, name, x_test, initialization_dic, outputList= outputList,
                                               leak = 0, endTime=endTime, sparse=True, modes=modes,
                                               timeStep=timeStep, initValue= initValue, rescaleFactor=rescaleFactor)
            output = results[modes.index("outputEqui")]
            output = np.reshape(output,(len(X1),len(X2)))

        C0 = 8.086075400626399e-07
        cstlist = [0.9999999999999998,0.1764705882352941,1.0,0.9999999999999998,0.1764705882352941,1.0,
                   0.018823529411764708,0.9999999999999998,0.1764705882352941,1.0,0.018823529411764708,0.018823529411764708]
        k1,k1n,k2,k3,k3n,k4,_,k5,k5n,k6,kd,_= cstlist
        TA = float(activInit/C0)
        TI = float(inhibInit/C0)
        E0 = float(enzymeInit/C0)
        nbUnits = [10,5]
        sparsities=[0.5,0.5]
        constantList = [0.9999999999999998,0.1764705882352941,1.0,0.9999999999999998,
                        0.1764705882352941,1.0,0.9999999999999998,0.1764705882352941,1.0,0.018823529411764708]
        constantList+=[constantList[-1]]

        if doODEvsTF or doTFvsPython:
            import tensorflow as tf
            from simulOfBioNN.nnUtils.chemTemplateNN import chemTemplateNNModel
            model = chemTemplateNNModel.chemTemplateNNModel(nbUnits=nbUnits,sparsities=sparsities,reactionConstants= constantList,
                                                            enzymeInitC=E0, activTempInitC=TA, inhibTempInitC=TI, randomConstantParameter=None, usingLog=False)
            model.compile(optimizer=tf.optimizers.Adam(),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            model.build(input_shape=(None,nbrInputs[0]))
            model.updateArchitecture(masksForTF, constantList, TA, TI, E0)
            model.force_rescale(rescaleFactor)

        X1 = X1/C0
        X2 = X2/C0
        x_test = np.array(x_test,dtype=np.float32)/C0

        print("=================FINISHED ODE SIMULATION========================")
        if doODEvsTF or doTFvsPython:
            outputTF = np.array(model.predConcentration(x_test,layerObserved=0,nodeObserved=nodeObserved))
            outputTF = np.reshape(outputTF,(len(X1),len(X2)))
            print(outputTF)

        courbs=[0,int(outputTF.shape[1]/2),outputTF.shape[1]-1,int(outputTF.shape[1]/3),int(2*outputTF.shape[1]/3)]

        if doODEvsTF:
            fitComparePlot(X1, X2, output, outputTF, courbs,
                           figname=os.path.join(experiment_path, "TFVSOdeX1.png"),
                           figname2=os.path.join(experiment_path, "TFVSOdeX2.png"), useLogX=False)
            fitComparePlot(X1, X2, output, outputTF, courbs,
                           figname=os.path.join(experiment_path, "TFVSOdelogX1.png"),
                           figname2=os.path.join(experiment_path, "TFVSOdelogX2.png"), useLogX=True)

        print("=================FINISHED TF SIMULATION========================")

        if doODEvsPython or doTFvsPython:
            outputPython = np.zeros((len(X1),len(X2)))
            pythonModel = pythonSolver(masks,k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kd,kd,TA,TI,E0*(rescaleFactor**0.5))
            for idx1,x1 in enumerate(X1):
                for idx2,x2 in enumerate(X2):
                    cp = pythonModel.computeCPonly(x_test[idx1*len(X1)+idx2]/rescaleFactor)
                    outputPython[idx1,idx2]=pythonModel.computeEquilibriumValue(cp,x_test[idx1*len(X1)+idx2]/rescaleFactor,observed=(2,nodeObserved))

        print("=================FINISHED raw PYTHON SIMULATION========================")
        if doODEvsPython:
            fitComparePlot(X1, X2, output, outputPython, courbs,
                           figname=os.path.join(experiment_path, "PythonVSOdeX1.png"),
                           figname2=os.path.join(experiment_path, "PythonVSOdeX2.png"), useLogX=False)
            fitComparePlot(X1, X2, output, outputPython, courbs,
                           figname=os.path.join(experiment_path, "PythonVSOdelogX1.png"),
                           figname2=os.path.join(experiment_path, "PythonVSOdelogX2.png"), useLogX=True)
        if doTFvsPython:
            fitComparePlot(X1, X2, outputTF, outputPython, courbs,
                           figname=os.path.join(experiment_path, "TFVSPythonX1.png"),
                           figname2=os.path.join(experiment_path, "TFVSPythonX2.png"), useLogX=False)
            fitComparePlot(X1, X2, outputTF, outputPython, courbs,
                           figname=os.path.join(experiment_path, "TFVSPythonlogX1.png"),
                           figname2=os.path.join(experiment_path, "TFVSPythonlogX2.png"), useLogX=True)

        fitComparePlot(X1, X2, outputTF, outputTF, courbs,
                       figname=os.path.join(experiment_path, "TFVSTFX1.png"),
                       figname2=os.path.join(experiment_path, "TFVSTFX2.png"), useLogX=False)
        fitComparePlot(X1, X2, outputTF, outputTF, courbs,
                       figname=os.path.join(experiment_path, "TFVSTFlogX1.png"),
                       figname2=os.path.join(experiment_path, "TFVSTFlogX2.png"), useLogX=True)