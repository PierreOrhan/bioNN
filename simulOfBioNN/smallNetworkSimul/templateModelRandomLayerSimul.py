"""

    This script was used to produce a random first initial layer on which we compare the simulation results and the fit.
        Moreover: the value of inhibitors and activators are selected as random from a law with two gaussian peak, one around 10**(-4) and one around 10**(-8)

    After the simulation, we randomly select an output neuron.
    On this neuron we display the same diagram where activator (inhibitors) are computed as the sum of activations (inhibitions).

"""
import numpy as np
import os
from simulOfBioNN.parseUtils.parser import generateNeuralNetwork,generateTemplateNeuralNetwork,read_file
from simulOfBioNN.simulNN.simulator import executeODESimulation
from simulOfBioNN.odeUtils.systemEquation import fPythonSparse
from simulOfBioNN.odeUtils.utils import readAttribute,obtainTemplateArray,obtainOutputArray
from simulOfBioNN.plotUtils.adaptivePlotUtils import colorDiagram,neuronPlot,plotEvolution,fitComparePlot
import sys
import pandas

def chemf(k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kd,TA,TI,E0,A,I,otherConcentrationsActiv=0,otherConcentrationsInhib=0):
    k1M = k1/(k1n+k2)
    k5M = k5/(k5n+k6)
    k3M = k3/(k3n+k4)
    Cactiv = k2*k1M*TA*E0
    CInhib = k6*k5M*k4*k3M*TI*E0*E0
    Kactiv = k1M*TA
    Kinhib = k3M*TI
    cp =kd*(1 + Kactiv*(A+otherConcentrationsActiv) + Kinhib*(I+otherConcentrationsInhib))
    return Cactiv*A/(cp + CInhib*I/cp)

def _createMask(nbrOutputNodes):
    mask=[]
    for idx in range(nbrOutputNodes):
        zero=np.zeros(2*nbrOutputNodes)
        zero[2*idx]=1
        zero[2*idx+1]=-1
        mask+=[zero]
    return np.array([mask])

def sample_bimodal_distrib(nbrInputs,peak1=-4,peak2=-8,sigma1=1,sigma2=1,mixture = 0.5):
    """
        Sample from a bimodal distribution around peak 1 and peak 2 based on gaussian law.
    :param nbrInputs: int, indicate the number of inputs
    :param peak1: int, indicate the power of 10 around which the first peak should be centered
    :param peak2: int, indicate the power of 10 around which the second peak should be centered
    :param sigma1: int, indicate the width in power of 10 around the peak1.
    :param sigma2: int, indicate the width in power of 10 around the peak2.
    :param mixture: proba for the mixture of the two.
    :return:
    """
    mixChoice = np.random.random(nbrInputs)
    gaussian1 = np.random.normal(peak1,sigma1,nbrInputs)
    gaussian2 = np.random.normal(peak2,sigma2,nbrInputs)

    powers = np.where(mixChoice>0.5,gaussian1,gaussian2)
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

    name = "templateModelFirstLayer/randomLayer100_sparse_rescaleOn_E_X_plot"
    endTime = 10000
    timeStep = 0.1

    nbrOutputNodes = 5
    nbrInputs = 100
    activatorsOnObserved = []
    inhibitorsOnObserved = []
    while(len(activatorsOnObserved)==0 and len(inhibitorsOnObserved)==0):
        masks= _sample_layer_architecture(nbrInputs,nbrOutputNodes,sparsity=.9)
        nodeObserved = np.random.randint(0,nbrOutputNodes,1)[0]
        # We look at the activators and inhibitors on the observed node, and verify that they have a length greater than one (while loop).
        for idx,m in enumerate(masks[0][nodeObserved]):
            if m==1:
                activatorsOnObserved += [idx]
            elif m==-1:
                inhibitorsOnObserved += [idx]

    print(masks)
    print("Observed nodes has "+str(len(activatorsOnObserved))+" activators and "+str(len(inhibitorsOnObserved))+" inhibitors")
    print(masks[0][nodeObserved])
    modes = ["verbose","outputPlot"]
    FULL = True
    outputMode = "last"
    outputList = None # ["X_1_"+str(nodeObserved)] #"all" or None or list of output species
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

    X1,X2,x_test,otherActivInitialC,otherInhibInitialC = _generate_initialConcentration_firstlayer(activatorsOnObserved,inhibitorsOnObserved,masks,nbrInputs,minLogSpace=-8,maxLogSpace=-4,nbrValue=10)

    print(X1)
    print(X2)

    # We realised that the rescale factor should be proportionate to the number of edges in order to keep competition low compare to the inhibition.
    # Moreover we observe that rescaling the template rather than the activation is probably better.
    computedRescaleFactor = np.sum([np.sum(m>0)+np.sum(m<0) for m in masks])
    print("Computed rescale factor is "+str(computedRescaleFactor))
    # Finally we observe that rescaling the number of enzyme up can also help to diminish the competition compare to the inhibition as the enzyme appear with a squared power in the last one

    enzymeInit = enzymeInit*(computedRescaleFactor**0.5)
    inhibInit = inhibInit
    activInit = activInit

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


    if useDerivativeLeak:
        results = executeODESimulation(fPythonSparse, name, x_test, initialization_dic, outputList= outputList,
                                       leak = leak, endTime=endTime, sparse=True, modes=modes,
                                       timeStep=timeStep, initValue= initValue, rescaleFactor=computedRescaleFactor)
    else:
        results = executeODESimulation(fPythonSparse, name, x_test, initialization_dic, outputList= outputList,
                                       leak = 0, endTime=endTime, sparse=True, modes=modes,
                                       timeStep=timeStep, initValue= initValue, rescaleFactor=computedRescaleFactor)

    if("outputPlot" in modes):
        shapeP = len(X1)*len(X2)
        nameDic = results[-1]
        X = results[modes.index("outputPlot")]
        myname=os.path.join(sys.path[0],name)
        if(outputMode=="all"):
            if outputList is None or type(outputList)==str:
                outputList = list(nameDic.keys())
        else:
            outputList = obtainOutputArray(nameDic)
        specialnameDic ={}
        for idx2,e in enumerate(outputList):
            specialnameDic[e] = idx2
        for e in range(X.shape[1]):
            displayX = np.moveaxis(X[:,e,:],0,1)
            plotEvolution(np.arange(0,endTime,timeStep),os.path.join(myname,str(e)), specialnameDic, displayX, wishToDisp=outputList, displaySeparate=True, displayOther=True)

    if("outputEqui" in modes):
        experiment_path = name
        C0=readAttribute(experiment_path,["C0"])["C0"]
        rescaleFactor = readAttribute(experiment_path,["rescaleFactor"])["rescaleFactor"]
        output = results[modes.index("outputEqui")]
        output = np.reshape(output,(len(X1),len(X2)))
        X1 = X1/(C0*rescaleFactor)
        X2 = X2/(C0*rescaleFactor)
        otherActivInitialC = otherActivInitialC/(C0*rescaleFactor)
        otherInhibInitialC = otherInhibInitialC/(C0*rescaleFactor)


        colorDiagram(X1,X2,output,"Initial concentration of X1","Initial concentration of X2","Equilibrium concentration of the output",figname=os.path.join(experiment_path, "neuralDiagramm.png"),equiPotential=False)
        neuronPlot(X1,X2,output,figname=os.path.join(experiment_path, "activationLogX1.png"),figname2=os.path.join(experiment_path, "activationLogX2.png"),useLogX = True, doShow= False)
        neuronPlot(X1,X2,output,figname=os.path.join(experiment_path, "activationX1.png"),figname2=os.path.join(experiment_path, "activationX2.png"),useLogX = False, doShow= False)
        df = pandas.DataFrame(X1)
        df.to_csv(os.path.join(experiment_path,"inputX1.csv"))
        df2 = pandas.DataFrame(X2)
        df2.to_csv(os.path.join(experiment_path,"inputX2.csv"))

        #we try to fit:
        nbrConstant = int(readAttribute(experiment_path,["Numbers_of_Constants"])["Numbers_of_Constants"])
        if nbrConstant == 12: #only one neuron, it is easy to extract cste values
            k1,k1n,k2,k3,k3n,k4,_,k5,k5n,k6,kd,_=[readAttribute(experiment_path,["k"+str(i)])["k"+str(i)] for i in range(0,nbrConstant)]
        else:
            k1,k1n,k2,k3,k3n,k4,_,k5,k5n,k6,kd,_= [0.9999999999999998,0.1764705882352941,1.0,0.9999999999999998,0.1764705882352941,1.0,0.018823529411764708,0.9999999999999998,0.1764705882352941,1.0,0.018823529411764708,0.018823529411764708]


        TA = activInit/C0
        TI = inhibInit/C0
        E0 = enzymeInit/C0
        k1M = k1/(k1n+k2)
        k5M = k5/(k5n+k6)
        k3M = k3/(k3n+k4)

        Cactiv = k2*k1M*TA*E0
        CInhib = k6*k5M*k4*k3M*TI*E0*E0
        Kactiv = k1M*TA
        Kinhib = k3M*TI
        print("Cactiv value is :"+str(Cactiv))
        print("Cinhib value is :"+str(CInhib))
        print("Kactiv value is :"+str(Kactiv))
        print("Kinhib value is :"+str(Kinhib))
        print("kd is :"+str(kd))
        print("template activator value is:"+str(TA))
        print("template inhibitor value is:"+str(TI))
        # print("other activators add a factor in the competition of:"+str(np.sum(otherActivInitialC,axis=1)))
        # print("other inhibitors add a factor in the competition of:"+str(np.sum(otherInhibInitialC,axis=1)))
        # print("while values for X1 are "+str(X1))
        # print("while values for X2 are "+str(X2))


        fitOutput = np.zeros((len(X1),len(X2)))
        for idx1,x1 in enumerate(X1):
            for idx2,x2 in enumerate(X2):
                fitOutput[idx1,idx2]= chemf(k1, k1n, k2, k3, k3n, k4, k5, k5n, k6, kd, TA, TI, E0,
                                            x1, x2,otherConcentrationsActiv=otherActivInitialC[idx1,idx2],otherConcentrationsInhib=otherInhibInitialC[idx1,idx2])
        fitOutput = np.array(fitOutput)


        colorDiagram(X1,X2,fitOutput,"Initial concentration of X1","Initial concentration of X2","Equilibrium concentration of the output",figname=os.path.join(experiment_path, "AnalyticNeuralDiagramm.png"),equiPotential=False)
        neuronPlot(X1,X2,fitOutput,figname=os.path.join(experiment_path, "AnalyticActivationX1Log.png"),figname2=os.path.join(experiment_path, "AnalyticActivationX2Log.png"),useLogX=True)
        neuronPlot(X1,X2,fitOutput,figname=os.path.join(experiment_path, "AnalyticActivationX1.png"),figname2=os.path.join(experiment_path, "AnalyticActivationX2.png"),useLogX=False, doShow= False)


        courbs=[0,int(fitOutput.shape[1]/2),fitOutput.shape[1]-1,int(fitOutput.shape[1]/3),int(2*fitOutput.shape[1]/3)]
        fitComparePlot(X1,X2,output,fitOutput,courbs,
                       figname=os.path.join(experiment_path, "fitComparisonX1.png"),
                       figname2=os.path.join(experiment_path, "fitComparisonX2.png"),useLogX=False)
        fitComparePlot(X1,X2,output,fitOutput,courbs,
                       figname=os.path.join(experiment_path, "fitComparisonLogX1.png"),
                       figname2=os.path.join(experiment_path, "fitComparisonLogX2.png"),useLogX=True)