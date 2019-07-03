"""

    Starting with a simple network, we try to obtain an idea of the usability of the obtained simulated chemf (chemical activation function for the neurons).

    To do so we change the reason for competition (either competition on template or/and on enzyme) by varying the concentration of enzyme and template.
    For all concentration we compute 2 indicator of usability: one for activation, one for inhibition.

    The indicator of activation is obtained by fixing the inhibition at 10**(-6) = Cinhib.
    Then we generate the chemf for activation function varying between 10**(-8) to 10**(-4).
    From 10**(-8) to 10**(-6)=Cinhib we call this the inhbition interval. The inhibition should win over this interval.
    From 10**(-6)=Cinhib to 10**(-4) we call this the activation interval. The activation should win over this interval.
    The measure is then defined as the (integral[over activation interval](Chemf)-integral[over inhibition interval](Chemf))integral[over activation interval](activInput)
    The denominator is used to scale the indicator.

    The indicator for inhibition is closely similar.

    Equations are solved in parallel on the CPU

"""
import numpy as np
import os
from simulOfBioNN.parseUtils.parser import generateNeuralNetwork,generateTemplateNeuralNetwork,read_file
from simulOfBioNN.simulNN.simulator import executeSimulation
from simulOfBioNN.odeUtils.systemEquation import f
from simulOfBioNN.odeUtils.utils import readAttribute,obtainTemplateArray,obtainOutputArray
from simulOfBioNN.plotUtils.adaptivePlotUtils import colorDiagram,neuronPlot,plotEvolution,fitComparePlot
import sys
import pandas



def indicator(Xlog, X,  output, indexSplit, sens=1, Xactiv=0):
    """
        Compute the approximative integral over the two domains, and then indicator.
    :param Xlog: logarithm of the x-axis
    :param X: x-axis
    :param output:
    :param indexSplit:
    :return: indicator value
    """
    assert Xlog.shape[0] == indexSplit * 2
    if sens==1:
        Iactiv = np.sum(output[indexSplit:len(output)-1] * (Xlog[indexSplit + 1:len(output)] - Xlog[indexSplit:len(output) - 1]))
        Iinhib = np.sum(output[0:indexSplit-1] * (Xlog[1:indexSplit] - Xlog[0:indexSplit - 1]))
        IactivInput = np.sum(X[indexSplit:len(output) - 1] * (Xlog[indexSplit + 1:len(output)] - Xlog[indexSplit:len(output) - 1]))
        return (Iactiv-Iinhib)/IactivInput
    else:
        Iactiv = np.sum(output[indexSplit:len(output)-1] * (Xlog[indexSplit + 1:len(output)] - Xlog[indexSplit:len(output) - 1]))
        Iinhib = np.sum(output[0:indexSplit-1] * (Xlog[1:indexSplit] - Xlog[0:indexSplit - 1]))
        IactivInput = np.sum(Xactiv * (Xlog[1:indexSplit] - Xlog[0:indexSplit - 1]))
        return (Iinhib-Iactiv)/IactivInput


def simulModelIndicator(name,nameFig,enzymeInit = 10**(-6),activInit =  10**(-8),inhibInit =  10**(-8), x2val = None, x1val = None, indexSplit = 10):
    """
        Generate an indicator using the indicator function after simulating the model over a range of inputs.
    :param name: directory where the network is located
    :param nameFig: directory to store figure
    :param enzymeInit: initial concentration of the enzyme
    :param activInit: initial concentration of the activators
    :param inhibInit: initial concentration of the inhibitors
    :param x2val: if not none, define fix initial concentration value for inhibitors
    :param x1val: if not none, define fix initial concentration value for activators
    :param indexSplit: the size of the test will be two time indexSplit.
    :return:
    """
    endTime = 10000
    timeStep = 0.1
    modes = ["verbose","outputEqui"]

    leak = 10**(-10)

    layerInit = 10**(-13) #initial concentation value for species in layers
    initValue = 10**(-13) #initial concentration value for all species.

    endoInit = 10**(-5) #only used if useEndo == True

    #generate the first layer concentration:

    if x1val is not None:
        assert x2val is None
        X1=np.array([x1val])
        X2 = np.concatenate((np.logspace(-8, -6, indexSplit), np.logspace(-6, -4, indexSplit)))
    else:
        assert x2val is not None
        X2 = np.array([x2val])
        X1 = np.concatenate((np.logspace(-8, -6, indexSplit), np.logspace(-6, -4, indexSplit)))
    # generate concentration for all different experiments:
    x_test=[]
    for x1 in X1:
        for x2 in X2:
            x_test+=[[x1,x2]]
    x_test = np.array(x_test)

    initialization_dic={}
    for layer in range(0,len(masks)): ## the first layer need not to be initiliazed
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
        results = executeSimulation(f, name, x_test, initialization_dic, outputList= outputList,
                                    leak = leak, endTime=endTime,sparse=False, modes=modes,
                                    timeStep=timeStep, initValue= initValue)
    else:
        results = executeSimulation(f, name, x_test, initialization_dic, outputList= outputList,
                                    leak = 0, endTime=endTime,sparse=False, modes=modes,
                                    timeStep=timeStep, initValue= initValue)

    if("outputEqui" in modes):
        experiment_path = name
        C0=readAttribute(experiment_path,["C0"])["C0"]
        rescaleFactor = readAttribute(experiment_path,["rescaleFactor"])["rescaleFactor"]
        output = results[modes.index("outputEqui")]
        output = np.reshape(output,(len(X1),len(X2)))

        X1 = X1/(C0*rescaleFactor)
        X2 = X2/(C0*rescaleFactor)

        X1log = np.log(X1)
        X2log = np.log(X2)

        experiment_path = nameFig
        colorDiagram(X1,X2,output,"Initial concentration of X1","Initial concentration of X2","Equilibrium concentration of the output",figname=os.path.join(experiment_path, "neuralDiagramm.png"),equiPotential=False)
        neuronPlot(X1,X2,output,figname=os.path.join(experiment_path, "activationX1.png"),figname2=os.path.join(experiment_path, "activationX2.png"),doShow=False)
        df = pandas.DataFrame(X1)
        df.to_csv(os.path.join(experiment_path,"inputX1.csv"))
        df2 = pandas.DataFrame(X2)
        df2.to_csv(os.path.join(experiment_path,"inputX2.csv"))

        if x1val is None:
            indicatorValue = indicator(X1log,X1,output[:,0],indexSplit)
        else:
            indicatorValue = indicator(X2log,X2,output[0,:],indexSplit,sens=-1,Xactiv= x1val/(C0*rescaleFactor))
    else:
        raise Exception("outputEqui should be in outputModes")
    return indicatorValue




if __name__ == '__main__':

    name = "templateModel/modelSearch"
    #generating the template:
    masks = np.array([np.array([[1,-1]])])
    outputList = None #"all" or None
    complexity="simple"
    useEndo = False  # if we want to use the complicated endo model
    useProtectionOnActivator = False
    useEndoOnInputs = False
    useEndoOnOutputs = True
    useDerivativeLeak = True
    if useEndo:
        generateTemplateNeuralNetwork(name,masks,complexity=complexity,useProtectionOnActivator=useProtectionOnActivator,
                                      useEndoOnOutputs=useEndoOnOutputs,useEndoOnInputs=useEndoOnInputs)
    else:
        generateTemplateNeuralNetwork(name,masks,complexity=complexity,endoConstants=None,useProtectionOnActivator=useProtectionOnActivator,
                                  useEndoOnInputs=useEndoOnInputs,useEndoOnOutputs=useEndoOnOutputs)

    enzyme = np.logspace(-8,-4,10)
    templateInit = np.logspace(-8,-4,10)
    indexSplit = 10

    indicatorMatrixActiv = np.zeros((enzyme.shape[0],templateInit.shape[0]))
    #first: generate indicator matrix for fixed inhibition
    x2Val = 10**(-6)
    for idx,e in enumerate(enzyme):
        for idx2,t in enumerate(templateInit):
            if not os.path.exists(os.path.join(os.path.join(name,"activation"),str(idx)+"_"+str(idx2))):
                os.makedirs(os.path.join(os.path.join(name,"activation"),str(idx)+"_"+str(idx2)))
            indicatorMatrixActiv[idx,idx2] = simulModelIndicator(name,os.path.join(os.path.join(name,"activation"),str(idx)+"_"+str(idx2)),enzymeInit = e,activInit = t,inhibInit = t, x2val = x2Val, x1val = None, indexSplit = indexSplit)
    df = pandas.DataFrame(indicatorMatrixActiv)
    df.to_csv(os.path.join(name,"indicatorMatrixActiv.csv"))
    colorDiagram(enzyme,templateInit,indicatorMatrixActiv,"Initial concentration of E","Initial concentration of template","Indicator of the model",figname=os.path.join(name, "modelIndicatorDiagrammActivation.png"),equiPotential=False)
    # second: generate indicator matrix for fixed activation
    indicatorMatrixInhib = np.zeros((enzyme.shape[0],templateInit.shape[0]))
    x1Val = 10**(-6)
    for idx,e in enumerate(enzyme):
        for idx2,t in enumerate(templateInit):
            if not os.path.exists(os.path.join(os.path.join(name,"inhibition"),str(idx)+"_"+str(idx2))):
                os.makedirs(os.path.join(os.path.join(name,"inhibition"),str(idx)+"_"+str(idx2)))
            indicatorMatrixInhib[idx,idx2] = simulModelIndicator(name,os.path.join(os.path.join(name,"inhibition"),str(idx)+"_"+str(idx2)),enzymeInit = e,activInit = t,inhibInit = t, x2val = None, x1val = x1Val, indexSplit = indexSplit)
    df2 = pandas.DataFrame(indicatorMatrixInhib)
    df2.to_csv(os.path.join(name,"indicatorMatrixInhib.csv"))
    colorDiagram(enzyme,templateInit,indicatorMatrixInhib,"Initial concentration of E","Initial concentration of template","Indicator of the model",figname=os.path.join(name, "modelIndicatorDiagramminhibition.png"),equiPotential=False)
