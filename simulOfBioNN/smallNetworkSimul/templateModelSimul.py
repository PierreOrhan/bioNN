"""
    Script for obtaining dynamic of small neural networks
    Here we proceed to the rendering of the function: output(X1,X2) for X1 activator and X2 inhibitor (killer template)
    Equations are solved in parallel on the CPU

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
    print("Cactiv value is :"+str(Cactiv))
    print("Cinhib value is :"+str(CInhib))
    print("kd is :"+str(kd))

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



if __name__ == '__main__':

    name = "templateModelFirstLayer/wideLayer100"
    endTime = 10000
    timeStep = 0.1

    nbrOutputNodes = 100
    masks= _createMask(nbrOutputNodes)
    # masks = np.array([np.array([[1,-1]])])
    modes = ["verbose","outputEqui"]
    FULL = True
    outputMode = "all"
    outputList = ["X_1_0"] #"all" or None or list of output species
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


    #generate the first layer concentration:
    if(FULL):
        # X1 = np.logspace(10**(-7),10**(-5),10**(-7))
        # X1 = np.concatenate((X1,np.arange( 10 ** (-5), 0.8*10 ** (-4), 10 ** (-6))))
        X1 = np.logspace(-7,-5,10)
        X1 = np.concatenate((X1,np.logspace( -5 , -4 , 10)))
        X2 = X1
        #X2 = X1
    else:
        X1 = np.array([10**(-8),10**(-6),10**(-4)])
        X2 = np.array([10**(-6)])


    # generate concentration for all different experiments:
    x_test=[]
    for x1 in X1:
        for x2 in X2:
            inputArray = []
            for neuron in range(nbrOutputNodes):
                inputArray += [x1,x2]
            x_test+=[inputArray]
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
        results = executeODESimulation(fPythonSparse, name, x_test, initialization_dic, outputList= outputList,
                                       leak = leak, endTime=endTime, sparse=True, modes=modes,
                                       timeStep=timeStep, initValue= initValue)
    else:
        results = executeODESimulation(fPythonSparse, name, x_test, initialization_dic, outputList= outputList,
                                       leak = 0, endTime=endTime, sparse=True, modes=modes,
                                       timeStep=timeStep, initValue= initValue)

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

        fitOutput = []
        for idx,x1 in enumerate(X1):
            fitOutput += [chemf(k1, k1n, k2, k3, k3n, k4, k5, k5n, k6, kd, TA, TI, E0, x1, X2,otherConcentrationsActiv=(nbrOutputNodes-1)*x1,otherConcentrationsInhib=(nbrOutputNodes-1)*X2)]
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