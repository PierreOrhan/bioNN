"""
    Script for obtaining dynamic of small neural networks
    Here we proceed to the rendering of the function: output(X1,X2) for X1 activator and X2 inhibitor (killer template)
    Equations are solved in parallel on the CPU

"""
import numpy as np
import os
from simulOfBioNN.parseUtils.parser import generateNeuralNetwork
from simulOfBioNN.simulNN.simulator import executeSimulation
from simulOfBioNN.odeUtils.systemEquation import f
from simulOfBioNN.odeUtils.utils import readAttribute
from simulOfBioNN.plotUtils.adaptivePlotUtils import colorDiagram,neuronPlot

name = "templateModel_activationSimulation_0"
endTime = 1000
masks = np.array([np.array([[1,-1,],[1,-1]]),np.array([[1,-1]])])
modes = ["verbose","outputEqui"]
generateNeuralNetwork(name,masks)
FULL = True

#generate the first layer concentration:
if(FULL):
    X1=np.arange(10**(-7),10**(-5),10**(-7))
    X1=np.concatenate((X1,np.arange( 10 ** (-5), 0.8*10 ** (-4), 10 ** (-6))))
    X2=np.arange( 10 ** (-8), 10 ** (-6), 10 ** (-8))
else:
    X1=np.array([10**(-6),10**(-5),10**(-4),10**(-2)])
    X2=np.array([10**(-7),10**(-6),5*10**(-6),10**(-5)])
# generate concentration for all different experiments:
x_test=[]
for x1 in X1:
    for x2 in X2:
        x_test+=[[x1,x2]]
x_test = np.array(x_test)

#generate other layer concentration, for initialization:
leak=10**(-13)
layerInit = 10**(-8)
enzymeInit = 10**(-6)
initialization_dic={}
for layer in range(1,len(masks)): ## the first layer need not to be initiliazed
    for node in range(masks[layer].shape[0]):
        initialization_dic["X_"+str(layer)+"_"+str(node)] = layerInit
initialization_dic["E"] = enzymeInit
initialization_dic["E2"] = enzymeInit

results = executeSimulation(f, name, x_test, initialization_dic,["X_1_0"], leak, endTime=endTime,sparse=False, modes=modes)

if("outputPlot" in modes):
    import matplotlib.pyplot as plt
    shapeP = len(X1)*len(X2)
    nameDic = results[-1]
    outArrayFullPlot = results[modes.index("output")]
    toObserve=range(0,shapeP,max(int(shapeP/100),1))
    time = np.arange(0,endTime,0.1)
    for t in toObserve:
        plt.figure()
        for idx,k in enumerate(nameDic.keys()):
            if idx in [nameDic["X_1_0"]]:
                plt.plot(time,outArrayFullPlot[t,:,idx],label=k)
                print("the observed min value is "+str(np.min(outArrayFullPlot[t,:,idx]))+" for "+k)
        plt.legend()
        plt.show()

if("outputEqui" in modes):
    experiment_path = name
    C0=readAttribute(experiment_path,["C0"])["C0"]
    output = results[modes.index("output")]
    colorDiagram(X1,X2,output,"Initial concentration of X1","Initial concentration of X2","Equilibrium concentration of the output",figname=os.path.join(experiment_path, "neuralDiagramm.png"),equiPotential=False)
    neuronPlot(X1/C0,X2/C0,output,figname=os.path.join(experiment_path, "activationX1.png"),figname2=os.path.join(experiment_path, "activationX2.png"))