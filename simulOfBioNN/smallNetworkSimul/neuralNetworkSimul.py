"""
    Script for obtaining dynamic of small neural network
    Here we proceed to the rendering of the function: output(X1,X2) for X1 activator and X2 inhibitor (killer template)
    Equations are solved in parallel on the CPU
"""



import numpy as np
from simulOfBioNN.parseUtils.parser import generateNeuralNetwork,read_file,parse
from simulOfBioNN.odeUtils.systemEquation import f, setToUnits, obtainJacobianMasks
from simulOfBioNN.odeUtils.utils import _getSpeciesArray,getSpeciesAtValue,saveAttribute
from simulOfBioNN.plotUtils.adaptivePlotUtils import neuronPlot,colorDiagram
from scipy.integrate import odeint
import pandas
from time import time as tm
import multiprocessing
"""
    Note on the threading library in python: due to the GIL lock, only the multiprocessing enable speed-up by launching new python environment for every process
"""
import os

def runForMultiprocess(X):
    speciesArray,time,KarrayA,stochio, maskA, maskComplementary, nameDic,myidx,output,outputPlot,falseResult,jacobKarrayA,jacobStochio,jacobMaskA,jacobMaskComplementary, coLeak  = X
    if(VISUALIZE_FULL):
        outfullPlot = np.zeros((speciesArray.shape[0],time.shape[0],speciesArray.shape[1]))
    for idx,species in enumerate(speciesArray):
        t0=tm()
        hmax=0.001
        derivLeak = coLeak
        X,dic=odeint(f,species,time,args=(KarrayA,stochio,maskA,maskComplementary,derivLeak),full_output=True,rtol=10**(-6),atol=10**(-12))
        print(str(idx)+" on "+str(len(speciesArray))+" for "+str(myidx)+" in "+str(tm()-t0))
        ##We make sure that the equilibrium is reached. we define it as a max variation of 1 percent over last 10% of total time steps (seconds)
        if(np.max(np.abs(X[max(0,len(X)-int(0.1*time.shape[0])),nameDic["X_1_0"]]-X[-1,nameDic["X_1_0"]]))>abs(X[-1,nameDic["X_1_0"]])*0.01
           or np.max(np.abs(X[max(0,len(X)-int(0.1*time.shape[0])),nameDic["X_0_0"]]-X[-1,nameDic["X_0_0"]]))>abs(X[-1,nameDic["X_0_0"]])*0.01):
            falseResult[idx]=1
        output[idx]=X[-1,nameDic["X_2_0"]]
        outputPlot[idx,:]=X[:,nameDic["X_2_0"]]
        if(VISUALIZE_FULL):
            outfullPlot[idx,:]=X
    if( VISUALIZE_FULL):
        return (output,outputPlot,falseResult,outfullPlot)
    return (output,outputPlot,falseResult)

name="neuronSimul11"
VISUALIZE_FULL = False
MODE = "FULL" #If we want to test a small number or parameters (use anything different than "FULL") or large (use "FULL")

masks=np.array([np.array([[1,-1,],[1,-1]]),np.array([[1,-1]])])
generateNeuralNetwork(name,masks)
parsedEquation,constants,nameDic=read_file(name+"/equations.txt",name+"/constants.txt")
KarrayA,stochio,maskA,maskComplementary = parse(parsedEquation,constants)

KarrayA,T0,C0,kDic=setToUnits(constants,KarrayA,stochio)

jacobKarrayA,jacobStochio,jacobMaskA,jacobMaskComplementary = obtainJacobianMasks(KarrayA,stochio,maskA)


leak=10**(-13)

#constant through the experiment:
e1 = 10**(-6)
e2 = 10**(-6)

y1 = 10**(-8)

if(MODE == "FULL"):
    X1=np.arange(10**(-7),10**(-5),10**(-7))
    X1=np.concatenate((X1,np.arange( 10 ** (-5), 0.8*10 ** (-4), 10 ** (-6))))
    X2=np.arange( 10 ** (-8), 10 ** (-6), 10 ** (-8))
else:
    X1=np.array([10**(-6),10**(-5),10**(-4),10**(-2)])
    X2=np.array([10**(-7),10**(-6),5*10**(-6),10**(-5)])

# X1=np.array([10**(-5)])
# X2=np.array([10**(-7)])

print("initialisation constant: time:"+str(T0)+" concentration:"+str(C0))

output=np.zeros((len(X1), len(X2)))
speciesArray=np.zeros((len(X1), len(X2),len(list(nameDic.keys()))))
for idx,x1 in enumerate(X1):
    for idxc,x2 in enumerate(X2):
        experiment_name=name+str(idx)+"_"+str(idxc)
        SDic = getSpeciesAtValue(nameDic, leak)
        SDic["X_0_0"]=x1
        SDic["X_0_1"]=x2
        SDic["X_1_0"]=y1
        # SDic["X_0_2"]=x1
        # SDic["X_0_3"]=x2
        SDic["X_1_1"]=y1
        SDic["X_2_0"]=y1
        SDic["E"]=e1
        SDic["E2"]=e2
        species = _getSpeciesArray(SDic,nameDic)
        species = species/C0
        speciesArray[idx,idxc]=species


endTime=10000
timeStep=0.1
time=np.arange(0,endTime,timeStep)
coLeak = leak/C0

##SAVE EXPERIMENT:
attributesDic={"e1":e1,"e2":e2,"y1":y1,"leak":leak,"T0":T0,"C0":C0,"lastTime":endTime,"time step":timeStep}
experiment_path=saveAttribute(name, attributesDic)


shapeP=speciesArray.shape[0]*speciesArray.shape[1]
speciesArray2=np.reshape(speciesArray,(shapeP,speciesArray.shape[2]))
#let us assign the right number of task in each process
num_workers = multiprocessing.cpu_count()-1
idxList = np.zeros(min(num_workers,shapeP),dtype=np.int) # store the idx of task for each process
reste = shapeP % min(num_workers,shapeP)

for i in range(len(idxList)):
    idxList[i] = int(i*shapeP/min(num_workers,shapeP))
cum = 1
for i in range(len(idxList)-reste,len(idxList)):
    idxList[i] = idxList[i] +cum
    cum = cum+1
#idxList=np.arange(0,shapeP+int(shapeP/min(num_workers,shapeP)),int(shapeP/min(num_workers,shapeP)))

outputCollector=[np.zeros(idxList[idx+1]-id) for idx,id in enumerate(idxList[:-1])]
outputPlotsCollector=[np.zeros((idxList[idx+1]-id,time.shape[0])) for idx,id in enumerate(idxList[:-1])]
falseResultsCollector=[np.zeros(idxList[idx+1]-id) for idx,id in enumerate(idxList[:-1])]


outputArray=np.zeros(speciesArray.shape[0]*speciesArray.shape[1])
outputArrayPlot=np.zeros((speciesArray.shape[0]*speciesArray.shape[1],time.shape[0]))
falseResult=np.zeros(speciesArray.shape[0]*speciesArray.shape[1])
if(VISUALIZE_FULL):
    outArrayFullPlot = np.zeros((speciesArray.shape[0]*speciesArray.shape[1],time.shape[0],len(list(nameDic.keys()))))

copyArgs=[[speciesArray2[id:idxList[idx+1]],time,KarrayA,
           stochio, maskA,maskComplementary,
           nameDic,idx,outputCollector[idx],outputPlotsCollector[idx],falseResultsCollector[idx],
           jacobKarrayA,jacobStochio, jacobMaskA, jacobMaskComplementary, coLeak] for idx,id in enumerate(idxList[:-1])]



with multiprocessing.Pool(processes= len(idxList[:-1])) as pool:
    myoutputs = pool.map(runForMultiprocess,copyArgs)
print("Finished computing, closing pool")
pool.close()
pool.join()
for idx,m in enumerate(myoutputs):
    outputArray[idxList[idx]:idxList[idx+1]] = m[0]
    outputArrayPlot[idxList[idx]:idxList[idx+1]] = m[1]
    falseResult[idxList[idx]:idxList[idx+1]] = m[2]
    if(VISUALIZE_FULL):
        outArrayFullPlot[idxList[idx]:idxList[idx+1]] = m[3]

output=np.reshape(outputArray,output.shape)
falseResultout=np.reshape(falseResult,output.shape)

# Let us save our result:
for p in ["false_result.csv","neural_equilibrium.csv","neural_plots.csv","neural_X1.csv","neural_X2.csv"]:
    if(os._exists(os.path.join(experiment_path, p))):
        print("Allready exists: renaming older")
        os.rename(os.path.join(experiment_path,p),os.path.join(experiment_path,p.split(".")[0]+"Old."+p.split(".")[1]))
df=pandas.DataFrame(falseResultout)
df.to_csv(os.path.join(experiment_path, "false_result.csv"))
df=pandas.DataFrame(output)
df.to_csv(os.path.join(experiment_path, "neural_equilibrium.csv"))
# df=pandas.DataFrame(outputArrayPlot)
# df.to_csv(os.path.join(experiment_path, "neural_plots.csv"))
df2=pandas.DataFrame(X1)
df2.to_csv(os.path.join(experiment_path, "neural_X1.csv"))
df2=pandas.DataFrame(X2)
df2.to_csv(os.path.join(experiment_path, "neural_X2.csv"))

# toObserve=range(0,shapeP,max(int(shapeP/100),1))
# if(not os.path.exists(os.path.join(experiment_path,"allSpecies"))):
#     os.mkdir(os.path.join(experiment_path,"allSpecies"))
# for t in toObserve:
#     df2=pandas.DataFrame(outArrayFullPlot[t])
#     df2.to_csv(os.path.join(experiment_path, "allSpecies/"+str(t)+".csv"))

if(VISUALIZE_FULL):
    import matplotlib.pyplot as plt
    toObserve=range(0,shapeP,max(int(shapeP/100),1))
    # for t in toObserve:
    #     if ((outputArrayPlot[t]*0==np.zeros(outputArrayPlot[t].shape)).all()):
    #         plt.plot(time,outputArrayPlot[t],label=str(t))
    # plt.legend()
    # plt.show()
    for t in toObserve:
        plt.figure()
        for idx,k in enumerate(nameDic.keys()):
            if idx in [nameDic["X_1_0"]]:
                plt.plot(time,outArrayFullPlot[t,:,idx],label=k)
                print("the observed min value is "+str(np.min(outArrayFullPlot[t,:,idx]))+" for "+k)
        plt.legend()
        plt.show()

# Plotting time:
colorDiagram(X1,X2,output,"Initial concentration of X1","Initial concentration of X2","Equilibrium concentration of the output",figname=os.path.join(experiment_path, "neuralDiagramm.png"),equiPotential=False)

neuronPlot(X1/C0,X2/C0,output,figname=os.path.join(experiment_path, "activationX1.png"),figname2=os.path.join(experiment_path, "activationX2.png"))


