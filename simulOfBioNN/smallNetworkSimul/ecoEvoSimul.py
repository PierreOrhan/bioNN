import numpy as np
from simulOfBioNN.odeUtils.systemEquation import f,setToUnits
from simulOfBioNN.parseUtils.parser import read_file,parse
from simulOfBioNN.parseUtils.ecoEvoGenerator import generateEcoEvoNetwork
from scipy.integrate import odeint
from simulOfBioNN.plotUtils.adaptivePlotUtils import colorDiagram
from simulOfBioNN.odeUtils.utils import saveAttribute,getSpeciesArray,getSpeciesAtLeak
from time import time as tm
import pandas
import os
import multiprocessing


def runForMultiprocess(X):
    speciesArray,time,KarrayA,stochio, maskA, maskComplementary, nameDic,myidx, coLeak,outputY1,outputY2  = X
    #outfullPlot = np.zeros((speciesArray.shape[0],time.shape[0],speciesArray.shape[1]))
    for idx,species in enumerate(speciesArray):
        t0=tm()
        derivLeak = coLeak
        X,dic=odeint(f,species,time,args=(KarrayA,stochio,maskA,maskComplementary,derivLeak),full_output=True,rtol=10**(-6),atol=10**(-12))
        print(str(idx)+" on "+str(len(speciesArray))+" for "+str(myidx)+" in "+str(tm()-t0))

        outputY1[idx,:]=X[:,nameDic["Y1"]]
        outputY2[idx,:]=X[:,nameDic["Y2"]]
        #outfullPlot[idx,:]=X
    return (outputY1,outputY2)#,outfullPlot)



generateEcoEvoNetwork("ecoEvoSimul")
parsedEquation,constants,nameDic=read_file("ecoEvoSimul/equations.txt","ecoEvoSimul/constants.txt")
KarrayA,stochio,maskA,maskComplementary=parse(parsedEquation,constants)
KarrayA,T0,C0,kDic=setToUnits(constants,KarrayA,stochio)

leak=10**(-13)

#constant through the experiment
e1 = 10**(-4)
e2 = 10**(-4)
y1 = 10**(-8)
y2 = 10**(-8)

a10 = 10**(-5)

A2onA1 = np.arange(1, 10, 0.1)
C = np.logspace(-9,-4, num = 50)

print("initialisation constant: time:"+str(T0)+" concentration:"+str(C0))

speciesArray=np.zeros((len(A2onA1), len(C),len(list(nameDic.keys()))))
for idx,rapport in enumerate(A2onA1):
    line=[]
    for idxc,c in enumerate(C):
        experiment_name="ecoEvoSimul/"+str(idx)+"_"+str(idxc)
        SDic = getSpeciesAtLeak(nameDic, leak)
        SDic["A1"]=a10
        SDic["A2"]=rapport*a10
        SDic["Y1"]=y1
        SDic["Y2"]=y2
        SDic["E"]=e1
        SDic["E2"]=e2
        SDic["C1"]=c
        SDic["C2"]=c
        species = getSpeciesArray(SDic,nameDic)
        species = species/C0
        speciesArray[idx,idxc]=species

endTime=1000
timeStep=10
time=np.arange(0,endTime,timeStep)
coLeak = leak/C0

name="ecoEvoSimul/10foisplusA1and100enzyme2"
##SAVE EXPERIMENT:
attributesDic={"e1":e1,"e2":e2,"y1":y1,"y2":y2,"a10":a10,"leak":leak,"T0":T0,"C0":C0,"lastTime":endTime,"time step":timeStep}
experiment_path=saveAttribute(name, attributesDic)


shapeP=speciesArray.shape[0]*speciesArray.shape[1]
speciesArray2=np.reshape(speciesArray,(shapeP,speciesArray.shape[2]))
#let us assign the right number of task in each process
num_workers = multiprocessing.cpu_count()-2
idxList = np.zeros(min(num_workers,shapeP),dtype=np.int) # store the idx of task for each process
reste = shapeP % min(num_workers,shapeP)

for i in range(len(idxList)):
    idxList[i] = int(i*shapeP/min(num_workers,shapeP))
cum = 1
for i in range(len(idxList)-reste,len(idxList)):
    idxList[i] = idxList[i] +cum
    cum = cum+1

outputArrayY1=np.zeros((speciesArray.shape[0]*speciesArray.shape[1],time.shape[0]))
outputArrayY2=np.zeros((speciesArray.shape[0]*speciesArray.shape[1],time.shape[0]))
#outArrayFullPlot = np.zeros((speciesArray.shape[0]*speciesArray.shape[1],time.shape[0],len(list(nameDic.keys()))))


copyArgs=[[speciesArray2[id:idxList[idx+1]],time,KarrayA, stochio, maskA, maskComplementary,nameDic,idx, coLeak ,
           np.zeros((idxList[idx+1]-id,time.shape[0])),np.zeros((idxList[idx+1]-id,time.shape[0]))] for idx,id in enumerate(idxList[:-1])]

with multiprocessing.Pool(processes= len(idxList[:-1])) as pool:
    myoutputs = pool.map(runForMultiprocess,copyArgs)
print("Finished computing, closing pool")
pool.close()
pool.join()
for idx,m in enumerate(myoutputs):
    outputArrayY1[idxList[idx]:idxList[idx+1]] = m[0]
    outputArrayY2[idxList[idx]:idxList[idx+1]] = m[1]
    #outArrayFullPlot[idxList[idx]:idxList[idx+1]] = m[2]

outputShape=(len(A2onA1),len(C))
outputY1 = np.reshape(outputArrayY1[:,-1],outputShape)
outputY2 = np.reshape(outputArrayY2[:,-1],outputShape)



# Let us save our result:
for p in ["Y1equi.csv","Y2equi.csv","Y1.csv","Y2.csv","a2on1.csv","c.csv"]:
    if(os._exists(os.path.join(experiment_path, p))):
        print("Allready exists: renaming older")
        os.rename(os.path.join(experiment_path,p),os.path.join(experiment_path,p.split(".")[0]+"Old."+p.split(".")[1]))
df=pandas.DataFrame(outputArrayY1)
df.to_csv(os.path.join(experiment_path,"Y1.csv"))
df=pandas.DataFrame(outputArrayY2)
df.to_csv(os.path.join(experiment_path,"Y2.csv"))
df=pandas.DataFrame(outputY1)
df.to_csv(os.path.join(experiment_path,"Y1equi.csv"))
df=pandas.DataFrame(outputY2)
df.to_csv(os.path.join(experiment_path,"Y2equi.csv"))
df2=pandas.DataFrame(A2onA1)
df2.to_csv(os.path.join(experiment_path,"a2on1.csv"))
df2=pandas.DataFrame(C)
df2.to_csv(os.path.join(experiment_path,"c.csv"))


colorDiagram(A2onA1,C,outputY1/outputY2,"Ratio A2/A1","Concentration of cooperative species, arbitrary unit",'Ratio: Y2/Y1',os.path.join(experiment_path,"diagrammRatio.png"),equiPotential=False)
colorDiagram(A2onA1,C,outputY1,"Ratio A2/A1","Concentration of cooperative species, arbitrary unit",'output concentration: Y1',os.path.join(experiment_path,"diagrammY1.png"),equiPotential=False)
colorDiagram(A2onA1,C,outputY2,"Ratio A2/A1","Concentration of cooperative species, arbitrary unit",'output concentration: Y2',os.path.join(experiment_path,"diagrammY2.png"),equiPotential=False)

