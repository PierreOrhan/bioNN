import numpy as np
import pandas
from simulOfBioNN.plotUtils.adaptivePlotUtils import colorDiagram

name="BetterView"
dfoutput=pandas.read_csv("equilibrium"+name+".csv")
output=dfoutput.values[:,1:]
dfoutput=pandas.read_csv("Y1"+name+".csv")
outputY1=dfoutput.values[:,1:]
dfoutput=pandas.read_csv("Y2"+name+".csv")
outputY2=dfoutput.values[:,1:]


dfA2onA1=pandas.read_csv("a2on1"+name+".csv")
A2onA1=np.reshape(dfA2onA1.values[:,1:],(dfA2onA1.values[:,1:].shape[0]))
dfC=pandas.read_csv("c"+name+".csv")
C=np.reshape(dfC.values[:,1:],(dfC.values[:,1:].shape[0]))

lineToKeep=colorDiagram(A2onA1,C,output,"Ratio A2/A1","Concentration of cooperative species, arbitrary unit",'Ratio: Y2/Y1',"diagramm"+name+".png")
colorDiagram(A2onA1,C,outputY1,"Ratio A2/A1","Concentration of cooperative species, arbitrary unit",'output concentration: Y1',"diagrammY1"+name+".png",equiPotential=False,lineToKeep=lineToKeep)
colorDiagram(A2onA1,C,outputY2,"Ratio A2/A1","Concentration of cooperative species, arbitrary unit",'output concentration: Y2',"diagrammY2"+name+".png",equiPotential=False,lineToKeep=lineToKeep)

# def getSpectiesAtLeak(nameDic,leak):
#     SDic={}
#     for k in nameDic.keys():
#         SDic[k]=leak
#     return SDic
# def getSpeciesArray(SDic,nameDic):
#     sArray=np.zeros(len(SDic.keys()))
#     for k in SDic.keys():
#         sArray[nameDic[k]]=SDic[k]
#     return sArray
#
# generateEcoEvoNetwork("ecoEvoSimul")
# parsedEquation,constants,nameDic=read_file("ecoEvoSimul/equations.txt","ecoEvoSimul/constants.txt")
# KarrayA,stochio,maskA,maskComplementary=parse(parsedEquation,constants)
# KarrayA,T0,C0,kDic=setToUnits(constants,KarrayA,stochio)
#
#
# leak=10**(-13)
#
# #constant through the experiment
# e1 = 10**(-8)
# e2 = 10**(-8)
# y1 = 10**(-8)
# y2 = 10**(-8)
#
# a10 = 10**(-6)
#
# A2onA1 = [1]
# C = [10**(-9),10**(-6),10**(-5)]
#
# print("initialisation constant: time:"+str(T0)+" concentration:"+str(C0))
# output=np.zeros((len(A2onA1),len(C)))
# speciesArray=np.zeros((len(A2onA1), len(C),len(list(nameDic.keys()))))
# for idx,rapport in enumerate(A2onA1):
#     line=[]
#     for idxc,c in enumerate(C):
#         experiment_name="ecoEvoSimul/"+str(idx)+"_"+str(idxc)
#         SDic = getSpectiesAtLeak(nameDic,leak)
#         SDic["A1"]=a10
#         SDic["A2"]=rapport*a10
#         SDic["Y1"]=y1
#         SDic["Y2"]=y2
#         SDic["E"]=e1
#         SDic["E2"]=e2
#         SDic["C1"]=c
#         SDic["C2"]=c
#         species = getSpeciesArray(SDic,nameDic)
#         species = species/C0
#         speciesArray[idx,idxc]=species
#
# time=np.arange(0,100000,1)
# for idx,rapport in enumerate(A2onA1):
#     for idxc,c in enumerate(C):
#         t=tm()
#         X,dic=odeint(f,speciesArray[idx,idxc],time,args=(KarrayA,stochio,maskA,maskComplementary),full_output=True)
#         print(str(tm()-t)+" is the time")
#         plot(time,experiment_name,nameDic,X,wishToDisp=["Y1","Y2"],displaySeparate=True,displayOther=False)