"""
    utilitarians for script
"""


import numpy as np
import sys,os

def getSpeciesAtLeak(nameDic, leak):
    SDic={}
    for k in nameDic.keys():
        SDic[k]=leak
    return SDic

def _getSpeciesArray(SDic,nameDic):
    sArray=np.zeros(len(SDic.keys()))
    for k in SDic.keys():
        sArray[nameDic[k]]=SDic[k]
    return sArray


def saveAttribute(experiment_name,attributesDic):
    """
        Save an experiment, for each attribute in attributeDic, save as: attribute + " has value: " + value
    :param experiment_name:
    :param attributesDic:
    :return:
    """
    result_path=os.path.join(sys.path[0],experiment_name)
    if(not os.path.exists(result_path)):
        os.mkdir(result_path)
    else:
        if(os.path.exists(os.path.join(result_path,"ExperimentDescriptif.txt"))):
            with open(os.path.join(result_path,"ExperimentDescriptif.txt"),'w') as file:
                os.rename(os.path.join(result_path,"ExperimentDescriptif.txt"),os.path.join(result_path,"ExperimentDescriptif_Old.txt"))
    with open(os.path.join(result_path,"ExperimentDescriptif.txt"),'w') as file:
        for c in attributesDic.keys():
            file.write(str(c)+" has value: "+str(attributesDic[c])+"\n")
    return result_path
def readAttribute(experiment_name,listToRead):
    """
        Read the description of the experiment
    :param experiment_name: path where it was saved using saveAttribute function
    :param listToRead: list of string from parameters to retrieve from the file
    :return:
    """
    with open(os.path.join(experiment_name,"ExperimentDescriptif.txt"),'r') as file:
        lines = file.readlines()
    dicOut={}
    for l in lines:
        if(l.split(" ")[0] in listToRead):
            dicOut[l.split(" ")[0]]=l.split(" ")[3].split("\n")[0]
    return dicOut


def findRightNumberProcessus(shapeP,num_workers):
    idxList = np.zeros(min(num_workers,shapeP)+1,dtype=np.int) # store the idx of task for each process
    reste = shapeP % min(num_workers,shapeP)

    for i in range(len(idxList)):
        idxList[i] = int(i*shapeP/min(num_workers,shapeP))
    cum = 1
    for i in range(len(idxList)-reste,len(idxList)):
        idxList[i] = idxList[i] +cum
        cum = cum+1
    return idxList

def obtainSpeciesArray(inputsArray,nameDic,leak,initializationDic,C0):
    speciesArray=np.zeros((inputsArray.shape[0], len(list(nameDic.keys()))))
    for idx,inputs in enumerate(inputsArray):
        SDic = getSpeciesAtLeak(nameDic, leak)
        for idx2,inputConcentration in enumerate(inputs):
            if "X_0_"+str(idx2) in nameDic.keys():
                SDic["X_0_"+str(idx2)]=inputConcentration
        if(initializationDic):
            for k in initializationDic.keys():
                if k in nameDic.keys(): #verify k is really used in our system
                    SDic[k]=initializationDic[k]
        species = _getSpeciesArray(SDic,nameDic) #We obtain from the dictionnary the array of concentration
        species = species/C0
        speciesArray[idx] = species
    return speciesArray

def obtainOutputDic(nameDic):
    """
        Find the output species, that is species of the last layer:
        species are sorted as: X_nbLayer_neuronPosition
    :param nameDic: name of all species of our systems.
    :return: the sorted list of species from the last layer, which might be interesting to overview.
    """
    max = 0
    for k in nameDic.keys():
        if("X"==k[0]):
            if(int(k.split("_")[1])>max):
                max = int(k.split("_")[1])
    outputDic = []
    for k in nameDic.keys():
        if "X_"+str(max) in k:
            if "X_"+str(max)+"_"+k.split("_")[2]==k: #k is really of the form of X_nbLayer_neuronPosition
                outputDic += [k]

    return np.sort(outputDic)

def obtainCopyArgs(modes,idxList,outputList,time,funcForSolver,speciesArray,KarrayA,stochio, maskA,maskComplementary,coLeak,nameDic):
    """
        Produce a list with each elements being a list of parameters used by a sub-process for integration.
        The merge is also dependent on the output modes that has been chosen.
    """
    if "outputPlot" and "outputEqui" in modes:
        outputCollector=[np.zeros((len(outputList), idxList[idx + 1] - id)) for idx, id in enumerate(idxList[:-1])]
        outputPlotsCollector=[np.zeros((len(outputList), idxList[idx + 1] - id, time.shape[0])) for idx, id in enumerate(idxList[:-1])]
        copyArgs=[[speciesArray[myId:idxList[idx+1]], time, funcForSolver,
                   (KarrayA,stochio, maskA,maskComplementary,coLeak),
                   {"mode":modes,"nameDic":nameDic,"idx":idx,"output":outputCollector[idx],
                    "outputPlot":outputPlotsCollector[idx],"outputList":outputList}] for idx, myId in enumerate(idxList[:-1])]
    elif "outputEqui" in modes:
        outputCollector=[np.zeros((len(outputList), idxList[idx + 1] - id)) for idx, id in enumerate(idxList[:-1])]
        copyArgs=[[speciesArray[myId:idxList[idx+1]], time, funcForSolver,
                   (KarrayA,stochio, maskA,maskComplementary,coLeak),
                   {"mode":modes,"nameDic":nameDic,"idx":idx,"output":outputCollector[idx],
                    "outputList":outputList}] for idx, myId in enumerate(idxList[:-1])]
    elif "outputPlot" in modes:
        outputPlotsCollector=[np.zeros((len(outputList), idxList[idx + 1] - id, time.shape[0])) for idx, id in enumerate(idxList[:-1])]
        copyArgs=[[speciesArray[myId:idxList[idx+1]], time, funcForSolver,
                   (KarrayA,stochio, maskA,maskComplementary,coLeak),
                   {"mode":modes,"nameDic":nameDic,"idx":idx,"outputPlot":outputPlotsCollector[idx],
                    "outputList":outputList}] for idx, myId in enumerate(idxList[:-1])]
    else:
        copyArgs=[[speciesArray[myId:idxList[idx+1]], time, funcForSolver,
                   (KarrayA,stochio, maskA,maskComplementary,coLeak),
                   {"mode":modes,"idx":idx}] for idx, myId in enumerate(idxList[:-1])]
    return copyArgs