"""
    utilitarians for script
"""


import numpy as np
import sys,os

def getSpeciesAtValue(nameDic, value):
    SDic={}
    for k in nameDic.keys():
        SDic[k]=value
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
    :return: dictionary with parameters as float.
    """
    with open(os.path.join(experiment_name,"ExperimentDescriptif.txt"),'r') as file:
        lines = file.readlines()
    dicOut={}
    for l in lines:
        if(l.split(" ")[0] in listToRead):
            dicOut[l.split(" ")[0]]=float(l.split(" ")[3].split("\n")[0])
    return dicOut


def findRightNumberProcessus(shapeP,num_workers):
    idxList = np.zeros(min(num_workers,shapeP)+1,dtype=np.int) # store the idx of task for each process
    reste = shapeP % min(num_workers,shapeP)

    for i in range(len(idxList)):
        idxList[i] = i*int(shapeP/min(num_workers,shapeP))
    cum = 1
    for i in range(len(idxList)-reste,len(idxList)):
        idxList[i] = idxList[i] +cum
        cum = cum+1
    return idxList

def obtainSpeciesArray(inputsArray, nameDic, initValue, initializationDic, C0):
    """
        Gives the array with the initial concentration value for all species.
    :param inputsArray: t*d array, where t is the number of test and d number of species in the first layer. Array of concentration for species in the first layer.
    :param nameDic: Dictionary, species k index is nameDic[k].
    :param initValue: value of concentration all other species should have.
    :param initializationDic: dictionary with initialization for other species than the first layer.
    :param C0: normalization value, divide all initial value.
    :return: an array of shape t*n, t: number of tests, n: number of species. The initial concentration for all species.
    """
    speciesArray=np.zeros((inputsArray.shape[0], len(list(nameDic.keys()))))
    for idx,inputs in enumerate(inputsArray):
        SDic = getSpeciesAtValue(nameDic, initValue)
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

def obtainOutputArray(nameDic):
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
            if "X_"+str(max)+"_"+k.split("_")[2]==k and k.split("_")[2].isdigit(): #k is really of the form of X_nbLayer_neuronPosition
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

def obtainCopyArgsLassie(modes,idxList,outputList,time,directory_for_network,parsedEquation,constants,coLeak,nameDic,speciesArray,lassieEx):
    """
        Produce a list with each elements being a list of parameters used by a sub-process for integration.
        The merge is also dependent on the output modes that has been chosen.
        Note that here we give for each thread a directory in which the thread will generate sub-directory for communication with LASSIE.
        :param lassieEx: the path to the executive file of lassie.
    """
    if "outputPlot" and "outputEqui" in modes:
        outputCollector=[np.zeros((len(outputList), idxList[idx + 1] - id)) for idx, id in enumerate(idxList[:-1])]
        outputPlotsCollector=[np.zeros((len(outputList), idxList[idx + 1] - id, time.shape[0])) for idx, id in enumerate(idxList[:-1])]
        copyArgs=[[speciesArray[myId:idxList[idx+1]], time, os.path.join(directory_for_network,str(idx)),parsedEquation,constants,coLeak,nameDic,lassieEx,
                   {"mode":modes,"nameDic":nameDic,"idx":idx,"output":outputCollector[idx],
                    "outputPlot":outputPlotsCollector[idx],"outputList":outputList}] for idx, myId in enumerate(idxList[:-1])]
    elif "outputEqui" in modes:
        outputCollector=[np.zeros((len(outputList), idxList[idx + 1] - id)) for idx, id in enumerate(idxList[:-1])]
        copyArgs=[[speciesArray[myId:idxList[idx+1]], time, os.path.join(directory_for_network,str(idx)),parsedEquation,constants,coLeak,nameDic,lassieEx,
                   {"mode":modes,"nameDic":nameDic,"idx":idx,"output":outputCollector[idx],
                    "outputList":outputList}] for idx, myId in enumerate(idxList[:-1])]
    elif "outputPlot" in modes:
        outputPlotsCollector=[np.zeros((len(outputList), idxList[idx + 1] - id, time.shape[0])) for idx, id in enumerate(idxList[:-1])]
        copyArgs=[[speciesArray[myId:idxList[idx+1]], time, os.path.join(directory_for_network,str(idx)),parsedEquation,constants,coLeak,nameDic,lassieEx,
                   {"mode":modes,"nameDic":nameDic,"idx":idx,"outputPlot":outputPlotsCollector[idx],
                    "outputList":outputList}] for idx, myId in enumerate(idxList[:-1])]
    else:
        copyArgs=[[speciesArray[myId:idxList[idx+1]], time, os.path.join(directory_for_network,str(idx)),parsedEquation,constants,coLeak,nameDic,lassieEx,
                   {"mode":modes,"idx":idx}] for idx, myId in enumerate(idxList[:-1])]
    return copyArgs

def obtainCopyArgsFixedPoint(idxList,modes,speciesArray,nameDic,outputList,masks,constants,chemicalModel="templateModel"):
    """
       Produce a list with each elements being a list of parameters used by a sub-process for computing equilibrium.
       The merge is also dependent on the output modes that has been chosen.
   """
    # constantArray,masks,X0,chemicalModel="templateModel",verbose=True

    ###Let us produce the constantArray:
    verbose = False
    if "verbose" in modes:
        verbose = True

    # for now we use the same constants for all equations...
    k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdI,kdT,TA,TI,E0 = constants
    k1 = [np.zeros(m.shape)+k1 for m in masks]
    k1n = [np.zeros(m.shape)+k1n for m in masks]
    k2 = [np.zeros(m.shape)+k2 for m in masks]
    k3 = [np.zeros(m.shape)+k3 for m in masks]
    k3n = [np.zeros(m.shape)+k3n for m in masks]
    k4 = [np.zeros(m.shape)+k4 for m in masks]
    k5 = [np.zeros(m.shape)+k5 for m in masks]
    k5n = [np.zeros(m.shape)+k5n for m in masks]
    k6 = [np.zeros(m.shape)+k6 for m in masks]
    kdT = [np.zeros(m.shape)+kdT for m in masks]
    kdI = [np.zeros(m.shape)+kdI for m in masks]
    TA0 = [np.where(m>0,TA,0) for m in masks]
    TI0 = [np.where(m<0,TI,0) for m in masks]

    constantArray = [k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdI,kdT,TA0,TI0,E0 ]

    if "outputEqui" in modes:
        outputCollector=[np.zeros((len(outputList), idxList[idx + 1] - id)) for idx, id in enumerate(idxList[:-1])]
        copyArgs=[[ speciesArray[myId:idxList[idx+1]],(constantArray,masks,chemicalModel,verbose),
                   {"mode":modes,"nameDic":nameDic,"idx":idx,"output":outputCollector[idx],
                    "outputList":outputList}] for idx, myId in enumerate(idxList[:-1])]
    else:
        copyArgs=[[ speciesArray[myId:idxList[idx+1]],(constantArray,masks,chemicalModel,verbose),
                   {"mode":modes,"idx":idx}] for idx, myId in enumerate(idxList[:-1])]
    return copyArgs




def _removeLastLayerFromDic(outputArray, nameDic):
    """
        Remove from name dic the species from the last layer, which we obtain with outputArray
    :param outputArray:
    :param nameDic:
    :return:
    """
    nameDic2 = {}
    layer = outputArray[0].split("_")[0] + "_" + outputArray[0].split("_")[1]
    for k in nameDic.keys():
        if not layer in k:
            nameDic2[k]=nameDic[k]
    return nameDic2

def rescaleInputConcentration(speciesArray=None,networkMask=None,nameDic=None,rescaleFactor=None):
    """
        This function enable to rescale concentrations when using more input species, or multi layers.
        Such rescale is crucial to obtain output of similar values despite having more species as inputs.
        The key point is that these species use shared enzymes of fixed concentration.
        As the number of species grows, the competition increases leading to very long reaction.
        In order to reduce computation time it is of importance to keep these reactions length in a reasonable amount of time.
        To do so, we diminish only the concentration of input species.
        One should either gave the network mask or dictionary with all the species names.
        Initial version:
            A better heuristic should be derived.
            For now we simply divide by the total number of nodes in the network.
    :param speciesArray: t*n-array, the concentration of every species. t: number of test, n: number of species
                         if None we only give the rescale factor.
    :param networkMask: optional, 2d-array or list of 2d-array with value in {0,1,-1}: mask of the network. (Same as considered by generateNeuralNetworkfunction)
                        using this is much faster.
    :param nameDic: optional, dictionary with name of species of the layers.
    :param rescaleFactor: optional,float, if not None then this value replace the value found by the default heuristic (that is the number of nodes)
    :return: modified speciesArray,rescaleFactor
    """
    firstLayer = []
    if networkMask is None and nameDic is None:
        raise Exception("please provide at least one mask")
    elif networkMask is not None:
        if not type(networkMask)==list:
            nbrNodes = networkMask.shape[0]*networkMask.shape[1] # IN FACT NBR OF TEMPLATE HERE....
        else:
            nbrNodes = np.sum([np.sum(np.where(m<0,1,0) + np.where(m>0,1,0)) for m in networkMask]) # IN FACT NBR OF TEMPLATE HERE....
    elif nameDic is not None:
        outputArray = obtainOutputArray(nameDic)
        lastLayerIndex = int(outputArray[0].split("_")[1])
        nbrNodes=len(outputArray)
        nameDic2 = _removeLastLayerFromDic(outputArray,nameDic)
        for i in range(lastLayerIndex): #while we are not at the first layer, we remove the last layer and add its nodes.
            outputArray = obtainOutputArray(nameDic2)
            nbrNodes+=len(outputArray)
            nameDic2 = _removeLastLayerFromDic(outputArray,nameDic2)
        firstLayer = outputArray
    if rescaleFactor is None:
        rescaleFactor=nbrNodes
    if not (speciesArray is None):
        for k in firstLayer:
            speciesArray[:,nameDic[k]] = speciesArray[:,nameDic[k]]/rescaleFactor
        print("Rescaled input species concentration by "+str(rescaleFactor)+" for "+str(firstLayer))
        return speciesArray,rescaleFactor
    return None,rescaleFactor

def obtainTemplateArray(nameDic = None,layer = None,masks = None,activ = None):
    """
          Gives back an array with the template name.
          Template are of the form: Templ_X_layerinput_nodeinput_X_layeroutput_nodeinput

          This function find the existing template from nameDic if they are provided.
          Otherwise it gives back the template with respect to the masks and the previous notations.

          One can also aks for template of a specific layer, or for activation or inhibition template.
          In these case a mask (list of 2d-array of integer) describing the network, in a neural network like way, must be provided.
                Mask axis are: 1st axis: layers, 2nd axis: output neurons, 3rd axis: input neurons.
                    So if mask[i][j,k]==1, neuron k from layer i-1 activates neurons j from layer i
    :param nameDic: dictionary with all species name as keys
    :param layer: integer or list of integer, indicate the layer in which the template shall be taken. We consider layer of inputs
    :param masks: (list of 2d-array of [-1,0,1]) describing the network, in a neural network like way.
    :param activ: if False, consider inhibiting template. If True, consider activator template.
    :return: Array with the template names.
    """
    templateArray=[]
    if nameDic is not None:
        if layer is not None or activ is not None:
            try:
                assert masks is not None
            except:
                raise Exception("Please provide a mask")
            if layer is not None:
                layerList = np.array(layer)
            else:
                layerList = np.arange(0,len(masks),1)
            if activ == None:
                keep = [-1,1]
            elif activ:
                keep = [1]
            else:
                keep = [-1]
            for k in nameDic.keys():
                if "Templ_" == k.split("_")[0]:
                    supposedLayerOutput= k.split("_")[2]
                    supposedNodeOutput= k.split("_")[3]
                    supposedLayerInput = k.split("_")[5]
                    supposedNodeInput  = k.split("_")[6]
                    # Case of inhibitor, we must remove the d:
                    if "d" in supposedNodeInput and -1 in keep:
                        supposedNodeInput = supposedNodeInput.split("d")[0]
                        if supposedNodeInput.isdigit() and supposedLayerInput.isdigit() and supposedNodeOutput.isdigit() and supposedLayerOutput.isdigit():
                            if int(supposedLayerInput) in layerList and masks[supposedLayerInput][supposedNodeOutput,supposedNodeInput] == -1:
                                if "Templ_X_"+supposedLayerOutput+"_"+supposedNodeOutput+"_X_"+supposedLayerInput+"_"+supposedNodeInput+"d" == k:
                                    templateArray +=[k]
                    else:
                        if 1 in keep and supposedNodeInput.isdigit() and supposedLayerInput.isdigit() and supposedNodeOutput.isdigit() and supposedLayerOutput.isdigit():
                            if int(supposedLayerInput) in layerList and masks[supposedLayerInput][supposedNodeOutput,supposedNodeInput] == 1:
                                if "Templ_X_"+supposedLayerOutput+"_"+supposedNodeOutput+"_X_"+supposedLayerInput+"_"+supposedNodeInput+"d" == k:
                                    templateArray +=[k]
        else:
            for k in nameDic.keys():
                if "Templ_" == k.split("_")[0]:
                    supposedLayerOutput= k.split("_")[2]
                    supposedNodeOutput= k.split("_")[3]
                    supposedLayerInput = k.split("_")[5]
                    supposedNodeInput  = k.split("_")[6]
                    if "d" in supposedNodeInput:
                        supposedNodeInput = supposedNodeInput.split("d")[0]
                        if supposedNodeInput.isdigit() and supposedLayerInput.isdigit() and supposedNodeOutput.isdigit() and supposedLayerOutput.isdigit():
                            if "Templ_X_"+supposedLayerOutput+"_"+supposedNodeOutput+"_X_"+supposedLayerInput+"_"+supposedNodeInput+"d"== k:
                                    templateArray +=[k]
                    else:
                        if supposedNodeInput.isdigit() and supposedLayerInput.isdigit() and supposedNodeOutput.isdigit() and supposedLayerOutput.isdigit():
                            if "Templ_X_"+supposedLayerOutput+"_"+supposedNodeOutput+"_X_"+supposedLayerInput+"_"+supposedNodeInput== k:
                                templateArray +=[k]
    else:
        try:
            assert masks is not None
        except:
            raise Exception("Please provide a mask as no nameDic was given")
        if layer is not None:
            layerList = np.array(layer)
        else:
            layerList = np.arange(0,len(masks),1)
        if activ == None:
            keep = [-1,1]
        elif activ:
            keep = [1]
        else:
            keep = [-1]

        for i in range(0,len(masks)):
            for output in range(masks[i].shape[0]):
                for input in range(masks[i].shape[1]):
                    if masks[i][output,input] in keep and i in layerList:
                        if masks[i][output,input]>0:
                            templateArray+=["Templ_X_"+str(i+1)+"_"+str(output)+"_X_"+str(i)+"_"+str(input)]
                        else:
                            templateArray+=["Templ_X_"+str(i+1)+"_"+str(output)+"_X_"+str(i)+"_"+str(input)+"d"]
    return templateArray