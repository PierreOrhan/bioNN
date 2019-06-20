"""
    This file contains all i/o operation to create bio-chemical models.

    Format is described in the function.
    Some special parsing heuristics are all-ready given, for example to create feed-forward neural network of biochemical reactions.
"""

import numpy as np
import pandas
import os

from simulOfBioNN.parseUtils.equationWriter import killingTemplateWrite, autocatalysisWrite, templateActivationWrite, \
    templateInhibWrite, templateRealInhibitionWrite, endonucleaseWrite,endonucleaseWrite2,templateProtection


def parse(equations,kdic):
    """
        Parse the equations into the correct masks and arrays for computations
    :param equations: a d*n array where d is the number of equations (one sided : -> ) and n the number of species
            The syntax should be the following, an array with the stochiometric coefficient of the species: [alpha,-beta] for alpha*A->beta*B for example
                Negative stochiometric coeff represent product when positive represent input.
    :param
        kdic: a d array, the reaction constant for the equation
    :return:  KarrayA,stochio, maskA these different mask are to feed to the function computing the derivative. It is more efficient to compute them beforehand.

    """
    # (a,n),(a,n,n),(a,n,n),(a,n,n)
    # KarrayA,inputStochio, maskA, maskComplementary
    maskA = np.zeros((equations.shape[0],equations.shape[1],equations.shape[1]))
    KarrayA = np.zeros(equations.shape)
    stochio = np.zeros((equations.shape[0],equations.shape[1],equations.shape[1]))+1
    print("parsing")
    for idx,e in enumerate(equations):
        # for species,coeff in enumerate(e):
        #     for species2,coeff2 in enumerate(e):
        #         if(coeff2>0 and coeff!=0): #species2 must be an input, and species should be either an input or an output species
        #             maskA[idx,species,species2] = 1
        #         if(coeff2>0):
        #             stochio[idx,species,species2] = coeff2
        #     if(coeff<0):
        #         KarrayA[idx,species] = -1*coeff*kdic[idx]
        #     elif(coeff>0):
        #         KarrayA[idx,species] = -1*kdic[idx]
        m = maskA[idx,e!=0]
        m[:,e>0] = 1
        maskA[idx,e!=0] = m

        s = stochio[idx,:]
        s[:,e>0] = e[e>0]
        stochio[idx,:] = s

        KarrayA[idx,e<0] = -1*e[e<0]*kdic[idx]
        KarrayA[idx,e>0] = -1*kdic[idx]

    maskComplementary=1-maskA
    return KarrayA,stochio,maskA,maskComplementary

def read_file(pathEquations,path):
    """
        Parse two file, on for the equations, one for the constants values
    :param pathEquations: path for the equations.
    :param path: path for the constants values.
    :return:
    """
    with open(pathEquations) as file:
        equations=file.readlines()
    nameDic={}
    position=0
    for idx,e in enumerate(equations):
        equations[idx]=e.split("\n")[0]

    #Discover all the name
    for e in equations:
        inputs=e.split("-")[0].split("+")
        for c in inputs:
            if("&" in c):
                namesInvolved=c.split("&")[1].split("*")
                for n in namesInvolved:
                    if n not in nameDic.keys():
                        nameDic[n] = position
                        position+=1
            else:
                namesInvolved=c.split("*")
                for n in namesInvolved:
                    if n not in nameDic.keys():
                        nameDic[n] = position
                        position+=1
        if(e.split("-")[1]!=""):
            products=e.split("-")[1].split("+")
            for c in products:
                if("&" in c):
                    namesInvolved=c.split("&")[1].split("*")
                    for n in namesInvolved:
                        if n not in nameDic.keys():
                            nameDic[n] = position
                            position+=1
                else:
                    namesInvolved=c.split("*")
                    for n in namesInvolved:
                        if n not in nameDic.keys():
                            nameDic[n] = position
                            position+=1
    nbSeparateSpecies=len(nameDic.keys())
    parsedEquation=np.zeros((len(equations),nbSeparateSpecies))
    for idx,e in enumerate(equations):
        inputs=e.split("-")[0].split("+")
        for c in inputs:
            if("&" in c):
                namesInvolved=c.split("&")[1].split("*")
                coeff=c.split("&")[0]
                for n in namesInvolved:
                    parsedEquation[idx,nameDic[n]]=parsedEquation[idx,nameDic[n]]+float(coeff)
            else:
                namesInvolved=c.split("*")
                coeff=1
                for n in namesInvolved:
                    parsedEquation[idx,nameDic[n]]=parsedEquation[idx,nameDic[n]]+float(coeff)
        if(e.split("-")[1]!=""):
            products=e.split("-")[1].split("+")
            for c in products:
                if("&" in c):
                    namesInvolved=c.split("&")[1].split("*")
                    coeff=c.split("&")[0]
                    for n in namesInvolved:
                        parsedEquation[idx,nameDic[n]]=parsedEquation[idx,nameDic[n]]+(-1)*float(coeff)
                else:
                    namesInvolved=c.split("*")
                    coeff=-1
                    for n in namesInvolved:
                        parsedEquation[idx,nameDic[n]]=parsedEquation[idx,nameDic[n]]+float(coeff)
    with open(path) as file:
        constants=file.readlines()
    for idx,c in enumerate(constants):
        constants[idx]=float(c.split("\n")[0])
    return parsedEquation,constants,nameDic


def generateLayer(nameInputs, nameOutputs, nameE, nameE2, mask, constantValues, endoConstants, activationWriter, inhibitionWriter, complexity=None,
                  pathEquations="models/equations.txt", pathConstants="models/constants.txt", nameEndo=None, useProtectionOnActivator=False):
    """
        Generate a layer with the given activation and inhibition writer.

    :param nameInputs: string array, the inputs species name
    :param nameOutputs: string array, the output species name
    :param mask: the mask connecting input to outputs
    :param constantValues: array with values for reactions constants.
    :param endoConstants: array with values for the reaction with the endonuclease.
    :param activationWriter: function to call for the activation. Should take args as other pre-defined activation function. If provided, the complexity arg is added at the end.
    :param inhibitionWriter: function to call for the inhibition. Should take args as other pre-defined inhibition function. If provided, the complexity arg is added at the end.
    :param complexity: string, optional parameter for activation or inhibition function that require it.
    :param nameEndo: name for the endonuclease, optional. Used when inhibitionWriter == templateInhibWrite.
                     In this case if none, then the EndoNuclease is not used, and a constant coefficient is used instead.
    :param useProtectionOnActivator: default to False. If True then the activators will be link to a protection reaction so that they are not degraded.
    :return: Parse the layer into the model
    """
    if(not os.path.exists(pathEquations)):
        if(not os.path.exists(pathEquations.split("/equations.txt")[0])):
            os.makedirs(pathEquations.split("/equations.txt")[0])
        with open(pathEquations,"w") as file:
            pass
    if(not os.path.exists(pathConstants)):
        if(not os.path.exists(pathEquations.split("/constants.txt")[0])):
            os.makedirs(pathEquations.split("/constants.txt")[0])
        with open(pathConstants,"w") as file:
            pass
    assert len(constantValues)==len(mask)
    inhibitedOutputs=[]
    activatorInputs=[]
    for idx,equation in enumerate(mask):
        for idx2,speciesAction in enumerate(equation):
            if(speciesAction>0):
                if complexity is not None:
                    if useProtectionOnActivator:
                        activationWriter(nameInputs[idx2]+"p", nameOutputs[idx], nameE, nameE2, constantValues[idx][idx2], pathEquations, pathConstants,complexity=complexity,templateName="Templ_"+nameOutputs[idx]+"_"+nameInputs[idx2])
                    else:
                        activationWriter(nameInputs[idx2], nameOutputs[idx], nameE, nameE2, constantValues[idx][idx2], pathEquations, pathConstants,complexity=complexity)
                    if complexity == "simple":
                        activatorInputs+=[[nameInputs[idx2],constantValues[idx][idx2][:3]]]
                    elif complexity == "normal":
                        activatorInputs+=[[nameInputs[idx2],constantValues[idx][idx2][:6]]]
                else:
                    if useProtectionOnActivator:
                        activationWriter(nameInputs[idx2]+"p", nameOutputs[idx], nameE, nameE2, constantValues[idx][idx2], pathEquations, pathConstants,templateName="Templ_"+nameOutputs[idx]+"_"+nameInputs[idx2])
                    else:
                        activationWriter(nameInputs[idx2], nameOutputs[idx], nameE, nameE2, constantValues[idx][idx2], pathEquations, pathConstants)
                    activatorInputs+=[[nameInputs[idx2],constantValues[idx][idx2][:6]]]
            elif(speciesAction<0):
                if complexity is not None:
                    inhibitionWriter(nameInputs[idx2], nameOutputs[idx], nameE, nameE2, constantValues[idx][idx2], pathEquations, pathConstants,complexity=complexity)
                else:
                    inhibitionWriter(nameInputs[idx2], nameOutputs[idx], nameE, nameE2, constantValues[idx][idx2], pathEquations, pathConstants)
                inhibitedOutputs+=[[nameOutputs[idx],constantValues[idx][idx2],endoConstants[idx]]]
    if inhibitionWriter == templateInhibWrite:
        for idx,nameY in enumerate(nameOutputs):
            if nameEndo is not None:
                endonucleaseWrite2(nameY, nameEndo, endoConstants[idx], pathEquations, pathConstants)
            else:
                endonucleaseWrite(nameY, endoConstants[idx], pathEquations, pathConstants)
        for idx,tupleY in enumerate(inhibitedOutputs):
            nameY,csteValues,endoCsteValues = tupleY
            if complexity is not None:
                templateRealInhibitionWrite(nameY,nameE,csteValues, pathEquations,pathConstants,complexity)
                if nameEndo is not None:
                    endonucleaseWrite2(nameY+"d",nameEndo, endoCsteValues, pathEquations, pathConstants)
                else:
                    endonucleaseWrite(nameY+"d", endoConstants[idx], pathEquations, pathConstants)
            else:
                templateRealInhibitionWrite(nameY,nameE,csteValues, pathEquations,pathConstants)
                if nameEndo is not None:
                    endonucleaseWrite2(nameY+"d",nameEndo, endoCsteValues, pathEquations, pathConstants)
                else:
                    endonucleaseWrite(nameY+"d", endoConstants[idx], pathEquations, pathConstants)
        if useProtectionOnActivator:
            for idx,tupleA in enumerate(activatorInputs):
                nameA,csteValues = tupleA
                if complexity is not None:
                    templateProtection(nameA,nameE,nameE2, csteValues, pathEquations, pathConstants, complexity=complexity)
                else:
                    templateProtection(nameA,nameE,nameE2, csteValues, pathEquations, pathConstants)
    else:
        for idx,nameY in enumerate(nameOutputs):
            endonucleaseWrite(nameY, endoConstants[idx], pathEquations, pathConstants)

    # We need to degrade inputs to the first layer (network inputs):
    if nameInputs[0].split("_")[1]=="0":
        for idx,nameA in enumerate(nameInputs):
            if inhibitionWriter == templateInhibWrite:
                if nameEndo is not None:
                    endonucleaseWrite2(nameA,nameEndo, endoConstants[idx+len(nameOutputs)], pathEquations, pathConstants)
                else:
                    endonucleaseWrite(nameA, endoConstants[idx], pathEquations, pathConstants)
            else:
                endonucleaseWrite(nameA, endoConstants[idx+len(nameOutputs)], pathEquations, pathConstants)
def generateNeuralNetwork(name,masks,activConstants=None,inhibConstants=None,endoConstant=None,erase=True):
    """
        Generate a neural network using the autocatalysis models.
            The reaction constants are believed to be similar among all reactions. (in future dev: either fully given, either modified by a small amount consistently)
            if not provided, we use values from Montagne paper.
    :param name: models name
    :param masks: neural network architecture: contains the mask at each layer
    :param activConstants: reactions constants for activations,
                           if not provided default to:
                                26*10**12 for activation reaction with 3 species (template + activator + polymerase)
                                26*10**6 for activation reaction with 2 species (complex + polymerase)
                                3 for reactions in the non-natural other way.
                                7*10**6 for activation reaction with 2 species (complex + nickase)
    :param inhibConstants:
    :param endoConstant: constant for endonuclease reaction default to 0.32
    :param erase: if we need to erase previous files for the equations, default to True (recommended)
    :return:
    """
    pathEquations = name+"/equations.txt"
    pathConstants = name+"/constants.txt"
    if(erase):
        if(os.path.exists(pathEquations)):
            with open(pathConstants,"w") as file:
                pass
        if(os.path.exists(pathConstants)):
            with open(pathEquations,"w") as file:
                pass
    if not activConstants:
        activConstants=[26*10**12,3,17,7*10**6,3,3]
    if not inhibConstants:
        inhibConstants=[26*10**6,3,17,26*10**6,3,3,26*10**12,3,17,0.03,26*10**6]
    if not endoConstant:
        endoConstant=0.32
    nameE="E"
    nameE2="E2"

    for l in range(len(masks)):
        nameInputs=["X_" + str(l) + "_" + str(idx) for idx in range(np.array(masks[l]).shape[1])]
        nameOutputs=["X_" + str(l+1) + "_" + str(idx) for idx in range(np.array(masks[l]).shape[0])]
        constantValues=[]
        for m in masks[l]:
            line=[]
            for m2 in m:
                if(m2>0):
                    line+=[activConstants]
                elif(m2<0):
                    line+=[inhibConstants]
                else:
                    line+=[[None]]
            constantValues+=[line]
        endoConstants=[endoConstant for _ in range(np.array(masks[l]).shape[0])]
        generateLayer(nameInputs, nameOutputs, nameE, nameE2, masks[l], constantValues, endoConstants,
                      activationWriter=autocatalysisWrite,inhibitionWriter=killingTemplateWrite,
                      pathEquations= pathEquations,pathConstants= pathConstants)

def generateTemplateNeuralNetwork(name, masks, complexity=None, activConstants=None, inhibConstants=None, endoConstants=[7*10**6,3,0.32], erase=True, useProtectionOnActivator=False):
    """
        Generate a neural network using the template models.
        For each reaction a template is used: all species of interests are now small adn strands.
        The reaction constants are believed to be similar among all reactions. (in future dev: either fully given, either modified by a small amount consistently)
        if not provided, we use values from Montagne paper.
    :param name: models name
    :param masks: neural network architecture: contains the mask at each layer
    :param complexity: to be chosen between [simple,normal,full], default to normal. Represent the level of chemical description we use.
                       Please refer to module equationWriter for more precisions.
    :param activConstants: if not provided default to: ONLY SUPPORTED FOR THE NORMAL MODE!!!
                            26*10**12 for activation reaction with 3 species (template + activator + polymerase)
                            26*10**6 for activation reaction with 2 species (complex + polymerase)
                            3 for reactions in the non-natural other way.
                            7*10**6 for activation reaction with 2 species (complex + nickase)
    :param inhibConstants: constants for activation, if not provided default to similar values than activConstants for similar reactions.
    :param endoConstants: constant for endonuclease reaction default to [7*10**6,0.32]
                          if is explicitely set at None, then we don't use endonuclease and replace it by a constant of value 0.32.
    :param erase: if we need to erase previous files for the equations, default to True (recommended)
    :param useProtectionOnActivator: default to False. If True then the activators will be link to a protection reaction so that they are not degraded.
    :return:
    """
    pathEquations = name+"/equations.txt"
    pathConstants = name+"/constants.txt"
    if(erase):
        if(os.path.exists(pathEquations)):
            with open(pathConstants,"w") as file:
                pass
        if(os.path.exists(pathConstants)):
            with open(pathEquations,"w") as file:
                pass
    if not activConstants:
        if complexity is not None:
            assert complexity=="normal" or complexity=="simple"
            if complexity=="normal":
                #k1,k-1,k2,k1n,k-1n,k2n
                activConstants=[26*10**12,3,17,7*10**6,3,3]
            elif complexity=="simple":
                #k1,k-1,k2
                activConstants=[26*10**12,3,17]
        else:
            activConstants=[26*10**12,3,17,7*10**6,3,3]
    if not inhibConstants:
        if complexity is not None:
            assert complexity=="normal" or complexity=="simple"
            if complexity=="normal":
                    #   k3,k-3,k4,k3n,k-3n,k4n,k5,k-6,k6
                inhibConstants=[26*10**12,3,17,7*10**6,3,3,26*10**12,3,17]
            elif complexity=="simple":
                    #   k3,k-3,k4,k5,k-6,k6
                inhibConstants=[26*10**12,3,17,26*10**12,3,17]
        else:
            inhibConstants=[26*10**12,3,17,7*10**6,3,3,26*10**12,3,17]
    if endoConstants is not None:
        nameEndo="Endo"
    else:
        nameEndo=None
    if endoConstants is None:
        endoConstants=0.32
    nameE="E"
    nameE2="E2"


    for l in range(len(masks)):
        nameInputs=["X_" + str(l) + "_" + str(idx) for idx in range(np.array(masks[l]).shape[1])]
        nameOutputs=["X_" + str(l+1) + "_" + str(idx) for idx in range(np.array(masks[l]).shape[0])]
        constantValues=[]
        for m in masks[l]:
            line=[]
            for m2 in m:
                if(m2>0):
                    line+=[activConstants]
                elif(m2<0):
                    line+=[inhibConstants]
                else:
                    line+=[[None]]
            constantValues+=[line]
        endoConstantsList=[endoConstants for _ in range(np.array(masks[l]).shape[0])]
        if l==0:
            endoConstantsList=np.concatenate((endoConstantsList,[endoConstants for _ in range(np.array(masks[l]).shape[1])]),axis=0)
        if complexity is None:
            generateLayer(nameInputs, nameOutputs, nameE, nameE2, masks[l], constantValues, endoConstantsList,
                          activationWriter=templateActivationWrite,inhibitionWriter=templateInhibWrite,
                          pathEquations= pathEquations,pathConstants= pathConstants,nameEndo=nameEndo,useProtectionOnActivator=useProtectionOnActivator)
        else:
            generateLayer(nameInputs, nameOutputs, nameE, nameE2, masks[l], constantValues, endoConstantsList,
                          activationWriter=templateActivationWrite,inhibitionWriter=templateInhibWrite,complexity=complexity,
                          pathEquations= pathEquations,pathConstants= pathConstants,nameEndo=nameEndo,useProtectionOnActivator=useProtectionOnActivator)



"""
    In the following part we define a parser rendering sparse matrix
    This is very usefull when considering huge system as one with thousand of species.
    The computation mask, tends to be very sparse and of a size too big to be use
"""
import sparse
#we use the sparse library which enable the use of multi dimensionnal array rather than just 2D arrays

def sparseParser(equations,kdic):
    """
        Parse the equations into the correct masks and arrays for computations, but as sparse matrix
    :param equations: a d*n array where d is the number of equations (one sided : -> ) and n the number of species
            The syntax should be the following, an array with the stochiometric coefficient of the species: [alpha,-beta] for alpha*A->beta*B for example
                Negative stochiometric coeff represent product when positive represent input.
    :param
        kdic: a d array, the reaction constant for the equation
    :return:  KarrayA,stochio, maskA these different mask are to feed to the function computing the derivative. It is more efficient to compute them beforehand.
              In the sparsity case, stochio is as bit different:
                We can't keep the ones everywhere => We use the fact that the log will only apply on elements in the data array from the maskA.
    """

    print("sparse parsing")

    coordMaskA2 = []
    valMaskA2 = []
    coordStockio2 = []
    valStockio2 = []
    coordKarrayA2 = []
    valkarrayA2 = []

    for idx,e in enumerate(equations):
        # First solution: really too slow
        # for species,coeff in enumerate(e):
        #     for species2,coeff2 in enumerate(e):
        #         if(coeff2>0 and coeff!=0): #species2 must be an input, and species should be either an input or an output species
        #             coordMaskA+=[[idx,species,species2]]
        #             valMaskA+=[1]
        #             coordStockio += [[idx,species,species2]]
        #             valStockio += [coeff2]
        #     if(coeff<0):
        #         coordKarrayA += [[idx,species]]
        #         valkarrayA += [-1*coeff*kdic[idx]]
        #     elif(coeff>0):
        #         coordKarrayA += [[idx,species]]
        #         valkarrayA += [-1*kdic[idx]]

        # Other solution:

        m = np.transpose(np.argwhere(e!=0))[0]
        m2 = np.transpose(np.argwhere(e>0))[0]
        # Take the union of the criteria:
        m3 = np.concatenate([m for _ in range(m2.shape[0])])
        m4 = np.concatenate([m2 for _ in range(m.shape[0])])
        coordMaskA2 += [np.stack([np.zeros(m3.shape[0]) + idx,m3,m4],axis = 1)]
        coordStockio2 += [np.stack([np.zeros(m3.shape[0]) + idx,m3,m4],axis = 1)]
        m5 = np.transpose(np.argwhere(e<0))[0]
        m5plus2 = np.concatenate((m5,m2))
        coordKarrayA2 += [np.stack([np.zeros(m5plus2.shape[0]) + idx,m5plus2], axis = 1)]

        valMaskA2 += [np.zeros(m3.shape[0]) + 1]
        valStockio2 += [np.concatenate([e[e>0] for _ in range(m.shape[0])])]
        valkarrayA2 += [np.concatenate([-1*e[e<0]*kdic[idx],[-1*kdic[idx] for _ in range(m2.shape[0])]])]

    #the format for the sparse library, concerning the coordinate matrix is (axis,nbdata) so we need to transpose here
    coordKarrayA = np.transpose(np.concatenate(coordKarrayA2))
    coordStockio = np.transpose(np.concatenate(coordStockio2))
    coordMaskA = np.transpose(np.concatenate(coordMaskA2))

    valMaskA = np.concatenate(valMaskA2)
    valStockio = np.concatenate(valStockio2)
    valkarrayA = np.concatenate(valkarrayA2)

    KarrayA = sparse.COO(coordKarrayA, valkarrayA, shape = equations.shape)
    stochio = sparse.COO(coordStockio, valStockio, shape = (equations.shape[0],equations.shape[1],equations.shape[1]))
    maskA = sparse.COO(coordMaskA, valMaskA , shape = (equations.shape[0],equations.shape[1],equations.shape[1]))
    maskComplementary = 1 - maskA
    return KarrayA,stochio,maskA,maskComplementary


def saveModelWeight(model,use_bias,dirName="weightDir"):
    """
        Save model weights so that in can be reused by the parser.
    :param model: keras trained model
    :param use_bias: if the model use bias or not.
    :param dirName: name of the directory where the weight shall be stored.
    :return
    """
    weights = model.get_weights()
    if(not os.path.exists(dirName)):
        os.mkdir(dirName)
    for idx,w in enumerate(weights):
        if(idx%2==0 and use_bias):    #ignore the bias:
            wNeg = np.zeros(w.shape) - 1
            wPos = np.zeros(w.shape) + 1
            wZeros = np.zeros(w.shape)
            wClipp = np.where(w<-0.2,wNeg,np.where(w>0.2,wPos,wZeros))
            df = pandas.DataFrame(wClipp)
            df.to_csv(dirName+"/weight_"+str(int(idx/2))+".csv")
        elif(not use_bias):
            wNeg = np.zeros(w.shape) - 1
            wPos = np.zeros(w.shape) + 1
            wZeros = np.zeros(w.shape)
            wClipp = np.where(w<-0.2,wNeg,np.where(w>0.2,wPos,wZeros))
            df = pandas.DataFrame(wClipp)
            df.to_csv(dirName+"/weight_"+str(idx)+".csv")