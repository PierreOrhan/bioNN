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
from simulOfBioNN.smallNetworkSimul.templateModelRandomNetworkSimul import computeCP as oldComputCP
from simulOfBioNN.smallNetworkSimul.templateModelRandomNetworkSimul import computeEquilibriumValue as oldComputeEquilibriumValue
from simulOfBioNN.smallNetworkSimul.templateModelRandomNetworkSimul import func as oldAllEquilibriumFunc
from scipy.optimize import minimize,root,brentq
import sys
import time
import pandas


def obtainBornSup(k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks):
    """
        Given approximate only for the enzyme competition term (cp), we compute the next approximate using G.
        The real value for the competitions term verify cp = G(cp,initialConditions)
    :param cp: float, competition over tempalte.
    :param E0: float, initial concentration in the enzyme
    :param X0: nbrInputs array, contains initial value for the inputs
    :param masks: masks giving the network topology.
    :return:
    """
    max_cp = 1
    olderX = [np.zeros(m.shape[1]) for m in masks]
    for layeridx,layer in enumerate(masks):
        layerEq = np.zeros(layer.shape[1])
        if(layeridx==0):
            for inpIdx in range(layer.shape[1]):
                #compute of Kactivs,Kinhibs;
                Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                #compute of "weights": sum of kactivs and kinhibs
                w_inpIdx = np.sum(Kactivs)+np.sum(Kinhibs)
                max_cp += w_inpIdx*X0[inpIdx]
                # saving values
                layerEq[inpIdx] = X0[inpIdx]
            olderX[layeridx] = layerEq
        else:
            for inpIdx in range(layer.shape[1]):
                #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                #Terms for the previous layers
                CactivsOld = np.where(masks[layeridx-1][inpIdx,:]>0,Cactiv0[layeridx-1][inpIdx],0)
                CinhibsOld = np.where(masks[layeridx-1][inpIdx,:]<0,Cinhib0[layeridx-1][inpIdx],0)
                Inhib = np.sum(CinhibsOld*olderX[layeridx-1]/kdT[layeridx-1][inpIdx])
                #computing of new equilibrium
                x_eq = np.sum(CactivsOld*olderX[layeridx-1]/kdT[layeridx-1][inpIdx])
                layerEq[inpIdx] = x_eq
                #compute of Kactivs,Kinhibs, for the current layer:
                Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                firstComplex = np.sum(np.where(layer[:,inpIdx]>0,Kactivs*x_eq,np.where(layer[:,inpIdx]<0,Kinhibs*x_eq,0)))
                #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                Inhib2 = np.sum(CinhibsOld*olderX[layeridx-1]/(kdT[layeridx-1][inpIdx]*k6[layeridx-1][inpIdx]))
                max_cp += firstComplex + Inhib2/E0*x_eq
            olderX[layeridx] = layerEq
    #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layers
    for outputsIdx in range(masks[-1].shape[0]):
        Cinhibs = np.where(masks[-1][outputsIdx,:]<0,Cinhib0[-1][outputsIdx],0)
        Cactivs = np.where(masks[-1][outputsIdx,:]>0,Cactiv0[-1][outputsIdx],0)
        x_eq = np.sum(Cactivs*olderX[-1]/(kdI[-1][outputsIdx]))
        Inhib2 = np.sum(Cinhibs*olderX[-1]/(kdT[-1][outputsIdx]*k6[-1][outputsIdx]))
        max_cp += Inhib2/E0*x_eq
    return max_cp


def cpEquilibriumFunc(cp,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks):
    """
        Given approximate only for the enzyme competition term (cp), we compute the next approximate using G.
        The real value for the competitions term verify cp = G(cp,initialConditions)
    :param cp: float, competition over tempalte.
    :param E0: float, initial concentration in the enzyme
    :param X0: nbrInputs array, contains initial value for the inputs
    :param masks: masks giving the network topology.
    :return:
    """
    new_cp = 1
    olderX = [np.zeros(m.shape[1]) for m in masks]
    for layeridx,layer in enumerate(masks):
        layerEq = np.zeros(layer.shape[1])
        if(layeridx==0):
            for inpIdx in range(layer.shape[1]):
                #compute of Kactivs,Kinhibs;
                Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                #compute of "weights": sum of kactivs and kinhibs
                w_inpIdx = np.sum(Kactivs)+np.sum(Kinhibs)
                x_eq = X0[inpIdx]/(1+E0*w_inpIdx/cp)
                # update for fixed point:
                new_cp += w_inpIdx*x_eq
                # saving values
                layerEq[inpIdx] = x_eq
            olderX[layeridx] = layerEq
        else:
            for inpIdx in range(layer.shape[1]):

                #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                #Terms for the previous layers
                CactivsOld = np.where(masks[layeridx-1][inpIdx,:]>0,Cactiv0[layeridx-1][inpIdx],0)
                CinhibsOld = np.where(masks[layeridx-1][inpIdx,:]<0,Cinhib0[layeridx-1][inpIdx],0)
                Inhib = np.sum(CinhibsOld*olderX[layeridx-1]/kdT[layeridx-1][inpIdx])
                #computing of new equilibrium
                x_eq = np.sum(CactivsOld*olderX[layeridx-1]/(kdI[layeridx-1][inpIdx]*cp+Inhib/cp))
                layerEq[inpIdx] = x_eq
                #compute of Kactivs,Kinhibs, for the current layer:
                Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                firstComplex = np.sum(np.where(layer[:,inpIdx]>0,Kactivs*x_eq,np.where(layer[:,inpIdx]<0,Kinhibs*x_eq,0)))
                #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                Inhib2 = np.sum(CinhibsOld*olderX[layeridx-1]/(kdT[layeridx-1][inpIdx]*k6[layeridx-1][inpIdx]))
                new_cp += firstComplex + Inhib2/(E0*cp)*x_eq
            olderX[layeridx] = layerEq
    #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layers
    for outputsIdx in range(masks[-1].shape[0]):
        Cinhibs = np.where(masks[-1][outputsIdx,:]<0,Cinhib0[-1][outputsIdx],0)
        Cactivs = np.where(masks[-1][outputsIdx,:]>0,Cactiv0[-1][outputsIdx],0)

        Inhib = np.sum(Cinhibs*olderX[-1]/kdT[-1][outputsIdx])
        x_eq = np.sum(Cactivs*olderX[-1]/(kdI[-1][outputsIdx]*cp+Inhib/cp))
        Inhib2 = np.sum(Cinhibs*olderX[-1]/(kdT[-1][outputsIdx]*k6[-1][outputsIdx]))
        new_cp += Inhib2/(E0*cp)*x_eq
    return cp - new_cp

def allEquilibriumFunc(cps,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks):
    """
        Given approximate for the different competitions term (in the vector cps), we compute the next approximate using G.
        The real value for the competitions term verify cps = G(cps,initialConditions)
    :param cps: 1+nbrTemplate array. In the first one we store the competition on the enzyme.
                                     In the other the competition on the template species.
    :param E0: float, initial concentration in the enzyme
    :param X0: nbrInputs array, contains initial value for the inputs
    :param masks: masks giving the network topology.
    :return:
    """
    # We move from an array to a list of 2d-array for the competition over each template:

    cp=cps[0]
    cpt = [np.reshape(cps[(l*m.shape[0]*m.shape[1]+1):((l+1)*m.shape[0]*m.shape[1]+1)],(m.shape[0],m.shape[1])) for l,m in enumerate(masks)]
    new_cpt = [np.zeros(l.shape)+1 for l in cpt]
    new_cp = 1

    olderX = [np.zeros(m.shape[1]) for m in masks]
    for layeridx,layer in enumerate(masks):
        layerEq = np.zeros(layer.shape[1])
        if(layeridx==0):
            for inpIdx in range(layer.shape[1]):
                #compute of Kactivs,Kinhibs;
                Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx]/cpt[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx]/cpt[layeridx][:,inpIdx],0)
                #compute of "weights": sum of kactivs and kinhibs
                w_inpIdx = np.sum(Kactivs)+np.sum(Kinhibs)
                x_eq = X0[inpIdx]/(1+E0*w_inpIdx/cp)
                # update for fixed point:
                new_cp += w_inpIdx*x_eq
                # saving values
                layerEq[inpIdx] = x_eq
            new_cpt[layeridx] = np.where(layer>0,(1+k1M[layeridx]*E0*layerEq/cp),np.where(layer<0,(1+k3M[layeridx]*E0*layerEq/cp),1))
            olderX[layeridx] = layerEq
        else:
            for inpIdx in range(layer.shape[1]):

                #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                #Terms for the previous layers
                CactivsOld = np.where(masks[layeridx-1][inpIdx,:]>0,Cactiv0[layeridx-1][inpIdx]/cpt[layeridx-1][inpIdx],0)
                CinhibsOld = np.where(masks[layeridx-1][inpIdx,:]<0,Cinhib0[layeridx-1][inpIdx]/cpt[layeridx-1][inpIdx],0)
                Inhib = np.sum(CinhibsOld*olderX[layeridx-1]/kdT[layeridx-1][inpIdx])
                #computing of new equilibrium
                x_eq = np.sum(CactivsOld*olderX[layeridx-1]/(kdI[layeridx-1][inpIdx]*cp+Inhib/cp))
                layerEq[inpIdx] = x_eq

                #compute of Kactivs,Kinhibs, for the current layer:
                Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx]/cpt[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx]/cpt[layeridx][:,inpIdx],0)
                #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                firstComplex = np.sum(np.where(layer[:,inpIdx]>0,Kactivs*x_eq,np.where(layer[:,inpIdx]<0,Kinhibs*x_eq,0)))
                #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                Inhib2 = np.sum(CinhibsOld*olderX[layeridx-1]/(kdT[layeridx-1][inpIdx]*k6[layeridx-1][inpIdx]))
                new_cp += firstComplex + Inhib2/(E0*cp)*x_eq
            new_cpt[layeridx] = np.where(layer>0,(1+k1M[layeridx]*E0*layerEq/cp),np.where(layer<0,(1+k3M[layeridx]*E0*layerEq/cp),1))
            olderX[layeridx] = layerEq
    #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layers
    for outputsIdx in range(masks[-1].shape[0]):
        Cinhibs = np.where(masks[-1][outputsIdx,:]<0,Cinhib0[-1][outputsIdx]/cpt[-1][outputsIdx],0)
        Cactivs = np.where(masks[-1][outputsIdx,:]>0,Cactiv0[-1][outputsIdx]/cpt[-1][outputsIdx],0)

        Inhib = np.sum(Cinhibs*olderX[-1]/kdT[-1][outputsIdx])
        x_eq = np.sum(Cactivs*olderX[-1]/(kdI[-1][outputsIdx]*cp+Inhib/cp))
        Inhib2 = np.sum(Cinhibs*olderX[-1]/(kdT[-1][outputsIdx]*k6[-1][outputsIdx]))
        new_cp += Inhib2/(E0*cp)*x_eq

    #we know that the root should be larger than 1 so we artificially add a root barrier at 1.
    diff_cps = np.zeros(cps.shape)
    if new_cp>=1:
        diff_cps[0] = cp - new_cp
    else:
        diff_cps[0] = 10**3
    for l,m in enumerate(masks):
        layer_new_cpt = np.reshape(new_cpt[l],(m.shape[0]*m.shape[1]))
        layer_old_cpt = cps[(l*m.shape[0]*m.shape[1]+1):((l+1)*m.shape[0]*m.shape[1]+1)]
        diff_cps[(l*m.shape[0]*m.shape[1]+1):((l+1)*m.shape[0]*m.shape[1]+1)] = layer_old_cpt - np.where(layer_new_cpt<1,layer_old_cpt + 10**3, layer_new_cpt)

    return diff_cps

def for_scalar(cps,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks):
    return np.sum(np.abs(allEquilibriumFunc(cps,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks)))

def computeCPonly(k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdI,kdT,TA0,TI0,E0,X0,masks,fittedValue=None):
    """
         This function computes the competition's value by solving a fixed point equation.
         It is based on the most simplest chemical model for the template model: no endonuclease; polymerase and nickase are considered together.
     :param k1, and the others k are the reactions constants
     :param X: array with initial concentration of the first layer.
     :param masks:
     :param fittedValue: value obtained with an analytical model that is an upper bound on the real value. if not provided we use 10**6
     :return:
     """
    # Computation of the different constants that are required.
    k1M = [k1[l]/(k1n[l]+k2[l]) for l in range(len(k1))] #(matrix operations)
    k3M = [k3[l]/(k3n[l]+k4[l]) for l in range(len(k3))]
    k5M = [k5[l]/(k5n[l]+k6[l]) for l in range(len(k5))]

    Kactiv0 = [k1M[l]*TA0[l] for l in range(len(k1M))] # element to element product
    Kinhib0 = [k3M[l]*TI0[l] for l in range(len(k3M))]
    Cactiv0 = [k2[l]*k1M[l]*TA0[l]*E0 for l in range(len(k1M))]
    Cinhib0 =[k6[l]*k5M[l]*k4[l]*k3M[l]*TI0[l]*E0*E0 for l in range(len(k3M))]

    if fittedValue is None:
        cp0max=obtainBornSup(k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks)# we start from an upper bound
    else:
        cp0max=fittedValue
    t0=time.time()
    computedCp = brentq(cpEquilibriumFunc,1,cp0max,args=(k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks))
    print("Ended brentq methods in "+str(time.time()-t0))
    return computedCp

def computeCP(k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdT,kdI,TA0,TI0,E0,X0,masks,initValue=None):
    """
        This function computes the competition's value by solving a fixed point equation.
        It is based on the most simplest chemical model for the template model: no endonuclease; polymerase and nickase are considered together.
    :param k1, and the others k are the reactions constants
    :param X0: array with initial concentration of the first layer.
    :param masks:
    :param initValue: initial value for cp, the competition over the enzymes. Can be computed with brentq method by fixing the competition over template...
    :return:
    """
    # Computation of the different constants that are required.
    k1M = [k1[l]/(k1n[l]+k2[l]) for l in range(len(k1))] #(matrix operations)
    k3M = [k3[l]/(k3n[l]+k4[l]) for l in range(len(k3))]
    k5M = [k5[l]/(k5n[l]+k6[l]) for l in range(len(k5))]

    Kactiv0 = [k1M[l]*TA0[l] for l in range(len(k1M))] # element to element product
    Kinhib0 = [k3M[l]*TI0[l] for l in range(len(k3M))]
    Cactiv0 = [k2[l]*k1M[l]*TA0[l]*E0 for l in range(len(k1M))]
    Cinhib0 =[k6[l]*k5M[l]*k4[l]*k3M[l]*TI0[l]*E0*E0 for l in range(len(k3M))]

    if initValue is None:
        cps0min=np.concatenate(([1],np.zeros(np.sum([m.shape[0]*m.shape[1] for m in masks]))+1)) # we start from an inferior bound
    else:
        cps0min=np.concatenate(([initValue],np.zeros(np.sum([m.shape[0]*m.shape[1] for m in masks]))+1))
    t0=time.time()
    #computedCp = minimize(for_scalar,cps0min,args=(k2,k4,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,TA0,TI0,E0,X0,masks),method="L-BFGS-B",bounds=[(1,None) for _ in cps0min])
    computedCp = root(allEquilibriumFunc,cps0min,args=(k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks))
    print("ended root methods in "+str(time.time()-t0))
    if not computedCp["success"]:
        print(computedCp["message"])
    return computedCp["x"]

def computeEquilibriumValue(cps,k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdT,kdI,TA0,TI0,E0,X0,masks,observed=None):
    """
        Given cp, compute the equilibrium values for all nodes in solutions.
    :param cps: Value for the competitions over enzyme and template obtained by fixed point strategy.
    :param observed: tuple, default to None. If provided, we only give the value for the species at the position observed.
    :return:
    """
    k1M = [k1[l]/(k1n[l]+k2[l]) for l in range(len(k1))] #(matrix operations)
    k3M = [k3[l]/(k3n[l]+k4[l]) for l in range(len(k3))]
    k5M = [k5[l]/(k5n[l]+k6[l]) for l in range(len(k5))]
    Kactiv0 = [k1M[l]*TA0[l] for l in range(len(k1M))] # element to element product
    Kinhib0 = [k3M[l]*TI0[l] for l in range(len(k3M))]
    Cactiv0 = [k2[l]*k1M[l]*TA0[l]*E0 for l in range(len(k1M))]
    Cinhib0 =[k6[l]*k5M[l]*k4[l]*k3M[l]*TI0[l]*E0*E0 for l in range(len(k3M))]

    # We move from an array to a list of 2d-array for the competition over each template:
    cp=cps[0]
    cpt = [np.reshape(cps[(l+1):(l+1+m.shape[0]*m.shape[1])],(m.shape[0],m.shape[1])) for l,m in enumerate(masks)]
    equilibriumValue = [np.zeros(m.shape[1]) for m in masks]+[np.zeros(masks[-1].shape[0])]
    for layeridx,layer in enumerate(masks):
        layerEq = np.zeros(layer.shape[1])
        if(layeridx==0):
            for inpIdx in range(layer.shape[1]):
                #compute of Kactivs,Kinhibs;
                Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx]/cpt[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx]/cpt[layeridx][:,inpIdx],0)
                #compute of "weights": sum of kactivs and kinhibs
                w_inpIdx = np.sum(Kactivs)+np.sum(Kinhibs)
                x_eq = X0[inpIdx]/(1+E0*w_inpIdx/cp)
                # saving values
                layerEq[inpIdx] = x_eq
            equilibriumValue[layeridx] = layerEq
        else:
            for inpIdx in range(layer.shape[1]):
                #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                #Terms for the previous layers
                CactivsOld = np.where(masks[layeridx-1][inpIdx,:]>0,Cactiv0[layeridx-1][inpIdx]/cpt[layeridx-1][inpIdx],0)
                CinhibsOld = np.where(masks[layeridx-1][inpIdx,:]<0,Cinhib0[layeridx-1][inpIdx]/cpt[layeridx-1][inpIdx],0)

                #compute of Kactivs,Kinhibs, for the current layer:
                Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx]/cpt[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx]/cpt[layeridx][:,inpIdx],0)
                w_nextLayer = E0*(np.sum(Kactivs*k2[layeridx][:,inpIdx])+np.sum(Kinhibs*k4[layeridx][:,inpIdx])) #the input influences the next layer

                Inhib = np.sum(CinhibsOld*equilibriumValue[layeridx-1]/kdT[layeridx-1][inpIdx])
                #computing of new equilibrium
                x_eq = np.sum(CactivsOld*equilibriumValue[layeridx-1]/(kdI[layeridx-1][inpIdx]*cp+Inhib/cp))
                layerEq[inpIdx] = x_eq
            equilibriumValue[layeridx] = layerEq

    #Compute equilibrium for last layers:
    layerEq = np.zeros(masks[-1].shape[0])
    for outputIdx in range(masks[-1].shape[0]):
        CactivsOld = np.where(masks[-1][outputIdx,:]>0,Cactiv0[-1][outputIdx]/cpt[-1][outputIdx],0)
        CinhibsOld = np.where(masks[-1][outputIdx,:]<0,Cinhib0[-1][outputIdx]/cpt[-1][outputIdx],0)

        Inhib = np.sum(CinhibsOld*equilibriumValue[-2]/kdT[-1][outputIdx])
        #computing of new equilibrium
        x_eq = np.sum(CactivsOld*equilibriumValue[-2]/(kdI[-1][outputIdx]*cp+Inhib/cp))
        layerEq[outputIdx]=x_eq
    equilibriumValue[-1] = layerEq

    if observed is not None:
        try:
            assert len(observed)==2
        except:
            raise Exception("please provide a tuple of size two for observed indicating the value to use.")
        return equilibriumValue[observed[0]][observed[1]]
    return equilibriumValue


def _createMask(nbrOutputNodes):
    mask=[]
    for idx in range(nbrOutputNodes):
        zero=np.zeros(2*nbrOutputNodes)
        zero[2*idx]=1
        zero[2*idx+1]=-1
        mask+=[zero]
    return np.array([mask])

def sample_bimodal_distrib(nbrInputs,peak1=-4,peak2=-8,sigma1=1.,sigma2=1.,mixture = 0.5):
    """
        Sample from a bimodal distribution around peak 1 and peak 2 based on gaussian law.
    :param nbrInputs: int, indicate the number of inputs
    :param peak1: int, indicate the power of 10 around which the first peak should be centered
    :param peak2: int, indicate the power of 10 around which the second peak should be centered
    :param sigma1: float, indicate the width in power of 10 around the peak1.
    :param sigma2: float, indicate the width in power of 10 around the peak2.
    :param mixture: proba for the mixture of the two.
    :return:
    """
    mixChoice = np.random.random(nbrInputs)
    gaussian1 = np.random.normal(peak1,sigma1,nbrInputs)
    gaussian2 = np.random.normal(peak2,sigma2,nbrInputs)

    powers = np.where(mixChoice>mixture,gaussian1,gaussian2)
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

    name = "templateModelTwoLayer/equiNetwork_100_newfit"
    endTime = 10000
    timeStep = 0.1

    # masks=[np.array([[1,-1,0,0],[0,0,1,-1]]),np.array([[1,-1]])]
    # nbrInputs=masks[0].shape[1]
    #
    # activatorsOnObserved = [0]
    # inhibitorsOnObserved = [1]
    # nodeObserved = 0
    nbrInputs=[100,10]
    nbrOutputs=[10,5]
    activatorsOnObserved = []
    inhibitorsOnObserved = []
    while(len(activatorsOnObserved)==0 and len(inhibitorsOnObserved)==0):
        masks=[]
        for idx,nbi in enumerate(nbrInputs):
            masks+= _sample_layer_architecture(nbi,nbrOutputs[idx],sparsity=.5)
        nodeObserved = np.random.randint(0,nbrOutputs[0],1)[0]
        # We look at the activators and inhibitors on the observed node, and verify that they have a length greater than one (while loop).
        for idx,m in enumerate(masks[0][nodeObserved]):
            if m==1:
                activatorsOnObserved += [idx]
            elif m==-1:
                inhibitorsOnObserved += [idx]

    print(masks)
    print("Observed nodes has "+str(len(activatorsOnObserved))+" activators and "+str(len(inhibitorsOnObserved))+" inhibitors")
    print(masks[0][nodeObserved])
    # For debug: we start by working with competition on template from the first node.
    templateObservedIdx = 1 + nodeObserved*masks[0].shape[1] + activatorsOnObserved[0]


    modes = ["verbose","outputEqui"]
    FULL = True
    outputMode = "all"
    outputList = ["X_1_"+str(nodeObserved)] #"all" or None or list of output species
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

    X1,X2,x_test,otherActivInitialC,otherInhibInitialC = _generate_initialConcentration_firstlayer(activatorsOnObserved,inhibitorsOnObserved,masks,nbrInputs[0],minLogSpace=-8,maxLogSpace=-4,nbrValue=5)

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

        # colorDiagram(X1,X2,output,"Initial concentration of X1","Initial concentration of X2","Equilibrium concentration of the output",figname=os.path.join(experiment_path, "neuralDiagramm.png"),equiPotential=False)
        # neuronPlot(X1,X2,output,figname=os.path.join(experiment_path, "activationLogX1.png"),figname2=os.path.join(experiment_path, "activationLogX2.png"),useLogX = True, doShow= False)
        # neuronPlot(X1,X2,output,figname=os.path.join(experiment_path, "activationX1.png"),figname2=os.path.join(experiment_path, "activationX2.png"),useLogX = False, doShow= False)
        # df = pandas.DataFrame(X1)
        # df.to_csv(os.path.join(experiment_path,"inputX1.csv"))
        # df2 = pandas.DataFrame(X2)
        # df2.to_csv(os.path.join(experiment_path,"inputX2.csv"))

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


        #Third fit, using the compute cp:
        myX_test=np.reshape(x_test,(len(X1),len(X2),nbrInputs[0]))/(C0*rescaleFactor)
        competitions = np.zeros((len(X1),len(X2)))
        oldCompetitions = np.zeros((len(X1),len(X2)))
        fitOutput = np.zeros((len(X1), len(X2)))
        oldFitOutput = np.zeros((len(X1), len(X2)))
        styleFit = np.zeros((len(X1),len(X2),1000))
        oldStyleFit = np.zeros((len(X1),len(X2),1000))
        testOfCp = np.logspace(-5,3,1000)/kd
        testOfCpt = np.logspace(-3,-1,1000)/kd
        courbs=[0,int(fitOutput.shape[1]/2),fitOutput.shape[1]-1,int(fitOutput.shape[1]/3),int(2*fitOutput.shape[1]/3)]

        k1 = [np.zeros(m.shape)+k1 for m in masks]
        k1n = [np.zeros(m.shape)+k1n for m in masks]
        k2 = [np.zeros(m.shape)+k2 for m in masks]
        k3 = [np.zeros(m.shape)+k3 for m in masks]
        k3n = [np.zeros(m.shape)+k3n for m in masks]
        k4 = [np.zeros(m.shape)+k4 for m in masks]
        k5 = [np.zeros(m.shape)+k5 for m in masks]
        k5n = [np.zeros(m.shape)+k5n for m in masks]
        k6 = [np.zeros(m.shape)+k6 for m in masks]
        kdT = [np.zeros(m.shape)+kd for m in masks]
        kdI = [np.zeros(m.shape)+kd for m in masks]

        TA0 = [np.where(m>0,TA,0) for m in masks]
        TI0 = [np.where(m<0,TI,0) for m in masks]
        k1M = [k1[l]/(k1n[l]+k2[l]) for l in range(len(k1))] #(matrix operations)
        k3M = [k3[l]/(k3n[l]+k4[l]) for l in range(len(k3))]
        k5M = [k5[l]/(k5n[l]+k6[l]) for l in range(len(k5))]
        Kactiv0 = [k1M[l]*TA0[l] for l in range(len(k1M))] # element to element product
        Kinhib0 = [k3M[l]*TI0[l] for l in range(len(k3M))]
        Cactiv0 = [k2[l]*k1M[l]*TA0[l]*E0 for l in range(len(k1M))]
        Cinhib0 =[k6[l]*k5M[l]*k4[l]*k3M[l]*TI0[l]*E0*E0 for l in range(len(k3M))]

        visuCPT = np.zeros((len(X1),len(X2),1000))
        outCPT = np.zeros((len(X1),len(X2)))
        for idx1,x1 in enumerate(X1):
            for idx2,x2 in enumerate(X2):
                oldCompetitions[idx1,idx2] = oldComputCP(k1[0][0,0], k1n[0][0,0], k2[0][0,0], k3[0][0,0], k3n[0][0,0], k4[0][0,0], k5[0][0,0], k5n[0][0,0], k6[0][0,0], kdT[0][0,0], TA, TI,E0,myX_test[idx1,idx2],masks)
                cpApproxim = computeCPonly(k1, k1n, k2, k3, k3n, k4, k5, k5n, k6, kdT,kdI, TA0, TI0, E0,myX_test[idx1,idx2],masks)
                cps = computeCP(k1, k1n, k2, k3, k3n, k4, k5, k5n, k6, kdT,kdI, TA0, TI0, E0,myX_test[idx1,idx2],masks,initValue=cpApproxim)
                competitions[idx1,idx2] = cps[0]
                outCPT[idx1,idx2] = cps[1]
                oldFitOutput[idx1,idx2] = oldComputeEquilibriumValue(oldCompetitions[idx1,idx2],myX_test[idx1,idx2],masks ,k1[0][0,0], k1n[0][0,0], k2[0][0,0], k3[0][0,0], k3n[0][0,0], k4[0][0,0], k5[0][0,0], k5n[0][0,0], k6[0][0,0], kdT[0][0,0], TA, TI,E0,observed=(1, nodeObserved))
                fitOutput[idx1,idx2] = computeEquilibriumValue(cps,k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdT,kdI,TA0,TI0,E0,myX_test[idx1,idx2],masks,observed=(1, nodeObserved))
                if idx2 in courbs and idx1==idx2:
                    testcps = []
                    for x in testOfCp:
                        testcps += [np.concatenate(([x],np.zeros(np.sum([m.shape[0]*m.shape[1] for m in masks]))+1))]
                    styleFit[idx1,idx2] = np.array([allEquilibriumFunc(testofcps,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,myX_test[idx1,idx2],masks)[0] for testofcps in testcps])
                    oldStyleFit[idx1,idx2] = np.array([oldAllEquilibriumFunc(x*kd,kdT[0][0,0],E0,k2[0][0,0],k4[0][0,0],k6[0][0,0],Cactiv,CInhib,Kactiv,Kinhib,masks,myX_test[idx1,idx2]) for x in testOfCp])
                    testCPTs=[]
                    for x in testOfCpt:
                        cptInitArray = np.zeros(np.sum([m.shape[0]*m.shape[1] for m in masks])-1)+1
                        cptInitArray[templateObservedIdx-1] = x
                        testCPTs += [np.concatenate(([oldCompetitions[idx1,idx2]*kd],cptInitArray))]
                    visuCPT[idx1,idx2] = np.array([allEquilibriumFunc(testCPT,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,myX_test[idx1,idx2],masks)[templateObservedIdx] for testCPT in testCPTs])
        # comparisons of plot
        fitComparePlot(X1, X2, output, fitOutput, courbs,
                       figname=os.path.join(experiment_path, "fixedPointVSOdeX1.png"),
                       figname2=os.path.join(experiment_path, "fixedPointVSOdeX2.png"), useLogX=False)
        fitComparePlot(X1, X2, output, fitOutput, courbs,
                       figname=os.path.join(experiment_path, "fixedPointVSOdelogX1.png"),
                       figname2=os.path.join(experiment_path, "fixedPointVSOdelogX2.png"), useLogX=True)
        fitComparePlot(X1, X2, output, oldFitOutput, courbs,
                       figname=os.path.join(experiment_path, "oldfixedPointVSOdeX1.png"),
                       figname2=os.path.join(experiment_path, "oldfixedPointVSOdeX2.png"), useLogX=False)
        fitComparePlot(X1, X2, output, oldFitOutput, courbs,
                       figname=os.path.join(experiment_path, "oldfixedPointVSOdelogX1.png"),
                       figname2=os.path.join(experiment_path, "oldfixedPointVSOdelogX2.png"), useLogX=True)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(19.2,10.8), dpi=100)
        cmap = plt.get_cmap('Dark2',X1.shape[0]*len(courbs))
        for idx1,x1 in enumerate(X1):
            for idx2,x2 in enumerate(X2):
                if idx2 in courbs and idx1==idx2:
                    ax.plot(testOfCp*kd,np.abs(styleFit[idx1,idx2,:])*kd,c=cmap(idx1*(courbs.index(idx2)+1)))
                    ax.plot(testOfCp*kd,np.abs(oldStyleFit[idx1,idx2,:]),c=cmap(idx1*(courbs.index(idx2)+1)),linestyle="--")
                    ax.axvline(competitions[idx1,idx2]*kd,c=cmap(idx1*(courbs.index(idx2)+1)),marker="o")
                    #ax.axvline(oldCompetitions[idx1,idx2],c=cmap(idx1*(courbs.index(idx2)+1)),marker="o",linestyle="--")
                    ax.axvline(kd,c=cmap(idx1*(courbs.index(idx2)+1)),marker="x",linestyle="-")
                    # ax.scatter(cpFitted[idx1,idx2],func(cpFitted[idx1,idx2],kd,Cactiv,CInhib,Kactiv,Kinhib,masks,myX_test[idx1,idx2]),c=cmap(idx1*(courbs.index(idx2)+1)),marker="x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.savefig("templateModelTwoLayer/equiNetwork_10_newfit/formOfcp.png")
        plt.show()
        print(competitions)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(19.2,10.8), dpi=100)
        cmap = plt.get_cmap('Dark2',X1.shape[0]*len(courbs))
        for idx1,x1 in enumerate(X1):
            for idx2,x2 in enumerate(X2):
                if idx2 in courbs and idx1==idx2:
                    # ax.plot(testOfCp,np.abs(styleFit[idx1,idx2,:]),c=cmap(idx1*(courbs.index(idx2)+1)))
                    ax.plot(testOfCpt*kd,np.abs(visuCPT[idx1,idx2,:])+10**(-10),c=cmap(idx1*(courbs.index(idx2)+1)),linestyle="--")
                    # ax.axvline(competitions[idx1,idx2],c=cmap(idx1*(courbs.index(idx2)+1)),marker="o")
                    ax.axvline(outCPT[idx1,idx2]*kd,c=cmap(idx1*(courbs.index(idx2)+1)),marker="o",linestyle="--")
                    # ax.scatter(cpFitted[idx1,idx2],func(cpFitted[idx1,idx2],kd,Cactiv,CInhib,Kactiv,Kinhib,masks,myX_test[idx1,idx2]),c=cmap(idx1*(courbs.index(idx2)+1)),marker="x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.savefig("templateModelTwoLayer/equiNetwork_10_newfit/formOfcp2.png")
        plt.show()
        print(outCPT)

