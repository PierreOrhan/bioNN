"""
    In this module we provide computation of a system's equilibrium using the specific chemical model: template.
    More model can be progressively added through the networkEquilibrium.
    This computation requires the solving of a multidimensional fixed-points equations.
        Under assumptions of a high concentration of templates (see model explanations), the solution can be more easily obtained by first approximating the competitions over the enzyme.
    We use scipy's optimize module, more explicitly the brentq and root methods.
"""
import numpy as np
from scipy.optimize import root,brentq
import time

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
def allEquilibriumFunc(cps,k6,k1M,k3M,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks):
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
def computeCPonly(k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdI,kdT,TA0,TI0,E0,X0,masks,fittedValue=None,verbose=True):
    """
         This function computes the competition's value by solving a fixed point equation.
         It is based on the most simplest chemical model for the template model: no endonuclease; polymerase and nickase are considered together.
     :param k1, and the others k are the reactions constants
     :param X0: array with initial concentration of the first layer.
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
    Cinhib0 =[np.stack([k6[l]*k5M[l]]*(k4[l].shape[1]),axis=1)*k4[l]*k3M[l]*TI0[l]*E0*E0 for l in range(len(k3M))]

    if fittedValue is None:
        cp0max=obtainBornSup(k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks)# we start from an upper bound
    else:
        cp0max=fittedValue
    t0=time.time()
    computedCp = brentq(cpEquilibriumFunc,1,cp0max,args=(k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks))
    if verbose:
        print("Ended brentq methods in "+str(time.time()-t0))
    return computedCp
def computeCPs(k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdT,kdI,TA0,TI0,E0,X0,masks,initValue=None,verbose=True):
    """
        This function computes the competitions' values by solving a fixed point equation.
        It is based on the most simplest chemical model for the template model: no endonuclease; polymerase and nickase are considered together.
    :param k1, and the others k are the reactions constants
    :param X0: array with initial concentration of the first layer.
    :param masks:
    :param initValue: initial value for cp, the competition over the enzymes. Can be computed with brentq method by fixing the competition over template...
    :param verbose: boolean, display time for computation but also error message if the solver failed.
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
    computedCp = root(allEquilibriumFunc,cps0min,args=(k6,k1M,k3M,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks))
    if verbose:
        print("ended root methods in "+str(time.time()-t0))
        if not computedCp["success"]:
            print(computedCp["message"])
    return computedCp["x"]
def computeEquilibriumValue(cps,k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdT,kdI,TA0,TI0,E0,X0,masks,observed=None):
    """
        Given cp, compute the equilibrium values for all nodes in solutions.
    :param cps: Value for the competitions over enzyme and template obtained by fixed point strategy.
    :param observed: tuple, default to None. If provided, we only give the value for the species at the position observed.
    :return: equilibrium value: equilibrium value for species in masks.
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


def networkEquilibrium(X0,constantArray,masks,chemicalModel="templateModel",verbose=True):
    """
        Compute the equilibrium values for the node of a network, using either an analytical fit or a fixed-point strategy.
    :param constantArray: constant to provide.
    :param masks: list of array provide the connectivity by link with [input,output] in {1 (activation),0 (no link), -1 (inhbition)}
    :param X0: initial value for the first layer
    :param chemicalModel: string. model to solver for. default to templateModel simple version.
    :param verbose: boolean, for solving time an additional informations.
    :return:
    """
    if chemicalModel=="templateModel":
        try:
            assert len(constantArray)== 14
        except:
            raise Exception("tempalte model needs 11 constant, initial value for Tp and Tpd and initial value for E0")
        k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdI,kdT,TA0,TI0,E0 = constantArray
        cpApproxim = computeCPonly(k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdI,kdT,TA0,TI0,E0,X0,masks,fittedValue=None,verbose=verbose)
        cps=np.concatenate(([cpApproxim],np.zeros(np.sum([m.shape[0]*m.shape[1] for m in masks]))+1))
        #cps = computeCPs(k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdT,kdI,TA0,TI0,E0,X0,masks,initValue=cpApproxim,verbose=verbose)
        return computeEquilibriumValue(cps,k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdT,kdI,TA0,TI0,E0,X0,masks,observed=None)
    raise Exception("currently supported model are: templateModel.")