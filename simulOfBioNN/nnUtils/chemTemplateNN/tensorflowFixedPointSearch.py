"""
    Provide function tensorflow's graph compatible to search for fix point in various chemical systems.
        implemented: #TODO write implemented stuff
"""
import tensorflow as tf
from simulOfBioNN.nnUtils.chemTemplateNN.tensorflowFixedPoint import brentq

@tf.function
def obtainBornSup(k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks):
    """
        Given approximate only for the enzyme competition term (cp), we compute the next approximate using G.
        The real value for the competitions term verify cp = G(cp,initialConditions)
    :param cp: float, competition over tempalte.
    :param E0: float, initial concentration in the enzyme
    :param X0: nbrInputs array, contains initial value for the inputs
    :param masks: masks giving the network topology. list of float32 tensor, we defined it with the shape [outputsNodes,inputsNodes]
    :return:
    """
    max_cp = tf.zeros(1,dtype=tf.float32)+1
    olderX = [tf.zeros(m.shape[1]) for m in masks]
    for layeridx in range(len(masks)):
        layer = masks[layeridx]
        layerEq = tf.TensorArray(dtype=tf.float32,size=layer.shape)
        if(layeridx==0):
            for inpIdx in range(layer.shape[1]):
                #compute of Kactivs,Kinhibs;
                Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                #compute of "weights": sum of kactivs and kinhibs
                w_inpIdx = tf.keras.backend.sum(Kactivs)+tf.keras.backend.sum(Kinhibs)
                max_cp += w_inpIdx*X0[inpIdx]
                # saving values
            olderX[layeridx] = X0
        else:
            for inpIdx in range(layer.shape[1]):
                #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                #Terms for the previous layers
                CactivsOld = tf.where(masks[layeridx-1][inpIdx,:]>0,Cactiv0[layeridx-1][inpIdx],0)
                CinhibsOld = tf.where(masks[layeridx-1][inpIdx,:]<0,Cinhib0[layeridx-1][inpIdx],0)
                #computing of new equilibrium
                x_eq = tf.keras.backend.sum(CactivsOld*olderX[layeridx-1]/kdT[layeridx-1][inpIdx])
                layerEq.write(inpIdx,x_eq)
                #compute of Kactivs,Kinhibs, for the current layer:
                Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                firstComplex = tf.keras.backend.sum(tf.where(layer[:,inpIdx]>0,Kactivs*x_eq,tf.where(layer[:,inpIdx]<0,Kinhibs*x_eq,0)))
                #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                Inhib2 = tf.keras.backend.sum(CinhibsOld*olderX[layeridx-1]/(kdT[layeridx-1][inpIdx]*k6[layeridx-1][inpIdx]))
                max_cp += firstComplex + tf.reshape(Inhib2/E0*x_eq,shape=()) #strange bug here
            olderX[layeridx] = layerEq.stack()
    #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layers
    for outputsIdx in range(masks[-1].shape[0]):
        Cinhibs = tf.where(masks[-1][outputsIdx,:]<0,Cinhib0[-1][outputsIdx],0)
        Cactivs = tf.where(masks[-1][outputsIdx,:]>0,Cactiv0[-1][outputsIdx],0)
        x_eq = tf.keras.backend.sum(Cactivs*olderX[-1]/(kdI[-1][outputsIdx]))
        Inhib2 = tf.keras.backend.sum(Cinhibs*olderX[-1]/(kdT[-1][outputsIdx]*k6[-1][outputsIdx]))
        max_cp += tf.reshape(Inhib2/E0*x_eq,shape=())
    return max_cp

@tf.function
def cpEquilibriumFunc(cp,args):
    """
        Given approximate only for the enzyme competition term (cp), we compute the next approximate using G.
        The real value for the competitions term verify cp = G(cp,initialConditions)
    :param cp: float, competition over tempalte.
    :param E0: float, initial concentration in the enzyme
    :param X0: nbrInputs array, contains initial value for the inputs
    :param masks: masks giving the network topology.
    :return:
    """
    k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks = args
    new_cp = tf.zeros(1,dtype=tf.float32)+1
    olderX = [tf.zeros(m.shape[1]) for m in masks]
    for layeridx in tf.range(len(masks)):
        layer = masks[layeridx]
        layerEq = tf.TensorArray(dtype=tf.float32,size=layer.shape)
        if(layeridx==0):
            for inpIdx in tf.range(layer.shape[1]):
                #compute of Kactivs,Kinhibs;
                Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                #compute of "weights": sum of kactivs and kinhibs
                w_inpIdx = tf.keras.backend.sum(Kactivs)+tf.keras.backend.sum(Kinhibs)
                x_eq = X0[inpIdx]/(1+E0*w_inpIdx/cp)
                # update for fixed point:
                new_cp+=w_inpIdx*x_eq
                # saving values
                layerEq.write(inpIdx,x_eq)
            olderX[layeridx] = layerEq.stack()
        else:
            for inpIdx in tf.range(layer.shape[1]):

                #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                #Terms for the previous layers
                CactivsOld = tf.where(masks[layeridx-1][inpIdx,:]>0,Cactiv0[layeridx-1][inpIdx],0)
                CinhibsOld = tf.where(masks[layeridx-1][inpIdx,:]<0,Cinhib0[layeridx-1][inpIdx],0)
                Inhib = tf.keras.backend.sum(CinhibsOld*olderX[layeridx-1]/kdT[layeridx-1][inpIdx])
                #computing of new equilibrium
                x_eq = tf.keras.backend.sum(CactivsOld*olderX[layeridx-1]/(kdI[layeridx-1][inpIdx]*cp+Inhib/cp))
                layerEq.write(inpIdx,x_eq)
                #compute of Kactivs,Kinhibs, for the current layer:
                Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                firstComplex = tf.keras.backend.sum(tf.where(layer[:,inpIdx]>0,Kactivs*x_eq,tf.where(layer[:,inpIdx]<0,Kinhibs*x_eq,0)))
                #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                Inhib2 = tf.keras.backend.sum(CinhibsOld*olderX[layeridx-1]/(kdT[layeridx-1][inpIdx]*k6[layeridx-1][inpIdx]))
                new_cp+=firstComplex + tf.reshape(Inhib2/(E0*cp)*x_eq,shape=())
            olderX[layeridx] = layerEq.stack()
    #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layers
    for outputsIdx in tf.range(masks[-1].shape[0]):
        Cinhibs = tf.where(masks[-1][outputsIdx,:]<0,Cinhib0[-1][outputsIdx],0)
        Cactivs = tf.where(masks[-1][outputsIdx,:]>0,Cactiv0[-1][outputsIdx],0)

        Inhib = tf.keras.backend.sum(Cinhibs*olderX[-1]/kdT[-1][outputsIdx])
        x_eq = tf.keras.backend.sum(Cactivs*olderX[-1]/(kdI[-1][outputsIdx]*cp+Inhib/cp))
        Inhib2 = tf.keras.backend.sum(Cinhibs*olderX[-1]/(kdT[-1][outputsIdx]*k6[-1][outputsIdx]))
        new_cp += tf.reshape(Inhib2/(E0*cp)*x_eq,shape=())
    return cp - new_cp

@tf.function
def computeCPonly(k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdI,kdT,TA0,TI0,E0,X0,masks):
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
    Cinhib0 =[tf.stack([k6[l]*k5M[l]]*(k4[l].shape[1]),axis=1)*k4[l]*k3M[l]*TI0[l]*E0*E0 for l in range(len(k3M))]
    newkdT = [tf.stack([kdT[l]]*(k4[l].shape[1]),axis=1) for l in range(len(k3M))]
    newkdI = [tf.stack([kdI[l]]*(k4[l].shape[1]),axis=1) for l in range(len(k3M))]

    print("X0 shape"+str(X0.shape))
    tf.print("X0 shape"+str(X0.shape))
    cp0max=obtainBornSup(k6,newkdT,newkdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks)# we start from an upper bound
    print("cp0max: "+str(cp0max))
    tf.print("cp0max: "+str(cp0max))
    computedCp = brentq(cpEquilibriumFunc,1.0,cp0max,args=(k6,newkdT,newkdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks))
    print("computedCp: "+str(computedCp))
    tf.print("computedCp: "+str(computedCp))
    return computedCp