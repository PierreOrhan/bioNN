"""
    Provide function tensorflow's graph compatible to search for fix point in various chemical systems.
        implemented: #TODO write implemented stuff
"""
import tensorflow as tf
from simulOfBioNN.nnUtils.chemTemplateNN.tensorflowFixedPoint import brentq

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
    olderX = None
    print("masks0 on graph:"+str(masks[0].to_tensor().graph))
    print("mask0 :"+str(masks[0].to_tensor()))
    print("in obtainBornsup, masks shape are:"+str(masks.shape[0])+" and type "+str(type(masks.shape[0])))
    print("and from layer: "+str(masks[0].shape)+" with to tensor:"+str(masks[0].to_tensor().shape))
    print(" by the way the layer is "+str(masks[0]))

    for layeridx in tf.range(masks.shape[0]):
        layer = masks[layeridx].to_tensor()
        print("looping , layer has shape: "+str(layer.shape))
        layerEq = tf.TensorArray(dtype=tf.float32,size=layer.shape)
        if(tf.equal(layeridx,0)):
            for inpIdx in tf.range(layer.shape[1]):
                #compute of Kactivs,Kinhibs;
                Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx].to_tensor()[:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx].to_tensor()[:,inpIdx],0)
                #compute of "weights": sum of kactivs and kinhibs
                w_inpIdx = tf.keras.backend.sum(Kactivs)+tf.keras.backend.sum(Kinhibs)
                max_cp += w_inpIdx*X0[inpIdx]
                # saving values
            olderX = X0
        else:
            for inpIdx in tf.range(layer.shape[1]):
                #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                #Terms for the previous layers
                CactivsOld = tf.where(masks[layeridx-1,inpIdx,:]>0,Cactiv0[layeridx-1,inpIdx],0)
                CinhibsOld = tf.where(masks[layeridx-1,inpIdx,:]<0,Cinhib0[layeridx-1,inpIdx],0)
                #computing of new equilibrium
                x_eq = tf.keras.backend.sum(CactivsOld*olderX/kdT[layeridx-1,inpIdx])
                layerEq.write(inpIdx,x_eq)
                #compute of Kactivs,Kinhibs, for the current layer:
                Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx].to_tensor()[:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx].to_tensor()[:,inpIdx],0)
                #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                firstComplex = tf.keras.backend.sum(tf.where(layer[:,inpIdx]>0,Kactivs*x_eq,tf.where(layer[:,inpIdx]<0,Kinhibs*x_eq,0)))
                #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                Inhib2 = tf.keras.backend.sum(CinhibsOld*olderX/(kdT[layeridx-1,inpIdx]*k6[layeridx-1,inpIdx]))
                max_cp += firstComplex + tf.reshape(Inhib2/E0*x_eq,shape=()) #strange bug here
            olderX= layerEq.stack()
    #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layers
    for outputsIdx in tf.range(masks[-1].shape[0]):
        Cinhibs = tf.where(masks[-1,outputsIdx,:]<0,Cinhib0[-1,outputsIdx],0)
        Cactivs = tf.where(masks[-1,outputsIdx,:]>0,Cactiv0[-1,outputsIdx],0)
        x_eq = tf.keras.backend.sum(Cactivs*olderX/(kdI[-1,outputsIdx]))
        Inhib2 = tf.keras.backend.sum(Cinhibs*olderX/(kdT[-1,outputsIdx]*k6[-1,outputsIdx]))
        max_cp += tf.reshape(Inhib2/E0*x_eq,shape=())
    return max_cp

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
    """
        We would like to store at each loop turn the result of the previous layer so we can use it at the next iteration.
        BUT the loop is built using tf.range, executing much faster but providing an indicator that is a tensor.
        We suggest two solutions in tensorflow:
            1) Simply: use a python variable that changes at each loop turn.
            2) Other possibility: 
                use a tensor of maximal size, and masks it at each loop turn using a mask.
                The mask should thus be provided as a variable.
        Indeed tensor does not support item assignment in tensorflow (and ragged tensor) 
        for example: e = tf.zeros((10,10))
                     e[10,:] = 4
        will fail!! Normally one would use a variable to compensate for this.
        BUT variable are not available for ragged tensor.
    """
    olderX = None
    # olderX = tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(m.shape[1])) for m in masks])
    for layeridx in tf.range(len(masks)):
        layer = masks[layeridx]
        layerEq = tf.TensorArray(dtype=tf.float32,size=layer.shape)
        if(layeridx==0):
            for inpIdx in tf.range(layer.shape[1]):
                #compute of Kactivs,Kinhibs;
                # In ragged tensor as Kactiv0 is, we cannot use slice on inner dimension, thus we are here required to convert it to a tensor.
                Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx].to_tensor()[:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx].to_tensor()[:,inpIdx],0)
                #compute of "weights": sum of kactivs and kinhibs
                w_inpIdx = tf.keras.backend.sum(Kactivs)+tf.keras.backend.sum(Kinhibs)
                x_eq = X0[inpIdx]/(1+E0*w_inpIdx/cp)
                # update for fixed point:
                new_cp+=w_inpIdx*x_eq
                # saving values
                layerEq.write(inpIdx,x_eq)
            olderX = layerEq.stack()
        else:
            for inpIdx in tf.range(layer.shape[1]):

                #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                #Terms for the previous layers
                CactivsOld = tf.where(masks[layeridx-1,inpIdx,:]>0,Cactiv0[layeridx-1,inpIdx],0)
                CinhibsOld = tf.where(masks[layeridx-1,inpIdx,:]<0,Cinhib0[layeridx-1,inpIdx],0)
                Inhib = tf.keras.backend.sum(CinhibsOld*olderX/kdT[layeridx-1,inpIdx])
                #computing of new equilibrium
                x_eq = tf.keras.backend.sum(CactivsOld*olderX/(kdI[layeridx-1,inpIdx]*cp+Inhib/cp))
                layerEq.write(inpIdx,x_eq)
                #compute of Kactivs,Kinhibs, for the current layer:
                Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx].to_tensor()[:,inpIdx],0) #This is also a matrix element wise multiplication
                Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx].to_tensor()[:,inpIdx],0)
                #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                firstComplex = tf.keras.backend.sum(tf.where(layer[:,inpIdx]>0,Kactivs*x_eq,tf.where(layer[:,inpIdx]<0,Kinhibs*x_eq,0)))
                #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                Inhib2 = tf.keras.backend.sum(CinhibsOld*olderX/(kdT[layeridx-1,inpIdx]*k6[layeridx-1,inpIdx]))
                new_cp+=firstComplex + tf.reshape(Inhib2/(E0*cp)*x_eq,shape=())
            olderX = layerEq.stack()
    #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layers
    for outputsIdx in tf.range(masks[-1].shape[0]):
        Cinhibs = tf.where(masks[-1,outputsIdx,:]<0,Cinhib0[-1,outputsIdx],0)
        Cactivs = tf.where(masks[-1,outputsIdx,:]>0,Cactiv0[-1,outputsIdx],0)

        Inhib = tf.keras.backend.sum(Cinhibs*olderX/kdT[-1,outputsIdx])
        x_eq = tf.keras.backend.sum(Cactivs*olderX/(kdI[-1,outputsIdx]*cp+Inhib/cp))
        Inhib2 = tf.keras.backend.sum(Cinhibs*olderX/(kdT[-1,outputsIdx]*k6[-1,outputsIdx]))
        new_cp += tf.reshape(Inhib2/(E0*cp)*x_eq,shape=())
    return cp - new_cp

def computeCPonly(vk1,vk1n,vk2,vk3,vk3n,vk4,vk5,vk5n,vk6,vkdI,vkdT,vTA0,vTI0,E0,X0,vmasks):
    """
         This function computes the competition's value by solving a fixed point equation.
         It is based on the most simplest chemical model for the template model: no endonuclease; polymerase and nickase are considered together.
     :param k1, and the others k are the reactions constants
     :param X0: array with initial concentration of the first layer.
     :param masks:
     :param fittedValue: value obtained with an analytical model that is an upper bound on the real value. if not provided we use 10**6
     :return:
     """

    # We load the tensor from the ragged Variable:
    k1 = vk1.getRagged()
    k1n = vk1n.getRagged()
    k2 = vk2.getRagged()
    k3 = vk3.getRagged()
    k3n = vk3n.getRagged()
    k4 = vk4.getRagged()
    k5 = vk5.getRagged()
    k5n = vk5n.getRagged()
    k6 = vk6.getRagged()
    kdI = vkdI.getRagged()
    kdT = vkdT.getRagged()
    TA0 = vTA0.getRagged()
    TI0 = vTI0.getRagged()
    masks = vmasks.getRagged()

    # Computation of the different constants that are required.
    k1M = k1/(k1n+k2)
    k3M = k3/(k3n+k4)
    k5M = k5/(k5n+k6)
    Kactiv0 = k1M*TA0 # element to element product
    Kinhib0 = k3M*TI0
    Cactiv0 = k2*k1M*TA0*E0
    # the shape 0 is defined in our ragged tensors whereas the shape 1 and 2 aren't

    Cinhib0 = k6*k5M*k4*k3M*TI0*E0*E0
    newkdT = kdT
    newkdI = kdI

    # Cinhib0 = tf.stack(tf.map_fn(lambda l :tf.stack([k6[l]*k5M[l]]*(k4[l].to_tensor().shape[1]),axis=1) ,tf.range(k3M.shape[0])))*k4*k3M*TI0*E0*E0
    # newkdT = tf.stack([tf.stack([kdT[l]]*(k4[l].to_tensor().shape[1]),axis=1) for l in tf.range(k3M.shape[0])])
    # newkdI = tf.stack([tf.stack([kdI[l]]*(k4[l].to_tensor().shape[1]),axis=1) for l in tf.range(k3M.shape[0])])

    # old: with lists
    # k1M = [k1[l]/(k1n[l]+k2[l]) for l in range(len(k1))] #(matrix operations)
    # k3M = [k3[l]/(k3n[l]+k4[l]) for l in range(len(k3))]
    # k5M = [k5[l]/(k5n[l]+k6[l]) for l in range(len(k5))]
    # Kactiv0 = [k1M[l]*TA0[l] for l in range(len(k1M))] # element to element product
    # Kinhib0 = [k3M[l]*TI0[l] for l in range(len(k3M))]
    # Cactiv0 = [k2[l]*k1M[l]*TA0[l]*E0 for l in range(len(k1M))]
    # Cinhib0 =[tf.stack([k6[l]*k5M[l]]*(k4[l].shape[1]),axis=1)*k4[l]*k3M[l]*TI0[l]*E0*E0 for l in range(len(k3M))]
    # newkdT = [tf.stack([kdT[l]]*(k4[l].shape[1]),axis=1) for l in range(len(k3M))]
    # newkdI = [tf.stack([kdI[l]]*(k4[l].shape[1]),axis=1) for l in range(len(k3M))]

    print("X0 shape"+str(X0.shape))
    tf.print("X0 shape"+str(X0.shape))

    print("computeCPonly masks shape"+str(masks.shape))
    print("computeCPonly masks0 to tensor shape"+str(masks[0].to_tensor().shape))
    print("computeCPonly masks0 on graph:"+str(masks[0].to_tensor().graph))

    cp0max=obtainBornSup(k6,newkdT,newkdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks)# we start from an upper bound
    print("cp0max: "+str(cp0max))
    tf.print("cp0max: "+str(cp0max))
    computedCp = brentq(cpEquilibriumFunc,1.0,cp0max,args=(k6,newkdT,newkdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks))
    print("computedCp: "+str(computedCp))
    tf.print("computedCp: "+str(computedCp))
    return computedCp


