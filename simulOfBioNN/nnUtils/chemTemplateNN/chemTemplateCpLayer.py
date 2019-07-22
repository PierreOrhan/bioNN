from tensorflow.python.keras.layers import Dense
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
import tensorflow as tf
from simulOfBioNN.nnUtils.chemTemplateNN.tensorflowFixedPointSearch import computeCPonly


class chemTemplateCpLayer(tf.keras.layers.Layer):
    """
        A layer that given the inputs should compute the competition and return it as a Tensor.
    """
    def __init__(self, deviceName, **kwargs):
        """
            Creates the VariableRaggedTensor object which store variable corresponding to ragged tensors.
            :param deviceName: device to use for computation
            :param kwargs:
        """
        super(chemTemplateCpLayer, self).__init__(**kwargs)
        self.deviceName=deviceName #the device on which the main operations will be conducted (forward and backward propagations)
        self.k1 = VariableRaggedTensor()
        self.k1n = VariableRaggedTensor()
        self.k2 = VariableRaggedTensor()
        self.k3 = VariableRaggedTensor()
        self.k3n = VariableRaggedTensor()
        self.k4 = VariableRaggedTensor()

        self.k5 = VariableRaggedTensor()
        self.k5n = VariableRaggedTensor()
        self.k6 = VariableRaggedTensor()
        self.kdI = VariableRaggedTensor()
        self.kdT = VariableRaggedTensor()

        self.TA0 = VariableRaggedTensor()
        self.TI0 = VariableRaggedTensor()
        self.Cinhib0 = VariableRaggedTensor()
        self.masks = VariableRaggedTensor(displayInfo=True)

        self.maxIndexToLookAt= tf.Variable(tf.constant(0,dtype=tf.int32),trainable=False) # variable needed in the tf.function... because of strange behavior with tf.TensorArray

    def add_weight(self,**kwargs):
        return super(chemTemplateCpLayer,self).add_weight(**kwargs)

    def build(self, input_shape,layerList):
        # We just change the way bias is added and remove it from trainable variable!
        input_shape = tf.TensorShape(input_shape)

        modelsConstantShape =[(layerList[0].units,input_shape[-1])]+[(layerList[idx+1].units,l.units) for idx,l in enumerate(layerList[:-1])]
        modelsOutputConstantShape = [l.units for l in layerList]
        self.k1.add_weight("k1",self,tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(layerList)]))
        self.k1n.add_weight("k1n",self,tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(layerList)]))
        self.k2.add_weight("k2",self,tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(layerList)]))
        self.k3.add_weight("k3",self,tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(layerList)]))
        self.k3n.add_weight("k3n",self,tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(layerList)]))
        self.k4.add_weight("k4",self,tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(layerList)]))
        self.TA0.add_weight("TA0",self,tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(layerList)]))
        self.TI0.add_weight("TI0",self,tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(layerList)]))
        self.Cinhib0.add_weight("Cinhib0",self,tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(layerList)]))

        self.k5.add_weight("k5",self,tf.RaggedTensor.from_row_lengths(values= tf.zeros(sum(modelsOutputConstantShape),dtype = tf.float32) ,row_lengths = modelsOutputConstantShape))
        self.k5n.add_weight("k5n",self,tf.RaggedTensor.from_row_lengths(values= tf.zeros(sum(modelsOutputConstantShape),dtype = tf.float32) ,row_lengths = modelsOutputConstantShape))
        self.k6.add_weight("k6",self,tf.RaggedTensor.from_row_lengths(values= tf.zeros(sum(modelsOutputConstantShape),dtype = tf.float32) ,row_lengths = modelsOutputConstantShape))
        self.kdI.add_weight("kdI",self,tf.RaggedTensor.from_row_lengths(values= tf.zeros(sum(modelsOutputConstantShape),dtype = tf.float32) ,row_lengths = modelsOutputConstantShape))
        self.kdT.add_weight("kdT",self,tf.RaggedTensor.from_row_lengths(values= tf.zeros(sum(modelsOutputConstantShape),dtype = tf.float32) ,row_lengths = modelsOutputConstantShape))

        self.E0 = tf.Variable(tf.constant(1,dtype=tf.float32),trainable=False,dtype=tf.float32)

        masks2 = tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(layerList)])
        self.masks.add_weight("masks",self,masks2)

        super(chemTemplateCpLayer,self).build(input_shape)
        print("cp layer successfully built")

    def assignConstantFromLayers(self,layerList):
        self.k1.assign(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k1)) for l in layerList])) #transpose enable to switch the shape format here as previously explained.
        self.k1n.assign(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k1n)) for l in layerList]))
        self.k2.assign(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k2)) for l in layerList]))
        self.k3.assign(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k3)) for l in layerList]))
        self.k3n.assign(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k3n)) for l in layerList]))
        self.k4.assign(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k4)) for l in layerList]))

        # 1d-tensors
        self.k5.assign(tf.RaggedTensor.from_row_lengths(values= tf.concat([l.k5 for l in layerList],axis=0),row_lengths=[l.k5.shape[0] for l in layerList]))
        self.k5n.assign(tf.RaggedTensor.from_row_lengths(values= tf.concat([l.k5n for l in layerList],axis=0),row_lengths=[l.k5n.shape[0] for l in layerList]))
        self.k6.assign(tf.RaggedTensor.from_row_lengths(values= tf.concat([l.k6 for l in layerList],axis=0),row_lengths=[l.k6.shape[0] for l in layerList]))
        self.kdI.assign(tf.RaggedTensor.from_row_lengths(values= tf.concat([l.kdI for l in layerList],axis=0),row_lengths=[l.kdI.shape[0] for l in layerList]))
        self.kdT.assign(tf.RaggedTensor.from_row_lengths(values= tf.concat([l.kdT for l in layerList],axis=0),row_lengths=[l.kdT.shape[0] for l in layerList]))

        self.TA0.assign(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.TA0)) for l in layerList]))
        self.TI0.assign(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.TI0)) for l in layerList]))
        self.Cinhib0.assign(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.Cinhib)) for l in layerList]))

        self.E0.assign(tf.constant(layerList[0].E0.read_value()))

    def assignMasksFromLayers(self,layerList):
        self.masks.assign(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.get_mask())) for l in layerList]))

    def lambdaComputeCpOnly(self,input):
        return self.computeCPonly(self.k1,self.k1n,self.k2,self.k3,self.k3n,self.k4,self.k5,self.k5n,self.k6,self.kdI,
                                 self.kdT,self.TA0,self.TI0,self.Cinhib0,self.E0,input,self.masks)
    @tf.function
    def __call__(self, inputs):
        #cps = tf.TensorArray(dtype=tf.float32,size=tf.shape(inputs)[0],dynamic_size=False,infer_shape=False)
        # for idx in tf.range(tf.shape(inputs)[0]):
        #     e = self.computeCPonly(self.k1,self.k1n,self.k2,self.k3,self.k3n,self.k4,self.k5,self.k5n,self.k6,self.kdI,
        #                            self.kdT,self.TA0,self.TI0,self.Cinhib0,self.E0,inputs[idx],self.masks)
        #     tf.print("compute cp is: "+str(e))
        #     print("compute cp is "+str(e))
        #     tf.print("compute cps is: "+str(cps))
        #     print("compute cps is "+str(cps))
        #     cps.write(idx,e)
        # print("the input is "+str(inputs))
        # print("the cps size is "+str(cps.size()))
        # print("the input size is "+str(tf.shape(inputs)[0]))
        # #gatheredCps = cps.gather(indices=tf.range(cps.size()))
        # gatheredCps = cps.stack()
        # print("gathered cps is "+str(gatheredCps))

        gatheredCps = tf.map_fn(self.lambdaComputeCpOnly,inputs)
        return gatheredCps

    @tf.function
    def obtainBornSup(self,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks):
        """
            Given approximate only for the enzyme competition term (cp), we compute the next approximate using G.
            The real value for the competitions term verify cp = G(cp,initialConditions)
        :param cp: float, competition over template.
        :param E0: float, initial concentration in the enzyme
        :param X0: nbrInputs array, contains initial value for the inputs
        :param masks: masks giving the network topology. list of float32 tensor, we defined it with the shape [outputsNodes,inputsNodes]
        :return:
        """
        max_cp = tf.fill([1],1.)
        olderX = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        for layeridx in tf.range(tf.shape(masks.to_tensor())[0]):
            layer = masks[layeridx].to_tensor()

            layerEq = tf.TensorArray(dtype=tf.float32,size=tf.shape(layer)[1])
            if(tf.equal(layeridx,0)):
                for inpIdx in tf.range(tf.shape(layer)[1]):
                    #compute of Kactivs,Kinhibs;
                    Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx].to_tensor()[:,inpIdx],0) #This is also a matrix element wise multiplication
                    Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx].to_tensor()[:,inpIdx],0)
                    #compute of "weights": sum of kactivs and kinhibs
                    w_inpIdx = tf.keras.backend.sum(Kactivs)+tf.keras.backend.sum(Kinhibs)
                    max_cp += w_inpIdx*X0[inpIdx]
                    # saving values
                olderX = olderX.scatter(indices=tf.range(tf.shape(X0)[0]),value=X0)
                self.maxIndexToLookAt.assign(tf.shape(X0)[0])
            else:
                stackOlderX = olderX.gather(tf.range(self.maxIndexToLookAt))
                for inpIdx in tf.range(tf.shape(layer)[1]):
                    #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                    #Terms for the previous layers
                    CactivsOld = tf.where(masks[layeridx-1,inpIdx,:]>0,Cactiv0[layeridx-1,inpIdx],0)
                    CinhibsOld = tf.where(masks[layeridx-1,inpIdx,:]<0,Cinhib0[layeridx-1,inpIdx],0)
                    #computing of new equilibrium
                    x_eq = tf.fill([1],tf.tensordot(stackOlderX,CactivsOld,axes=[[0],[0]])/kdT[layeridx-1,inpIdx])
                    layerEq.write(inpIdx,x_eq)
                    #compute of Kactivs,Kinhibs, for the current layer:
                    Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx].to_tensor()[:,inpIdx],0)
                    Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx].to_tensor()[:,inpIdx],0)
                    #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                    firstComplex = tf.fill([1],tf.keras.backend.sum(tf.where(layer[:,inpIdx]>0,Kactivs*x_eq,tf.where(layer[:,inpIdx]<0,Kinhibs*x_eq,0))))
                    #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                    Inhib2 = tf.fill([1],tf.tensordot(CinhibsOld,stackOlderX,axes=[[0],[0]])/(kdT[layeridx-1,inpIdx]*k6[layeridx-1,inpIdx]))
                    max_cp +=  Inhib2/E0*x_eq + firstComplex
                verticalStack = layerEq.stack()
                layerEqStack = tf.transpose(verticalStack)[0]
                olderX = olderX.scatter(indices=tf.range(tf.shape(layerEqStack)[0]),value=layerEqStack)
                self.maxIndexToLookAt.assign(tf.shape(layerEqStack)[0])
        #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layers
        stackOlderX = olderX.gather(tf.range(self.maxIndexToLookAt))
        for outputsIdx in tf.range(tf.shape(masks[-1].to_tensor())[0]):
            Cinhibs = tf.where(masks[-1,outputsIdx,:]<0,Cinhib0[-1,outputsIdx],0)
            Cactivs = tf.where(masks[-1,outputsIdx,:]>0,Cactiv0[-1,outputsIdx],0)
            x_eq = tf.fill([1],tf.tensordot(Cactivs,stackOlderX,axes=[[0],[0]])/(kdI[-1,outputsIdx]))
            Inhib2 = tf.fill([1],tf.tensordot(Cinhibs,stackOlderX,axes=[[0],[0]])/(kdT[-1,outputsIdx]*k6[-1,outputsIdx]))
            max_cp += Inhib2/E0*x_eq
        return max_cp

    @tf.function
    def cpEquilibriumFunc(self,cp,args):
        """
            Given approximate only for the enzyme competition term (cp), we compute the next approximate using G.
            The real value for the competitions term verify cp = G(cp,initialConditions)
        :return:
        """
        k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks = args
        new_cp = tf.fill([1],1.)
        """
            We would like to store at each loop turn the result of the previous layer so we can use it at the next iteration.
            BUT the loop is built using tf.range, executing much faster but providing an indicator that is a tensor.
            
            We then need to us the TensorArray with a dynamic size!
            To gather this array:
                we first tried to store the shape with a TensorArray of size 1.
                    But at each loop, the value would be reset to 0, especially if we have a forked ( if/else).
                Thus we switched for a class Variable, and used assign method. 
                
            Indeed tensor does not support item assignment in tensorflow (and ragged tensor) 
            for example: e = tf.zeros((10,10))
                         e[10,:] = 4
            will fail!! Normally one would use a variable to compensate for this.
            BUT variable are not available for ragged tensor.
        """
        olderX = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        for layeridx in tf.range(tf.shape(masks.to_tensor())[0]):
            layer = masks[layeridx].to_tensor()
            # print("looping cpEqui , layer has shape: "+str(layer.shape))
            layerEq = tf.TensorArray(dtype=tf.float32,size=tf.shape(layer)[1])
            if(tf.equal(layeridx,0)):
                for inpIdx in tf.range(tf.shape(layer)[1]):
                    #compute of Kactivs,Kinhibs;
                    # In ragged tensor as Kactiv0 is, we cannot use slice on inner dimension, thus we are here required to convert it to a tensor.
                    Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx].to_tensor()[:,inpIdx],0)
                    Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx].to_tensor()[:,inpIdx],0)
                    #compute of "weights": sum of kactivs and kinhibs
                    w_inpIdx = tf.keras.backend.sum(Kactivs)+tf.keras.backend.sum(Kinhibs)
                    x_eq = X0[inpIdx]/(1+E0*w_inpIdx/cp)
                    # update for fixed point:
                    new_cp += w_inpIdx*x_eq
                    # saving values
                    layerEq.write(inpIdx,x_eq)
                verticalStack = layerEq.stack()
                layerEqStack = tf.transpose(verticalStack)[0]
                olderX = olderX.scatter(indices=tf.range(tf.shape(layerEqStack)[0]),value=layerEqStack)
                self.maxIndexToLookAt.assign(tf.shape(layerEqStack)[0])
            else:
                stackOlderX = olderX.gather(tf.range(self.maxIndexToLookAt))
                for inpIdx in tf.range(tf.shape(layer)[1]):
                    #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                    #Terms for the previous layers
                    CactivsOld = tf.where(masks[layeridx-1,inpIdx,:]>0,Cactiv0[layeridx-1,inpIdx],0)
                    CinhibsOld = tf.where(masks[layeridx-1,inpIdx,:]<0,Cinhib0[layeridx-1,inpIdx],0)
                    Inhib = tf.tensordot(CinhibsOld,stackOlderX,axes=[[0],[0]])/kdT[layeridx-1,inpIdx]
                    #computing of new equilibrium
                    x_eq = tf.tensordot(CactivsOld,stackOlderX,axes=[[0],[0]])/(kdI[layeridx-1,inpIdx]*cp+Inhib/cp)
                    layerEq.write(inpIdx,x_eq)
                    #compute of Kactivs,Kinhibs, for the current layer:
                    Kactivs = tf.where(layer[:,inpIdx]>0,Kactiv0[layeridx].to_tensor()[:,inpIdx],0)
                    Kinhibs = tf.where(layer[:,inpIdx]<0,Kinhib0[layeridx].to_tensor()[:,inpIdx],0)
                    #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                    firstComplex = tf.fill([1],tf.keras.backend.sum(tf.where(layer[:,inpIdx]>0,Kactivs*x_eq,tf.where(layer[:,inpIdx]<0,Kinhibs*x_eq,0))))
                    #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                    Inhib2 = tf.tensordot(CinhibsOld,stackOlderX,axes=[[0],[0]])/(kdT[layeridx-1,inpIdx]*k6[layeridx-1,inpIdx])
                    new_cp += firstComplex+ Inhib2/(E0*cp)*x_eq
                verticalStack = layerEq.stack()
                layerEqStack = tf.transpose(verticalStack)[0]
                olderX = olderX.scatter(indices=tf.range(tf.shape(layerEqStack)[0]),value=layerEqStack)
                self.maxIndexToLookAt.assign(tf.shape(layerEqStack)[0])
        #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layers
        stackOlderX = olderX.gather(tf.range(self.maxIndexToLookAt))
        for outputsIdx in tf.range(tf.shape(masks[-1].to_tensor())[0]):
            Cinhibs = tf.where(masks[-1,outputsIdx,:]<0,Cinhib0[-1,outputsIdx],0)
            Cactivs = tf.where(masks[-1,outputsIdx,:]>0,Cactiv0[-1,outputsIdx],0)

            Inhib = tf.tensordot(Cinhibs,stackOlderX,axes=[[0],[0]])/kdT[-1,outputsIdx]
            x_eq = tf.tensordot(Cactivs,stackOlderX,axes=[[0],[0]])/(kdI[-1,outputsIdx]*cp+Inhib/cp)
            Inhib2 = tf.tensordot(Cinhibs,stackOlderX,axes=[[0],[0]])/(kdT[-1,outputsIdx]*k6[-1,outputsIdx])
            new_cp += Inhib2/(E0*cp)*x_eq
        new_cp = new_cp - cp
        return new_cp*(-1)

    @tf.function
    def computeCPonly(self,vk1,vk1n,vk2,vk3,vk3n,vk4,vk5,vk5n,vk6,vkdI,vkdT,vTA0,vTI0,vCInhib0,E0,X0,vmasks):
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
        # k5 = vk5.getRagged()
        # k5n = vk5n.getRagged()
        k6 = vk6.getRagged()
        kdI = vkdI.getRagged()
        kdT = vkdT.getRagged()
        TA0 = vTA0.getRagged()
        TI0 = vTI0.getRagged()
        masks = vmasks.getRagged()
        Cinhib0 = vCInhib0.getRagged()

        # Computation of the different constants that are required.
        k1M = k1/(k1n+k2)
        k3M = k3/(k3n+k4)
        # k5M = k5/(k5n+k6)
        Kactiv0 = k1M*TA0 # element to element product
        Kinhib0 = k3M*TI0
        Cactiv0 = k2*k1M*TA0*E0
        # the shape 0 is defined in our ragged tensors whereas the shape 1 and 2 aren't
        newkdT = kdT
        newkdI = kdI

        cp0max=self.obtainBornSup(k6,newkdT,newkdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks)# we start from an upper bound
        computedCp = self.brentq(self.cpEquilibriumFunc,tf.fill([1],1.),cp0max,args=(k6,newkdT,newkdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks))
        return computedCp

    @tf.function
    def brentq(self, f, xa, xb,args=(),xtol=tf.constant(10**(-12)), rtol=tf.constant(4.4408920985006262*10**(-16)),iter=tf.constant(100)):
        xpre = tf.fill([1],0.) + xa
        xcur = tf.fill([1],0.) + xb
        xblk = tf.fill([1],0.)
        fblk = tf.fill([1],0.)
        spre = tf.fill([1],0.)
        scur = tf.fill([1],0.)
        fpre = f(xpre, args)
        fcur = f(xcur, args)
        if tf.math.greater((fpre*fcur)[0],0.):
            return tf.fill([1],0.)
        if tf.equal(fpre[0],0):
            return xpre
        if tf.equal(fcur[0],0):
            return xcur

        for i in tf.range(iter):
            if tf.less((fpre*fcur)[0],0):
                xblk = xpre
                fblk = fpre
                spre = xcur - xpre
                scur = xcur - xpre
            if tf.less(tf.abs(fblk)[0],tf.abs(fcur)[0]):
                xpre = xcur
                xcur = xblk
                xblk = xpre

                fpre = fcur
                fcur = fblk
                fblk = fpre

            delta = (xtol + rtol*tf.abs(xcur))/2
            sbis = (xblk - xcur)/2
            if tf.equal(fcur[0],0) or tf.less(tf.abs(sbis)[0],delta[0]):
                i = iter #BREAK FAILS HERE!!! ==> strange behavior?
            else:
                if tf.greater(tf.abs(spre)[0],delta[0]) and tf.less(tf.abs(fcur)[0],tf.abs(fpre)[0]):
                    if tf.equal(xpre[0],xblk[0]):
                        # /* interpolate */
                        stry = -fcur*(xcur - xpre)/(fcur - fpre)
                    else :
                        # /* extrapolate */
                        dpre = (fpre - fcur)/(xpre - xcur)
                        dblk = (fblk - fcur)/(xblk - xcur)
                        stry = -fcur*(fblk*dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))

                    mymin = tf.minimum(tf.abs(spre), 3*tf.abs(sbis) - delta) #Here would not understand...
                    spre=tf.where(tf.less(2*tf.abs(stry)-mymin,0),scur,sbis)
                    scur=tf.where(tf.less(2*tf.abs(stry)-mymin,0),stry,sbis)
                else:
                    # /* bisect */
                    spre = sbis
                    scur = sbis
                xpre = xcur
                fpre = fcur
                if tf.greater(tf.abs(scur)[0],delta[0]):
                    xcur += scur
                else:
                    if tf.greater(sbis[0],0):
                        xcur += delta
                    else:
                        xcur += -delta
            fcur = f(xcur, args)
        return xcur


class VariableRaggedTensor():
    def __init__(self,displayInfo=False):
        """
            We give a recursive definition for the variable of the ragged tensor.
        :param displayInfo:
        """
        self.displayInfo = displayInfo

    def add_weight(self,name, layer, raggedTensor):
        """
            Add (and creates) the variables to the layer given as input
        :param name: name to use for this RaggedTensor
        :param layer: layer to which the variable will be added
        :param raggedTensor:
        :return:
        """
        self.var_rowsplits = layer.add_weight(name=name+"/var_rowsplits",
                                              dtype=raggedTensor.row_splits.dtype,
                                              shape=raggedTensor.row_splits.shape,
                                              initializer=tf.initializers.Zeros(),
                                              trainable=False)
        self.var_rowsplits.assign(raggedTensor.row_splits)
        if type(raggedTensor.values)==tf.RaggedTensor:
            self.var_values = VariableRaggedTensor(self.displayInfo)
            self.var_values.add_weight(name+"1",layer, raggedTensor.values)
            self.multiDim = True
        else:
            self.var_values = layer.add_weight(name=name+"/var_values",
                                               dtype=raggedTensor.values.dtype,
                                               shape=raggedTensor.values.shape,
                                               initializer=tf.initializers.Zeros(),
                                               trainable=False)
            self.var_values.assign(raggedTensor.values)
            self.multiDim = False
        # self.shape0 = layer.add_weight(name=name+"/shape0",
        #                                dtype=tf.int64,
        #                                 shape=1,
        #                                 initializer=tf.constant_initializer(raggedTensor.shape[0]),
        #                                 trainable=False)

    def assign(self,raggedTensor):
        self.var_rowsplits.assign(raggedTensor.row_splits)
        if self.multiDim:
            self.var_values.assign(raggedTensor.values) # assign of the VariableRaggedTensor object
        else:
            self.var_values.assign(raggedTensor.values) #assign of the Variable object
        # self.shape0.assign(raggedTensor.shape[0])

    @tf.function
    def getRagged(self):
        if self.displayInfo:
            pass
        if self.multiDim:
            return tf.RaggedTensor.from_row_splits(row_splits=self.var_rowsplits.read_value(),values=self.var_values.getRagged())
        return tf.RaggedTensor.from_row_splits(row_splits=self.var_rowsplits.read_value(),values=self.var_values.read_value())
