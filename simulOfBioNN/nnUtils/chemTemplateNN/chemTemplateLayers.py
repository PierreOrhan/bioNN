from tensorflow.python.keras.layers import Dense
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import nn
from simulOfBioNN.nnUtils.clippedSparseBioDenseLayer import weightFixedAndClippedConstraint,sparseInitializer,constant_initializer,layerconstantInitiliaizer




class chemTemplateLayer(Dense):
    """
        Layers with a strange activation function of the form: activator*kernelActivator/(1+activator*kernelActivator+sumALLInhib*kernelInhibitor)
        Adds a constant bias to the input
        :param theta: bias to be added after each multipication
    """
    def __init__(self, deviceName, sparsity=0.9, min=-1, max=1, usingLog=True, **kwargs):
        """

            :param deviceName: device to use for computation
            :param biasValue: either None(random but constant through layer) or 1d-array of size units (nb of output neurons).
            :param sparsity: fraction of weight that should remain at 0
            :param min: min value for weights
            :param max: max value for weights
            :param kwargs:
        """
        super(chemTemplateLayer, self).__init__(**kwargs)
        self.supports_masking = False
        self.sparseInitializer = sparseInitializer(sparsity, minval=min, maxval=max)
        self.deviceName=deviceName #the device on which the main operations will be conducted (forward and backward propagations)

        self.usingLog = usingLog
            # when using log, the layer sill takes value in the normal scale ( so the model should apply some exponential) but gives the log of the equilibrium value

    def build(self, input_shape):
        # We just change the way bias is added and remove it from trainable variable!
        input_shape = tf.TensorShape(input_shape)
        if input_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        if not input_shape.is_fully_defined():
            print("the input shape used for build is not fully defined")

        self.kernel = self.add_weight(
            'kernel',
            shape=[input_shape[-1], self.units],
            initializer=self.sparseInitializer,
            regularizer=self.kernel_regularizer,
            constraint=weightFixedAndClippedConstraint(self.sparseInitializer),
            dtype=self.dtype,
            trainable=True)
        self.bias = None


        variableShape=(input_shape[-1],self.units)
        self.mask = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)

        self.k1 = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.k1n = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.k2 = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.k3 = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.k3n = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.k4 = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.TA0 = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.TI0 = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)

        #only one inhibition by outputs (units):
        self.k5 = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k5n = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k6 = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.kdI = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.kdT = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)


        self.E0 = tf.Variable(tf.constant(1,dtype=tf.float32),trainable=False,dtype=tf.float32)
        self.rescaleFactor = tf.Variable(1,dtype=tf.float32,trainable=False)

        #other intermediates variable:
        self.k1M = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.Cactiv = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.k3M = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.Cinhib = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.Kactiv = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.Kinhib = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)

        self.k5M = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.firstLayerTA0 = tf.Variable(tf.zeros(variableShape[0],dtype=tf.float32),trainable=False)
        self.firstLayerK1M = tf.Variable(tf.zeros(variableShape[0],dtype=tf.float32),trainable=False)
        self.firstLayerkdT = tf.Variable(tf.zeros(variableShape[0],dtype=tf.float32),trainable=False)
        self.firstLayerk2 = tf.Variable(tf.zeros(variableShape[0],dtype=tf.float32),trainable=False)


        self.built = True
        print("Layer successfully built")


    def get_rescaleFactor(self):
        Tminus = tf.cast(tf.fill(self.kernel.shape,-1),dtype=tf.float32)
        Tplus = tf.cast(tf.fill(self.kernel.shape,1),dtype=tf.float32)
        Tzero = tf.cast(tf.fill(self.kernel.shape,0),dtype=tf.float32)
        clippedKernel=tf.where(tf.less(self.kernel,-0.2),Tminus,tf.where(tf.less(0.2,self.kernel),Tplus,Tzero))
        rescaleFactor = tf.keras.backend.sum(tf.where(tf.less(clippedKernel,0.),Tplus,Tzero)) + tf.keras.backend.sum(tf.where(tf.less(0.,clippedKernel),Tplus,Tzero))
        return rescaleFactor

    def get_mask(self):
        Tminus = tf.cast(tf.fill(self.kernel.shape,-1),dtype=tf.float32)
        Tplus = tf.cast(tf.fill(self.kernel.shape,1),dtype=tf.float32)
        Tzero = tf.cast(tf.fill(self.kernel.shape,0),dtype=tf.float32)
        clippedKernel=tf.where(tf.less(self.kernel,-0.2),Tminus,tf.where(tf.less(0.2,self.kernel),Tplus,Tzero))
        return clippedKernel

    def set_mask(self,mask):
        Tminus = tf.cast(tf.fill(self.kernel.shape,-1),dtype=tf.float32)
        Tplus = tf.cast(tf.fill(self.kernel.shape,1),dtype=tf.float32)
        Tzero = tf.cast(tf.fill(self.kernel.shape,0),dtype=tf.float32)

        clippedKernel=tf.where(tf.less(self.kernel,-0.2),Tminus,tf.where(tf.less(0.2,self.kernel),Tplus,Tzero))
        newClippedKernel = tf.where(mask*clippedKernel>0.,clippedKernel,
                                    tf.where(mask*clippedKernel<0.,
                                             (-1)*clippedKernel,
                                             mask))

        newKernel = tf.where(tf.less(self.kernel,-0.2),
                                tf.where(tf.less(newClippedKernel,0.),
                                         self.kernel,
                                         tf.where(tf.less(0.,newClippedKernel),(-1)*self.kernel,0.)), # same or symetric
                                tf.where(tf.less(0.2,self.kernel),
                                         tf.where(tf.less(0.,newClippedKernel),
                                                  self.kernel,
                                                  tf.where(tf.less(newClippedKernel,0.),(-1)*self.kernel,0.)),
                                         tf.where(tf.less(newClippedKernel,0.),
                                                    -1.,
                                                    tf.where(tf.less(0,newClippedKernel),
                                                             1.,
                                                             self.kernel))
                                         ))

        self.kernel.assign(newKernel)
        self.mask.assign(tf.where(tf.less(self.kernel,-0.2),-1.,tf.where(tf.less(0.2,self.kernel),1.,0.)))


    def set_constants(self,constantArray,enzymeInit,activInitC,inhibInitC,computedRescaleFactor):
        """
            Define the ops assigning the values for the network constants.
        :return:
        """
        self.rescaleFactor.assign(computedRescaleFactor)
        enzymeInitTensor = enzymeInit*(computedRescaleFactor**0.5)
        self.k1.assign(tf.fill(self.k1.shape,constantArray[0]))
        self.k1n.assign(tf.fill(self.k1n.shape,constantArray[1]))
        self.k2.assign(tf.fill(self.k2.shape,constantArray[2]))
        self.k3.assign(tf.fill(self.k3.shape,constantArray[3]))
        self.k3n.assign(tf.fill(self.k3n.shape,constantArray[4]))
        self.k4.assign(tf.fill(self.k4.shape,constantArray[5]))
        self.k5.assign(tf.fill(self.k5.shape,constantArray[6]))
        self.k5n.assign(tf.fill(self.k5n.shape,constantArray[7]))
        self.k6.assign(tf.fill(self.k6.shape,constantArray[8]))
        self.kdI.assign(tf.fill(self.kdI.shape,constantArray[9]))
        self.kdT.assign(tf.fill(self.kdT.shape,constantArray[10]))
        self.TA0.assign(tf.fill(self.TA0.shape,activInitC))
        self.TI0.assign(tf.fill(self.TI0.shape,inhibInitC))
        self.E0.assign(tf.constant(enzymeInitTensor,dtype=tf.float32))

        #used in the first layer:
        self.firstLayerTA0.assign(tf.fill(self.firstLayerTA0.shape,activInitC))
        self.firstLayerK1M.assign(tf.fill(self.firstLayerK1M.shape,constantArray[0]/(constantArray[1]+constantArray[2])))
        self.firstLayerkdT.assign(tf.fill(self.firstLayerkdT.shape,constantArray[10]))
        self.firstLayerk2.assign(tf.fill(self.firstLayerk2.shape,constantArray[2]))

        #intermediate values for faster computations:
        self.k1M.assign(self.k1/(self.k1n+self.k2))
        self.Cactiv.assign(self.k2*self.k1M*self.E0*self.TA0)
        self.k5M.assign(self.k5/(self.k5n+self.k6))
        self.k3M.assign(self.k3/(self.k3n+self.k4))
        self.Cinhib.assign(tf.stack([self.k6*self.k5M]*(self.k4.shape[0]),axis=0)*self.k4*self.k3M*self.E0*self.E0*self.TI0)
        self.Kactiv.assign(self.k1M*self.TA0)
        self.Kinhib.assign(self.k3M*self.TI0)

        self.mask.assign(tf.where(tf.less(self.kernel,-0.2),-1.,tf.where(tf.less(0.2,self.kernel),1.,0.)))

        self.cstList = [self.k1,self.k1n,self.k2,self.k3,self.k3n,self.k4,self.k5,self.k5n,self.k6,self.kdI,self.kdT,self.TA0,self.E0,
                        self.k1M,self.Cactiv,self.Cinhib,self.Kactiv,self.Kinhib,self.k5M,self.k3M,self.firstLayerTA0,self.firstLayerK1M,
                        self.firstLayerkdT,self.firstLayerk2]
        self.cstListName = ["self.k1","self.k1n","self.k2","self.k3","self.k3n","self.k4","self.k5","self.k5n","self.k6","self.kdI","self.kdT","self.TA0","self.E0",
                            "self.k1M","self.Cactiv","self.Cinhib","self.Kactiv","self.Kinhib","self.k5M","self.k3M","self.firstLayerTA0","self.firstLayerK1M",
                            "self.firstLayerkdT","self.firstLayerk2"]

    @tf.function
    def rescale(self,rescaleFactor):
        """
            Rescale the enzyme value
        :param rescaleFactor: 0d Tensor: float with the new rescale
        :return:
        """
        self.E0.assign(self.E0.read_value()*(rescaleFactor**0.5)/(self.rescaleFactor.read_value()**0.5))
        self.Cactiv.assign(self.k2*self.k1M*self.E0*self.TA0)
        self.Cinhib.assign(tf.stack([self.k6*self.k5M]*(self.k4.shape[0]),axis=0)*self.k4*self.k3M*self.E0*self.E0*self.TI0)
        self.rescaleFactor.assign(rescaleFactor)

    def call(self, inputs, cps = None, isFirstLayer=False):

        if isFirstLayer:
            bOnA = inputs - self.firstLayerTA0 - cps / (self.firstLayerK1M * self.E0)
            olderInput = tf.where(tf.equal(self.firstLayerK1M * self.E0 * self.firstLayerTA0 / cps, 0), inputs, 1 / 2 * (bOnA + (bOnA ** 2 + 4 * inputs * cps / (self.firstLayerK1M * self.E0))))
        else:
            olderInput = inputs

        #For batch computation: k1m * tf.transpose(inputs) doesn't work and we need to add a 1 axis in the end and use only *
        #Just above we use rank1 vector so should keep rank2 batches of input
        cpsExpand = tf.expand_dims(cps,-1)
        tf.assert_rank(cpsExpand,3)
        tf.assert_rank(cps,2)
        olderInputExpand=tf.expand_dims(olderInput,-1)
        tf.assert_rank(olderInputExpand,3)
        olderInputMidExpand = tf.expand_dims(olderInput,1)

        #clipped version
        # Cactivs= tf.where(self.mask>0,self.Cactiv,0)
        # Cinhibs = tf.where(self.mask<0,self.Cinhib,0)
        # Inhib = tf.divide(tf.matmul(olderInput,Cinhibs),self.kdT)
        # if self.usingLog:
        #     x_eq_clipped = tf.math.log(tf.matmul(olderInput,Cactivs))-tf.math.log(self.kdI*cp+Inhib/cp)
        # else:
        #     x_eq_clipped = tf.matmul(olderInput,Cactivs)/(self.kdI*cp+Inhib/cp)

        Cactivs= tf.where(self.mask > 0, self.Cactiv / (1 + self.k1M * self.E0 * olderInputExpand / cpsExpand), 0)
        Cinhibs = tf.where(self.mask < 0, self.Cinhib / (1 + self.k3M * self.E0 * olderInputExpand / cpsExpand), 0)
        tf.assert_rank(Cactivs,3)
        tf.assert_rank(Cinhibs,3)
        Inhib = tf.squeeze(tf.matmul(olderInputMidExpand,Cinhibs),axis=1)/self.kdT
        x_eq_clipped = tf.squeeze(tf.matmul(olderInputMidExpand,Cactivs),axis=1)/(self.kdI * cps + Inhib / cps)

        #unclipped version:
        Cactivs_unclipped= tf.where(self.kernel > 0, self.Cactiv * self.kernel/(1 + self.k1M * self.E0 * olderInputExpand / cpsExpand), 0)
        Cinhibs_unclipped = tf.where(self.kernel < 0, (-1) * self.Cinhib * self.kernel / (1 + self.k3M * self.E0 * olderInputExpand / cpsExpand), 0)
        tf.assert_rank(Cactivs_unclipped,3)
        tf.assert_rank(Cinhibs_unclipped,3)
        #CAREFUL: now the cactivs has taken the batch size, it is of rank 3 : [None,inputdims,outputdims]
        # THUS WE NEED: [None,1,inputdims] to use the matmul, and then squeeze the result!
        Inhib_unclipped = tf.squeeze(tf.matmul(olderInputMidExpand,Cinhibs_unclipped),axis=1)/self.kdT
        x_eq_unclipped = tf.squeeze(tf.matmul(olderInputMidExpand,Cactivs_unclipped),axis=1)/(self.kdI * cps + Inhib_unclipped / cps)
        # if self.usingLog:
        #     x_eq_unclipped = tf.math.log(tf.matmul(olderInput,Cactivs_unclipped))-tf.math.log(self.kdI*cp+Inhib_unclipped/cp)
        # else:
        #     x_eq_unclipped = tf.matmul(olderInput_unclipped,Cactivs_unclipped)/(self.kdI*cp+Inhib_unclipped/cp)
        tf.assert_rank(tf.squeeze(tf.matmul(olderInputMidExpand,Cinhibs_unclipped),axis=1),2,message="compute not good dims")
        outputs =  tf.stop_gradient(x_eq_clipped - x_eq_unclipped) + x_eq_unclipped
        tf.assert_rank(outputs,2,message="outputs not good dims")
        #outputs = chemTemplateClippedMatMul(self.deviceName, inputs, self.kernel, cp, self.Cactiv, self.Cinhib, self.kdI,self.kdT)
        return outputs

    def get_config(self):
        config = {'E0':float(self.E0.read_value())} #'cp for last batch': float(self.cp.read_value()),
        base_config = super(chemTemplateLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf.function
    def layer_cp_born_sup(self,olderInput):
        """
            Computes the influence of the layer to compute a superior born to the "competition" fixed point
        :param olderInput:
        :return:
        """
        with tf.device(self.deviceName):
            Kactivs = tf.where(self.mask>0,self.Kactiv,0)
            Kinhibs = tf.where(self.mask<0,self.Kinhib,0)
            #need to take axis = 1 here:
            w_inpIdx = tf.keras.backend.sum(Kactivs,axis=1)+tf.keras.backend.sum(Kinhibs,axis=1)
            firstComplex = tf.tensordot(olderInput[0],w_inpIdx,axes=[[0],[0]]) # template complexes in the layer

            Cactivs= tf.where(self.mask>0,self.Cactiv,0)
            Cinhibs = tf.where(self.mask<0,self.Cinhib,0)
            x_eq = tf.matmul(olderInput,Cactivs)/self.kdT

            #need to compute the influence of inhibition template here produced:
            Inhib2 = tf.matmul(olderInput,Cinhibs)/(self.kdT*self.k6)
            layer_cp = tf.keras.backend.sum(firstComplex) + tf.keras.backend.sum(Inhib2*x_eq/self.E0)

            return layer_cp,x_eq

    @tf.function
    def layer_cp_equilibrium(self,cp,input,isFirstLayer=False):
        with tf.device(self.deviceName):
            # Kactivs = tf.where(self.mask>0,self.Kactiv,0)
            # Kinhibs = tf.where(self.mask<0,self.Kinhib,0)
            # #need to take axis = 1 here:
            # w_inpIdx = tf.keras.backend.sum(Kactivs,axis=1)+tf.keras.backend.sum(Kinhibs,axis=1)
            layer_cp = 0
            if isFirstLayer:
                EandTemplate = self.firstLayerK1M*self.E0*self.firstLayerTA0/cp

                bOnA = input -self.firstLayerTA0 - cp/(self.firstLayerK1M*self.E0)
                tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_nan(bOnA),1,0)),0,message=" bOnA has nan")

                olderInput = tf.where(tf.equal(EandTemplate,0),input,1/2*(bOnA + (bOnA**2+4*input*cp/(self.firstLayerK1M*self.E0))**0.5))

                tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_nan(olderInput),1,0)),0,message=" olderInput has nan")
                tf.debugging.assert_greater_equal(olderInput,0.,message="older input has negative element")

                cp_with_Input = tf.where(tf.math.is_inf(olderInput),cp*self.firstLayerTA0/self.E0,self.firstLayerK1M*olderInput*self.firstLayerTA0/(1+self.firstLayerK1M*self.E0*olderInput/cp))
                tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_inf(cp_with_Input),1,0)),0,message=" cp_with_Input has inf")
                layer_cp += tf.keras.backend.sum(tf.where(tf.equal(EandTemplate,0),0.,cp_with_Input)) #first layer non-linearity!

                layer1input =self.firstLayerk2*self.firstLayerK1M/self.firstLayerkdT*self.firstLayerTA0*self.E0/cp*olderInput/(1+self.firstLayerK1M*self.E0/cp*olderInput)

                templateComplex = tf.where(self.mask>0,self.Kactiv*tf.transpose(layer1input)/(1+self.k1M*self.E0*tf.transpose(layer1input)/cp),
                                           tf.where(self.mask<0,self.Kinhib*tf.transpose(layer1input)/(1+self.k3M*self.E0*tf.transpose(layer1input)/cp),0)) # template complexes in the layer
                templateComplexCp = tf.keras.backend.sum(templateComplex)
                layer_cp += templateComplexCp

                # Be careful here: olderInput might have taken infinite value for cp very large, so we compute it again but divided by cp!
                Cactivs= tf.where(self.mask>0,self.Cactiv/(1+self.k1M*self.E0*tf.transpose(layer1input)/cp),0)
                Cinhibs = tf.where(self.mask<0,self.Cinhib/(1+self.k3M*self.E0*tf.transpose(layer1input)/cp),0)
                Inhib = tf.matmul(layer1input,Cinhibs)/self.kdT
                x_eq = tf.matmul(layer1input,Cactivs)/(self.kdI*cp+Inhib/cp)
                Inhib2 = tf.matmul(layer1input,Cinhibs)/(self.kdT*self.k6)

                layer_cp += tf.keras.backend.sum(Inhib2*x_eq/(self.E0*cp))
            else:
                olderInput = input
                templateComplex = tf.where(self.mask>0,self.Kactiv*tf.transpose(olderInput)/(1+self.k1M*self.E0*tf.transpose(olderInput)/cp),
                                           tf.where(self.mask<0,self.Kinhib*tf.transpose(olderInput)/(1+self.k3M*self.E0*tf.transpose(olderInput)/cp),0)) # template complexes in the layer
                templateComplexCp = tf.keras.backend.sum(templateComplex)# template complexes in the layer
                Cactivs= tf.where(self.mask>0,self.Cactiv/(1+self.k1M*self.E0*tf.transpose(olderInput)/cp),0)
                Cinhibs = tf.where(self.mask<0,self.Cinhib/(1+self.k3M*self.E0*tf.transpose(olderInput)/cp),0)
                Inhib = tf.matmul(olderInput,Cinhibs)/self.kdT
                x_eq = tf.matmul(olderInput,Cactivs)/(self.kdI*cp+Inhib/cp)
                Inhib2 = tf.matmul(olderInput,Cinhibs)/(self.kdT*self.k6)
                layer_cp += tf.keras.backend.sum(Inhib2*x_eq/(self.E0*cp)) + templateComplexCp

            return layer_cp,x_eq

    def print_constants(self):
        for idx,c in enumerate(self.cstList):
            if tf.equal(tf.rank(c),2):
                tf.print(self.cstList[idx][0,0],self.cstListName[idx]," rank 2")
            elif tf.equal(tf.rank(c),1):
                tf.print(self.cstList[idx][0],self.cstListName[idx]," rank 1")
            else:
                tf.print(self.cstList[idx],self.cstListName[idx]," rank 0")