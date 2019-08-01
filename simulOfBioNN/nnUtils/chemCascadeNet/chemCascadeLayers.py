from tensorflow.python.keras.layers import Dense
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import nn
from simulOfBioNN.nnUtils.clippedSparseBioDenseLayer import weightFixedAndClippedConstraint,sparseInitializer,constant_initializer,layerconstantInitiliaizer




class chemCascadeLayer(Dense):
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
        super(chemCascadeLayer, self).__init__(**kwargs)
        self.supports_masking = False
        self.sparseInitializer = sparseInitializer(sparsity, minval=min, maxval=max)
        self.deviceName=deviceName #the device on which the main operations will be conducted (forward and backward propagations)

        self.usingLog = usingLog
        # When using log:
        #   Inputs should be log(inputs)  ==> the layer will apply some log on them
        #   self.Xglobal also is computed as log(Xglobal)
        #   Finally: model gives back the log of the inputs
        #   All other variable are considered in their normal scale.... ( even Template fixed concentration as well as competition)

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

        #only one template starts each cascade:
        self.TA0 = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.TI0 = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)

        #only one inhibition by outputs (units):
        self.k5 = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k5n = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k6 = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.kdI = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.kdT = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.kdT2 = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)


        self.E0 = tf.Variable(tf.constant(1,dtype=tf.float32),trainable=False,dtype=tf.float32)
        self.rescaleFactor = tf.Variable(1,dtype=tf.float32,trainable=False)
        self.Xglobal = tf.Variable(tf.constant(1,dtype=tf.float32),trainable=False,dtype=tf.float32)

        #other intermediates variable:
        self.k1M = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.k3M = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.k5M = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)

        self.k1g = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k1ng = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k2g = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k3g = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k3ng = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k4g = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k1Mg = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k3Mg = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)

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

    def set_constants(self,constantArray,enzymeInit,activInitC,inhibInitC,computedRescaleFactor,Xglobal):
        """
            Define the ops assigning the values for the network constants.
        :return:
        """
        tf.assert_equal(len(constantArray),17)

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

        self.Xglobal.assign(tf.constant(Xglobal,dtype=tf.float32))

        self.k1g.assign(tf.fill(self.k1g.shape,constantArray[11]))
        self.k1ng.assign(tf.fill(self.k1ng.shape,constantArray[12]))
        self.k2g.assign(tf.fill(self.k2g.shape,constantArray[13]))
        self.k3g.assign(tf.fill(self.k3g.shape,constantArray[14]))
        self.k3ng.assign(tf.fill(self.k3ng.shape,constantArray[15]))
        self.k4g.assign(tf.fill(self.k4g.shape,constantArray[16]))

        self.k1Mg.assign(self.k1g/(self.k1ng+self.k2g))
        self.k3Mg.assign(self.k3g/(self.k3ng+self.k4g))

        #now we compute other intermediate values:

        self.k1M.assign(self.k1/(self.k1n+self.k2))

        self.k5M.assign(self.k5/(self.k5n+self.k6))
        self.k3M.assign(self.k3/(self.k3n+self.k4))

        self.mask.assign(tf.where(tf.less(self.kernel,-0.2),-1.,tf.where(tf.less(0.2,self.kernel),1.,0.)))

    def rescale(self,rescaleFactor):
        """
            Rescale the enzyme value
        :param rescaleFactor: 0d Tensor: float with the new rescale
        :return:
        """
        self.E0.assign(self.E0.read_value()*(rescaleFactor**0.5)/(self.rescaleFactor.read_value()**0.5))
        self.rescaleFactor.assign(rescaleFactor)

    def call(self, inputs, cps = None, isFirstLayer=False):
        if cps is None:
            cps = tf.ones((tf.shape(inputs)[0],1))*(2.*10**6)

        if self.usingLog:
            Template_Activation_eq = self.TA0/(1+self.k1Mg*tf.exp(self.Xglobal)*self.E0/cps)
                #similarly the complexes of the template binding with Xglobal and enzyme reach equilibrium:
                # X1s = T1/(k2g*E) = TA:E:Xglobal/E
            X1s = self.k1Mg*Template_Activation_eq*tf.exp(self.Xglobal)
        else:
            Template_Activation_eq = self.TA0/(1+self.k1Mg*self.Xglobal*self.E0/cps)
                #similarly the complexes of the template binding with Xglobal and enzyme reach equilibrium:
                #X1s = T1/(k2g*E) = TA:E:Xglobal/E
            X1s = self.k1Mg*Template_Activation_eq*self.Xglobal

        #clipped version
        activBias = tf.sum(tf.where(self.mask>0,tf.math.log(self.k2*self.k1M*self.E0/(self.kdT*cps)),0),axis=1)+tf.math.log(self.k2G*X1s*self.E0/cps)
        activKernel = tf.where(self.mask>0,self.mask,0)

        if self.usingLog:
            activations = tf.where(activKernel>0,activKernel*inputs,1)
        else:
            activations = tf.where(activKernel>0,activKernel*tf.math.log(inputs),1)
        logActivations= tf.sum(activations,axis=1)

        if self.usingLog:
            pT_eq = tf.reduce_prod(tf.where(self.mask<0,self.k4*self.k3M*self.E0*tf.exp(inputs)/(cps*self.kdT2),1),axis=1)* \
                    self.k3Mg*self.TI0*self.E0*tf.exp(self.Xglobal)/(self.kdI*(cps+self.k3Mg*tf.exp(self.Xglobal)*self.E0))
        else:
            pT_eq = tf.reduce_prod(tf.where(self.mask<0,self.k4*self.k3M*self.E0*inputs/(cps*self.kdT2),1),axis=1)* \
                    self.k3Mg*self.TI0*self.E0*self.Xglobal/(self.kdI*(cps+self.k3Mg*self.Xglobal*self.E0))
        inhib = self.k4*self.k5M*self.E0/cps*pT_eq

        if self.usingLog:
            A_log= logActivations + activBias - tf.math.log(self.kdI+inhib)
        else:
            A_log = tf.exp(logActivations + activBias)/(self.kdI+inhib)


        #unclipped version:
        activBias_unclipped = tf.sum(tf.where(self.kernel>0,tf.math.log(self.k2*self.k1M*self.E0/(self.kdT*cps)),0),axis=1)+tf.math.log(self.k2G*X1s*self.E0/cps)
        activKernel_unclipped = tf.where(self.kernel>0,self.kernel,0)

        if self.usingLog:
            activations_unclipped = tf.where(activKernel_unclipped>0,activKernel_unclipped*inputs,1)
        else:
            activations_unclipped = tf.where(activKernel_unclipped>0,activKernel_unclipped*tf.math.log(inputs),1)
        logActivations_unclipped = tf.sum(activations_unclipped,axis=1)

        if self.usingLog:
            pT_eq_unclipped = tf.reduce_prod(tf.where(self.kernel<0,self.k4*self.k3M*self.E0*tf.exp(inputs)/(cps*self.kdT2),1),axis=1)* \
                    self.k3Mg*self.TI0*self.E0*tf.exp(self.Xglobal)/(self.kdI*(cps+self.k3Mg*tf.exp(self.Xglobal)*self.E0))
        else:
            pT_eq_unclipped = tf.reduce_prod(tf.where(self.kernel<0,self.k4*self.k3M*self.E0*inputs/(cps*self.kdT2),1),axis=1)* \
                    self.k3Mg*self.TI0*self.E0*self.Xglobal/(self.kdI*(cps+self.k3Mg*self.Xglobal*self.E0))
        inhib_unclipped = self.k4*self.k5M*self.E0/cps*pT_eq_unclipped

        if self.usingLog:
            A_log_unclipped= logActivations_unclipped + activBias_unclipped - tf.math.log(self.kdI+inhib_unclipped)
        else:
            A_log_unclipped = tf.exp(logActivations_unclipped + activBias)/(self.kdI+inhib_unclipped)

        outputs =  tf.stop_gradient(A_log - A_log_unclipped) + A_log_unclipped

        return outputs


    @tf.function
    def hornerMultiX(self,Xs,hornerMask):
        """
            Execute horner algorithm on a vector X giving:
                1+X[0]+X[0]*X[1]+X[0]*X[1]*X[2]+..
            but using the horner strategy, for elements of X that appear to have a positive boolean value in Xs.
            The algorithm is executed on the second axis of vector X to accomodate for batches
        :param Xs: the vector of interest should be of rank 2
        :return:
        """
        tf.assert_equal(tf.rank(Xs),2)
        p = 1
        for idx in tf.range(Xs.shape[-1]-1,-1,-1):
            p += 1 + tf.where(tf.less(0,hornerMask),Xs[:,idx]*p,p-1)
        return p

    @tf.function
    def layer_cp_born_sup(self,input):
        """
            Computes the influence of the layer to compute a superior born to the "competition" fixed point
        :param olderInput:
        :return:
        """
        with tf.device(self.deviceName):
            if self.usingLog:
                XActivs = self.k1*tf.tensordot(self.k1M,tf.exp(input)[0],axes=[[0],[0]])*self.E0/(self.kdT)
                Xinhibs = self.k3*tf.tensordot(self.k3M,tf.exp(input)[0],axes=[[0],[0]])*self.E0/(self.kdT)
            else:
                XActivs = self.k1*tf.tensordot(self.k1M,input[0],axes=[[0],[0]])*self.E0/(self.kdT)
                Xinhibs = self.k3*tf.tensordot(self.k3M,input[0],axes=[[0],[0]])*self.E0/(self.kdT)


            hornerActiv = self.hornerMultiX(XActivs,self.mask)
            hornerInhib = self.hornerMultiX(Xinhibs,self.mask)

            if self.usingLog:
                X1s = self.k1Mg*self.TA0*tf.exp(self.Xglobal)
            else:
                X1s = self.k1Mg*self.TA0*self.Xglobal

            activCP = X1s*hornerActiv
            inhibCP_no_pT = X1s*hornerInhib

            #Compute the cp due to the activation:
            activBias = tf.sum(tf.where(self.mask>0,tf.math.log(self.k2*self.k1M*self.E0/(self.kdT)),0),axis=1)+tf.math.log(self.k2G*X1s*self.E0)
            activKernel = tf.where(self.mask>0,self.mask,0)
            activations = tf.where(activKernel>0,activKernel*input,1)
            logActivationsSum = tf.sum(tf.math.log(activations),axis=1)

            if self.usingLog:
                pT_eq = tf.reduce_prod(tf.where(self.mask<0,self.k4*self.E0*tf.tensordot(self.k3M,tf.exp(input)[0],axes=[[0],[0]])/(self.kdT2),1),axis=1)* \
                        self.k3Mg*self.TI0*self.E0*tf.exp(self.Xglobal)/(self.kdI*(1+self.k3Mg*tf.exp(self.Xglobal)*self.E0))
            else:
                pT_eq = tf.reduce_prod(tf.where(self.mask<0,self.k4*self.E0*tf.tensordot(self.k3M,input,axes=[[0],[0]])/(self.kdT2),1),axis=1)* \
                        self.k3Mg*self.TI0*self.E0*self.Xglobal/(self.kdI*(1+self.k3Mg*self.Xglobal*self.E0))
            inhib = self.k4*self.k5M*self.E0*pT_eq

            if self.usingLog:
                x_eq = logActivationsSum + activBias - tf.math.log(self.kdI+inhib)
                pT_cp = self.k5M*tf.exp(x_eq)*pT_eq
            else:
                x_eq = tf.exp(logActivationsSum + activBias)/(self.kdI+inhib)
                pT_cp = self.k5M*x_eq*pT_eq

            layer_cp = activCP + inhibCP_no_pT + pT_cp

            return layer_cp,x_eq

    @tf.function
    def layer_cp_equilibrium(self,cp,input,isFirstLayer=False):
        with tf.device(self.deviceName):

            if self.usingLog:
                XActivs = self.k1*tf.tensordot(self.k1M,tf.exp(input)[0],axes=[[0],[0]])*self.E0/(self.kdT*cp)
                Xinhibs = self.k3*tf.tensordot(self.k3M,tf.exp(input)[0],axes=[[0],[0]])*self.E0/(self.kdT*cp)
            else:
                XActivs = self.k1*tf.tensordot(self.k1M,input[0],axes=[[0],[0]])*self.E0/(self.kdT*cp)
                Xinhibs = self.k3*tf.tensordot(self.k3M,input[0],axes=[[0],[0]])*self.E0/(self.kdT*cp)


            hornerActiv = self.hornerMultiX(XActivs,self.mask)
            hornerInhib = self.hornerMultiX(Xinhibs,self.mask)

            if self.usingLog:
                Template_Activation_eq = self.TA0/(1+self.k1Mg*tf.exp(self.Xglobal)*self.E0/cp)
                    #similarly the complexes of the template binding with Xglobal and enzyme reach equilibrium:
                    # X1s = T1/(k2g*E) = TA:E:Xglobal/E
                X1s = self.k1Mg*Template_Activation_eq*tf.exp(self.Xglobal)
            else:
                Template_Activation_eq = self.TA0/(1+self.k1Mg*self.Xglobal*self.E0/cp)
                    #similarly the complexes of the template binding with Xglobal and enzyme reach equilibrium:
                    #X1s = T1/(k2g*E) = TA:E:Xglobal/E
                X1s = self.k1Mg*Template_Activation_eq*self.Xglobal

            activCP = X1s*hornerActiv
            inhibCP_no_pT = X1s*hornerInhib

            #Compute the cp due to the activation:
            activBias = tf.sum(tf.where(self.mask>0,tf.math.log(self.k2*self.k1M*self.E0/(self.kdT*cp)),0),axis=1)+tf.math.log(self.k2G*X1s*self.E0/cp)
            activKernel = tf.where(self.mask>0,self.mask,0)
            activations = tf.where(activKernel>0,activKernel*input,1)
            logActivationsSum = tf.sum(tf.math.log(activations),axis=1)

            if self.usingLog:
                pT_eq = tf.reduce_prod(tf.where(self.mask<0,self.k4*self.E0*tf.tensordot(self.k3M,tf.exp(input)[0],axes=[[0],[0]])/(cp*self.kdT2),1),axis=1)*\
                            self.k3Mg*self.TI0*self.E0*tf.exp(self.Xglobal)/(self.kdI*(cp+self.k3Mg*tf.exp(self.Xglobal)*self.E0))
            else:
                pT_eq = tf.reduce_prod(tf.where(self.mask<0,self.k4*self.E0*tf.tensordot(self.k3M,input[0],axes=[[0],[0]])/(cp*self.kdT2),1),axis=1)* \
                        self.k3Mg*self.TI0*self.E0*self.Xglobal/(self.kdI*(cp+self.k3Mg*self.Xglobal*self.E0))
            inhib = self.k4*self.k5M*self.E0/cp*pT_eq

            if self.usingLog:
                x_eq = logActivationsSum + activBias - tf.math.log(self.kdI+inhib)
                pT_cp = self.k5M*tf.exp(x_eq)*pT_eq
            else:
                x_eq = tf.exp(logActivationsSum + activBias)/(self.kdI+inhib)
                pT_cp = self.k5M*x_eq*pT_eq

            layer_cp = activCP + inhibCP_no_pT + pT_cp

            return layer_cp,x_eq

    @tf.function
    def layer_XgCp_born_sup(self):
        return tf.sum(self.k1Mg*self.TA0*self.E0) +\
               tf.sum(self.k3Mg*self.TI0*self.E0)


    @tf.function
    def layer_XgCp(self,cp):
        return tf.sum(self.k1Mg*self.TA0*self.E0/(cp+self.k1Mg*self.E0*self.Xglobal)) + \
               tf.sum(self.k3Mg*self.TI0*self.E0/(cp+self.k3Mg*self.E0*self.Xglobal))

    @tf.function
    def get_inhib_and_output(self,cps,inputs,isFirstLayer=False):
        if isFirstLayer:
            Kactivs = tf.where(self.mask>0,self.Kactiv,0)
            Kinhibs = tf.where(self.mask<0,self.Kinhib,0)
            w_inpIdx = tf.keras.backend.sum(Kactivs,axis=1)+tf.keras.backend.sum(Kinhibs,axis=1)
            olderInput = tf.divide(inputs,(1+self.E0*w_inpIdx/cps)) #need to rescale the initial inputs too
        else:
            olderInput = inputs

        #clipped version
        Cactivs= tf.where(self.mask>0,self.Cactiv,0)
        Cinhibs = tf.where(self.mask<0,self.Cinhib,0)
        Inhib = tf.divide(tf.matmul(olderInput,Cinhibs),self.kdT)
        x_eq_clipped = tf.matmul(olderInput,Cactivs)/(self.kdI*cps+Inhib/cps)

        return x_eq_clipped,Inhib,cps*cps*self.kdI