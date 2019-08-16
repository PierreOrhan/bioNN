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
        #degradation cst for the pseudo-template
        self.kdpT = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        #degradation cst for the output
        self.kd = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        #degradation cst for the activation cascaded template
        self.kdT = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        #degradation cst for the inhibition cascaded template
        self.kdT2 = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)

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
        tf.assert_equal(len(constantArray),19)

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
        self.kdpT.assign(tf.fill(self.kdpT.shape,constantArray[9]))
        self.kd.assign(tf.fill(self.kd.shape,constantArray[10]))
        self.kdT.assign(tf.fill(self.kdT.shape,constantArray[11]))
        self.kdT2.assign(tf.fill(self.kdT.shape,constantArray[12]))
        self.TA0.assign(tf.fill(self.TA0.shape,activInitC))
        self.TI0.assign(tf.fill(self.TI0.shape,inhibInitC))
        self.E0.assign(tf.constant(enzymeInitTensor,dtype=tf.float32))

        self.Xglobal.assign(tf.constant(Xglobal,dtype=tf.float32))

        self.k1g.assign(tf.fill(self.k1g.shape,constantArray[13]))
        self.k1ng.assign(tf.fill(self.k1ng.shape,constantArray[14]))
        self.k2g.assign(tf.fill(self.k2g.shape,constantArray[15]))
        self.k3g.assign(tf.fill(self.k3g.shape,constantArray[16]))
        self.k3ng.assign(tf.fill(self.k3ng.shape,constantArray[17]))
        self.k4g.assign(tf.fill(self.k4g.shape,constantArray[18]))

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
        cps = tf.expand_dims(cps,-1)
        tf.assert_rank(cps,3)
        inputs=tf.expand_dims(inputs,-1)
        tf.assert_rank(inputs,3)

        if self.usingLog:
            QgijTemplate_eq = self.TA0*self.k2g*self.k1Mg*tf.exp(self.Xglobal)/(1+self.k1Mg*self.E0*tf.exp(self.Xglobal)/tf.squeeze(cps,axis=-1))
            QdgijTemplate_eq = self.TI0*self.k4g*self.k3Mg*tf.exp(self.Xglobal)/(1+self.k3Mg*self.E0*tf.exp(self.Xglobal)/tf.squeeze(cps,axis=-1))
        else:
            QgijTemplate_eq = self.TA0*self.k2g*self.k1Mg*self.Xglobal/(1+self.k1Mg*self.E0*self.Xglobal/tf.squeeze(cps,axis=-1))
            QdgijTemplate_eq = self.TI0*self.k4g*self.k3Mg*self.Xglobal/(1+self.k3Mg*self.E0*self.Xglobal/tf.squeeze(cps,axis=-1))

        #clipped version
        activBias = tf.keras.backend.sum(tf.where(self.mask > 0, tf.math.log(self.k2 * self.k1M * self.E0/(self.kdT*cps)), 0), axis=1) + \
                    tf.math.log(QgijTemplate_eq * self.E0 /tf.squeeze(cps,axis=-1))
        activKernel = tf.where(self.mask>0,self.mask,0.)
        if self.usingLog:
            activations = tf.where(activKernel>0,activKernel*inputs,0.)
        else:
            activations = tf.where(activKernel>0,activKernel*tf.math.log(inputs),0.)
        logActivationsSum= tf.keras.backend.sum(activations,axis=1)
        if self.usingLog:
            pT_eq = tf.keras.backend.sum(tf.where(self.mask<0,tf.math.log(self.k4 * self.E0 *self.k3M/(cps*self.kdT2)),0),axis=1) + \
                    tf.keras.backend.sum(tf.where(self.mask < 0,(-1)*self.mask * inputs, 0), axis=1) + \
                    tf.math.log(QdgijTemplate_eq*self.E0/tf.squeeze(cps,axis=-1)) #VERIFY SQUEEZING ECT....
            inhib = self.k6*self.k5M*self.E0*tf.exp(pT_eq)/(self.kdpT*tf.squeeze(cps,axis=-1))
            inhibition = tf.where(tf.equal(tf.exp(pT_eq),0),tf.math.log(self.kd),
                                  (tf.math.log(self.k6*self.k5M*self.E0/(self.kdpT*tf.squeeze(cps,axis=-1)))+pT_eq)+tf.math.log(self.kd/inhib+1))
            x_eq = logActivationsSum + activBias - inhibition
        else:
            pT_eq = tf.reduce_prod(tf.where(self.mask<0,self.k4*self.k3M*inputs*self.E0/(cps*self.kdT2),1),axis=1)* \
                    self.k3Mg*self.TI0*self.E0*self.Xglobal/(self.kdpT*(tf.squeeze(cps,axis=-1)+self.k3Mg*self.Xglobal*self.E0))
            inhib = self.k6*self.k5M*self.E0/tf.squeeze(cps,axis=-1)*pT_eq
            inhibition = tf.where(tf.equal(tf.exp(pT_eq),0),tf.math.log(self.kd),
                                  (tf.math.log(self.k6*self.k5M*self.E0/(self.kdpT*tf.squeeze(cps,axis=-1)))+pT_eq)+tf.math.log(self.kd/inhib+1))
            x_eq = logActivationsSum + activBias - inhibition

        #unclipped version:
        activBias_unclipped = tf.keras.backend.sum(tf.where(self.kernel > 0, tf.math.log(self.k2 * self.k1M * self.E0/(self.kdT*cps)), 0), axis=1) + \
                    tf.math.log(QgijTemplate_eq * self.E0 /tf.squeeze(cps,axis=-1))
        activKernel_unclipped = tf.where(self.kernel>0,self.mask,0.)
        if self.usingLog:
            activations_unclipped = tf.where(activKernel_unclipped>0,activKernel_unclipped*inputs,0.)
        else:
            activations_unclipped = tf.where(activKernel_unclipped>0,activKernel_unclipped*tf.math.log(inputs),0.)
        logActivationsSum_unclipped= tf.keras.backend.sum(activations_unclipped,axis=1)
        if self.usingLog:
            pT_eq_unclipped = tf.keras.backend.sum(tf.where(self.kernel<0,tf.math.log(self.k4 * self.E0 *self.k3M/(cps*self.kdT2) ),0),axis=1) + \
                    tf.keras.backend.sum(tf.where(self.kernel < 0,(-1)*self.kernel * inputs, 0), axis=1) + \
                    tf.math.log(QdgijTemplate_eq*self.E0/tf.squeeze(cps,axis=-1)) #VERIFY SQUEEZING ECT....
            inhib_unclipped = self.k6*self.k5M*self.E0*tf.exp(pT_eq_unclipped)/(self.kdpT*tf.squeeze(cps,axis=-1))
            inhibition_unclipped = tf.where(tf.equal(tf.exp(pT_eq_unclipped),0),tf.math.log(self.kd),
                                  (tf.math.log(self.k6*self.k5M*self.E0/(self.kdpT*tf.squeeze(cps,axis=-1)))+pT_eq_unclipped)+tf.math.log(self.kd/inhib_unclipped+1))
            x_eq_unclipped = logActivationsSum_unclipped + activBias_unclipped - inhibition_unclipped
        else:
            pT_eq_unclipped = tf.reduce_prod(tf.where(self.kernel<0,self.k4*self.k3M*inputs*self.E0/(cps*self.kdT2),1),axis=1)* \
                    self.k3Mg*self.TI0*self.E0*self.Xglobal/(self.kdpT*(tf.squeeze(cps,axis=-1)+self.k3Mg*self.Xglobal*self.E0))
            inhib_unclipped = self.k6*self.k5M*self.E0/tf.squeeze(cps,axis=-1)*pT_eq_unclipped
            inhibition_unclipped = tf.where(tf.equal(tf.exp(pT_eq_unclipped),0),tf.math.log(self.kd),
                                  (tf.math.log(self.k6*self.k5M*self.E0/(self.kdpT*tf.squeeze(cps,axis=-1)))+pT_eq_unclipped)+tf.math.log(self.kd/inhib_unclipped+1))
            x_eq_unclipped = logActivationsSum_unclipped + activBias_unclipped - inhibition_unclipped

        outputs =  tf.stop_gradient(x_eq - x_eq_unclipped) + x_eq_unclipped

        tf.assert_rank(x_eq_unclipped,2)
        tf.assert_rank(outputs,2)

        return outputs


    @tf.function
    def hornerMultiX(self,Xs,hornerMask,kp):
        """
            Execute horner algorithm on a vector X giving:
                X[0]+X[0]*X[1]+X[0]*X[1]*X[2]+..
            but using the horner strategy, for elements of X that appear to have a positive boolean value in Xs.
        :param Xs: the vector of interest should be of rank 2
        :return:
        """
        tf.assert_equal(tf.rank(Xs),2,message=str(Xs.shape))
        tf.assert_equal(Xs.shape[1],hornerMask.shape[1])
        p = tf.fill([hornerMask.shape[1]],0.)
        for idx in tf.range(hornerMask.shape[1]-1,-1,-1):
            p = tf.where(tf.greater(hornerMask[idx,:],0.),Xs[idx]*(1+kp[idx]*p),p)
        return p

    @tf.function
    def layer_cp_equilibrium(self, cp, input,verbose=False):
        tf.assert_rank(input,2,message="Input should have a shape like [1,??] the shape is"+str(input.shape))
        with tf.device(self.deviceName):
            #In the following computation we did not solve the case where input could have infinite value:
            if not self.usingLog:
                tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_inf(input),1,0)),0,message="inf detected in inputs cascade")

            # Here we observed a strange behavior and used the set_shape strategy to give more information to the solver.
            if self.usingLog:
                Qijm = self.k1M*tf.transpose(tf.exp(input))*self.E0/(self.kdT*cp)
                Qdijm = self.k3M*tf.transpose(tf.exp(input))*self.E0/(self.kdT*cp)
                Qijm.set_shape(self.k1M.shape)
                Qdijm.set_shape(self.k3M.shape)
            else:
                Qijm = self.k1M*tf.transpose(input)*self.E0/(self.kdT*cp)
                Qdijm = self.k3M*tf.transpose(input)*self.E0/(self.kdT*cp)
                Qijm.set_shape(self.k1M.shape)
                Qdijm.set_shape(self.k3M.shape)

            hornerActiv = self.hornerMultiX(Qijm,self.mask,self.k2)
            hornerInhib = self.hornerMultiX(Qdijm,(-1)*self.mask,self.k4)

            if verbose:
                tf.print(hornerActiv,"hornerActiv")
                tf.print(hornerInhib,"hornerInhib")

            if self.usingLog:
                # We must be careful of the case where cpg --> +infinity
                Template_eqXg = self.TA0*tf.exp(self.Xglobal)/(1+self.k1Mg*self.E0*tf.exp(self.Xglobal)/cp)
                QgijTemplate_eq = self.TA0*self.k2g*self.k1Mg*tf.exp(self.Xglobal)/(1+self.k1Mg*self.E0*tf.exp(self.Xglobal)/cp)
            else:
                Template_eqXg = self.TA0*self.Xglobal/(1+self.k1Mg*self.E0*self.Xglobal/cp)
                QgijTemplate_eq = self.TA0*self.k2g*self.k1Mg*self.Xglobal/(1+self.k1Mg*self.E0*self.Xglobal/cp)
            tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_inf(QgijTemplate_eq),1,0)),0,message="inf detected in QgijTemplate_eq")
            activCP = QgijTemplate_eq * hornerActiv + self.k1Mg*Template_eqXg

            if verbose:
                tf.print(activCP,"activCP")

            if self.usingLog:
                # We must be careful of the case where cpg --> +infinity
                #tf.where(tf.math.is_inf(self.k1Mg*tf.exp(self.Xglobal)/cpg),self.TA0/(self.E0*cpInv),self.TA0*self.k1Mg*tf.exp(self.Xglobal)/(cpg + self.k1Mg *tf.exp(self.Xglobal) * self.E0 *cpInv))
                TemplateI_eqXg = self.TI0*tf.exp(self.Xglobal)/(1+self.k3Mg*self.E0*tf.exp(self.Xglobal)/cp)
                QdgijTemplate_eq = self.TI0*self.k4g*self.k3Mg*tf.exp(self.Xglobal)/(1+self.k3Mg*self.E0*tf.exp(self.Xglobal)/cp)
            else:
                TemplateI_eqXg = self.TI0*self.Xglobal/(1+self.k3Mg*self.E0*self.Xglobal/cp)
                QdgijTemplate_eq = self.TI0*self.k4g*self.k3Mg*self.Xglobal/(1+self.k3Mg*self.E0*self.Xglobal/cp)
            tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_inf(QdgijTemplate_eq),1,0)),0,message="inf detected in QdgijTemplate_eq")

            inhibCP_no_pT = QdgijTemplate_eq * hornerInhib + self.k3Mg*TemplateI_eqXg


            #Here we might encounter NaN in the case where Xglobal is 0 and hornerActiv or hornerInhib exceed machine precision.
            #   In these case the unknown result should be found by plugging in the computation of hornerActiv and hornerInhib a fraction of cpg....
            #   To begin with we took cpg = 1 ... thus the incertitude comes from the fact that the initial value of Xglobal is null in which case no production is possible
            #   In this case, activCp and inhibCP_no_pT takes value of 0!
            activCP = tf.where(tf.math.is_nan(activCP),0.,activCP)
            inhibCP_no_pT = tf.where(tf.math.is_nan(inhibCP_no_pT),0.,inhibCP_no_pT)

            activBias = tf.keras.backend.sum(tf.where(self.mask > 0, tf.math.log(self.k2 * self.k1M * self.E0/(self.kdT*cp)), 0), axis=0) +\
                        tf.math.log(QgijTemplate_eq * self.E0 /cp)
            if verbose:
                tf.print(tf.keras.backend.sum(tf.where(self.mask > 0, tf.math.log(self.k2 * self.k1M * self.E0/(self.kdT*cp)), 0), axis=0),"first sum for activBias")
                tf.print(tf.math.log(QgijTemplate_eq * self.E0 /cp)," log for activ Bias")

            activKernel = tf.where(self.mask>0,self.mask,0)
            #when the input is 0 so -inf in log, its multiplication by 0 gives NaN.
            if self.usingLog:
                activations = tf.where(activKernel>0,activKernel*tf.transpose(input) ,0)
            else:
                activations = tf.where(activKernel>0,tf.math.log(activKernel*tf.transpose(input)),0)
            logActivationsSum = tf.keras.backend.sum(activations,axis=0)

            if self.usingLog:
                pT_eq = tf.keras.backend.sum(tf.where(self.mask<0,tf.math.log(self.k4 * self.E0 *self.k3M/(cp*self.kdT2) ),0),axis=0) +\
                        tf.keras.backend.sum(tf.where(self.mask < 0,(-1)*self.mask * tf.transpose(input), 0), axis=0) + \
                        tf.math.log(QdgijTemplate_eq*self.E0/cp)
                    # tf.math.log(tf.where(tf.math.is_inf(self.k3Mg*tf.exp(self.Xglobal)/cpg),self.TI0/self.kdpT,self.k3Mg*self.TI0*self.E0*tf.exp(self.Xglobal)/cpg*cpInv/(self.kdpT*(1+cpInv*self.k3Mg*tf.exp(self.Xglobal)*self.E0/cpg))))
                inhib = self.k6*self.k5M*self.E0*tf.exp(pT_eq)/(self.kdpT*cp)
                inhibition = tf.where(tf.equal(tf.exp(pT_eq),0),tf.math.log(self.kd),
                                      (tf.math.log(self.k6*self.k5M*self.E0/(self.kdpT*cp))+pT_eq)+tf.math.log(self.kd/inhib+1))
                x_eq = logActivationsSum + activBias - inhibition
                if verbose:
                    tf.print(tf.keras.backend.sum(tf.where(self.mask<0,tf.math.log(self.k4 * self.E0 *self.k3M/(cp*self.kdT2) ),0),axis=0),"sum 1 for pT")
                    tf.print(tf.keras.backend.sum(tf.where(self.mask < 0, (-1)*self.mask * tf.transpose(input), 0), axis=0),"sum 2 for pT")
                    tf.print(tf.math.log(QdgijTemplate_eq*self.E0/cp),"log last for pT")
                    tf.print(pT_eq,"pT_eq")
                    tf.print(inhib,"inhib")
                    tf.print(inhibition,"inhibition")
                    tf.print(logActivationsSum,"logActivationsSum")
                    tf.print(tf.exp(logActivationsSum + activBias),"tf.exp(logActivationsSum + activBias)")
                    tf.print(activBias,"activBias")
                pT_cp = tf.where(tf.equal(tf.exp(pT_eq),0),0.,
                                self.k5M*tf.exp(logActivationsSum+activBias)/(
                                         self.kd/tf.exp(pT_eq) + self.k6*self.k5M*self.E0/(cp*self.kdpT)))
                if verbose:
                    tf.print(x_eq,"x_eq")
                    tf.print(pT_cp,"pT_cp")
            else:
                pT_eq = tf.reduce_prod(tf.where(self.mask < 0,
                                                self.k4 * self.E0 * tf.transpose(input) * self.k3M/ (self.kdT2*cp), 1), axis=0) * \
                                                QdgijTemplate_eq*self.E0/cp
                inhib = self.k6 * self.k5M * self.E0*pT_eq /(self.kdpT*cp)
                if verbose:
                    tf.print(self.k4 * self.E0 * tf.transpose(input) * self.k3M/ (self.kdT2*cp),"self.k4 * self.E0 * tf.transpose(input) * self.k3M/ (self.kdT2*cp)")
                    tf.print(pT_eq,"pT_eq")
                    tf.print(inhib,"inhib")
                    tf.print(logActivationsSum,"logActivationsSum")
                    tf.print(tf.exp(logActivationsSum + activBias),"tf.exp(logActivationsSum + activBias)")
                    tf.print(activBias,"activBias")
                x_eq = tf.exp(logActivationsSum + activBias)/(self.kd+inhib)
                pT_cp = tf.where(tf.equal(pT_eq,0),0.,
                                 self.k5M*tf.exp(logActivationsSum+activBias)/(self.kd/pT_eq + self.k6*self.k5M*self.E0/(self.kdpT*cp)))


            tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_nan(activCP),1,0)),0,message="nan detected in activCP of cascade layer")
            tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_nan(inhibCP_no_pT),1,0)),0,message="nan detected in inhibCP_no_pT of cascade layer")
            tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_nan(pT_cp),1,0)),0,
                                      message="nan detected in pT_cp of cascade layer pT_eq:"+str(pT_eq)+" cp:"+str(cp))

            layer_cp = activCP + inhibCP_no_pT + pT_cp

            tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_nan(x_eq),1,0)),0,message="nan detected in outputs of cascade layer")

            return tf.keras.backend.sum(layer_cp),tf.expand_dims(x_eq,axis=0)

    def updateXglobal(self,Xglobal):
        self.Xglobal.assign(Xglobal)

    def displayVariable(self):
        tf.print(self.k1,"self.k1")
        tf.print(self.Xglobal,"self.Xglobal")
        tf.print(self.rescaleFactor,"self.rescaleFactor")
        tf.print(self.E0,"self.E0")
        tf.print(self.kdT2,"self.kdT2")
        tf.print(self.kdT,"self.kdT")
        tf.print(self.kd,"self.kd")
        tf.print(self.kdpT,"self.kdpT")
        tf.print(self.k6,"self.k6")
        tf.print(self.k5n,"self.k5n")
        tf.print(self.k5,"self.k5")
        tf.print(self.TI0 ,"self.TI0 ")
        tf.print(self.TA0 ,"self.TA0 ")
        tf.print(self.k4,"self.k4")
        tf.print(self.k3n,"self.k3n")
        tf.print(self.k3,"self.k3")
        tf.print(self.k2 ,"self.k2 ")
        tf.print(self.k1n,"self.k1n")
        tf.print(self.k1M ,"self.k1M ")
        tf.print(self.k3M ,"self.k3M ")
        tf.print(self.k5M,"self.k5M")
        tf.print(self.k1g,"self.k1g")
        tf.print(self.k1ng,"self.k1ng")
        tf.print(self.k2g,"self.k2g")
        tf.print(self.k3g,"self.k3g")
        tf.print(self.k3ng,"self.k3ng")
        tf.print(self.k4g,"self.k4g")
        tf.print(self.k1Mg ,"self.k1Mg ")
        tf.print(self.k3Mg,"self.k3Mg")

