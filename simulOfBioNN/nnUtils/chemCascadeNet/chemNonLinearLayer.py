from tensorflow.python.keras.layers import Dense
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import nn
from simulOfBioNN.nnUtils.clippedSparseBioDenseLayer import weightFixedAndClippedConstraint,sparseInitializer,constant_initializer,layerconstantInitiliaizer




class chemNonLinearLayer(Dense):
    """
        Layer providing a non-linear output,
            based on the saturation of its templates which are not degraded.
        Can be see as a per-element ops.... Each neuron has only one output (mask is identity)
        :param theta: bias to be added after each multipication
    """
    def __init__(self, deviceName, usingLog=True, **kwargs):
        """

            :param deviceName: device to use for computation
            :param usingLog: if working in log concentration or not.
            :param kwargs:
        """
        super(chemNonLinearLayer, self).__init__(**kwargs)
        self.supports_masking = False
        self.deviceName=deviceName #the device on which the main operations will be conducted (forward and backward propagations)
        self.usingLog = usingLog
        # when using log, the layer sill takes value in the normal scale ( so the model should apply some exponential) but gives the log of the equilibrium value

    def build(self):
        # We just change the way bias is added and remove it from trainable variable!


        self.kernel = None
        self.bias = None

        variableShape=(None,self.units)
        self.k1 = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k1n = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.k2 = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.TA0 = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.kdT = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.E0 = tf.Variable(tf.constant(1,dtype=tf.float32),trainable=False,dtype=tf.float32)
        self.rescaleFactor = tf.Variable(1,dtype=tf.float32,trainable=False)
        self.k1M = tf.Variable(tf.zeros(variableShape[-1],dtype=tf.float32),trainable=False)
        self.built = True
        print("Layer successfully built")


    def set_constants(self,constantArray,enzymeInit,activInitC,computedRescaleFactor):
        """
            Define the ops assigning the values for the network constants.
        :return:
        """
        tf.assert_equal(len(constantArray),4)
        self.rescaleFactor.assign(computedRescaleFactor)
        enzymeInitTensor = enzymeInit*(computedRescaleFactor**0.5)
        self.k1.assign(tf.fill(self.k1.shape,constantArray[0]))
        self.k1n.assign(tf.fill(self.k1n.shape,constantArray[1]))
        self.k2.assign(tf.fill(self.k2.shape,constantArray[2]))

        self.kdT.assign(tf.fill(self.kdT.shape,constantArray[3]))
        self.TA0.assign(tf.fill(self.TA0.shape,activInitC))

        self.E0.assign(tf.constant(enzymeInitTensor,dtype=tf.float32))

        #now we compute other intermediate values:
        self.k1M.assign(self.k1/(self.k1n+self.k2))

    def rescale(self,rescaleFactor):
        """
            Rescale the enzyme value
        :param rescaleFactor: 0d Tensor: float with the new rescale
        :return:
        """
        self.E0.assign(self.E0.read_value()*(rescaleFactor**0.5)/(self.rescaleFactor.read_value()**0.5))
        self.rescaleFactor.assign(rescaleFactor)

    def get_rescaleFactor(self):
        return self.units

    def call(self, inputs, cps = None, isFirstLayer=False):
        if cps is None:
            cps = tf.ones((tf.shape(inputs)[0],1))*(2.*10**6)
        else:
            tf.assert_rank(cps,2)
        tf.assert_rank(inputs,2)

        if self.usingLog:
            tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_inf(tf.exp(inputs)),1,0)),0,message="inf detected in exponential inputs call of NL")
        else:
            tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_inf(inputs),1,0)),0,message="inf detected in inputs call of NL")

        if isFirstLayer:
            if self.usingLog:
                olderInput = ((tf.exp(inputs) - self.TA0 - cps/(self.k1M*self.E0))+
                                        ((cps/(self.k1M*self.E0)-tf.exp(inputs)+self.TA0)**2+4*tf.exp(inputs)*cps/(self.k1M*self.E0))**0.5)/2
                olderInput = tf.where(tf.math.is_nan(olderInput),tf.nn.relu(inputs-self.TA0),olderInput)
            else:
                olderInput = ((inputs - self.TA0 - cps/(self.k1M*self.E0))+
                                    ((cps/(self.k1M*self.E0)-inputs+self.TA0)**2+4*cps*inputs/(self.k1M*self.E0))**0.5)/2
                olderInput = tf.where(tf.math.is_nan(olderInput),tf.nn.relu(inputs-self.TA0),olderInput)
        else:
            if self.usingLog:
                olderInput = tf.exp(inputs)
            else:
                olderInput = inputs

        if self.usingLog:
            outputs = tf.math.log(self.k2*self.TA0*olderInput/(self.kdT*(cps/(self.k1M*self.E0)+olderInput)))
        else:
            outputs =  self.k2*self.TA0*olderInput/(self.kdT*(cps/(self.k1M*self.E0)+olderInput))

        return outputs

    @tf.function
    def layer_cp_equilibrium_FromInv(self, cpInv, input, isFirstLayer=False):
        with tf.device(self.deviceName):
            #In the following computation we did not solve the case where input could have infinite value:
            if not self.usingLog:
                tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_inf(input),1,0)),0,message="inf detected in inputs")
            if isFirstLayer:
                if self.usingLog:
                    # Due to the presence of the root, it is impossible for us to write the equatation to make it possible for the computer to compute that
                    #   when cpInv -> 0 we have odlerInput -> (olderInput - TA0)
                    #   and a NaN wil be computed instead...
                    # Therefore we make the test of the presence of a Nan and gives back the approximated equation if it is the case!
                    olderInput = (((cpInv *self.k1M * self.E0)*(tf.exp(input) - self.TA0) - 1) +
                                              ((1 - ((cpInv * self.k1M * self.E0) *(tf.exp(input) - self.TA0)))**2 + 4 * tf.exp(input)*(cpInv * self.k1M * self.E0)) ** 0.5)/(2*cpInv * self.k1M * self.E0)
                    olderInput = tf.where(tf.math.is_nan(olderInput),tf.nn.relu(tf.exp(input)-self.TA0),olderInput)
                else:
                    olderInput = (((cpInv *self.k1M * self.E0)*(input - self.TA0) - 1) +
                                  ((1 - ((cpInv * self.k1M * self.E0) *(input - self.TA0)))**2 + 4 * input*(cpInv * self.k1M * self.E0)) ** 0.5)/(2*cpInv * self.k1M * self.E0)
                    olderInput = tf.where(tf.math.is_nan(olderInput),tf.nn.relu(input-self.TA0),olderInput)
            else:
                if self.usingLog:
                    olderInput= tf.exp(input)
                else:
                    olderInput = input

            layer_cp = tf.where(tf.math.is_inf(olderInput),self.k1M*self.TA0/(1/olderInput+ self.k1M*cpInv*self.E0), self.k1M*self.TA0*olderInput/(1+ self.k1M*olderInput*cpInv*self.E0))
            if self.usingLog:
                x_eq = tf.where(tf.math.is_inf(olderInput),tf.math.log(self.k1M*self.k2*self.E0*cpInv*self.TA0/(self.kdT*(1/olderInput+self.k1M*self.E0*cpInv))),tf.math.log(self.k1M*self.k2*self.E0*cpInv*self.TA0*olderInput/(self.kdT*(1+self.k1M*self.E0*cpInv*olderInput))))
            else:
                x_eq = self.k1M*self.k2*self.E0*cpInv*self.TA0*olderInput/(self.kdT*(1+self.k1M*self.E0*cpInv*olderInput))
            tf.print(tf.math.is_inf(olderInput),"tf.math.is_inf(olderInput)")
            tf.print(layer_cp,"layer_cp")
            return tf.keras.backend.sum(layer_cp),x_eq