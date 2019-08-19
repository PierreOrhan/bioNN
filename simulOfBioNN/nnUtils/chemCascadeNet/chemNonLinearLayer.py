from tensorflow.python.keras.layers import Dense

import tensorflow as tf


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
                bOnA = tf.exp(inputs) - self.TA0 - cps/(self.k1M * self.E0)
                olderInput = 0.5*(bOnA + ((bOnA)**2 + 4 * tf.exp(inputs)*cps/(self.k1M * self.E0))**0.5)
                olderInput = tf.where(tf.math.is_nan(olderInput),tf.exp(inputs),olderInput)
            else:
                bOnA = inputs - self.TA0 - cps/(self.k1M * self.E0)
                olderInput = 0.5*(bOnA + ((bOnA)**2 + 4 * inputs*cps/(self.k1M * self.E0))**0.5)
                olderInput = tf.where(tf.math.is_nan(olderInput),inputs,olderInput)
        else:
            if self.usingLog:
                olderInput = tf.exp(inputs)
            else:
                olderInput = inputs

        if self.usingLog:
            x_eq = tf.where(tf.math.is_inf(olderInput),
                            tf.math.log(self.k2*self.TA0/self.kdT),
                            tf.where(tf.equal(olderInput,0.),
                                     0.,
                                     tf.math.log((self.k1M*self.k2*self.E0*self.TA0/self.kdT)*(olderInput/(cps+self.k1M*self.E0*olderInput)))))
        else:
            x_eq = self.k1M*self.k2*self.E0*self.TA0*olderInput/(self.kdT*cps*(1+self.k1M*self.E0*olderInput/cps))
        return x_eq

    @tf.function
    def layer_cp_equilibrium(self, cp, input, isFirstLayer=False):
        with tf.device(self.deviceName):
            #In the following computation we did not solve the case where input could have infinite value:
            if not self.usingLog:
                tf.debugging.assert_equal(tf.keras.backend.sum(tf.where(tf.math.is_inf(input),1,0)),0,message="inf detected in inputs")
            if isFirstLayer:
                if self.usingLog:
                    # Due to the presence of the root, it is impossible for us to write the equatation to make it possible for the computer to compute that
                    #   when cp -> +infinity we have olderInput -> input
                    #   and a NaN wil be computed instead...
                    # Therefore we make the test of the presence of a Nan and gives back the approximated equation if it is the case!
                    bOnA = tf.exp(input) - self.TA0 - cp/(self.k1M * self.E0)
                    olderInput = 0.5*(bOnA + ((bOnA)**2 + 4 * tf.exp(input)*cp/(self.k1M * self.E0))**0.5)
                    olderInput = tf.where(tf.math.is_nan(olderInput),tf.exp(input),olderInput)
                else:
                    bOnA = input - self.TA0 - cp/(self.k1M * self.E0)
                    olderInput = 0.5*(bOnA + ((bOnA)**2 + 4 * input*cp/(self.k1M * self.E0))**0.5)
                    olderInput = tf.where(tf.math.is_nan(olderInput),input,olderInput)
            else:
                if self.usingLog:
                    olderInput= tf.exp(input)
                else:
                    olderInput = input

            layer_cp = tf.where(tf.math.equal(olderInput,0),0.,self.k1M*self.TA0/(1/olderInput+ self.k1M*self.E0/cp))
            if tf.equal(tf.math.is_inf(tf.keras.backend.sum(layer_cp)),False):
                if self.usingLog:
                    x_eq = tf.where(tf.math.is_inf(olderInput),
                                    tf.math.log(self.k2*self.TA0/self.kdT),
                                    tf.where(tf.equal(olderInput,0.),
                                        0.,
                                        tf.math.log((self.k1M*self.k2*self.E0*self.TA0/self.kdT)*(olderInput/(cp+self.k1M*self.E0*olderInput)))))
                else:
                    x_eq = self.k1M*self.k2*self.E0*self.TA0*olderInput/(self.kdT*cp*(1+self.k1M*self.E0*olderInput/cp))
            else: #The infinite value of layer_cp ends the loop so we don't compute x_eq...
                x_eq = self.k1M*input #Value doesn't matter
            return tf.keras.backend.sum(layer_cp),x_eq

    def obtainNonLinearityShape(self,input,cps,isFirstLayer=False):
        input = tf.convert_to_tensor(input,dtype=tf.float32)
        if isFirstLayer:
            if self.usingLog:
                bOnA = tf.exp(input) - self.TA0[0] - cps/(self.k1M[0] * self.E0)
                olderInput = 0.5*(bOnA + ((bOnA)**2 + 4 * tf.exp(input)*cps/(self.k1M[0] * self.E0))**0.5)
            else:
                bOnA = input - self.TA0[0] - cps/(self.k1M[0] * self.E0)
                olderInput = 0.5*(bOnA + ((bOnA)**2 + 4 * input*cps/(self.k1M[0] * self.E0))**0.5)
                olderInput = tf.where(tf.math.is_nan(olderInput),input,olderInput)
        else:
            if self.usingLog:
                olderInput = tf.exp(input)
            else:
                olderInput = input

        if self.usingLog:
            x_eq = tf.where(tf.math.is_inf(olderInput),
                            tf.math.log(self.k2[0]*self.TA0[0]/self.kdT[0]),
                            tf.where(tf.equal(olderInput,0.),
                                     0.,
                                     tf.math.log((self.k1M[0]*self.k2[0]*self.E0*self.TA0[0]/self.kdT[0])*(olderInput/(cps+self.k1M[0]*self.E0*olderInput)))))
        else:
            x_eq = self.k1M[0]*self.k2[0]*self.E0*self.TA0[0]*olderInput/(self.kdT[0]*cps*(1+self.k1M[0]*self.E0*olderInput/cps))
        return x_eq


    def displayVariable(self):
        tf.print(self.k1,"self.k1")
        tf.print(self.k1n,"self.k1n")
        tf.print(self.k2 ," self.k2 ")
        tf.print(self.TA0,"self.TA0")
        tf.print(self.kdT,"self.kdT")
        tf.print(self.E0,"self.E0")
        tf.print(self.rescaleFactor,"self.rescaleFactor")
        tf.print(self.k1M,"self.k1M")