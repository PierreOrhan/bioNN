from tensorflow.python.keras.layers import Dense
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import nn
from simulOfBioNN.nnUtils.clippedSparseBioDenseLayer import weightFixedAndClippedConstraint,sparseInitializer,constant_initializer,layerconstantInitiliaizer

def chemTemplateClippedTensorDot(deviceName, inputs, kernel, rank, cp,CinhibMat,CactivMat,kd):
    """
        Clipped tensorDot on which we apply on a sigmoid function of the form activator*kernelActivator/1+activator*kernelActivator+sumALLInhib*kernelInhibitor
        Clipping follow the rule: weights<0.2 take value -1, weighs>0.2 take value 1 and other take value 0.
    """
    with tf.device(deviceName):
        with ops.name_scope(None,default_name="clippedTensorDotOp",values=[inputs, kernel]) as scope:
            Tminus = tf.cast(tf.fill(kernel.shape,-1),dtype=tf.float32)
            Tplus = tf.cast(tf.fill(kernel.shape,1),dtype=tf.float32)
            Tzero = tf.cast(tf.fill(kernel.shape,0),dtype=tf.float32)
            clippedKernel=tf.where(tf.less(kernel,-0.2),Tminus,tf.where(tf.less(kernel,0.2),Tzero,Tplus))
            kernelInhibitor=tf.stop_gradient(tf.where(tf.less(kernel,0),clippedKernel,Tzero)*(-1))
            inhibFiltered = kernelInhibitor * CinhibMat/cp    # We multiply the filtered by Cinhib and divide it by cp
            kernelActivator = tf.stop_gradient(tf.where(tf.less(kernel,0),Tzero,clippedKernel))
            activatorFiltered = kernelActivator * CactivMat
            inhibition = tf.stop_gradient(tf.tensordot(inputs,inhibFiltered, [[rank - 1], [0]]))
            activation = tf.stop_gradient(tf.tensordot(inputs, activatorFiltered, [[rank - 1], [0]]))

            ## same operation but unclipped:
            Inhib = tf.where(tf.less(kernel,0),kernel,Tzero)*(-1)*CinhibMat/cp
            Activ = tf.where(tf.less(kernel,0),Tzero,kernel)*CactivMat
            activation2 = tf.tensordot(inputs, Activ, [[rank - 1], [0]])
            inhibition2 = tf.tensordot(inputs, Inhib, [[rank - 1], [0]])
            forBackProp = tf.divide(activation2,cp*kd+inhibition2)

            outputs = tf.stop_gradient(tf.divide(activation,cp*kd+inhibition)-forBackProp)
            return tf.add(forBackProp,outputs,name=scope)

def chemTemplateClippedMatMul(deviceName, inputs, kernel,cp,CinhibMat,CactivMat,kd):
    '''
        Clipped matmul on which we apply on a sigmoid function of the form activator*kernelActivator/1+activator*kernelActivator+sumALLInhib*kernelInhibitor
        Clipping follow the rule: weights<0.2 take value -1, weighs>0.2 take value 1 and other take value 0.
    '''
    with tf.device(deviceName):
        with ops.name_scope(None,default_name="clippedMatMulOp", values = [inputs, kernel]) as scope:

            if len(inputs.shape.as_list())==1:
                inputs = tf.stack([inputs])

            Tminus = tf.cast(tf.fill(kernel.shape,-1),dtype=tf.float32)
            Tplus = tf.cast(tf.fill(kernel.shape,1),dtype=tf.float32)
            Tzero = tf.cast(tf.fill(kernel.shape,0),dtype=tf.float32)
            #clipp the kernel at 0:
            clippedKernel=tf.stop_gradient(tf.where(tf.less(kernel,-0.2),Tminus,tf.where(tf.less(kernel,0.2),Tzero,Tplus)))
            kernelInhibitor=tf.stop_gradient(tf.where(tf.less(kernel,0),clippedKernel,Tzero)*(-1))
            inhibFiltered = kernelInhibitor * CinhibMat/cp    # We multiply the filtered by Cinhib and divide it by cp
            inhibition = tf.stop_gradient(tf.matmul(inputs,inhibFiltered))
            #kernel for activators:
            kernelActivator = tf.stop_gradient(tf.where(tf.less(kernel,0),Tzero,clippedKernel))
            activatorFiltered = kernelActivator * CactivMat
            activation = tf.stop_gradient(tf.matmul(inputs, activatorFiltered))

            ## same operation but unclipped:
            Inhib = tf.where(tf.less(kernel,0),kernel,Tzero)*(-1)*CinhibMat
            Activ = tf.where(tf.less(kernel,0),Tzero,kernel)*CactivMat
            activation2 = tf.matmul(inputs, Activ)
            inhibition2 = tf.matmul(inputs, Inhib)
            forBackProp = tf.divide(activation2,cp*kd+inhibition2)

            outputs = tf.stop_gradient(tf.divide(activation,cp*kd+inhibition)-forBackProp)
            return tf.add(forBackProp,outputs,name=scope)


class chemTemplateLayer(Dense):
    """
        Layers with a strange activation function of the form: activator*kernelActivator/(1+activator*kernelActivator+sumALLInhib*kernelInhibitor)
        Adds a constant bias to the input
        :param theta: bias to be added after each multipication
    """
    def __init__(self, deviceName, sparsity=0.9, min=-1, max=1, **kwargs):
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

    def build(self, input_shape):
        # We just change the way bias is added and remove it from trainable variable!
        input_shape = tf.TensorShape(input_shape)
        if input_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        if not input_shape.is_fully_defined():
            print("the input shape used for build is not fully defined")
        print("the input shape has rank"+str(input_shape.rank))
        self.input_spec = tf.keras.layers.InputSpec(shape = input_shape,dtype=tf.float32,min_ndim=2)
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


        self.E0 = tf.Variable(0,trainable=False,dtype=tf.float32)
        self.rescaleFactor = tf.Variable(1,dtype=tf.float32,trainable=False)

        #The competition, used with batches in mind!
        self.cps = tf.Variable(tf.zeros(input_shape[0],dtype=tf.float32), trainable=False)

        #other intermediates variable:
        self.k1M = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.Cactiv = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.k5M = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.k3M = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)
        self.Cinhib = tf.Variable(tf.zeros(variableShape,dtype=tf.float32),trainable=False)

        self.built = True


    def get_rescaleOps(self):
        Tone = tf.cast(tf.fill(self.kernel.shape,1),dtype=tf.float32)
        Tzero = tf.cast(tf.zeros(self.kernel.shape),dtype=tf.float32)
        rescaleFactor = tf.keras.backend.sum(tf.where(tf.less(self.kernel,0),Tone,Tzero)) + tf.keras.backend.sum(tf.where(tf.less(self.kernel,0),Tone,Tzero))
        return rescaleFactor

    def get_mask(self):
        return self.kernel

    def set_constants(self,constantArray,activInitC,inhibInitC,enzymeInit,computedRescaleFactor):
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
        self.E0.assign(enzymeInitTensor)

        #now we compute other intermediate values:

        self.k1M.assign(self.k1/(self.k1n+self.k2))
        self.Cactiv.assign(self.k2*self.k1M*self.E0*self.TA0)
        self.k5M.assign(self.k5/(self.k5+self.k6))
        self.k3M.assign(self.k3/(self.k3n+self.k4))
        self.Cinhib.assign(tf.stack([self.k6*self.k5M/self.kdT]*(self.k4.shape[1]),axis=1)*self.k4*self.k3M*self.E0*self.E0*self.TI0)

    def update_cp(self,cps):
        self.cps.assign(cps)

    def rescale(self,rescaleFactor):
        """
            Rescale the enzyme value
        :param rescaleFactor: 0d Tensor: float with the new rescale
        :return:
        """
        self.E0.assign(self.E0.read_value()*(rescaleFactor**0.5)/(self.rescaleFactor.read_value()**0.5))
        self.rescaleFactor.assign(rescaleFactor)

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs=chemTemplateClippedTensorDot(self.deviceName, inputs, self.kernel, rank, self.cps, self.Cactiv, self.Cinhib, self.kdI)
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = chemTemplateClippedMatMul(self.deviceName, inputs, self.kernel, self.cps, self.Cactiv, self.Cinhib, self.kdI)
        return outputs

    def get_config(self):
        config = {'cps for last batch': float(self.cps.read_value()), 'E0':float(self.E0.read_value())}
        base_config = super(chemTemplateLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

