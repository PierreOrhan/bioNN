from tensorflow.python.keras.layers import Dense
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.eager import context
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import common_shapes
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import nn
from simulOfBioNN.nnUtils.classicalTfNet.clippedSparseBioDenseLayer import weightFixedAndClippedConstraint,sparseInitializer,constant_initializer,layerconstantInitiliaizer

def sigClippedTensorDot(deviceName,inputs,kernel,rank):
    """
        Clipped tensorDot on which we apply on a sigmoid function of the form activator*kernelActivator/1+activator*kernelActivator+sumALLInhib*kernelInhibitor
        Clipping follow the rule: weights<0.2 take value -1, weighs>0.2 take value 1 and other take value 0.
    """
    with tf.device(deviceName):
        with ops.name_scope("clippedTensorDotOp", [inputs, kernel]) as scope:
            Tminus = tf.cast(tf.fill(kernel.shape,-1),dtype=tf.float32)
            Tplus = tf.cast(tf.fill(kernel.shape,1),dtype=tf.float32)
            Tzero = tf.cast(tf.fill(kernel.shape,0),dtype=tf.float32)
            clippedKernel=tf.where(tf.less(kernel,-0.2),Tminus,tf.where(tf.less(kernel,0.2),Tzero,Tplus))
            #compute sum of inhibitors:
            Xzero=tf.cast(tf.fill(kernel.shape[:-1],0),dtype=tf.float32)
            minLine=tf.keras.backend.min(clippedKernel,axis=-1)
            filtered=tf.transpose(tf.stack([tf.where(minLine<0,-1*minLine,Xzero)]))
            sumALLInhib=tf.stop_gradient(tf.matmul(inputs,filtered))
            #kernel for activators:
            kernelActivator = tf.stop_gradient(tf.where(tf.less(kernel,0),Tzero,clippedKernel))
            activator = tf.stop_gradient(tf.tensordot(inputs, kernelActivator, [[rank - 1], [0]],name="clipped"))

            forBackProp = standard_ops.tensordot(inputs,kernel,[[rank - 1], [0]],name="normal")
            outputs = tf.stop_gradient(tf.divide(activator,1+activator+sumALLInhib)-forBackProp)
            return tf.add(forBackProp,outputs,name=scope)

def sigClippedMatMul(deviceName,inputs,kernel):
    '''
        Clipped matmul on which we apply on a sigmoid function of the form activator*kernelActivator/1+activator*kernelActivator+sumALLInhib*kernelInhibitor
        Clipping follow the rule: weights<0.2 take value -1, weighs>0.2 take value 1 and other take value 0.
    '''
    with tf.device(deviceName):
        with ops.name_scope("clippedMatMulOp", [inputs, kernel]) as scope:
            Tminus = tf.cast(tf.fill(kernel.shape,-1),dtype=tf.float32)
            Tplus = tf.cast(tf.fill(kernel.shape,1),dtype=tf.float32)
            Tzero = tf.cast(tf.fill(kernel.shape,0),dtype=tf.float32)
            #clipp the kernel at 0:
            clippedKernel=tf.stop_gradient(tf.where(tf.less(kernel,-0.2),Tminus,tf.where(tf.less(kernel,0.2),Tzero,Tplus)))
            #compute sum of inhibitors:
            Xzero=tf.cast(tf.fill(kernel.shape[:-1],0),dtype=tf.float32)
            minLine=tf.keras.backend.min(clippedKernel,axis=-1)
            filtered=tf.transpose(tf.stack([tf.where(minLine<0,-1*minLine,Xzero)]))
            sumALLInhib=tf.stop_gradient(tf.matmul(inputs,filtered))
            #kernel for activators:
            kernelActivator=tf.stop_gradient(tf.where(tf.less(kernel,0),Tzero,clippedKernel))
            product=tf.stop_gradient(tf.matmul(inputs, kernelActivator))
            forBackProp = tf.matmul(inputs,kernel,name="normal")
            outputs = tf.stop_gradient(tf.divide(product,1+product+sumALLInhib)-forBackProp)
            return tf.add(forBackProp,outputs,name=scope)


class clippedSparseBioSigLayer(Dense):
    """
        Layers with a strange activation function of the form: activator*kernelActivator/(1+activator*kernelActivator+sumALLInhib*kernelInhibitor)
        Adds a constant bias to the input
        :param theta: bias to be added after each multipication
    """

    def __init__(self,deviceName,biasValue=[1.0], fractionZero=0.9, min=-1, max=1, **kwargs):
        """

            :param deviceName: device to use for computation
            :param biasValue: either None(random but constant through layer) or 1d-array of size units (nb of output neurons).
            :param fractionZero: fraction of weight that should remain at 0
            :param min: min value for weights
            :param max: max value for weights
            :param kwargs:
        """
        super(clippedSparseBioSigLayer, self).__init__(**kwargs)
        self.supports_masking = True
        if biasValue:
            self.hasPredefinedBias = True
            self.theta = K.cast_to_floatx(biasValue)
        else:
            self.hasPredefinedBias = False
        self.sparseInitializer = sparseInitializer(fractionZero, minval=min, maxval=max)
        self.deviceName=deviceName #the device on which the main operations will be conducted (forward and backward propagations)

    def build(self, input_shape):
        # We just change the way bias is added and remove it from trainable variable!
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.sparseInitializer,
            regularizer=self.kernel_regularizer,
            constraint=weightFixedAndClippedConstraint(self.sparseInitializer),
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            if(self.hasPredefinedBias):
                value = [self.theta[idx] for idx in range(self.units)]
                self.bias_initializer = constant_initializer(value)
            else:
                self.bias_initializer = layerconstantInitiliaizer(-1,1)
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=None,
                constraint=None,
                dtype=self.dtype,
                trainable=False)  # the bias cannot be trained
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs=sigClippedTensorDot(self.deviceName,inputs,self.kernel,rank)
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = sigClippedMatMul(self.deviceName,inputs,self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def get_config(self):
        config = {'theta': float(self.theta)}
        base_config = super(clippedSparseBioSigLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

