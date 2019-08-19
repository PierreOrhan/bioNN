from tensorflow.python.keras.layers import Dense
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.eager import context
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import common_shapes
from tensorflow.python.keras import backend as K
import tensorflow as tf
from simulOfBioNN.nnUtils.classicalTfNet.clippedSparseBioDenseLayer import weightFixedAndClippedConstraint,sparseInitializer,constant_initializer,layerconstantInitiliaizer

def clippedTensorDot(inputs, kernel, rank, bias ,rate, rateInhib):
    """
        Clipped tensorDot on which we apply on a sigmoid function of the form activator*kernelActivator/1+activator*kernelActivator+sumALLInhib*kernelInhibitor
        Clipping follow the rule: weights<0.2 take value -1, weighs>0.2 take value 1 and other take value 0.
    """
    Tminus = tf.cast(tf.fill(kernel.shape,-1),dtype=tf.float32)
    Tplus = tf.cast(tf.fill(kernel.shape,1),dtype=tf.float32)
    Tzero = tf.cast(tf.fill(kernel.shape,0),dtype=tf.float32)
    clippedKernel = tf.where(tf.less(kernel,-0.2),Tminus,tf.where(tf.less(kernel,0.2),Tzero,Tplus))

    # Kernel for activators:
    kernelActivator = tf.where(tf.less(clippedKernel,0),Tzero,clippedKernel)
    kernelInhibitor = tf.where(tf.less(clippedKernel,0.),clippedKernel,Tzero)
    activator = standard_ops.tensordot(inputs, kernelActivator, [[rank - 1], [0]],name="clippedActivator")
    inhibitor = standard_ops.tensordot(inputs,kernelInhibitor,[[rank - 1], [0]],name="clippedInhibitor")

    # For backpropagation:
    kernelActivator_bp = tf.where(tf.less(kernel,0),Tzero,kernel)
    kernelInibhitor_bp = tf.where(tf.less(kernel,0.),kernel,Tzero)
    bp_activator = standard_ops.tensordot(inputs,kernelActivator_bp,[[rank - 1], [0]],name="normalActivator")
    bp_inhibitor = standard_ops.tensordot(inputs,kernelInibhitor_bp,[[rank - 1], [0]],name="normalInhibitor")

    forBackProp = tf.divide(rate*bp_activator-bias+bp_inhibitor,bp_activator)

    outputs = tf.stop_gradient(tf.divide(rate*activator-bias+inhibitor,activator)-forBackProp)
    return tf.add(forBackProp,outputs)

def clippedMatMul(inputs, kernel,bias, rate, rateInhib):
    '''
        Clipped matmul on which we apply on a sigmoid function of the form activator*kernelActivator/1+activator*kernelActivator+sumALLInhib*kernelInhibitor
        Clipping follow the rule: weights<0.2 take value -1, weighs>0.2 take value 1 and other take value 0.
    '''
    Tminus = tf.cast(tf.fill(kernel.shape,-1),dtype=tf.float32)
    Tplus = tf.cast(tf.fill(kernel.shape,1),dtype=tf.float32)
    Tzero = tf.cast(tf.fill(kernel.shape,0),dtype=tf.float32)
    #clipp the kernel at 0:
    clippedKernel=tf.stop_gradient(tf.where(tf.less(kernel,-0.2),Tminus,tf.where(tf.less(kernel,0.2),Tzero,Tplus)))

    # Kernel for activators:
    kernelActivator = tf.where(tf.less(clippedKernel,0),Tzero,clippedKernel)
    kernelInhibitor = tf.where(tf.less(clippedKernel,0),clippedKernel,Tzero)
    activator = tf.matmul(inputs, kernelActivator, name="clippedActivator")
    inhibitor = tf.matmul(inputs,kernelInhibitor, name="clippedInhibitor")

    # For backpropagation:
    kernelActivator_bp = tf.where(tf.less(kernel,0),Tzero,kernel)
    kernelInibhitor_bp = tf.where(tf.less(kernel,0),kernel,Tzero)
    bp_activator = tf.matmul(inputs,kernelActivator_bp,name="normalActivator")
    bp_inhibitor = tf.matmul(inputs,kernelInibhitor_bp,name="normalInhibitor")
    forBackProp = tf.divide(rate*bp_activator+-bias+rateInhib*bp_inhibitor,bp_activator)

    outputs = tf.stop_gradient(tf.divide(rate*activator+-bias+rateInhib*inhibitor,activator)-forBackProp)
    return tf.add(forBackProp,outputs)


class autoRegulLayer(Dense):
    """
        Layers with an activation function of the form: (WX-bias)/(activator*kernelActivator)
        :param biasValue: bias to be added after each multipication
    """

    def __init__(self,biasValue=[1.0], fractionZero=0.9, min=-1, max=1, rate = 10., rateInhib = 10. ,use_bias = True, gpuName=None, **kwargs):
        """

            :param biasValue: either None(random but constant through layer) or 1d-array of size units (nb of output neurons).
            :param fractionZero: fraction of weight that should remain at 0
            :param min: min value for weights
            :param max: max value for weights
            :param kwargs:
            :param rate: float constant, the rate of production
            :param gpuName: for compatibility reasons, not used
        """
        super(autoRegulLayer, self).__init__(**kwargs)
        self.supports_masking = True
        if not biasValue is None:
            self.hasPredefinedBias = True
            self.theta = K.cast_to_floatx(biasValue)
        else:
            self.hasPredefinedBias = False
        assert fractionZero<=1 and fractionZero>=0
        self.sparseInitializer = sparseInitializer(fractionZero, minval=min, maxval=max)
        assert type(rate)==float
        self.rate = rate
        assert type(rateInhib) == float
        self.rateInhib = rateInhib
        self.use_bias = use_bias

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
            outputs = clippedTensorDot(inputs, self.kernel, rank,self.bias,self.rate, self.rateInhib)
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = clippedMatMul(inputs, self.kernel,self.bias,self.rate, self.rateInhib)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

