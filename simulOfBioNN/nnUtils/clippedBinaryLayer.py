"""
    Definition of a binary layer with tensorflow.
"""

from tensorflow.python.keras.layers import Dense
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops

import tensorflow as tf


class _weightFixedAndClippedConstraint(constraints.Constraint):
    """
        Constraint on weights values:
            Restrict weights between -1 and 1
    """
    def __call__(self, w):
        return tf.clip_by_value(w,tf.convert_to_tensor(-1.0),tf.convert_to_tensor(1.0))

def _clippedTensorDot(deviceName, inputs, kernel, rank):
    with tf.device(deviceName):
        with ops.name_scope("clippedTensorDotOp", [inputs, kernel]) as scope:
            Tminus = tf.cast(tf.fill(kernel.shape,-1),dtype=tf.float32)
            Tplus = tf.cast(tf.fill(kernel.shape,1),dtype=tf.float32)
            clippedKernel=tf.where(tf.less(kernel,0),Tminus,Tplus)
            forBackProp = standard_ops.tensordot(inputs,kernel,[[rank - 1], [0]],name="normal")
            outputs = tf.stop_gradient(tf.tensordot(inputs, clippedKernel, [[rank - 1], [0]],name="clipped")-forBackProp)
            return tf.add(forBackProp,outputs,name=scope)
def _clippedMatMul(deviceName, inputs, kernel):
    '''
        After extensive research, it appeared that if you want the following behavior:
            Y = def_op(f(x)) in forward pass
            grad = grad(def_op(x))
        (In our case f is the clipping and def_op is a tf basic operation like MatMul), the only solution that doesn't use C re-implementation of the ops is to use:
            Y = def_op(f(x))-def_op(x)+def_op(x) and to forbid gradient computing on the first part of the graph.
        Nonetheless, such writing double forward time for this op, making it very bad, a C implementation should be used instead!
    '''
    with tf.device(deviceName):
        with ops.name_scope("clippedMatMul", [inputs, kernel]) as scope:
            Tminus = tf.cast(tf.fill(kernel.shape,-1),dtype=tf.float32)
            Tplus = tf.cast(tf.fill(kernel.shape,1),dtype=tf.float32)
            clippedKernel=tf.where(tf.less(kernel,0),Tminus,Tplus)
            forBackProp = tf.matmul(inputs,kernel,name="normal")
            outputs = tf.stop_gradient(tf.matmul(inputs, clippedKernel,name="clipped")-forBackProp)
            return tf.add(forBackProp,outputs,name=scope)

class clippedBinaryLayer(Dense):
    """
        A binary layer.
            On prediction, weights are projected at either -1 or 1
            On backpropagaton the real value of weight is used.
            Finally we also binarize the activation by clipping it at -1 or 1.
    """
    def __init__(self,deviceName,**kwargs):
        """
            The initialization of a binary layer only needs a device name and classical args for Dense layer, see tensorflow doc for them.
            Main operations will be conducted on this device.
        :param deviceName: The name of the device
        :param kwargs: classical argument for dense layer.
        """
        super(clippedBinaryLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.deviceName=deviceName  #the device on which the main operations will be conducted (forward and backward propagations)

    def build(self, input_shape):
        """
            Building function, based on the original one from Dense.
            We implemented a constraint on the weight and bias: clipped between -1 and 1.
            Initialization also takes place in this range of values.
            Please note that training is enabled in bias.
        :param input_shape:
        :return:
        """
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
            initializer=initializers.random_uniform(-1,1),
            regularizer=self.kernel_regularizer,
            constraint=_weightFixedAndClippedConstraint(), # where we have our constraint
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=initializers.random_uniform(-1,1),
                regularizer=None,
                constraint=_weightFixedAndClippedConstraint(),
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        """
            Apply a clipped matrix multiplication and sign activation function.
            Backpropagation is made on non-clipped and without sign expression.
        :param inputs:
        :return:
        """
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs=_clippedTensorDot(self.deviceName, inputs, self.kernel, rank)
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = _clippedMatMul(self.deviceName, inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation==tf.nn.softmax:
            outputs = self.activation(outputs)  # pylint: disable=not-callable
            #We must not clip the activation on either -1 or 1, because the activation is the softmax!!
            return outputs
        #We must clip the activation on either -1 or 1, use of sign function:
        with tf.device(self.deviceName):
            with ops.name_scope("signOp", [outputs]) as scope:
                return tf.add(tf.stop_gradient(tf.sign(outputs)-outputs),outputs,name=scope) #Use of stop_gradient enable rigorous backpropagation here