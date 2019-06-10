"""
    Definition of a sparse layer with tensorflow.
    This layer starts with a forced sparsity.
    It also clips its weights in forward pass to {-1,0,1}, increasing sparsity.
    Bias are constant by layer and not trainable.
"""


from tensorflow import constant_initializer
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from numpy.random import choice
import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops

import tensorflow as tf


class weightFixedAndClippedConstraint(constraints.Constraint):
    """
        Constraint on weights values:
            If the weight was initially set to 0, then we don't allow it to move, it is fixed at 0.
        In version 0_2_0 we add a new clip for values: we restrict them between -1 and 1
    """
    def __init__(self, sparseInitializer):
        """
        :param sparseInitializer: the initializer, if called it will have created a mask for the weigh that we will use here.
                                  The mask should have 1 for weights that should be clipped to 0.
        """
        self.sparseInitializer=sparseInitializer

    def __call__(self, w):
        r1=tf.where(self.sparseInitializer.mask==0,w,tf.zeros(w.shape))
        return tf.clip_by_value(r1,tf.convert_to_tensor(-1.0),tf.convert_to_tensor(1.0))

def sparse_random_uniform(newshape,shape,indices,minval=0,maxval=None,dtype=dtypes.float32,seed=None,name=None):
    '''
        This operator is built as a combination of tensorflow ops. It is very slow but gives the possibility to set a distribution that only takes into account
        the point that will have a none-zero value in the model (during and after training)
    '''
    with ops.name_scope(name, "sparse_random_uniform", [newshape,shape, minval, maxval,indices]) as name:
        init = random_ops.random_uniform(newshape,minval,maxval, dtype, seed=seed,name=name)
        params = tf.concat([tf.zeros((shape[0],1)),init],axis=1)
        finalInit = []
        for e in range(shape[0]):
            finalInit += [tf.gather(params[e],indices=indices[e])]
        res1 = tf.stack(finalInit)

        ## second solution:
        # Xindices = tf.keras.backend.repeat_elements(tf.expand_dims(e,axis=1),indices.shape[1],axis=1)
        # MatIndices = tf.concatenate([Xindices,indices],axis=1)
        # res2 = params[MatIndices[:,:,0],MatIndices[:,:,1]]

        return res1


class sparseInitializer(initializers.Initializer):
    """
        Initializer that generates tensors with a sparse uniform distribution.
    """
    def __init__(self, fractionZero, minval=0, maxval=None, seed=None, dtype=dtypes.float32):
        """
            Initializer that generates tensors with a sparse uniform distribution.
            :param fractionZero: The fraction of weight that are initialized as zero
            :param minval: A python scalar or a scalar tensor. Lower bound of the range of random values to generate.
            :param maxval: A python scalar or a scalar tensor. Upper bound of the range of random values to generate.  Defaults to 1 for float types.
            :param seed: A Python integer. Used to create random seeds. See `tf.set_random_seed` for behavior.
            :param dtype: Default data type, used if no `dtype` argument is provided when calling the initializer.
        """
        self.fractionZero = fractionZero
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        # We reduce the shape to take into account the fraction of values that are zeros
        if len(shape) != 2:
            raise ValueError("shape should be a tupple of length 2")
        # We distribute uniformly among lines our zeros, this is a good practice, especially if the input sparsity is also uniformly scattered

        numberOfZeroInLine = int(self.fractionZero * shape[1])
        # we want at least one zero
        if (numberOfZeroInLine > 0):
            newshape = (shape[0], shape[1] - numberOfZeroInLine)
            indices = self._computeCuts(shape,numberOfZeroInLine)
            self.mask=self._getweightMask(shape,indices)
            return sparse_random_uniform(newshape,shape,indices,self.minval,self.maxval,dtype=dtype, seed=self.seed)
        else:
            # The number of enforced-input for each output neurons is less than one for a uniform distribution over output neurons.
            # This might occur only if the asked sparsity is really low or if the number of neurons is small.
            # In both case we just set the sparsity to 0 for this layer in order to stay simple
            print("Asked sparsity for this layer is too small or the layer is very small itself, setting 0 sparsity")
            finalInit = random_ops.random_uniform(shape, self.minval, self.maxval, dtype, seed=self.seed)
            self.mask=self._getweightMask(shape,[])
            return finalInit

    def _computeCuts(self,shape,numberOfZeroInLine):
        """
            Produce indices array for use with the tf.gather function.
            The params array shall be created by tensorflow random operator, with the first variable as 0.
            Our indices then follow the following rules: we add 0 for zeros weight, otherwize we add the position in the future random array, starting from 1.
            :param shape: The shape of the layer
            :param numberOfZeroInLine: The number of zero weights induced in each neurons
            :return: shape[1]-d array, an array to use in a tf.gather function.
        """
        indices=np.zeros((shape[0],shape[1]),dtype = np.int64)
        for e in range(shape[0]):
            posOfFull = choice(shape[1], shape[1]-numberOfZeroInLine, replace=False)
            posOfFull = np.sort(posOfFull)
            indices[e,posOfFull] = np.arange(1,shape[1]-numberOfZeroInLine+1)

        return indices

    def _getweightMask(self,shape,indices):
        mask=np.where(indices==0,np.zeros(shape)+1,np.zeros(shape))
        return mask

def clippedTensorDot(deviceName,inputs,kernel,rank):
    """
        Clipped tensorDot
        Clipping follow the rule: weights<0.2 take value -1, weighs>0.2 take value 1 and other take value 0.
    """
    with tf.device(deviceName):
        with ops.name_scope("clippedTensorDotOp", [inputs, kernel]) as scope:
            Tminus = tf.cast(tf.fill(kernel.shape,-1),dtype=tf.float32)
            Tplus = tf.cast(tf.fill(kernel.shape,1),dtype=tf.float32)
            Tzero = tf.cast(tf.fill(kernel.shape,0),dtype=tf.float32)
            clippedKernel=tf.where(tf.less(kernel,-0.2),Tminus,tf.where(tf.less(kernel,0.2),Tzero,Tplus))
            forBackProp = standard_ops.tensordot(inputs,kernel,[[rank - 1], [0]],name="normal")
            outputs = tf.stop_gradient(tf.tensordot(inputs, clippedKernel, [[rank - 1], [0]],name="clipped")-forBackProp)
            return tf.add(forBackProp,outputs,name=scope)

def clippedMatMul(deviceName,inputs,kernel):
    '''
        Clipped Matrix multiplication
        Clipping follow the rule: weights<0.2 take value -1, weighs>0.2 take value 1 and other take value 0.
            After extensive research, it appeared that if you want the following behavior:
                Y = def_op(f(x)) in forward pass
                grad = grad(def_op(x))
            (In our case f is the clipping and def_op is a tf basic operation like MatMul), the only solution that doesn't use C re-implementation of the ops is to use:
                Y = def_op(f(x))-def_op(x)+def_op(x) and to forbid gradient computing on the first part of the graph.
            Nonetheless, such writing double forward time for this op, making it very bad, a C implementation should be used instead!

    '''
    with tf.device(deviceName):
        with ops.name_scope("clippedMatMulOp", [inputs, kernel]) as scope:
            Tminus = tf.cast(tf.fill(kernel.shape,-1),dtype=tf.float32)
            Tplus = tf.cast(tf.fill(kernel.shape,1),dtype=tf.float32)
            Tzero = tf.cast(tf.fill(kernel.shape,0),dtype=tf.float32)
            clippedKernel=tf.stop_gradient(tf.where(tf.less(kernel,-0.2),Tminus,tf.where(tf.less(kernel,0.2),Tzero,Tplus)))
            # forBackProp = gen_math_ops.mat_mul(inputs,kernel,name="MyMatMul")
            forBackProp = tf.matmul(inputs,kernel,name="normal")
            outputs = tf.stop_gradient(tf.matmul(inputs, clippedKernel,name="clipped")-forBackProp)
            return tf.add(forBackProp,outputs,name=scope)

class layerconstantInitiliaizer(initializers.Initializer):
    """
        Initiliazer for the bias. Generates random but constant bias for the weights.

    """
    def __init__(self, minval=0, maxval=None, seed=None, dtype=dtypes.float32):
        """
            Initiliazer for the bias. Generates random but constant bias for the weights.
            :param minval: A python scalar or a scalar tensor. Lower bound of the range of random values to generate.
            :param maxval: A python scalar or a scalar tensor. Upper bound of the range of random values to generate.  Defaults to 1 for float types.
            :param seed: A Python integer. Used to create random seeds. See `tf.set_random_seed` for behavior.
            :param dtype: Default data type, used if no `dtype` argument is provided when calling the initializer.
        """
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        return self._layer_random_uniform(shape,self.minval,self.maxval,dtype,self.seed)

    def _layer_random_uniform(self,shape,minval=0,maxval=None,dtype=dtypes.float32,seed=None,name=None):
        '''
            Operation fo the random initialization of the bias.
        '''
        with ops.name_scope(name, "layer_random_uniform", [shape, minval, maxval]) as name2:
            init = random_ops.random_uniform([1],minval,maxval, dtype, seed=seed,name=name2)
            return tf.keras.backend.repeat_elements(init,shape[0],axis=0)


class clippedSparseBioDenseLayer(Dense):
    """
        A layer parametrize by sparsity, applied for each output neurons at every stage.
        The bias value is constant through each layer.
        It can either be initialized randomly or given (for repeated tests)
   """
    def __init__(self,deviceName,biasValue=None, fractionZero=0.9, min=-1, max=1, **kwargs):
        """

            :param deviceName: device to use for computation
            :param biasValue: either None(random but constant through layer) or 1d-array of size units (nb of output neurons).
            :param fractionZero: fraction of weight that should remain at 0
            :param min: min value for weights
            :param max: max value for weights
            :param kwargs:
        """
        super(clippedSparseBioDenseLayer, self).__init__(**kwargs)
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
            outputs=clippedTensorDot(self.deviceName,inputs,self.kernel,rank)
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = clippedMatMul(self.deviceName,inputs,self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def get_config(self):
        if(self.hasPredefinedBias):
            config = {'biasValues': float(self.theta)}
            base_config = super(clippedSparseBioDenseLayer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        return super(clippedSparseBioDenseLayer, self).get_config()
