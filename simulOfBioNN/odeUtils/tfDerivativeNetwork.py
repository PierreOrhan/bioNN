"""
    In this module we propose a neural network as the derivative of the chemical system.
    The goal is to be able to be plugged-in directly with tensorflow's tools.
"""
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import RepeatVector,Multiply
from tensorflow.python.keras import backend
from tensorflow import sparse


class derivativeNetwork(models.Model):

    def __init__(self, deviceIdx, KarrayA, inputStoichio, maskA, maskComplementary, derivLeak, isSparse = False):
        """
            Builds the neural network according to the provided masks.
            We begin with a dense coding, but prepare to switch to the sparse case, which will probably appear.
            In such case, tensorflow only enable product of at most 2D sparse matrix (rank==2 in tensorflow terms) against some dense matrix.
        :param deviceIdx: the device on which the computation shall be done
        :param KarrayA: A 2d-array, if sparse
        :param inputStoichio:
        :param maskA:
        :param maskComplementary:
        :param derivLeak:
        """
        super(derivativeNetwork, self).__init__()

        try:
            assert len(KarrayA.shape) == 2
            assert len(inputStoichio.shape) == 3
            assert len(maskA.shape) == 3
            assert len(maskComplementary.shape) == 3
        except:
            print("Wrong shape for masks")
            raise

        self.repeat = RepeatVector(maskA.shape[1])
        self.maskIdxList = []
        self.maskComplementaryList = []
        self.stoichioList = []
        self.deviceIdx = deviceIdx
        self.isSparse = isSparse

        if not isSparse:
            for m in range(maskA.shape[0]):
                self.maskIdxList += [tf.convert_to_tensor(maskA[m])]
                self.maskComplementaryList += [tf.convert_to_tensor(maskComplementary[m])]
                self.stoichioList += [tf.convert_to_tensor(inputStoichio[m])]

                self.tfmask = tf.convert_to_tensor(maskA)
                self.tfMaskComplementary = tf.convert_to_tensor(maskComplementary)
                self.tfStoichio = tf.convert_to_tensor(inputStoichio)

        else:
            """
                In tensorflow, the element-wise product is not defined for sparse matrix.
                Therefore two possible solution is offered to us:
                    We can either compute a boolean mask for each row (of last axis) of the 3D masks, and compute on these masks and then aggregate.
                    We can also try to use the sparse_embedding_lookup function.
                We implement the second solution.        
                       
            """
            self.maskWeightList = []
            ## First we need to convert to the look-up made of two sparse matrix
            ## In the sparse matrix module, .coords is defined as a (ndim,nnz) shaped array.
            for m in range(maskA.shape[0]):
                coords = []
                idxValues = []
                weightValues = []
                for idx,e in maskA.coords[0]:
                    if(e==m):
                        coords += [maskA.coords[1:,idx]]
                        idxValues += [maskA.coords[2,idx]]
                        weightValues += [inputStoichio.data[idx]]
                if(len(coords)>0):
                    self.maskIdxList += [sparse.SparseTensor(indices=coords, values=idxValues, dense_shape=maskA.shape[1:])]
                    self.maskWeightList += [sparse.SparseTensor(indices=coords, values=weightValues,dense_shape=maskA.shape[1:])]

        self.Karray = tf.convert_to_tensor(KarrayA)
        self.derivLeak = tf.convert_to_tensor(derivLeak,dtype=self.Karray.dtype)

    def call(self, inputs, training=None, mask=None):
        """
            Compute the derivative for every species, on the device given by self.deviceIdx (set at initialization of the object).
        :param inputs: 1d-array or tensor of the concentration of input species
        :param training: legacy, do not use
        :param mask: legacy, do not use
        :return: The derivative for every species in a Tensor.
        """
        # with tf.device(self.deviceIdx):
        if(not self.isSparse):
            #outputs = []
            # for idx,mask2 in enumerate(self.maskIdxList):
            #     # The following expression is described precisely in the docs.
            #     outputs += [backend.sum(tf.multiply(tf.log(tf.multiply(mask2,X)+self.maskComplementaryList[idx]),self.stoichioList[idx]),axis=1)]
            # x = tf.multiply(
            #         tf.exp(backend.stack(outputs,axis=0)),
            #         self.Karray)
            x = tf.multiply(
                            tf.exp(backend.sum(tf.multiply(tf.log(tf.multiply(self.tfmask,inputs)+self.tfMaskComplementary),self.tfStoichio),axis=2)),
                            self.Karray)
            derivatives = backend.sum(x,axis=0) + self.derivLeak
            return tf.reshape(derivatives,(1,derivatives.shape[0]))
        else:
            outputs=[]
            x2 = tf.log(inputs)
            for idx,mask2 in enumerate(self.masklist):
                outputs += [tf.exp(tf.nn.embedding_lookup_sparse(x2,mask2,self.maskWeightList[idx],combiner="sum"))]
            x = tf.multiply(
                    backend.stack(outputs,axis=0),
                    self.Karray)
            return backend.sum(x,axis=0) + self.derivLeak

