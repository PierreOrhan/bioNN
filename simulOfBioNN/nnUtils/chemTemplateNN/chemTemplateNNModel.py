"""
    Module for the Neural Network model we use in TF 2.0
        Tf 2.0 is still in beta and far from being perfectly documented. We found additional insights here: https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/.
"""


import tensorflow as tf
import numpy as np
from simulOfBioNN.nnUtils.chemTemplateNN.tensorflowFixedPointSearch import computeCPonly
from simulOfBioNN.nnUtils.chemTemplateNN.chemTemplateLayers import chemTemplateLayer

class chemTemplateNNModel(tf.keras.Model):
    """
        Implement the model we derived from our mathematical analysis at equilibrium of a CRN implemented with the template model.
        We make two assumption:
            First we neglect variations of templates concentrations. This assumption makes our model biased
            Secondly, at backpropagation we neglect the derivative of cp with respect to either a template concentration or a node concentration.
                (TODO: prove this second assumption can be made)
                This second assumption makes our training heuristic less correct but far more easy to implement.
        We provide support for the simple model:
            exonuclease reactions model at order 1, polymerase and nickase are grouped under the same enzyme.
    """
    def __init__(self, sess, useGPU, nbUnits, sparsities):
        """
            Initialization of the model:
                    Here we simply add layers, they remain to be built once the input shape is known.
                    Next steps are made at the build level, when the input shape becomes usable.
            Be careful: we here consider the system to have been rescaled by using T0 and C0 constants.
        :param sess: tensorflow session: we use to to obtain GPU or CPU name where we dispatch some of our operations.
        :param useGPU: boolean, if True: we use GPU, else CPU.
        :param nbUnits: list, for each layer its number of units
        :param sparsities: the value of sparsity in each layer
                """
        super(chemTemplateNNModel,self).__init__(name="")
        nbLayers = len(nbUnits)

        if tf.test.is_gpu_available():
            Deviceidx = "/gpu:0"
        else:
            Deviceidx = "/cpu:0"
        # Test of the inputs
        assert len(sparsities)==len(nbUnits)

        self.layerList = []
        for e in range(nbLayers):
            self.layerList += [chemTemplateLayer(Deviceidx,units=nbUnits[e],sparsity=sparsities[e],dynamic=True)]

    def build(self,input_shape,reactionConstants,enzymeInitC,activTempInitC, inhibTempInitC, randomConstantParameter=None):
        """
            Build the model and make the following initialization steps:
                1) Intermediate layer use a random heuristic to obtain a sparse initialization
                2) Based on the initialized topology, we can rescale our system.
                        We diminish the concentration of template.
                        We increase the concentration of enzyme.
                3) We then provide each layer with initialization values for their real weights.
                    The kernel of each layer is based on weights in {-1,0,1},
                    but at computation time we multiply these weights by a value describing the chemistry.
                4) If the random constant parameter is provided (default to None), we use a gaussian noise to randomise:
                        1) the concentration of template.
                        2) the constant parameters for each reactions.
        :param input_shape:
        :param reactionConstants: standard value for the constants
        :param enzymeInitC: concentration value for the enzyme that proved to be working well for a small neuron.
                             Will be rescaled to compensate for larger number.
        :param activTempInitC: concentration value for the activating template that proved to be working well for a small neuron.
                                Will be rescaled to compensate for larger number.
        :param inhibTempInitC: concentration value for the inhibiting template that proved to be working well for a small neuron.
                                Will be rescaled to compensate for larger number.
        :param randomConstantParameter: randomConstantParameter: tuple of size 2 of list, default to None. First value is used for template (list of size 2),
                                        Second value for reactions constant (list of size reaction constants).
        :return:
        """
        self.rescaleFactor = tf.Variable(0,trainable=False,dtype=tf.float32)
        # Initiate variable for the call to the herited build which will compile the call and thus need it.
        # CONCERNING THE SHAPE WE DO NOT USE TRADITIONNAL (INPUT,OUTPUT) format in the way we compute cp.
        #   Therefore we define the following heuristic here:
        #       ==> at the model level we store the reaction constants with this shape format (output,input) so that they are used for cp computation
        #       ==> at the layer model we store the reaction constants with the (input,ouput) shape format
        modelsConstantShape =[(self.layerList[0].units,input_shape[-1])]+[(self.layerList[idx+1].units,l.units) for idx,l in enumerate(self.layerList[:-1])]
        modelsOutputConstantShape = [l.units for l in self.layerList]

        print("models shapes are "+str(modelsConstantShape))
        print("outputs shapes are"+str(modelsOutputConstantShape))
        self.k1 = [tf.Variable(tf.zeros(modelsConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]
        self.k1n = [tf.Variable(tf.zeros(modelsConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]
        self.k2 = [tf.Variable(tf.zeros(modelsConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]
        self.k3 = [tf.Variable(tf.zeros(modelsConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]
        self.k3n = [tf.Variable(tf.zeros(modelsConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]
        self.k4 = [tf.Variable(tf.zeros(modelsConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]
        self.TA0 = [tf.Variable(tf.zeros(modelsConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]
        self.TI0 = [tf.Variable(tf.zeros(modelsConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]

        self.k5 = [tf.Variable(tf.zeros(modelsOutputConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]
        self.k5n = [tf.Variable(tf.zeros(modelsOutputConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]
        self.k6 = [tf.Variable(tf.zeros(modelsOutputConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]
        self.kdI = [tf.Variable(tf.zeros(modelsOutputConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]
        self.kdT = [tf.Variable(tf.zeros(modelsOutputConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]

        self.E0 = tf.Variable(tf.zeros(1,dtype=tf.float32),trainable=False)
        self.masks = [tf.Variable(tf.zeros(modelsConstantShape[idx],dtype=tf.float32),trainable=False) for idx,l in enumerate(self.layerList)]

        super(chemTemplateNNModel,self).build(input_shape)

        assert len(reactionConstants) == 11
        assert type(enzymeInitC) == float
        assert type(activTempInitC) == float
        assert type(inhibTempInitC) == float
        if randomConstantParameter is not None:
            assert type(randomConstantParameter) == tuple

        self.rescaleFactor.assign(tf.keras.backend.sum([l.get_rescaleOps() for l in self.layerList],axis=-1))
        for l in self.layerList:
            l.set_constants(reactionConstants,enzymeInitC,activTempInitC,inhibTempInitC,self.rescaleFactor)
        for idx,l in enumerate(self.layerList):
            self.k1[idx].assign(tf.transpose(l.k1)) #transpose enable to switch the shape format here as previously explained.
            self.k1n[idx].assign(tf.transpose(l.k1n))
            self.k2[idx].assign(tf.transpose(l.k2))
            self.k3[idx].assign(tf.transpose(l.k3))
            self.k3n[idx].assign(tf.transpose(l.k3n))
            self.k4[idx].assign(tf.transpose(l.k4))
            self.k5[idx].assign(tf.transpose(l.k5))
            self.k5n[idx].assign(tf.transpose(l.k5n))
            self.k6[idx].assign(tf.transpose(l.k6))
            self.kdI[idx].assign(tf.transpose(l.kdI))
            self.kdT[idx].assign(tf.transpose(l.kdT))
            self.TA0[idx].assign(tf.transpose(l.TA0))
            self.TI0[idx].assign(tf.transpose(l.TI0))
            self.masks[idx].assign(tf.transpose(l.get_mask().numpy()))
        self.E0.assign(self.layerList[0].E0)

    def call(self, inputs, training=None, mask=None):
        """
            Call function for out chemical model.
                At training time we verify whether the model has changed total number of template due to backpropagation.
                    In this case we compute new values for the rescale factor, and update in all template the real weights.
        :param inputs: tensor for the inputs, if of rank >= 2, we need to split it.
        :param training: if we are in training or inference mode (we save time of checking model change)
        :param mask: we don't use it here...
        :return:
        """
        # if len(inputs.shape)==1:
        #     inputs=tf.stack(inputs)
        print("tensorflow is executing eagerly: "+str(tf.executing_eagerly()))
        # print("model is executing eagerly: "+str(self.run_eagerly))
        from tensorflow.python.eager import context
        print("context is executing eagerly:"+str(context.executing_eagerly()))
        if len(inputs.shape)>1:
            X0s = tf.convert_to_tensor(inputs)
            if training:
                self.verifyMask()
            result = self.funcTesting(X0s)
        else:
            tf.print("using normal pipeline")
            X0s = tf.convert_to_tensor(inputs)
            result = self.funcTesting(X0s)
        return result

    @tf.function
    def verifyMask(self):
        tf.print("starting mask verifying")
        newRescaleFactor = tf.keras.backend.sum([l.get_rescaleOps() for l in self.layerList],axis=-1)
        if(False in tf.equal(self.rescaleFactor,newRescaleFactor)):
            self.rescaleFactor.assign(newRescaleFactor)
            for l in self.layerList:
                l.rescale(self.rescaleFactor)
        # We also need to update the information the model has: E0 and the masks might have change
        self.E0.assign(self.layerList[0].E0)
        for idx in range(len(self.layerList)): #becareful! enumerate is not yet supported in tf.function
            self.masks[idx].assign(tf.transpose(self.layerList[idx].get_mask().numpy()))
        tf.print("ended mask update")

    @tf.function
    def lambdaComputecp(self,input):
        return computeCPonly(self.k1,self.k1n,self.k2,self.k3,self.k3n,self.k4,self.k5,self.k5n,self.k6,self.kdI,
                             self.kdT,self.TA0,self.TI0,self.E0,input,self.masks)
    @tf.function
    def funcTesting(self,inputs):
        print("starting cp computing")
        tf.print("starting cp computing")
        #rescaling of the inputs
        inputs = inputs/self.rescaleFactor
        cps = tf.map_fn(self.lambdaComputecp,inputs)
        print("ended cp computing,starting output computation")
        tf.print("ended cp computing,starting output computation")
        for l in self.layerList:
            l.update_cp(cps)
        x = self.layerList[0](inputs)
        for l in self.layerList[1:]:
            tf.print("next layer")
            x = l(x)
        tf.print("ended output computaton")
        return x

