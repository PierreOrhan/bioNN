"""
    Module for the Neural Network model we use in TF 2.0
        Tf 2.0 is still in beta and far from being perfectly documented. We found additional insights here: https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/.
"""


import tensorflow as tf
import numpy as np
from simulOfBioNN.nnUtils.chemTemplateNN.tensorflowFixedPointSearch import computeCPonly
from simulOfBioNN.nnUtils.chemTemplateNN.chemTemplateLayers import chemTemplateLayer
from simulOfBioNN.nnUtils.chemTemplateNN.chemTemplateCpLayer import chemTemplateCpLayer
from tensorflow import initializers

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

        self.cpLayer = chemTemplateCpLayer(Deviceidx)

        self.built = False

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

        # CONCERNING THE SHAPE WE DO NOT USE TRADITIONNAL (INPUT,OUTPUT) format in the way we compute cp.
        #   Therefore we define the following heuristic here:
        #       ==> at the model level we store the reaction constants with this shape format (output,input) so that they are used for cp computation
        #       ==> at the layer model we store the reaction constants with the (input,ouput) shape format

        # We need to Initiate variable for the call to the herited build which will compile the call and thus need it.
        # BUT at the time of this function the model is still in eager mode, and the graph activated in the non-eager mode doesn't get all informations
        # THUS this instantation shall be made in non-eager mode, and the only moment to catch it is within the call function (see call).
        #self.non_eager_building(input_shape)


        super(chemTemplateNNModel,self).build(input_shape)
        modelsConstantShape =[(input_shape[-1]),self.layerList[0].units]+[(l.units,self.layerList[idx+1].units) for idx,l in enumerate(self.layerList[:-1])]
        for idx,l in enumerate(self.layerList):
            l.build(modelsConstantShape[idx])

        self.cpLayer.build(input_shape,self.layerList)

        print("mask for cpLayer shape are "+str([[k.shape for k in e] for e in self.cpLayer.masks.getRagged()]))
        mask = tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.get_mask())) for l in self.layerList])
        print("mask for all layer is computed as "+str([[k.shape for k in e] for e in mask]))

        assert len(reactionConstants) == 11
        assert type(enzymeInitC) == float
        assert type(activTempInitC) == float
        assert type(inhibTempInitC) == float
        if randomConstantParameter is not None:
            assert type(randomConstantParameter) == tuple

        self.rescaleFactor.assign(tf.keras.backend.sum([l.get_rescaleOps() for l in self.layerList],axis=-1))
        for l in self.layerList:
            l.set_constants(reactionConstants,enzymeInitC,activTempInitC,inhibTempInitC,self.rescaleFactor)

        mask = tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.get_mask())) for l in self.layerList])
        print("mask for all layer is computed as "+str([[k.shape for k in e] for e in mask]))

        self.cpLayer.assignConstantFromLayers(self.layerList)
        self.cpLayer.assignMasksFromLayers(self.layerList)
        self.cpLayer.E0.assign(self.layerList[0].E0)
        self.built = True
        print("At end of build")
        print("mask0 for cp layer is computed as "+str(self.cpLayer.masks.getRagged()[0].shape))

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
        if(self.built):
            tf.print("model has well been built and a call is tried")
            mask = tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.get_mask())) for l in self.layerList])
            tf.print("mask for all layer is computed as "+str(mask[0].shape))
            tf.print("mask0 for cp layer is computed as "+str(self.cpLayer.masks.getRagged()[0].shape))
            self.cpLayer.assignMasksFromLayers(self.layerList)
            tf.print("mask0 for cpLayer after assign is computed as "+str(self.cpLayer.masks.getRagged()[0].shape))
        """
            We would like to proceed with a batching point of view.
            The problem here, is that tf.map_fn creates a graph for each realisation, making us loose the initialization on the current graph...
            Thus we cannot use it here, while this has not been fixed in tensorflow!
        """
        if(self.built):
            if len(inputs.shape)>1:
                # X0s = tf.convert_to_tensor(inputs)
                if training:
                    self.verifyMask()
                result = self.funcTesting(inputs)
            else:
                tf.print("using normal pipeline")
                # X0s = tf.convert_to_tensor(inputs)
                result = self.funcTesting(inputs)
        else:
            result = tf.zeros(10,tf.float32)
        return result

    def verifyMask(self):
        tf.print("starting mask verifying")
        newRescaleFactor = tf.keras.backend.sum([l.get_rescaleOps() for l in self.layerList],axis=-1)
        if(False in tf.equal(self.rescaleFactor,newRescaleFactor)):
            self.rescaleFactor.assign(newRescaleFactor)
            for l in self.layerList:
                l.rescale(self.rescaleFactor)
        # We also need to update the information the model has: E0 and the masks might have change
        self.cpLayer.E0.assign(self.layerList[0].E0)
        self.cpLayer.masks.assign(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.get_mask())) for l in self.layerList]))
        tf.print("ended mask update")

    def funcTesting(self,inputs):
        inputs = inputs/self.rescaleFactor
        """ 
            tf.map_fn creates a graph for each realisation, making us loose the initialization on the current graph...
            Thus we cannot use it here, while this has not been fixed in tensorflow!
        """
        gatheredCps = self.cpLayer(inputs)
        print("ended cp computing,starting output computation")
        tf.print("ended cp computing,starting output computation")
        for l in self.layerList:
            l.update_cp(gatheredCps)
        x = self.layerList[0](inputs)
        for l in self.layerList[1:]:
            tf.print("next layer")
            x = l(x)
        tf.print("ended output computaton")
        return x