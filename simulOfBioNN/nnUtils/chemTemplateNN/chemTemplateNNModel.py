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
    def __init__(self, sess, useGPU, nbUnits, sparsities, reactionConstants,enzymeInitC,activTempInitC, inhibTempInitC, randomConstantParameter=None):
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
            print("computing on GPU")
            Deviceidx = "/gpu:0"
        else:
            Deviceidx = "/cpu:0"
        # Test of the inputs
        assert len(sparsities)==len(nbUnits)

        self.layerList = []
        for e in range(nbLayers):
            self.layerList += [chemTemplateLayer(Deviceidx,units=nbUnits[e],sparsity=sparsities[e],dynamic=False)]

        self.cpLayer = chemTemplateCpLayer(Deviceidx)

        self.reactionConstants = reactionConstants
        self.enzymeInitC = enzymeInitC
        self.activTempInitC = activTempInitC
        self.inhibTempInitC = inhibTempInitC
        self.randomConstantParameter =randomConstantParameter

        assert len(self.reactionConstants) == 11
        assert type(self.enzymeInitC) == float
        assert type(self.activTempInitC) == float
        assert type(self.inhibTempInitC) == float
        if self.randomConstantParameter is not None:
            assert type(self.randomConstantParameter) == tuple
            #TODO: implement randomized constants

        self.writer = tf.summary.create_file_writer("tfOUT")
        self.writer.set_as_default()


        self.built = False
        self.gathered = None

    def build(self,input_shape):
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

        modelsConstantShape =[(input_shape[-1],self.layerList[0].units)]+[(l.units,self.layerList[idx+1].units) for idx,l in enumerate(self.layerList[:-1])]
        input_shapes = [input_shape]+[(input_shape[0],l.units) for l in self.layerList[:-1]]
        for idx,l in enumerate(self.layerList):
            l.build(input_shapes[idx])

        self.cpLayer.build(input_shape,self.layerList)

        self.rescaleFactor.assign(tf.keras.backend.sum([l.get_rescaleFactor() for l in self.layerList], axis=-1))
        for l in self.layerList:
            l.set_constants(self.reactionConstants,self.enzymeInitC,self.activTempInitC,self.inhibTempInitC,self.rescaleFactor)

        self.cpLayer.assignConstantFromLayers(self.layerList)
        self.cpLayer.assignMasksFromLayers(self.layerList)
        self.cpLayer.E0.assign(self.layerList[0].E0)
        self.meanGatheredCps = tf.Variable(0,dtype=tf.float32,trainable=False)
        self.built = True

        super(chemTemplateNNModel,self).build(input_shape)
        print("model successfully built")

    @tf.function
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
        """
            We would like to proceed with a batching point of view.
            The problem here, is that tf.map_fn creates a graph for each realisation, making us loose the initialization on the current graph...
            Thus we cannot use it here, while this has not been fixed in tensorflow!
        """
        inputs = tf.convert_to_tensor(inputs)
        if training:
            self.verifyMask()
        inputs = inputs/self.rescaleFactor
        gatheredCps = tf.stop_gradient(self.cpLayer(inputs))
        self.meanGatheredCps.assign(tf.reduce_mean(gatheredCps))
        #tf.summary.scalar("mean_cp",data=tf.reduce_mean(gatheredCps),step=tf.summary.experimental.get_step())
        x = self.layerList[0](inputs,cps=gatheredCps)
        for l in self.layerList[1:]:
            x = l(x,cps=gatheredCps)
        return x

    @tf.function
    def verifyMask(self):
        newRescaleFactor = tf.stop_gradient(tf.keras.backend.sum([l.get_rescaleFactor() for l in self.layerList], axis=-1))
        if(not tf.equal(self.rescaleFactor,newRescaleFactor)):
            self.rescaleFactor.assign(newRescaleFactor)
            for l in self.layerList:
                l.rescale(self.rescaleFactor)
        # We also need to update the information the model has: E0 and the masks might have change
        self.cpLayer.E0.assign(self.layerList[0].E0)
        self.cpLayer.assignMasksFromLayers(self.layerList)

    def updateArchitecture(self,masks,reactionsConstants,enzymeInitC,activTempInitc,inhibTempInitC):
        for idx,m in enumerate(masks):
            self.layerList[m].set_mask(m)

        self.reactionConstants = reactionsConstants
        self.enzymeInitC = enzymeInitC
        self.activTempInitC = activTempInitc
        self.inhibTempInitC = inhibTempInitC

        self.rescaleFactor.assign(tf.keras.backend.sum([l.get_rescaleFactor() for l in self.layerList], axis=-1))
        for l in self.layerList:
            l.set_constants(self.reactionConstants,self.enzymeInitC,self.activTempInitC,self.inhibTempInitC,self.rescaleFactor)

        self.cpLayer.assignConstantFromLayers(self.layerList)
        self.cpLayer.assignMasksFromLayers(self.layerList)

    def logCp(self,epoch,logs=None):
        cp = self.meanGatheredCps
        tf.summary.scalar('mean cps', data= cp, step=epoch)
        return cp
