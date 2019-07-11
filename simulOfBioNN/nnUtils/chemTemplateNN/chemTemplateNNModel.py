import tensorflow as tf
import numpy as np
from simulOfBioNN.odeUtils.equilibrium import obtainBornSup,computeCPonly
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
    def __init__(self, sess, useGPU, nbUnits, sparsities, inputShape, reactionConstants,
                  enzymeInitC, activTempInitC, inhibTempInitC, randomConstantParameter=None):
        """
            Initialization of the model:
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
            Be carefull: we here consider the system to have been rescaled by using T0 and C0 constants.
        :param sess: tensorflow session: we use to to obtain GPU or CPU name where we dispatch some of our operations.
        :param useGPU: boolean, if True: we use GPU, else CPU.
        :param nbUnits: list, for each layer its number of units
        :param sparsities: the value of sparsity in each layer
        :param inputShape: int, size of the input.
                    We need it here because we need to compute at initialization time the rescale factor.
                    This way we don't have to compute it at every initialization.
        :param reactionConstants: standard value for the constant
        :param enzymeInitC: concentration value for the enzyme that proved to be working well for a small neuron.
                            Will be rescaled to compensate for larger number.
        :param activTempInitC: concentration value for the activating template that proved to be working well for a small neuron.
                            Will be rescaled to compensate for larger number.
        :param inhibTempInitC: concentration value for the inhibiting template that proved to be working well for a small neuron.
                            Will be rescaled to compensate for larger number.
        :param randomConstantParameter: tuple of size 2 of list, default to None. First value is used for template (list of size 2),
                                                                                  Second value for reactions constant (list of size reaction constants).
        """
        super(chemTemplateNNModel,self).__init__(name="")
        nbLayers = len(nbUnits)
        if sess is not None:
            self.sess = sess
        else:
            self.sess = tf.Session()
        deviceList = self.sess.list_devices()
        print("at initialization:"+str(tf.executing_eagerly()))
        if useGPU:
            for l in deviceList:
                if "GPU" in l.name:
                    Deviceidx = l.name
                    break
        else:
            for l in deviceList:
                if "CPU" in l.name:
                    Deviceidx= l.name
                    break

        # Test of the inputs
        assert len(sparsities)==len(nbUnits)
        assert len(reactionConstants) == 11
        assert type(enzymeInitC) == float
        assert type(activTempInitC) == float
        assert type(inhibTempInitC) == float
        if randomConstantParameter is not None:
            assert type(randomConstantParameter) == tuple
        self.layerList = []
        inputShapes = [inputShape]+nbUnits[:-1]
        for e in range(nbLayers):
            self.layerList += [chemTemplateLayer(Deviceidx,inputShapes[e],units=nbUnits[e],sparsity=sparsities[e])]
        #We force the initialization of the kernels in each layer, making sure they have a mask:
        # for l in self.layerList:
        #     sess.run(tf.global_variables_initializer())
        self.rescaleFactor = np.sum([l.get_rescaleOps().numpy() for l in self.layerList])
        print("rescale Factor: "+str(self.rescaleFactor))
        for l in self.layerList:
            l.set_constants(reactionConstants,enzymeInitC,activTempInitC,inhibTempInitC,self.rescaleFactor)
            l.rescale(self.rescaleFactor)

        self.k1 = [l.k1 for l in self.layerList]
        self.k1n = [l.k1n for l in self.layerList]
        self.k2 = [l.k2 for l in self.layerList]
        self.k3 = [l.k3 for l in self.layerList]
        self.k3n = [l.k3n for l in self.layerList]
        self.k4 = [l.k4 for l in self.layerList]
        self.k5 = [l.k5 for l in self.layerList]
        self.k5n = [l.k5n for l in self.layerList]
        self.k6 = [l.k6 for l in self.layerList]
        self.kdI = [l.kdI for l in self.layerList]
        self.kdT = [l.kdT for l in self.layerList]
        self.TA0 = [l.TA0 for l in self.layerList]
        self.TI0 = [l.TI0 for l in self.layerList]
        self.E0 = self.layerList[0].E0
        self.masks = [l.get_mask().numpy() for l in self.layerList]


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
        def func(input):
            # rescaling:
            if training:
                newRescaleFactor = np.sum([l.get_rescaleOps().numpy() for l in self.layerList])
                if self.rescaleFactor!= newRescaleFactor:
                    self.rescaleFactor = newRescaleFactor
                    for l in self.layerList:
                        l.rescale(self.rescaleFactor)
            # We also need to update the information the model has: we gather from all layers their weights matrix
                self.k1 = [l.k1 for l in self.layerList]
                self.k1n = [l.k1n for l in self.layerList]
                self.k2 = [l.k2 for l in self.layerList]
                self.k3 = [l.k3 for l in self.layerList]
                self.k3n = [l.k3n for l in self.layerList]
                self.k4 = [l.k4 for l in self.layerList]
                self.k5 = [l.k5 for l in self.layerList]
                self.k5n = [l.k5n for l in self.layerList]
                self.k6 = [l.k6 for l in self.layerList]
                self.kdI = [l.kdI for l in self.layerList]
                self.kdT = [l.kdT for l in self.layerList]
                self.TA0 = [l.TA0 for l in self.layerList]
                self.TI0 = [l.TI0 for l in self.layerList]
                self.E0 = self.layerList[0].E0
                self.masks = [l.get_mask().numpy() for l in self.layerList]

            #rescaling of the inputs
            input = input/self.rescaleFactor

            #We compute competition ==> dependant on the inputs, results of a fixed-point solve.
            cp = computeCPonly(self.k1,self.k1n,self.k2,self.k3,self.k3n,self.k4,self.k5,self.k5n,self.k6,self.kdI,
                               self.kdT,self.TA0,self.TI0,self.E0,input,self.masks,fittedValue=None,verbose=False)

            for l in self.layerList:
                l.updateCp(cp)

            x = self.layerList[0](inputs)
            for l in self.layerList[1:]:
                x = l(x)
            return x
        print("tensorflow is executing eagerly: "+str(tf.executing_eagerly()))
        print("model is executing eagerly: "+str(self.run_eagerly))
        from tensorflow.python.eager import context
        print("context is executing eagerly:"+str(context.executing_eagerly()))
        X0s = inputs
        result = [func(X0) for X0 in X0s]
        return tf.convert_to_tensor(np.array(result))

