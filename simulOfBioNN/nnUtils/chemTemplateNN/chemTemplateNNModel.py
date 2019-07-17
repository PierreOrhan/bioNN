"""
    Module for the Neural Network model we use in TF 2.0
        Tf 2.0 is still in beta and far from being perfectly documented. We found additional insights here: https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/.
"""


import tensorflow as tf
import numpy as np
from simulOfBioNN.nnUtils.chemTemplateNN.tensorflowFixedPointSearch import computeCPonly,cpComputer
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
        self.non_eager_building(input_shape)

        super(chemTemplateNNModel,self).build(input_shape)

        print("tensorflow is executing eagerly: "+str(tf.executing_eagerly()))
        from tensorflow.python.eager import context
        print("context is executing eagerly:"+str(context.executing_eagerly()))
        print("masks shape are:"+str(self.masks.shape))
        print("masks0 shape are "+str(self.masks.getRagged()[0].to_tensor().shape))

        assert len(reactionConstants) == 11
        assert type(enzymeInitC) == float
        assert type(activTempInitC) == float
        assert type(inhibTempInitC) == float
        if randomConstantParameter is not None:
            assert type(randomConstantParameter) == tuple

        self.rescaleFactor.assign(tf.keras.backend.sum([l.get_rescaleOps() for l in self.layerList],axis=-1))
        for l in self.layerList:
            l.set_constants(reactionConstants,enzymeInitC,activTempInitC,inhibTempInitC,self.rescaleFactor)


        self.k1= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k1)) for l in self.layerList])) #transpose enable to switch the shape format here as previously explained.
        self.k1n= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k1n)) for l in self.layerList]))
        self.k2= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k2)) for l in self.layerList]))
        self.k3= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k3)) for l in self.layerList]))
        self.k3n= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k3n)) for l in self.layerList]))
        self.k4= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k4)) for l in self.layerList]))
        self.k5= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k5)) for l in self.layerList]))
        self.k5n= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k5n)) for l in self.layerList]))
        self.k6= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.k6)) for l in self.layerList]))
        self.kdI= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.kdI)) for l in self.layerList]))
        self.kdT= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.kdT)) for l in self.layerList]))
        self.TA0= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.TA0)) for l in self.layerList]))
        self.TI0= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.TI0)) for l in self.layerList]))
        self.masks= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.get_mask())) for l in self.layerList]))


        self.E0.assign(self.layerList[0].E0)
        self.built = True

    def non_eager_building(self,input_shape):
        """
            When calling model.build parent method, tensorflow switches to non-eager mode.
            This non-eager mode has in its graph the previously defines tensor or variable ...
            BUT for some the initialization seems lost, [e.g dimensions  ( switching to None for my case )]
            This creates a failure at the time where the tensorflow model wishes to verify the call on this graph.

            Objectively this doesn't happen for tf.Variable object but occurs for RaggedTensor or Tensor....

            Therefore we could have concluded the following good practice:
                In the function call, in case the model has not been built, loads the tensor.
            BUT it goes against TF 2.0 and is a bad hack as next problem will reveal:

            SECOND PROBLEM:
            Each time we refer to a function define with a @tf.function annotation, tensorflow backend will build a NEW graph.
            in this new graph the tensors loose there initialization, and thus referring to a class's tensor or ragged tensor fails.
            The solution for tensor is to use Variable, but for ragged tensor this can't be done, as they can't be represented by a variable.
            To solve this problem two solutions can be adopted:
            First:
                We could define the @tf.function at the call level, where the graph is being made and nowhere afterwards where classes' ragged tensor are necessary.
                BUT we realised soon that the use of either tf.map_fn, or a loop of the form for .. in tf.range(
                    would also create new graphs. A solve could be to instentiate once again at each graph run...
                    but in the end we could not call the raggedtensor.shape for a tf.range( loop.
                    We conclude by the following: ragged tensor should not be used as if with TF 2.0,
                    which in its spirit asks to act on variable of Tensor, variable defined outside of every tf.function nesting.
            Second:
                We realized that ragged tensor are essentially built with two tensors. Thus we define a new class that stores the Variable object built on these tensor
                and propose a method to give a new ragged tensor at creation time.
                    To prevent any conflict these variable are not trainable...

        :return:
        """
        # To speed up function we use tf.rank in @tf.function rather then numpy range.
        # This gives an iterated "variable" that is no more an int but a tensor, thus unable to index list in graph mode.
        # Rather than a list of tensor we thus represent our network using ragged tensor where the last two dimensions (out of 3) are ragged !
        # !!CAREFULL!! : One problem is that these ragged tensor cannot be used as variable for now in tensorflow...

        modelsConstantShape =[(self.layerList[0].units,input_shape[-1])]+[(self.layerList[idx+1].units,l.units) for idx,l in enumerate(self.layerList[:-1])]
        modelsOutputConstantShape = [l.units for l in self.layerList]
        self.k1 = VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(self.layerList)]))
        self.k1n = VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(self.layerList)]))
        self.k2 = VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(self.layerList)]))
        self.k3 = VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(self.layerList)]))
        self.k3n = VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(self.layerList)]))
        self.k4 = VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(self.layerList)]))
        self.TA0 = VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(self.layerList)]))
        self.TI0 = VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32)) for idx,l in enumerate(self.layerList)]))

        self.k5 = VariableRaggedTensor(tf.RaggedTensor.from_row_lengths(values= tf.zeros(sum(modelsOutputConstantShape),dtype = tf.float32) ,row_lengths = modelsOutputConstantShape))
        self.k5n = VariableRaggedTensor(tf.RaggedTensor.from_row_lengths(values= tf.zeros(sum(modelsOutputConstantShape),dtype = tf.float32) ,row_lengths = modelsOutputConstantShape))
        self.k6 = VariableRaggedTensor(tf.RaggedTensor.from_row_lengths(values= tf.zeros(sum(modelsOutputConstantShape),dtype = tf.float32) ,row_lengths = modelsOutputConstantShape))
        self.kdI = VariableRaggedTensor(tf.RaggedTensor.from_row_lengths(values= tf.zeros(sum(modelsOutputConstantShape),dtype = tf.float32) ,row_lengths = modelsOutputConstantShape))
        self.kdT = VariableRaggedTensor(tf.RaggedTensor.from_row_lengths(values= tf.zeros(sum(modelsOutputConstantShape),dtype = tf.float32) ,row_lengths = modelsOutputConstantShape))

        self.E0 = tf.Variable(tf.zeros(1,dtype=tf.float32),trainable=False)
        self.masks = VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(modelsConstantShape[idx],dtype=tf.float32))for idx,l in enumerate(self.layerList)]))



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
        # if len(inputs.shape)==1:
        #     inputs=tf.stack(inputs)
        # if(not self.built):
        #     self.non_eager_building(inputs.shape)
        """
            We would like to proceed with a batching point of view.
            The problem here, is that tf.map_fn creates a graph for each realisation, making us loose the initialization on the current graph...
            Thus we cannot use it here, while this has not been fixed in tensorflow!
        """
        inputs = inputs/self.rescaleFactor
        cps = tf.TensorArray(dtype=tf.float32,size=inputs.shape[0])
        for idx in tf.range(inputs.shape[0]):
            # if(not self.built):
            #     self.non_eager_building(inputs.shape)
            cps.write(idx,computeCPonly(self.k1,self.k1n,self.k2,self.k3,self.k3n,self.k4,self.k5,self.k5n,self.k6,self.kdI,
                                        self.kdT,self.TA0,self.TI0,self.E0,inputs[idx],self.masks))
        if len(inputs.shape)>1:
            # X0s = tf.convert_to_tensor(inputs)
            if training:
                self.verifyMask()
            result = self.funcTesting(inputs)
        else:
            tf.print("using normal pipeline")
            # X0s = tf.convert_to_tensor(inputs)
            result = self.funcTesting(inputs)
        return result

    def verifyMask(self):
        tf.print("starting mask verifying")
        newRescaleFactor = tf.keras.backend.sum([l.get_rescaleOps() for l in self.layerList],axis=-1)
        if(False in tf.equal(self.rescaleFactor,newRescaleFactor)):
            self.rescaleFactor.assign(newRescaleFactor)
            for l in self.layerList:
                l.rescale(self.rescaleFactor)
        # We also need to update the information the model has: E0 and the masks might have change
        self.E0.assign(self.layerList[0].E0)
        self.masks= VariableRaggedTensor(tf.stack([tf.RaggedTensor.from_tensor(tf.transpose(l.get_mask()))] for l in self.layerList))
        tf.print("ended mask update")

    def lambdaComputecp(self):
        return lambda input: computeCPonly(self.k1,self.k1n,self.k2,self.k3,self.k3n,self.k4,self.k5,self.k5n,self.k6,self.kdI,
                             self.kdT,self.TA0,self.TI0,self.E0,input,self.masks)

    def funcTesting(self,inputs):

        print("tensorflow is executing eagerly: "+str(tf.executing_eagerly()))
        from tensorflow.python.eager import context
        print("funcTesting context is executing eagerly:"+str(context.executing_eagerly()))
        print("funcTesting masks0 on graph:"+str(self.masks[0].to_tensor().graph))
        print("funcTesting mask0 :"+str(self.masks[0].to_tensor()))

        print("starting cp computing")
        tf.print("starting cp computing")
        #rescaling of the inputs
        inputs = inputs/self.rescaleFactor

        """ 
            tf.map_fn creates a graph for each realisation, making us loose the initialization on the current graph...
            Thus we cannot use it here, while this has not been fixed in tensorflow!
        """

        cps = tf.TensorArray(dtype=tf.float32,size=inputs.shape[0])
        for idx in tf.range(inputs.shape[0]):
            cps.write(idx,computeCPonly(self.k1,self.k1n,self.k2,self.k3,self.k3n,self.k4,self.k5,self.k5n,self.k6,self.kdI,
                                       self.kdT,self.TA0,self.TI0,self.E0,inputs[idx],self.masks))

        gatheredCps = cps.gather(indices=tf.range(cps.size()))
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


class VariableRaggedTensor():
    def __init__(self,raggedTensor):
        self.var_rowsplits = tf.Variable(raggedTensor.row_splits,trainable=False)
        self.multiDim = False
        if type(raggedTensor.values)==tf.RaggedTensor:
            self.var_values = VariableRaggedTensor(raggedTensor.values)
            self.multiDim = True
        else:
            self.var_values = tf.Variable(raggedTensor.values,trainable=False)
        self.shape0 = tf.Variable(raggedTensor.shape[0],trainable=False)
    @tf.function
    def getRagged(self):
        if self.multiDim:
            stackList=[]
            for s in tf.range(self.shape0):
                stackList+=[self.var_values.getRagged()[s]]
            return tf.RaggedTensor.from_row_splits(row_splits=self.var_rowsplits,values=tf.stack(stackList))
        return tf.RaggedTensor.from_row_splits(row_splits=self.var_rowsplits,values=self.var_values)
