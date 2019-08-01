"""
    Module for the Neural Network model we use in TF 2.0
        Tf 2.0 is still in beta and far from being perfectly documented. We found additional insights here: https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/.
"""


import tensorflow as tf
import numpy as np
from simulOfBioNN.nnUtils.chemCascadeNet.chemCascadeLayers import chemCascadeLayer
from simulOfBioNN.nnUtils.chemCascadeNet.chemNonLinearLayer import chemNonLinearLayer


class chemCascadeNNModel(tf.keras.Model):
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
    def __init__(self, nbUnits, sparsities, reactionConstantsCascade,reactionConstantsNL,
                 enzymeInitC,activTempInitC, inhibTempInitC,
                 activTempInitCNL,sizeInput,XglobalinitC,
                 randomConstantParameter=None, usingLog = True, usingSoftmax = True ):
        """
            Initialization of the model:
                    Here we simply add layers, they remain to be built once the input shape is known.
                    Next steps are made at the build level, when the input shape becomes usable.
            Be careful: we here consider the system to have been rescaled by using T0 and C0 constants.
        :param nbUnits: list, for each layer its number of units
        :param sparsities: the value of sparsity in each layer
                """
        super(chemCascadeNNModel, self).__init__(name="")
        nbLayers = len(nbUnits)

        if tf.test.is_gpu_available():
            print("computing on GPU")
            Deviceidx = "/gpu:0"
        else:
            Deviceidx = "/cpu:0"
        # Test of the inputs
        assert len(sparsities)==len(nbUnits)

        self.nlLayerList = [chemNonLinearLayer(Deviceidx,units=sizeInput,usingLog=usingLog)]
        self.CascadeLayerList = []
        self.layerList = []
        for e in range(nbLayers):
            self.layerList += [(chemCascadeLayer(Deviceidx, units=nbUnits[e], sparsity=sparsities[e], dynamic=False, usingLog=usingLog),chemNonLinearLayer(Deviceidx, units=nbUnits[e], usingLog=usingLog))]
            self.CascadeLayerList += [self.layerList[-1][0]]
            self.nlLayerList += [self.layerList[-1][1]]


        self.reactionConstantsCascade = reactionConstantsCascade
        self.reactionConstantsNL = reactionConstantsNL
        self.enzymeInitC = enzymeInitC
        self.XglobalinitC = XglobalinitC
        self.activTempInitC = activTempInitC
        self.inhibTempInitC = inhibTempInitC
        self.activTempInitCNL = activTempInitCNL
        self.randomConstantParameter =randomConstantParameter


        assert len(self.reactionConstantsCascade) == 17
        assert len(self.reactionConstantsNL) == 4
        assert type(self.enzymeInitC) == float
        assert type(self.activTempInitC) == float
        assert type(self.inhibTempInitC) == float
        if self.randomConstantParameter is not None:
            assert type(self.randomConstantParameter) == tuple
            #TODO: implement randomized constants

        # If the log mode is activated:
        #   - initial inputs should be provided as log(value)
        #   - each layer will gives the log(value at equilibrium) as output instead of "value at equilibrium"
        self.usingLog = usingLog
        self.usingSoftmax = usingSoftmax

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

        # We need to Initiate variable for the call to the herited build which will compile the call and thus need it.
        # BUT at the time of this function the model is still in eager mode, and the graph activated in the non-eager mode doesn't get all informations
        # THUS this instantation shall be made in non-eager mode, and the only moment to catch it is within the call function (see call).
        #self.non_eager_building(input_shape)

        modelsConstantShape = [(input_shape[-1],self.CascadeLayerList[0].units)] + [(l.units, self.CascadeLayerList[idx + 1].units) for idx, l in enumerate(self.CascadeLayerList[:-1])]
        input_shapes = [input_shape]+[(input_shape[0],l.units) for l in self.CascadeLayerList[:-1]]
        for idx,l in enumerate(self.CascadeLayerList):
            l.build(input_shapes[idx])
        for idx,l in enumerate(self.nlLayerList):
            l.build()

        self.rescaleFactor.assign(tf.keras.backend.sum([l.get_rescaleFactor() for l in self.CascadeLayerList], axis=-1))
        self.nlLayerList[0].set_constants(self.reactionConstantsNL,self.enzymeInitC,self.activTempInitCNL,self.rescaleFactor)
        for idx,l in enumerate(self.CascadeLayerList):
            l.set_constants(self.reactionConstantsCascade, self.enzymeInitC, self.activTempInitC, self.inhibTempInitC, self.rescaleFactor,self.XglobalinitC)
            self.nlLayerList[idx].set_constants(self.reactionConstantsNL,self.enzymeInitC,self.activTempInitCNL,self.rescaleFactor)

        self.meanGatheredCps = tf.Variable(0,dtype=tf.float32,trainable=False)
        self.built = True
        self.mycps = tf.Variable(tf.zeros(1),dtype=tf.float32,trainable=False)
        super(chemCascadeNNModel, self).build(input_shape)

        print("model successfully built")

    def force_rescale(self,rescaleFactor):
        """
            Force the rescale factor to take the input value instead of the number of nodes in the graph.
        :param rescaleFactor: rescale value
        :return:
        """
        if not self.built:
            raise Exception("model should be built before calling this function")

        for l in self.CascadeLayerList:
            l.rescale(rescaleFactor)
        for l in self.nlLayerList:
            l.rescale(rescaleFactor)

        self.rescaleFactor.assign(rescaleFactor)

    def greedy_set_cps(self,inputs):
        inputs = tf.convert_to_tensor(inputs,dtype=tf.float32)
        cps = self.obtainCp(inputs)
        tf.print(cps)
        tf.print(tf.nn.moments(cps,axes=[0]))
        self.mycps.assign([tf.reduce_mean(cps)])

    @tf.function
    def call(self, inputs, training=None, mask=None):
        """
            Call function for out chemical model.
                At training time we verify whether the model has changed total number of template due to backpropagation.
                    In this case we compute new values for the rescale factor, and update in all template the real weights.
        :param inputs: tensor for the inputs, if of rank >= 2, we need to split it.
                       if self.usinglog is true, the inputs should be in logarithmic scale...
        :param training: if we are in training or inference mode (we save time of checking model change)
        :param mask: we don't use it here...
        :return:
        """
        """
            We would like to proceed with a batching point of view.
            The problem here, is that tf.map_fn creates a graph for each realisation, making us loose the initialization on the current graph...
            Thus we cannot use it here, while this has not been fixed in tensorflow!
        """
        inputs = tf.convert_to_tensor(inputs,dtype=tf.float32)

        if training:
            self.verifyMask()

        inputs = inputs

        gatheredCps = tf.stop_gradient(tf.fill([tf.shape(inputs)[0]],tf.reshape(self._obtainCp(inputs[0]),())))
        gatheredCps = tf.reshape(gatheredCps,((tf.shape(inputs)[0],1)))
        tf.assert_equal(tf.shape(gatheredCps),(tf.shape(inputs)[0],1))
        #
        # gatheredCps = tf.stop_gradient(self.obtainCp(inputs))
        # gatheredCps = tf.fill([tf.shape(inputs)[0]],tf.reshape(self.mycps,()))
        # gatheredCps = tf.reshape(gatheredCps,((tf.shape(inputs)[0],1)))

        #self.meanGatheredCps.assign(tf.reduce_mean(gatheredCps))
        #tf.summary.scalar("mean_cp",data=tf.reduce_mean(gatheredCps),step=tf.summary.experimental.get_step())


        x = self.layerList[0][1](inputs, cps=gatheredCps, isFirstLayer=True)
        for idx,l in enumerate(self.layerList):
            x = l[0](x,cps=gatheredCps)
            x = l[1](x,cps=gatheredCps)

        if self.usingSoftmax:
            if self.usingLog:
                s = tf.keras.activations.softmax(tf.exp(x))
            else:
                s = tf.keras.activations.softmax(x)
        else:
            s = x
        return s

    @tf.function
    def predConcentration(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        inputs = inputs/self.rescaleFactor
        gatheredCps = tf.stop_gradient(self.obtainCp(inputs))
        #self.meanGatheredCps.assign(tf.reduce_mean(gatheredCps))
        #tf.summary.scalar("mean_cp",data=tf.reduce_mean(gatheredCps),step=tf.summary.experimental.get_step())
        x = self.nlLayerList[0](inputs, cps=gatheredCps, isFirstLayer=True)
        for l in self.layerList:
            x = l[0](x,cps=gatheredCps)
            x = l[1](x,cps=gatheredCps)
        return x


    @tf.function
    def obtainCp(self,inputs):
        gatheredCps = tf.map_fn(self._obtainCp,inputs,parallel_iterations=32,back_prop=False)

        return gatheredCps

    @tf.function
    def _obtainCp(self,input):
        #first we obtain the born sup:
        if(input.shape.rank<2):
            input = tf.reshape(input,(1,tf.shape(input)[0]))
        bornsupcp,x = self.nlLayerList[0].layer_cp_born_sup(input)
        layercp = tf.fill([1],1.)
        for l in self.layerList:
            layercp,x = l[0].layer_cp_born_sup(x)
            bornsupcp += layercp
            layercp,x = l[1].layer_cp_born_sup(x)
            bornsupcp += layercp
        #then we solve the fixed point:

        bornsupcp = tf.where(tf.math.is_inf(bornsupcp),1.*10**(20),bornsupcp) # float32 manages value up to 10**40 ...

        cpmin = tf.fill([1],1.)
        cp = self.brentq(self._computeCPdiff,cpmin,bornsupcp,input)
        return cp

    @tf.function
    def _computeCPdiff(self,cp,input):
        new_cp = tf.fill([1],1.)
        layercp,x = self.nlLayerList[0].layer_cp_equilibrium(cp, input, isFirstLayer=True)
        new_cp += layercp
        for l in self.layerList:
            layercp,x = l[0].layer_cp_equilibrium(cp,x)
            new_cp += layercp
            layercp,x = l[1].layer_cp_equilibrium(cp,x)
            new_cp += layercp
        new_cp = new_cp-cp
        return (-1)*new_cp

    # def measureModelRelevancy(self,input):
    #     inputs = tf.stack([input])
    #     if(input.shape.rank<2):
    #         input = tf.reshape(input,(1,tf.shape(input)[0]))
    #     cp = self.obtainCp(inputs)
    #     Inhib = []
    #     cp2kd = []
    #     x,b,c=self.CascadeLayerList[0].get_inhib_and_output(cp, input, isFirstLayer=True)
    #     Inhib += [b]
    #     cp2kd += [c]
    #     for l in self.CascadeLayerList[1:]:
    #         x,b,c=self.CascadeLayerList[0].get_inhib_and_output(cp, x, isFirstLayer=True)
    #         Inhib += [b]
    #         cp2kd += [c]
    #     return Inhib,cp2kd

    def lamda_computeCPdiff(self,input):
        if(input.shape.rank<2):
            input = tf.reshape(input,(1,tf.shape(input)[0]))
        return lambda cp: self._computeCPdiff(cp,input)

    @tf.function
    def getFunctionStyle(self,cpArray,X0):
        """
            Given an input, compute the value of f(cp) for all cp in cpArray (where f is the function we would like to find the roots with brentq)
        :param cpArray:
        :param X0:
        :return:
        """
        cpArray=tf.cast(tf.convert_to_tensor(cpArray),dtype=tf.float32)
        X0 = tf.cast(tf.convert_to_tensor(X0),dtype=tf.float32)
        func = self.lamda_computeCPdiff(X0)
        gatheredCps = tf.map_fn(func,cpArray,parallel_iterations=32,back_prop=False)
        return gatheredCps

    @tf.function
    def verifyMask(self):
        newRescaleFactor = tf.stop_gradient(tf.keras.backend.sum([l[0].get_rescaleFactor()+l[1].get_rescaleFactor() for l in self.layerList], axis=-1)+self.nlLayerList[0].get_rescaleFactor())
        if(not tf.equal(self.rescaleFactor,newRescaleFactor)):
            self.rescaleFactor.assign(newRescaleFactor)
            self.nlLayerList[0].rescale(self.rescaleFactor)
            for l in self.layerList:
                l[0].rescale(self.rescaleFactor)
                l[1].rescale(self.rescaleFactor)

    def updateArchitecture(self,masks,reactionsConstants,reactionConstantsNL,
                           enzymeInitC,activTempInitC,inhibTempInitC,
                           activTempInitCNL):
        for idx,m in enumerate(masks):
            self.CascadeLayerList[idx].set_mask(tf.convert_to_tensor(m, dtype=tf.float32))

        self.reactionConstantsCascade = reactionsConstants
        self.reactionConstantsNL = reactionConstantsNL
        self.activTempInitCNL = activTempInitCNL
        self.enzymeInitC = enzymeInitC
        self.activTempInitC = activTempInitC
        self.inhibTempInitC = inhibTempInitC

        self.rescaleFactor.assign(tf.keras.backend.sum([l[0].get_rescaleFactor()+l[1].get_rescaleFactor() for l in self.layerList], axis=-1)+self.nlLayerList[0].get_rescaleFactor())
        self.nlLayerList[0].set_constants(self.reactionConstantsCascade, self.enzymeInitC, self.activTempInitC, self.inhibTempInitC)
        for l in self.layerList:
            l[0].set_constants(self.reactionConstantsCascade, self.enzymeInitC, self.activTempInitC, self.inhibTempInitC, self.rescaleFactor,self.XglobalinitC)
            l[1].set_constants(self.reactionConstantsNL,self.enzymeInitC,self.activTempInitCNL,self.rescaleFactor)

    def logCp(self,epoch,logs=None):
        cp = self.meanGatheredCps
        tf.summary.scalar('mean cps', data= cp, step=epoch)
        return cp

    @tf.function
    def brentq(self, f, xa, xb,args=(),xtol=tf.constant(10**(-12)), rtol=tf.constant(4.4408920985006262*10**(-16)),iter=tf.constant(100)):
        xpre = tf.fill([1],0.) + xa
        xcur = tf.fill([1],0.) + xb
        xblk = tf.fill([1],0.)
        fblk = tf.fill([1],0.)
        spre = tf.fill([1],0.)
        scur = tf.fill([1],0.)
        fpre = f(xpre, args)
        fcur = f(xcur, args)
        tf.assert_less((fpre*fcur)[0],0.)

        if tf.equal(fpre[0],0):
            return xpre
        if tf.equal(fcur[0],0):
            return xcur
        else:
            tf.assert_greater(fcur,0.)

        for i in tf.range(iter):
            if tf.less((fpre*fcur)[0],0):
                xblk = xpre
                fblk = fpre
                spre = xcur - xpre
                scur = xcur - xpre
            if tf.less(tf.abs(fblk)[0],tf.abs(fcur)[0]):
                xpre = xcur
                xcur = xblk
                xblk = xpre

                fpre = fcur
                fcur = fblk
                fblk = fpre

            delta = (xtol + rtol*tf.abs(xcur))/2
            sbis = (xblk - xcur)/2
            if tf.equal(fcur[0],0) or tf.less(tf.abs(sbis)[0],delta[0]):
                break #BREAK FAILS HERE!!! ==> strange behavior?
            else:
                if tf.greater(tf.abs(spre)[0],delta[0]) and tf.less(tf.abs(fcur)[0],tf.abs(fpre)[0]):
                    if tf.equal(xpre[0],xblk[0]):
                        # /* interpolate */
                        stry = -fcur*(xcur - xpre)/(fcur - fpre)
                    else :
                        # /* extrapolate */
                        dpre = (fpre - fcur)/(xpre - xcur)
                        dblk = (fblk - fcur)/(xblk - xcur)
                        stry = -fcur*(fblk*dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))

                    mymin = tf.minimum(tf.abs(spre), 3*tf.abs(sbis) - delta) #Here would not understand...
                    spre=tf.where(tf.less(2*tf.abs(stry)-mymin,0),scur,sbis)
                    scur=tf.where(tf.less(2*tf.abs(stry)-mymin,0),stry,sbis)
                else:
                    # /* bisect */
                    spre = sbis
                    scur = sbis
                xpre = xcur
                fpre = fcur
                if tf.greater(tf.abs(scur)[0],delta[0]):
                    xcur += scur
                else:
                    if tf.greater(sbis[0],0):
                        xcur += delta
                    else:
                        xcur += -delta
            fcur = f(xcur, args)
        return xcur