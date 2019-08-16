"""
    Module for the Neural Network model we use in TF 2.0
        Tf 2.0 is still in beta and far from being perfectly documented. We found additional insights here: https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/.
"""


import tensorflow as tf
from simulOfBioNN.nnUtils.chemTemplateNN.chemTemplateLayers import chemTemplateLayer
# from simulOfBioNN.nnUtils.chemTemplateNN.chemTemplateCpLayer import chemTemplateCpLayer

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
    def __init__(self, nbUnits, sparsities, reactionConstants,enzymeInitC,activTempInitC, inhibTempInitC, randomConstantParameter=None, usingLog = True, usingSoftmax = True ):
        """
            Initialization of the model:
                    Here we simply add layers, they remain to be built once the input shape is known.
                    Next steps are made at the build level, when the input shape becomes usable.
            Be careful: we here consider the system to have been rescaled by using T0 and C0 constants.
        :param nbUnits: list, for each layer its number of units
        :param sparsities: the value of sparsity in each layer
                """
        super(chemTemplateNNModel,self).__init__(name="",dtype=tf.float64)
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
            self.layerList += [chemTemplateLayer(Deviceidx,units=nbUnits[e],sparsity=sparsities[e],dynamic=False,usingLog=usingLog,dtype=tf.float64)]

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
        self.rescaleFactor = tf.Variable(0,trainable=False,dtype=tf.float64)

        # We need to Initiate variable for the call to the herited build which will compile the call and thus need it.
        # BUT at the time of this function the model is still in eager mode, and the graph activated in the non-eager mode doesn't get all informations
        # THUS this instantation shall be made in non-eager mode, and the only moment to catch it is within the call function (see call).
        #self.non_eager_building(input_shape)

        modelsConstantShape =[(input_shape[-1],self.layerList[0].units)]+[(l.units,self.layerList[idx+1].units) for idx,l in enumerate(self.layerList[:-1])]
        input_shapes = [input_shape]+[(input_shape[0],l.units) for l in self.layerList[:-1]]
        for idx,l in enumerate(self.layerList):
            l.build(input_shapes[idx])

        self.rescaleFactor.assign(tf.keras.backend.sum([l.get_rescaleFactor() for l in self.layerList], axis=-1))
        for l in self.layerList:
            l.set_constants(self.reactionConstants,self.enzymeInitC,self.activTempInitC,self.inhibTempInitC,self.rescaleFactor)

        self.meanGatheredCps = tf.Variable(0,dtype=tf.float64,trainable=False)
        self.built = True
        self.mycps = tf.Variable(tf.zeros(1,dtype=tf.float64),trainable=False)
        super(chemTemplateNNModel,self).build(input_shape)

        print("model successfully built")

    def force_rescale(self,rescaleFactor):
        """
            Force the rescale factor to take the input value instead of the number of nodes in the graph.
        :param rescaleFactor: rescale value
        :return:
        """
        if not self.built:
            raise Exception("model should be built before calling this function")
        for l in self.layerList:
            l.rescale(rescaleFactor)
        self.rescaleFactor.assign(rescaleFactor)

    def greedy_set_cps(self,inputs):
        inputs = tf.convert_to_tensor(inputs,dtype=tf.float64)
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
        :param training: if we are in training or inference mode (we save time of checking model change)
        :param mask: we don't use it here...
        :return:
        """
        """
            We would like to proceed with a batching point of view.
            The problem here, is that tf.map_fn creates a graph for each realisation, making us loose the initialization on the current graph...
            Thus we cannot use it here, while this has not been fixed in tensorflow!
        """
        inputs = tf.cast(tf.convert_to_tensor(inputs),dtype=tf.float64)

        if training:
            self.verifyMask()
        inputs = inputs/self.rescaleFactor

        if self.usingLog:
            inputs = tf.exp(inputs)

        gatheredCps = tf.stop_gradient(tf.fill([tf.shape(inputs)[0]],tf.reshape(self._obtainCp(inputs[0]),())))
        gatheredCps = tf.reshape(gatheredCps,((tf.shape(inputs)[0],1)))
        tf.assert_equal(tf.shape(gatheredCps),(tf.shape(inputs)[0],1))
        #
        # gatheredCps = tf.stop_gradient(self.obtainCp(inputs))
        # gatheredCps = tf.fill([tf.shape(inputs)[0]],tf.reshape(self.mycps,()))
        # gatheredCps = tf.reshape(gatheredCps,((tf.shape(inputs)[0],1)))

        #self.meanGatheredCps.assign(tf.reduce_mean(gatheredCps))
        #tf.summary.scalar("mean_cp",data=tf.reduce_mean(gatheredCps),step=tf.summary.experimental.get_step())

        x = self.layerList[0](inputs,cps=gatheredCps,isFirstLayer=True)
        for l in self.layerList[1:]:
            if self.usingLog:
                x = l(tf.exp(x),cps=gatheredCps)
            else:
                x = l(x,cps=gatheredCps)
        if self.usingSoftmax:
            if self.usingLog:
                s = tf.keras.activations.softmax(tf.exp(x))
            else:
                s = tf.keras.activations.softmax(x)
        else:
            s = x
        return s


    def predConcentration(self, inputs, layerObserved,nodeObserved):
        inputs = tf.convert_to_tensor(inputs,dtype=tf.float64)
        tf.print(inputs.shape,"inputs shape")
        tf.assert_rank(inputs,2)
        tf.print("rescale factor for TF:",self.rescaleFactor)
        inputs = inputs/self.rescaleFactor
        gatheredCps = tf.stop_gradient(self.obtainCp(inputs))
        #self.meanGatheredCps.assign(tf.reduce_mean(gatheredCps))
        #tf.summary.scalar("mean_cp",data=tf.reduce_mean(gatheredCps),step=tf.summary.experimental.get_step())
        tf.print(inputs[0]," first input after rescale")
        x = self.layerList[0](inputs,cps=gatheredCps,isFirstLayer=True)
        if layerObserved==0:
            tf.print("outputs shape:",x.shape)
            return x[:,nodeObserved],gatheredCps
        tf.print(x,"x")
        tf.print(gatheredCps,"gatheredcps")
        for idx,l in enumerate(self.layerList[1:]):
            x = l(x,cps=gatheredCps)
            tf.print(x,"x")
            tf.print(idx,"idx")
            if tf.equal(idx,layerObserved-1):
                return x[:,nodeObserved],gatheredCps

    @tf.function
    def obtainCp(self,inputs):
        gatheredCps = tf.map_fn(self._obtainCp,inputs,parallel_iterations=32,back_prop=False)

        return gatheredCps

    @tf.function
    def _obtainCp(self,input):

        if(input.shape.rank<2):
            input = tf.reshape(input,(1,tf.shape(input)[0]))
        #first we obtain the born sup:

        bornsupIncremental = tf.cast(tf.fill([1],1.),dtype=tf.float64)
        bornsupcpdiff = self._computeCPdiff(bornsupIncremental,input)
        #tf.print("ended initial bornsup with 1 :",bornsupIncremental," found value",bornsupcpdiff)
        #tf.print("starting while")
        for idx in tf.range(20): #stop after 20 iteration
            bornsupIncremental = bornsupIncremental * 10.
            bornsupcpdiff = self._computeCPdiff(bornsupIncremental,input)
            if tf.greater(bornsupcpdiff[0],0.):
                #tf.print("ended bornsup loop",idx," found value",bornsupcpdiff, "using as sup:",bornsupIncremental)
                break
            #tf.print("ended bornsup loop",idx," found value",bornsupcpdiff)
            # if tf.equal(idx,20):
            #
                # bornsupcp0,x = self.layerList[0].layer_cp_born_sup(input)
                # layercp = tf.fill([1],1.)
                # for l in self.layerList[1:]:
                #     layercp,x = l.layer_cp_born_sup(x)
                #     bornsupcp0 += layercp
                # #tf.print("had to use layer born sup and found",bornsupcp0)
                # bornsupIncremental = tf.reshape(bornsupcp0,bornsupIncremental.shape)

        cpmin = tf.cast(tf.fill([1],1.),dtype=tf.float64)
        cp = self.brentq(self._computeCPdiff,cpmin,bornsupIncremental,input)
        return cp

    @tf.function
    def _computeCPdiff(self,cp,input):
        new_cp = tf.cast(tf.fill([1],1.),dtype =tf.float64)
        layercp,x = self.layerList[0].layer_cp_equilibrium(cp,input,isFirstLayer=True)
        new_cp += layercp
        for l in self.layerList[1:]:
            layercp,x = l.layer_cp_equilibrium(cp,x)
            new_cp += layercp
        new_cp = new_cp-cp
        return (-1)*new_cp

    def measureModelRelevancy(self,input):
        inputs = tf.stack([input])
        if(input.shape.rank<2):
            input = tf.reshape(input,(1,tf.shape(input)[0]))
        cp = self.obtainCp(inputs)
        Inhib = []
        cp2kd = []
        x,b,c=self.layerList[0].get_inhib_and_output(cp,input,isFirstLayer=True)
        Inhib += [b]
        cp2kd += [c]
        for l in self.layerList[1:]:
            x,b,c=self.layerList[0].get_inhib_and_output(cp,x,isFirstLayer=True)
            Inhib += [b]
            cp2kd += [c]
        return Inhib,cp2kd

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
        cpArray=tf.cast(tf.convert_to_tensor(cpArray),dtype=tf.float64)
        X0 = tf.cast(tf.convert_to_tensor(X0),dtype=tf.float64)
        func = self.lamda_computeCPdiff(X0)
        gatheredCps = tf.map_fn(func,cpArray,parallel_iterations=32,back_prop=False)
        return gatheredCps

    @tf.function
    def verifyMask(self):
        newRescaleFactor = tf.stop_gradient(tf.keras.backend.sum([l.get_rescaleFactor() for l in self.layerList], axis=-1))
        if(not tf.equal(self.rescaleFactor,newRescaleFactor)):
            self.rescaleFactor.assign(newRescaleFactor)
            for l in self.layerList:
                l.rescale(self.rescaleFactor)

    def updateArchitecture(self,masks,reactionsConstants,enzymeInitC,activTempInitC,inhibTempInitC):
        for idx,m in enumerate(masks):
            self.layerList[idx].set_mask(tf.convert_to_tensor(m,dtype=tf.float64))

        self.reactionConstants = reactionsConstants
        self.enzymeInitC = enzymeInitC
        self.activTempInitC = activTempInitC
        self.inhibTempInitC = inhibTempInitC

        self.rescaleFactor.assign(tf.keras.backend.sum([l.get_rescaleFactor() for l in self.layerList], axis=-1))
        for l in self.layerList:
            l.set_constants(self.reactionConstants,self.enzymeInitC,self.activTempInitC,self.inhibTempInitC,self.rescaleFactor)

    def logCp(self,epoch,logs=None):
        cp = self.meanGatheredCps
        tf.summary.scalar('mean cps', data= cp, step=epoch)
        return cp

    @tf.function
    def brentq(self, f, xa, xb,args=(),xtol=tf.constant(10**(-12),dtype=tf.float64), rtol=tf.constant(4.4408920985006262*10**(-16),dtype=tf.float64),iter=tf.constant(100)):
        xpre = tf.cast(tf.fill([1],0.),dtype=tf.float64) + xa
        xcur = tf.cast(tf.fill([1],0.),dtype=tf.float64)+ xb
        xblk = tf.cast(tf.fill([1],0.),dtype=tf.float64)
        fblk = tf.cast(tf.fill([1],0.),dtype=tf.float64)
        spre = tf.cast(tf.fill([1],0.),dtype=tf.float64)
        scur = tf.cast(tf.fill([1],0.),dtype=tf.float64)
        fpre = f(xpre, args)
        fcur = f(xcur, args)

        if tf.equal(fpre[0],0):
            return xpre
        if tf.equal(fcur[0],0):
            return xcur
        else:
            tf.assert_less((fpre*fcur)[0],tf.cast(0,dtype=tf.float64),message="seems like fpre*fcur have same sign",summarize=1)
            tf.assert_greater(fcur,tf.cast(0,dtype=tf.float64))

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