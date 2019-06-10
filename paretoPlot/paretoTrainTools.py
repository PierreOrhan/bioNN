'''
    In this module we provide functions for creating pareto plots, and training tool using python multiprocessing.
'''

from simulOfBioNN.nnUtils.plotUtils import *
from simulOfBioNN.nnUtils.dataUtils import loadMnist
import multiprocessing
import os
import pandas
from paretoPlot.paretoConfigUtils import _getConfig

def _trainAndTest(sess,GPUidx, x_train, y_train, x_test, x_test_noise, y_test, resultPath, nbUnits=[], use_bias=True, flatten=True, epochs=5, biasValue=[], fractionZero=[], verbose=False):
    import tensorflow as tf
    from simulOfBioNN.nnUtils.clippedSparseBioDenseLayer import clippedSparseBioDenseLayer
    assert len(fractionZero)==len(nbUnits)
    nbLayers = len(fractionZero)
    if(GPUidx):
        GPUname = sess.list_devices()[GPUidx].name
    else:
        for device in sess.list_devices():
            if(device.device_type == "GPU"):
                GPUname = device.name
                break
    layerList=[]
    if(flatten):
        layerList+=[tf.keras.layers.Flatten(input_shape=(x_train.shape[1], x_train.shape[2]))]
    for e in range(nbLayers-1):
        layerList+=[clippedSparseBioDenseLayer(GPUname, biasValue=biasValue[e], fractionZero=fractionZero[e], units=nbUnits[e], activation=tf.nn.relu, use_bias=use_bias)]
        #layerList+=[tf.keras.layers.BatchNormalization(beta_initializer=biasInit[e])]
    layerList+=[clippedSparseBioDenseLayer(GPUname, biasValue=biasValue[-1], fractionZero=fractionZero[nbLayers - 1], units=10, activation=tf.nn.softmax, use_bias=use_bias)]
    #model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(100,))] +
    model = tf.keras.models.Sequential(layerList)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs,verbose=verbose)
    # print(model.summary())
    _,acc=model.evaluate(x_test, y_test,verbose=verbose)
    _,accNoise=model.evaluate(x_test_noise, y_test,verbose=verbose)
    weights=model.get_weights()
    #   we compute the weight clipping:
    noneZeroWeight=0
    zeroWeight=0
    for idx,wLayer in enumerate(weights):
        if(use_bias):
            if(idx%2==0): #we are not looking at bias
                noneZeroWeight+=len(wLayer[wLayer>0.2])+len(wLayer[wLayer<-0.2])
                zeroWeight+=wLayer.shape[0]*wLayer.shape[1]-(len(wLayer[wLayer>0.2])+len(wLayer[wLayer<-0.2]))
        else:
            noneZeroWeight+=len(wLayer[wLayer>0.2])+len(wLayer[wLayer<-0.2])
            zeroWeight+=wLayer.shape[0]*wLayer.shape[1]-(len(wLayer[wLayer>0.2])+len(wLayer[wLayer<-0.2]))
    model.save(resultPath+"model.h5")
    del model
    return acc,accNoise,noneZeroWeight,zeroWeight

def trainMultiProcess(X):
    """
        Launch repeat training, validation test on normal and provided noisy data.
        The tip to launch parallel training is to modify the configuration with core_config.gpu_options.allow_growth = True.
        Naively Tensorflow allocate the whole memory for the first session created, which we don't want.
    :param X: tupple containing the following parameters:
             resultPath : string,directory to store results
             gpuName: int, index of the GPU to use (often: 0 and 1 are CPU,CPU_XLA, 2,3 : GPU0,GPU1XLA ...
                           set to none to let us automatically choose the first encountered GPU
             x_train: 3d-array, image data set for train
             y_train: 2d-array, result data set for train
             x_test: 3d-array, image data set for test
             x_test_noise: 3d-array, noisy image data set for test
             y_test: 2d-array, result data set for test
             nbU: int, number of units for each layer
             use_bias: bool, If using bias or not
             flatten: bool, if the data set is already flattent or not
             epochs: int, number of epochs for training
             biasValues: 1d-array, bias in each layer.
             zeroFrac: int, amount of sparsity
             idx: str use for naming the saved data
             repeat: number of repetition
    :return: collectedAcc,collectedAccNoise,collectedWeightsNoneZeros,collectedWeightsZeros
             Arrays with the results of the training.
    """
    import tensorflow as tf
    resultPath,gpuName,x_train,y_train,x_test,x_test_noise,y_test,nbU,use_bias,flatten,epochs,biasValues,zeroFrac,idx,repeat = X

    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    collectedAcc=[]
    collectedAccNoise=[]
    collectedWeightsNoneZeros=[]
    collectedWeightsZeros=[]

    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True
    sess = tf.Session(config=core_config)
    #tf.keras.backend.set_session(sess)
    print(str(idx)+" launched")
    with sess:
        for r in range(repeat):
            print(str(idx)+" started "+str(r+1)+" on "+str(repeat))
            acc,accNoise,weightsNZ,weightsZ=_trainAndTest(sess,gpuName,x_train,y_train,x_test,x_test_noise,y_test,resultPath+"_"+str(idx)+"_"+str(r),nbUnits=nbU,
                                                                 use_bias=use_bias,flatten=flatten,epochs=epochs,biasValue=biasValues[:len(zeroFrac)+1],
                                                                 fractionZero=zeroFrac,verbose=False)
            print(str(idx)+" finished "+str(r+1)+" on "+str(repeat))
            collectedAcc+=[acc]
            collectedAccNoise+=[accNoise]
            collectedWeightsNoneZeros+=[weightsNZ]
            collectedWeightsZeros+=[weightsZ]
    sess.close()
    #tf.keras.backend.clear_session()

    print(str(idx)+" giving result ")

    return collectedAcc,collectedAccNoise,collectedWeightsNoneZeros,collectedWeightsZeros

def _saveTrainingResults(myOutputs,resultPath):
    unitsAcc = []
    unitsAccNoise = []
    unitsWeights = []
    unitsWeightsZ = []
    for m in myOutputs:
        unitsAcc+=[m[0]]
        unitsAccNoise+=[m[1]]
        unitsWeights+=[m[2]]
        unitsWeightsZ+=[m[3]]
    #save datas
    resultArray=np.array(unitsAcc)
    noiseResultArray=np.array(unitsAccNoise)
    resultWeight=np.array(unitsWeights)
    resultWeightZ=np.array(unitsWeightsZ)

    df=pandas.DataFrame(resultArray)
    df.to_csv(resultPath+"_acc.csv")
    df=pandas.DataFrame(noiseResultArray)
    df.to_csv(resultPath+"_acc_noise.csv")
    df=pandas.DataFrame(resultWeight)
    df.to_csv(resultPath+"_nbrNoneZeroWeights.csv")
    df=pandas.DataFrame(resultWeightZ)
    df.to_csv(resultPath+"_nbrZeroWeights.csv")

def train(listOfRescale,Batches,Sparsity,NbUnits,Initial_Result_Path,epochs,repeat,use_bias,fashion=False):
    """
        Train a variety of neural network on Batches architecture, a list of string name for architecture.
        Sparsity and NbUnits should be dictionary which keys are the architecture names.
        The training is also made for different rescale size of the inputs.
        They should define 3d_list with respectively the sparsity and number of units desired for each layer.
        The training program save all result in the Initial_Result_Path folder.
        Results for different rescale and different batches are saved in separate sub-directory.
    :param listOfRescale: list with the rescale size, often [1,2,4], that is the scale to divide each size of the image.
    :param Batches: see above
    :param Sparsity: see above
    :param NbUnits: see above
    :param Initial_Result_Path:
    :param fashion: if Fashion_mnist rather than mnist: use True.
    :param epochs: int, number of epochs per training
    :param repeat: int, number of repeat
    :param use_bias: boolean, True if using bias
    :return:
    """
    gpuIdx = None
    for ridx,r in enumerate(listOfRescale):
        print("________________________SWITCHING RESCALE_____________________")
        p="../Data/mnist"
        x_train,x_test,y_train,y_test,x_test_noise=loadMnist(rescaleFactor=r,fashion=fashion,size=None,mean=0,var=0.01,path=p)
        flatten = True #we have to fatten
        for idxb,batch in enumerate(Batches):
            sparsityMat = Sparsity[batch]
            nbUnitMat = NbUnits[batch]
            RESULT_PATH=Initial_Result_Path+str(r)+"/"+str(batch)+"/"
            argsList=[]
            for i in range(len(sparsityMat)):
                sparsity=sparsityMat[i]
                units=nbUnitMat[i]
                for idx,u in enumerate(units):
                    biasValues=[np.random.rand(1) for _ in range(len(sparsity[idx]))]
                    argsList += [[RESULT_PATH, gpuIdx, x_train, y_train, x_test, x_test_noise, y_test, u, use_bias, flatten, epochs, biasValues, sparsity[idx], str(i)+"_"+str(idx),repeat]]
            with multiprocessing.Pool(processes= len(argsList)) as pool:
                batchOutputs = pool.map(trainMultiProcess,argsList)
            pool.close()
            pool.join()
            print("Finished computing, closing pool")

            _saveTrainingResults(batchOutputs,RESULT_PATH)

def paretoPlot(listOfRescale,Batches,colors,colors3,Initial_Result_Path):
    """
        Produce the pareto plot for a solution created by paretoExperiment.Train, in the directory Initial_Result_Path
    :return: save the fig in Initial_Result_Path
    """
    figPareto0 = plt.figure(figsize=(19.2,10.8), dpi=100)
    axPareto0 = figPareto0.add_subplot(1,1,1)
    for ridx,r in enumerate(listOfRescale):
        figPareto = plt.figure(figsize=(19.2,10.8), dpi=100)
        axPareto = figPareto.add_subplot(1,1,1)
        N0=[]
        R=[]
        for idxb,batch in enumerate(Batches):
            RESULT_PATH=Initial_Result_Path+str(r)+"/"+str(batch)+"/"
            #Plot management:
            colorPareto=colors[batch]
            df=pandas.read_csv(RESULT_PATH+"_acc.csv")
            dfNoise=pandas.read_csv(RESULT_PATH+"_acc_noise.csv")
            dfNoneZeroWeights=pandas.read_csv(RESULT_PATH+"_nbrNoneZeroWeights.csv")
            dfZeroWeights=pandas.read_csv(RESULT_PATH+"_nbrZeroWeights.csv")
            result=df.values[:,1:]
            resultNoise=dfNoise.values[:,1:]
            nbNoneZero=dfNoneZeroWeights.values[:,1:]
            nbZero=dfZeroWeights.values[:,1:]

            axPareto.scatter(nbNoneZero,result,c=colorPareto,label=batch)
            N0+=[nbNoneZero]
            R+=[result]
        res=np.concatenate([res for res in R])
        n=np.concatenate([n for n in N0])
        axPareto0.scatter(n,res,c=colors3[r],label="rescale:"+str(r)) #[colorPareto0 for _ in nbNoneZero]
        # axPareto0.set_xscale("log")

        axPareto.legend(scatterpoints=1)
        figPareto.tight_layout()
        figPareto.savefig(Initial_Result_Path+str(r)+"/paretoPlot.png")
    axPareto0.set_xlabel("number of non-zero weight after training",fontsize="xx-large")
    axPareto0.set_ylabel("Accuracy against test set",fontsize="xx-large")
    axPareto0.tick_params(labelsize="xx-large")
    axPareto0.legend(scatterpoints=1)
    figPareto0.tight_layout()
    axPareto0.set_title("Pareto plot for validation against Mnist",fontsize="xx-large")
    figPareto0.savefig(Initial_Result_Path+"/paretoPlot.png")




