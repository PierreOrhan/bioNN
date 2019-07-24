"""
    Provide a simple train function to train a network with tensorflow.
"""

from simulOfBioNN.nnUtils.plotUtils import displayEmbeddingHeat,plotWeight
from simulOfBioNN.nnUtils.dataUtils import loadMnist
from simulOfBioNN.parseUtils.parser import saveModelWeight,sparseParser,read_file,generateTemplateNeuralNetwork
from simulOfBioNN.odeUtils.systemEquation import setToUnits
from simulOfBioNN.nnUtils.clippedSparseBioDenseLayer import clippedSparseBioDenseLayer
from simulOfBioNN.nnUtils.clippedBinaryLayer import clippedBinaryLayer
from simulOfBioNN.nnUtils.clippedSparseBioSigLayer import clippedSparseBioSigLayer
from simulOfBioNN.nnUtils.chemTemplateNN.chemTemplateNNModel import chemTemplateNNModel
import tensorflow as tf

import os
import numpy as np


def train(savePath):
    """
        Train some neural network, and save the weight, aka the architecture, so that in can be used by our parser module.
    :param: savePath: path where the network will be saved
    :return directory for weight
             accuracy
             testing_x_set : a set of inputs for test
             testing_y_set : a set of outputs for test
             nnAnswer : the answer for the raw nn on the test set
    """

    import tensorflow as tf

    x_train,x_test,y_train,y_test,x_test_noise=loadMnist(rescaleFactor=2,fashion=False,size=None,mean=0,var=0.01,path="../../Data/mnist")

    archlist=["binary","sig","sparseNormal"]
    architecture=archlist[2]
    nbUnits = [50,50,10,10]
    nbLayers = len(nbUnits)
    use_bias = True
    epochs = 5
    sess=tf.Session()
    with sess.as_default():
        GPUidx = sess.list_devices()[0].name
        layerList=[]
        layerList+=[tf.keras.layers.Flatten(input_shape=(x_train.shape[1], x_train.shape[2]))]
        if(architecture=="binary"):
            for e in range(nbLayers-1):
                layerList+=[clippedBinaryLayer(GPUidx,units=nbUnits[e], activation=None,use_bias=use_bias)]
            layerList+=[clippedBinaryLayer(GPUidx,units=10, activation=tf.nn.softmax,use_bias=use_bias)]
        elif(architecture=="sig"):
            for e in range(nbLayers-1):
                layerList+=[clippedSparseBioSigLayer(GPUidx,units=nbUnits[e], activation=None,use_bias=use_bias)]
            layerList+=[clippedSparseBioSigLayer(GPUidx,units=10, activation=tf.nn.softmax,use_bias=use_bias)]
        else:
            for e in range(nbLayers-1):
                layerList+=[clippedSparseBioDenseLayer(GPUidx,units=nbUnits[e], activation=tf.keras.activations.relu,use_bias=use_bias)]
            layerList+=[clippedSparseBioDenseLayer(GPUidx,units=10, activation=tf.nn.softmax,use_bias=use_bias)]
        model = tf.keras.models.Sequential(layerList)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        #model.build(input_shape=x_train.shape)
        model.fit(x_train, y_train, epochs=epochs,verbose=True)
        print(model.summary())
        _,acc=model.evaluate(x_test, y_test)
        _,accNoise=model.evaluate(x_test_noise, y_test)

        nnAnswer = model.predict(x_test)


        # activs=[tf.placeholder(dtype=tf.float32) for _ in layerList]
        # inputs = tf.placeholder(dtype=tf.float32,shape=(None,x_train.shape[1],x_train.shape[2]))
        # activs[0]=layerList[0](inputs)
        # for idx,l in enumerate(layerList[1:]):
        #     activs[idx+1] = l(activs[idx])
        # activation=sess.run(activs,feed_dict={inputs:x_train})
        # names = ["activation of layer"+str(idx) for idx in range(len(layerList))]
        # for idx,a in enumerate(activation):
        #     displayEmbeddingHeat(a,0.1,name=names[idx])

        savePath = os.path.join(savePath,"weightDir")
        plotWeight(model,use_bias)
        saveModelWeight(model,use_bias,savePath)

        print("Ended Training")
    sess.close()
    del model
    del sess
    return savePath,acc,x_test,y_test,nnAnswer


def _findConstant(savePath):
    pathForDeterminingConstant = os.path.join(savePath,"toDefineModelConstant")

    # Let us determine the constant that goes with the chemical model, by simply defining a small network.
    smallMasks=[np.array([[1,-1]])]
    complexity= "simple"
    useProtectionOnActivator = False
    useEndoOnOutputs = True
    useEndoOnInputs = False
    generateTemplateNeuralNetwork(pathForDeterminingConstant,smallMasks,complexity=complexity,useProtectionOnActivator=useProtectionOnActivator,
                                  useEndoOnOutputs=useEndoOnOutputs,useEndoOnInputs=useEndoOnInputs)
    parsedEquation,constants,nameDic=read_file(pathForDeterminingConstant + "/equations.txt", pathForDeterminingConstant + "/constants.txt")
    KarrayA,stochio,maskA,maskComplementary = sparseParser(parsedEquation,constants)
    _,T0,C0,_=setToUnits(constants,KarrayA,stochio)
    constantList = [0.9999999999999998,0.1764705882352941,1.0,0.9999999999999998,
                    0.1764705882352941,1.0,0.9999999999999998,0.1764705882352941,1.0,0.018823529411764708]
    constantList+=[constantList[-1]]
    enzymeInit = 5*10**(-7)/C0
    activInit =  10**(-4)/C0
    inhibInit =  10**(-4)/C0
    return constantList,enzymeInit,activInit,inhibInit


def trainWithChemTemplateNN(savePath):


    x_train,x_test,y_train,y_test,x_test_noise=loadMnist(rescaleFactor=2,fashion=False,size=None,mean=0,var=0.01,path="../../Data/mnist")
    if(np.max(x_test)<=1):
        x_test = np.array(x_test*255,dtype=np.int)
        x_train = np.array(x_train*255,dtype=np.int)
    else:
        x_test = np.array(x_test,dtype=np.int)
        x_train = np.array(x_train,dtype=np.int)
    unique = list(np.sort(np.unique(x_test)))
    myLogSpace = np.logspace(-8,-4,len(unique))
    x_test = myLogSpace[x_test]
    x_test = np.reshape(x_test,(x_test.shape[0],(x_test.shape[1]*x_test.shape[2]))).astype(dtype=np.float32)
    x_train = myLogSpace[x_train]
    x_train = np.reshape(x_train,(x_train.shape[0],(x_train.shape[1]*x_train.shape[2]))).astype(dtype=np.float32)


    constantList,enzymeInit,activInit,inhibInit = _findConstant(savePath)
    nbUnits = [50,50,10,10]
    sparsities = [0.9,0.9,0.9,0.9]
    use_bias = False
    useGPU = False
    epochs = 1
    my_batchsize = 32

    device_name = tf.test.gpu_device_name()
    if not tf.test.is_gpu_available():
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    model = chemTemplateNNModel(None,useGPU=useGPU,nbUnits=nbUnits,sparsities=sparsities,reactionConstants= constantList, enzymeInitC=enzymeInit, activTempInitC=activInit,
                                inhibTempInitC=inhibInit, randomConstantParameter=None)
    print("model is running eagerly: "+str(model.run_eagerly))
    # model.run_eagerly=True
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.build(input_shape=(None,x_train.shape[-1]))
    print("testing against example:")

    # writer = tf.summary.create_file_writer("tfOUT")
    # tf.summary.trace_on(graph=True, profiler=True)
    # res = model.call(x_test[:my_batchsize])
    # print(res)
    # with writer.as_default():
    #     tf.summary.trace_export(
    #         name="my_func_trace",
    #         step=0,
    #         profiler_outdir="tfOUT")

    # print("training:")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tfOUT", histogram_freq=1 ,profile_batch = 2)
    cp_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=model.logCp)

    model.fit(x_train[:100], y_train[:100],batch_size=my_batchsize,epochs=epochs,verbose=True,callbacks=[tensorboard_callback,cp_callback])
    #
    # print("finished the call, trying to print")
    # print(model.summary())
    # _,acc=model.evaluate(x_test, y_test)
    # _,accNoise=model.evaluate(x_test_noise, y_test)
    #
    # nnAnswer = model.predict(x_test)
    #
    #
    # # activs=[tf.placeholder(dtype=tf.float32) for _ in layerList]
    # # inputs = tf.placeholder(dtype=tf.float32,shape=(None,x_train.shape[1],x_train.shape[2]))
    # # activs[0]=layerList[0](inputs)
    # # for idx,l in enumerate(layerList[1:]):
    # #     activs[idx+1] = l(activs[idx])
    # # activation=sess.run(activs,feed_dict={inputs:x_train})
    # # names = ["activation of layer"+str(idx) for idx in range(len(layerList))]
    # # for idx,a in enumerate(activation):
    # #     displayEmbeddingHeat(a,0.1,name=names[idx])
    #
    # savePath = os.path.join(savePath,"weightDir")
    # plotWeight(model,use_bias)
    # saveModelWeight(model,use_bias,savePath)
    #
    # print("Ended Training")
    # sess.close()
    #del model
    # del sess
    return savePath #,acc,x_test,y_test,nnAnswer

if __name__ == '__main__':
    import sys
    p1 = os.path.join(sys.path[0],"..")
    p3 = os.path.join(p1,"trainingWithChemicalNN")
    if not os.path.exists(p3):
        os.makedirs(p3)
    trainWithChemTemplateNN(p3)
