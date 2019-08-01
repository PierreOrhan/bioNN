"""
    Provide a simple train function to train a network with tensorflow.
"""

from simulOfBioNN.nnUtils.plotUtils import displayEmbeddingHeat,plotWeight
from simulOfBioNN.nnUtils.dataUtils import loadMnist
from simulOfBioNN.parseUtils.parser import saveModelWeight,sparseParser,read_file,generateTemplateNeuralNetwork
from simulOfBioNN.odeUtils.systemEquation import setToUnits

from simulOfBioNN.nnUtils.chemCascadeNet.chemCascadeNNModel import chemCascadeNNModel
import tensorflow as tf
import multiprocessing as mlp

import os
import numpy as np
from simulOfBioNN.nnUtils.neurVorConcSet import VoronoiSet
import matplotlib.pyplot as plt

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
    return constantList,enzymeInit,activInit,inhibInit,C0

def trainWithChemTemplateNN(savePath):

    x_train,x_test,y_train,y_test,x_test_noise=loadMnist(rescaleFactor=2,fashion=False,size=None,mean=0,var=0.01,path="../../../Data/mnist")
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

    constantList,enzymeInit,activInit,inhibInit,C0 = _findConstant(savePath)

    #in a first time we consider the activInitNL as similar:
    activInitNL = activInit
    XglobalInit = 8
    reactionConstantsNL = constantList[:3]+[constantList[10]]
    constantList = constantList + constantList[:6]

    nbUnits = [10,10,5,3]
    sparsities = [0.1,0.1,0.1,0.]
    use_bias = False
    epochs = 10
    my_batchsize = 32
    x_train = x_train/C0
    x_test = x_test/C0


    #Instead of MNIST we try simple VORONOI task:
    # barycenters=np.log(np.array([[5*10**(-6),10**(-4)],[10**(-5),5*10**(-6)],[10**(-4),10**(-4)]])/C0)
    # set=VoronoiSet(barycenters)
    # x_train,y_train=set.generate(100000)
    # x_train = np.asarray(x_train,dtype=np.float32)
    # x_test, y_test=set.generate(1000)
    # x_test = np.asarray(x_test,dtype=np.float32)
    # print(y_test)
    # colors = ["r","g","b"]
    # for idx,x in enumerate(x_test):
    #     plt.scatter(x[0],x[1],c=colors[y_test[idx]])
    # for b in barycenters:
    #     plt.scatter(b[0],b[1],c="m",marker="x")
    # plt.show()

    usingLog = True
    usingSoftmax = True

    if usingLog:
        x_train = np.log(x_train)
        x_test = np.log(x_test)


    model = chemCascadeNNModel(nbUnits=nbUnits, sparsities=sparsities, reactionConstantsCascade= constantList,
                               reactionConstantsNL= reactionConstantsNL,
                               enzymeInitC=enzymeInit, activTempInitC=activInit, inhibTempInitC=inhibInit,
                               activTempInitCNL=activInitNL,sizeInput=x_train.shape[1],
                               randomConstantParameter=None, usingLog=usingLog, usingSoftmax=usingSoftmax,XglobalinitC=XglobalInit)
    print("model is running eagerly: "+str(model.run_eagerly))
    # model.run_eagerly=True
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  #loss = tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
                  #metrics=[tf.keras.metrics.MeanSquaredError()])
    model.build(input_shape=(None,x_train.shape[-1]))
    print("testing against example:")
    #model.greedy_set_cps(x_train[:my_batchsize])
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tfOUT", histogram_freq=1 ,profile_batch = 2)
    model.fit(x_train[:], y_train[:],epochs=10,verbose=True,validation_data=(x_test,y_test))#,callbacks=[tensorboard_callback])



if __name__ == '__main__':
    #mlp.set_start_method('fork',force=True)
    #tf.debugging.set_log_device_placement(True)
    import sys
    p1 = os.path.join(sys.path[0],"..")
    p3 = os.path.join(p1,"trainingWithChemicalNN")
    if not os.path.exists(p3):
        os.makedirs(p3)
    device_name = tf.test.gpu_device_name()
    if not tf.test.is_gpu_available():
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    trainWithChemTemplateNN(p3)
