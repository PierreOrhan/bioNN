"""
    Provide a simple train function to train a network with tensorflow.
"""

from simulOfBioNN.nnUtils.plotUtils import displayEmbeddingHeat,plotWeight
from simulOfBioNN.nnUtils.dataUtils import loadMnist
from simulOfBioNN.parseUtils.parser import saveModelWeight,sparseParser,read_file,generateTemplateNeuralNetwork
from simulOfBioNN.odeUtils.systemEquation import setToUnits
from mpl_toolkits.mplot3d import Axes3D

from simulOfBioNN.nnUtils.chemCascadeNet.chemCascadeNNModel import chemCascadeNNModel
import tensorflow as tf
import multiprocessing as mlp
import time
import sys
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
    enzymeInit = 10**(-4)/C0
    activInit =  10**(-6)/C0
    inhibInit =  10**(-4)/C0
    return constantList,enzymeInit,activInit,inhibInit,C0


def getSetForMNIST():
    x_train,x_test,y_train,y_test,x_test_noise=loadMnist(rescaleFactor=2,fashion=False,size=None,mean=0,var=0.01,path="../../../Data/mnist")
    if(np.max(x_test)<=1):
        x_test = np.array(x_test*255,dtype=np.int)
        x_train = np.array(x_train*255,dtype=np.int)
    else:
        x_test = np.array(x_test,dtype=np.int)
        x_train = np.array(x_train,dtype=np.int)
    unique = list(np.sort(np.unique(x_test)))
    myLogSpace = np.logspace(-8,-6,len(unique))
    x_test = myLogSpace[x_test]
    x_test = np.reshape(x_test,(x_test.shape[0],(x_test.shape[1]*x_test.shape[2]))).astype(dtype=np.float32)
    x_train = myLogSpace[x_train]
    x_train = np.reshape(x_train,(x_train.shape[0],(x_train.shape[1]*x_train.shape[2]))).astype(dtype=np.float32)
    return x_train,x_test,y_train,y_test

def trainWithChemTemplateNN(savePath):

    x_train,x_test,y_train,y_test = getSetForMNIST()

    constantList,enzymeInit,activInit,inhibInit,C0 = _findConstant(savePath)

    #in a first time we consider the activInitNL as similar:
    activInitNL = activInit
    XglobalInit = 8.
    reactionConstantsNL = constantList[:3]+[constantList[10]]
    constantList = constantList + [constantList[-1],constantList[-1]] + constantList[:6]

    nbUnits = [10,10,5,3]
    sparsities = [0.9,0.9,0.9,0.8]
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


    enzymeInits = np.logspace(-10,-8,3)/C0
    print(enzymeInits)
    bornsups =[]
    bornsupsMean = []
    bornsupsCPG = []
    cpg = np.arange(1,70,10)
    for e in enzymeInits:
        print("REACTIONS CONSTANT FOR NL LAYER",reactionConstantsNL)
        model = chemCascadeNNModel(nbUnits=nbUnits, sparsities=sparsities, reactionConstantsCascade= constantList,
                                   reactionConstantsNL= reactionConstantsNL,
                                   enzymeInitC=float(e), activTempInitC=activInit, inhibTempInitC=inhibInit,
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
        print("======== Model: layers constants ========")

        model.force_rescale(1.)
        cpg = np.arange(10**(-8),np.sum([l[0].layer_XgCp_born_sup() for l in model.layerList]),(np.sum([l[0].layer_XgCp_born_sup() for l in model.layerList])-10**(-8))/10)

        res = np.zeros((cpg.shape[0],10))
        x = []
        y = []
        for idx,cpgi in enumerate(cpg):
            res[idx],cpInv = model.getFunctionStyleFromsize(size = 10,cpg=cpgi, X0=x_train[0])
            x+=[cpg]
            y+=[cpInv]
        print(res)
        x = np.array(x)
        y = np.array(y)
        print(x)
        print(y)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap=plt.get_cmap("tab20")
        for idx,r in enumerate(res):
            ax.plot(y[idx],r,c=cmap(idx),label="sup: "+str(cpg[-1]))
        plt.show()
        plt.savefig(os.path.join(sys.path[0],"cascadePlots")+"/"+str(e).replace(".","")+".png")

        # print(cpg,"cpg")
        # print("cpgSup",np.sum([l[0].layer_XgCp_born_sup() for l in model.layerList]))

    #     X = tf.convert_to_tensor(x_train[0],dtype=tf.float32)
    #     res=[]
    #     supCPG =[]
    #     cpg = np.arange(10**(-8),np.sum([l[0].layer_XgCp_born_sup() for l in model.layerList]),(np.sum([l[0].layer_XgCp_born_sup() for l in model.layerList])-10**(-8))/10)
    #
    #     for cpgi in cpg:
    #         print(cpgi,"cpgi")
    #         res+=[model.bornsup_fromCpg(tf.convert_to_tensor(cpgi,dtype=tf.float32),X)]
    #     bornsups += [res]
    #     bornsupsMean += [np.mean(res)]
    #     bornsupsCPG +=[np.sum([l[0].layer_XgCp_born_sup() for l in model.layerList])]
    #     print(res,"res")
    #     print(bornsups,"bornsups")
    # plt.plot(enzymeInits,bornsupsMean)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
    # fig = plt.figure()
    # plt.plot(enzymeInits,bornsupsCPG)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
    #
    # fig = plt.figure()
    # cmap = plt.get_cmap("tab20")
    # for idx,b in enumerate(bornsups):
    #     plt.plot(cpg,b,c=cmap(idx),label=str(idx))
    # plt.legend()
    # plt.show()
    #
    # tf.print(np.sum([l[0].layer_XgCp_born_sup() for l in model.layerList]),"born sup cpg")
    # print(cpg.shape)


    #model.greedy_set_cps(x_train[:my_batchsize])
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tfOUT", histogram_freq=1 ,profile_batch = 2)

    #model.fit(x_train[:], y_train[:],epochs=10,verbose=True,validation_data=(x_test,y_test))#,callbacks=[tensorboard_callback])
    return res


def computeCpRootFunction(x_test,model,path):
    tfCompetitions = np.zeros((len(x_test)),dtype=np.float64)
    fitOutput = np.zeros(len(x_test),dtype=np.float64)
    tfstyleFit = np.zeros((len(x_test),1000),dtype=np.float64)
    testOfCp = np.array(np.logspace(5,5.5,1000),dtype=np.float64)

    courbs=[0,int(fitOutput.shape[0]/2),fitOutput.shape[0]-1,int(fitOutput.shape[0]/3),int(2*fitOutput.shape[0]/3)]

    for idx1,x in enumerate(x_test):
        if idx1 in courbs:
            assert len(x.shape)==1
            x = tf.convert_to_tensor(x,dtype=tf.float32)
            print(x.shape,"x shape ",x," x")
            t0=time.time()
            tfCompetitions[idx1] = model._obtainCp(x)
            print("Ended tensorflow brentq methods in "+str(time.time()-t0))
            t0=time.time()
            tfstyleFit[idx1] = np.reshape(np.array(model.getFunctionStyle(testOfCp,x)),(1000))
            print("Obtain the landscape with tf in  "+str(time.time()-t0))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(19.2,10.8), dpi=100)
    cmap = plt.get_cmap('Dark2',x_test.shape[0]*len(courbs))
    for idx1,x in enumerate(x_test):
        if idx1 in courbs:
            ax.plot(testOfCp,np.abs(tfstyleFit[idx1,:]),c=cmap(idx1*(courbs.index(idx1)+1)),linestyle="--")
            ax.axvline(tfCompetitions[idx1],c=cmap(idx1*(courbs.index(idx1)+1)),marker="x",linestyle="-")
    ax.tick_params(labelsize="xx-large")
    ax.set_xlabel("cp",fontsize="xx-large")
    ax.set_ylabel("f(cp)-cp",fontsize="xx-large")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig(os.path.join(path,"formOfcp.png"))
    plt.show()

def displayCPFunc(path):
    x_train,x_test,y_train,y_test = getSetForMNIST()
    sizeInput = x_train.shape[-1]
    nbUnits = [100,100,10]
    sparsities = [0,0,0]
    constantList,enzymeInitC,activTempInitC,inhibTempInitC,C0 = _findConstant(path)
    x_train = x_train/C0
    x_test = x_test/C0
    XglobalinitC = 8.
    reactionConstantsNL = constantList[:3]+[constantList[10]]
    reactionConstantsCascade = constantList + [constantList[-1],constantList[-1]] + constantList[:6]
    activTempInitCNL = activTempInitC
    TAglobalInitC = activTempInitC
    cstGlobalInitC = [0.9999999999999998,0.1764705882352941,1.0,0.018823529411764708]
    usingLog = True
    usingSoftmax = False
    if usingLog:
        x_train = np.log(x_train)
        x_test = np.log(x_test)

    model = chemCascadeNNModel(nbUnits, sparsities, reactionConstantsCascade,reactionConstantsNL,
                               enzymeInitC,activTempInitC, inhibTempInitC,
                               activTempInitCNL,sizeInput,XglobalinitC,
                               TAglobalInitC,cstGlobalInitC,
                               randomConstantParameter=None, usingLog = usingLog, usingSoftmax = usingSoftmax )
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.build(input_shape=(None,sizeInput))
    computeCpRootFunction(x_test,model,path)

def obtainActivationShapes(model,C0,path):
    #first: plot the non-linearity
    print("PLOT OF NON-LINEARITIES")
    inputs = np.log(np.logspace(-8,4,1000)/C0)
    cps = model.mycps
    output1 = model.firstNlLayer.obtainNonLinearityShape(inputs, cps, isFirstLayer=True)
    outputs=[]
    for l in model.layerList:
        outputs+=[l[1].obtainNonLinearityShape(inputs,cps)]
    fig, ax = plt.subplots(figsize=(19.2,10.8), dpi=100)
    cmap = plt.get_cmap('Dark2',len(outputs)+1)
    ax.plot(inputs,output1,c=cmap(0),label="Initial layer")
    for idx,x in enumerate(outputs):
        ax.plot(inputs,x,c=cmap(idx+1),label="NL n°"+str(idx))
    ax.tick_params(labelsize="xx-large")
    fig.savefig(os.path.join(path,"nlActivations.png"))
    plt.legend()
    plt.show()
    print("BIAS FOR LINEARITIES")
    #Second: simply display the activation and inhibition bias
    for idx,l in enumerate(model.layerList):
        print(l[0].measureBias(cps)," for layer n°",idx)

def train(path):
    x_train,x_test,y_train,y_test = getSetForMNIST()
    sizeInput = x_train.shape[-1]
    nbUnits = [100,100,10]
    sparsities = [0.5,0.5,0]
    constantList,enzymeInitC,activTempInitC,inhibTempInitC,C0 = _findConstant(path)
    x_train = x_train/C0
    x_test = x_test/C0
    XglobalinitC = 8.
    reactionConstantsNL = constantList[:3]+[constantList[10]]
    reactionConstantsCascade = constantList + [constantList[-1],constantList[-1]] + constantList[:6]
    activTempInitCNL = activTempInitC
    TAglobalInitC = activTempInitC
    cstGlobalInitC = [0.9999999999999998,0.1764705882352941,1.0,0.018823529411764708]
    usingLog = True
    usingSoftmax = False
    if usingLog:
        x_train = np.log(x_train)
        x_test = np.log(x_test)

    model = chemCascadeNNModel(nbUnits, sparsities, reactionConstantsCascade,reactionConstantsNL,
                               enzymeInitC,activTempInitC, inhibTempInitC,
                               activTempInitCNL,sizeInput,XglobalinitC,
                               TAglobalInitC,cstGlobalInitC,
                               randomConstantParameter=None, usingLog = usingLog, usingSoftmax = usingSoftmax )
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.build(input_shape=(None,sizeInput))
    model.greedy_set_cps(inputs=x_train[:32])

    obtainActivationShapes(model,C0,path)

    #model.fit(x_train,y_train,verbose=True)


if __name__ == '__main__':
    p1 = os.path.join(sys.path[0],"..")
    p3 = os.path.join(p1,"trainingWithChemicalNN")
    if not os.path.exists(p3):
        os.makedirs(p3)
    device_name = tf.test.gpu_device_name()
    if not tf.test.is_gpu_available():
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    train(p3)
