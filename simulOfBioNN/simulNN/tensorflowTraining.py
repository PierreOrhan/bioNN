"""
    Provide a simple train function to train a network with tensorflow.
"""

from simulOfBioNN.nnUtils.plotUtils import plotWeight
from simulOfBioNN.nnUtils.dataUtils import loadMnist
from simulOfBioNN.parseUtils.parser import saveModelWeight,sparseParser,read_file,generateTemplateNeuralNetwork
from simulOfBioNN.odeUtils.systemEquation import setToUnits
from simulOfBioNN.nnUtils.classicalTfNet.clippedSparseBioDenseLayer import clippedSparseBioDenseLayer
from simulOfBioNN.nnUtils.classicalTfNet.clippedBinaryLayer import clippedBinaryLayer
from simulOfBioNN.nnUtils.classicalTfNet.clippedSparseBioSigLayer import clippedSparseBioSigLayer
import tensorflow as tf

import os
import numpy as np
from simulOfBioNN.nnUtils.neurVorConcSet import VoronoiSet
import matplotlib.pyplot as plt

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
    return constantList,enzymeInit,activInit,inhibInit,C0


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

    constantList,enzymeInit,activInit,inhibInit,C0 = _findConstant(savePath)
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

    # if usingLog:
    #     x_train = np.log(x_train)
    #     x_test = np.log(x_test)


    # model = chemTemplateNNModel(nbUnits=nbUnits,sparsities=sparsities,reactionConstants= constantList, enzymeInitC=enzymeInit, activTempInitC=activInit,
    #                             inhibTempInitC=inhibInit, randomConstantParameter=None,usingLog=usingLog,usingSoftmax=usingSoftmax)
    # print("model is running eagerly: "+str(model.run_eagerly))
    # # model.run_eagerly=True
    # model.compile(optimizer=tf.optimizers.Adam(),
    #               loss='sparse_categorical_crossentropy',
    #               #loss = tf.keras.losses.MeanSquaredError(),
    #               metrics=['accuracy'])
    #               #metrics=[tf.keras.metrics.MeanSquaredError()])
    # model.build(input_shape=(None,x_train.shape[-1]))
    # print("testing against example:")
    # model.greedy_set_cps(x_train[:my_batchsize])
    # #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tfOUT", histogram_freq=1 ,profile_batch = 2)
    # model.fit(x_train[:], y_train[:],epochs=10,verbose=True)#,callbacks=[tensorboard_callback])

    #res = model.call(x_train[:10])

    # print("computing cps at equilibrium")
    # import time
    # t = time.time()
    # cps = model.obtainCp(tf.convert_to_tensor(x_train[:100],dtype=tf.float32))
    # print("ended computing of cps in ",time.time()-t)
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(19.2,10.8), dpi=100)
    # plt.scatter(range(cps.shape[0]),cps[:,0],c="b")
    # plt.yscale("log")
    # plt.ylim(1,10**9)
    # plt.title("competition found for the mnist dataset under the initialization architecture")
    # plt.show()
    # plt.savefig("cp_rescaleof"+str(forcedRescaleFactor))

    # res = model.call(x_test[:10])
    # concentration = model.predConcentration(x_test[:10])
    # print(res)
    # print(concentration)
    a = 1
    b = 1
    @tf.function
    def mylogActivation(x):
        return tf.math.log(a * tf.math.exp(x)/(b + tf.math.exp(x)))

    model2 = tf.keras.Sequential()
    model2.add(tf.keras.layers.Dense(100,activation=mylogActivation))
    model2.add(tf.keras.layers.Dense(100,activation=mylogActivation))
    model2.add(tf.keras.layers.Dense(10,activation=mylogActivation))
    if usingSoftmax:
        model2.add(tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax))
    model2.compile(optimizer=tf.optimizers.Adam(),
                   #loss=tf.keras.losses.BinaryCrossentropy(),
                   loss='sparse_categorical_crossentropy',
                   #loss = tf.keras.losses.MeanSquaredError(),
                   metrics=['accuracy']
                   #metrics=[tf.keras.metrics.MeanSquaredError()]
                   )
    model2.build(input_shape=(None,x_train.shape[-1]))
    model2.fit(np.log(x_train[:]), y_train[:],epochs=epochs,verbose=True)

    # import time
    # t0= time.time()
    # res = model.call(x_test[:1000])
    # print("computed res in "+str(time.time()-t0))
    # #
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



    # colors = ["r","g","b"]
    # Y = model2.call(x_test).numpy()
    # Ychem = model.call(x_test).numpy()
    # for idx,x in enumerate(x_test):
    #     plt.scatter(x[0],x[1],c=colors[np.argmax(Y[idx])])
    #     plt.scatter(x[0],x[1],c=colors[np.argmax(Ychem[idx])],marker="x")
    # plt.show()


    #res = model.call(x_test[:1000])
    # import matplotlib.pyplot as plt
    # X = np.stack([np.logspace(-8,-4,100,dtype=np.float32)],axis=1)/C0
    # plt.plot(np.logspace(-8,-4,100)/C0,function_to_infer(X),c="r")
    # plt.plot(np.logspace(-8,-4,100)/C0,np.reshape(model2.call(X),(100)),c="b")
    # # plt.plot(np.logspace(-8,-4,1000)/C0,np.reshape(model.call(X),(1000)),c="g",linestyle="-")
    # plt.show()


    #return savePath,model,model2 #,cps #,acc,x_test,y_test,nnAnswer

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

def testSomeActivation():
    epochs = 1
    usingLog = True
    usingSoftmax = True

    barycenters=np.log(np.array([[5*10**(-6),10**(-4)],[10**(-5),5*10**(-6)],[10**(-4),10**(-4)]])/C0)
    set=VoronoiSet(barycenters)
    x_train,y_train=set.generate(100000)
    x_train = np.asarray(x_train,dtype=np.float32)
    x_test, y_test=set.generate(1000)
    x_test = np.asarray(x_test,dtype=np.float32)
    print(y_test)
    colors = ["r","g","b"]
    for idx,x in enumerate(x_test):
        plt.scatter(x[0],x[1],c=colors[y_test[idx]])
    for b in barycenters:
        plt.scatter(b[0],b[1],c="m",marker="x")
    plt.show()

    size = 10

    beta_list = np.logspace(-2,2,size)
    alpha_list = np.logspace(-2,2,size)

    scores = np.zeros((size,size))
    from tqdm import  tqdm
    argsList=[]
    for idx1,b in tqdm(enumerate(beta_list)):
        for idx2,a in enumerate(alpha_list):
            argsList +=[[b,a,barycenters,usingSoftmax,x_train,y_train,x_test,y_test,epochs]]
            def mylogActivation(x):
                return tf.math.log(a * tf.math.exp(x)/(b + tf.math.exp(x)))
            s=0
            for e in range(3):
                A =np.log(np.logspace(-2,2,1000))
                # plt.plot(A,mylogActivation(A).numpy())
                # plt.show()
                model2 = tf.keras.Sequential()
                model2.add(tf.keras.layers.Dense(10,activation=mylogActivation))
                model2.add(tf.keras.layers.Dense(10,activation=mylogActivation))
                model2.add(tf.keras.layers.Dense(len(barycenters),activation=mylogActivation))
                if usingSoftmax:
                    model2.add(tf.keras.layers.Dense(len(barycenters),activation=tf.keras.activations.softmax))
                model2.compile(optimizer=tf.optimizers.Adam(),
                               #loss=tf.keras.losses.BinaryCrossentropy(),
                               loss='sparse_categorical_crossentropy',
                               #loss = tf.keras.losses.MeanSquaredError(),
                               metrics=['accuracy']
                               #metrics=[tf.keras.metrics.MeanSquaredError()]
                               )
                model2.build(input_shape=(None,x_train.shape[-1]))
                print("starting fit")
                model2.fit(x_train[:], y_train[:],epochs=epochs,verbose=False)
                print("ended fit")
                answer = np.argmax(model2.predict(x_test[:]),axis=1)
                del model2
                s += np.sum(np.where(answer==y_test,1,0))/y_test.shape[0]
            scores[idx1,idx2] = s

    import pandas as pd
    df =pd.DataFrame(scores)
    df.to_csv("results")

    import matplotlib.colors as clr
    norm = clr.Normalize(vmin=np.min(scores), vmax=np.max(scores))
    cmap = plt.get_cmap("Oranges")
    fig, ax = plt.subplots(figsize=(19.2,10.8), dpi=100)
    ax.imshow(scores,cmap=cmap)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm,ax=ax,norm=norm)
    cbar.ax.set_ylabel("scores",fontsize="xx-large")
    cbar.ax.tick_params(labelsize="xx-large")
    plt.show()
    plt.savefig("results.png")