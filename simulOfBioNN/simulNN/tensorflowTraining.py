"""
    Provide a simple train function to train a network with tensorflow.
"""

from simulOfBioNN.nnUtils.plotUtils import displayEmbeddingHeat,plotWeight
from simulOfBioNN.nnUtils.dataUtils import loadMnist
from simulOfBioNN.parseUtils.parser import saveModelWeight
from simulOfBioNN.nnUtils.clippedSparseBioDenseLayer import clippedSparseBioDenseLayer
from simulOfBioNN.nnUtils.clippedBinaryLayer import clippedBinaryLayer
from simulOfBioNN.nnUtils.clippedSparseBioSigLayer import clippedSparseBioSigLayer
import os
import sys


def train():
    """
        Train some neural network, and save the weight, aka the architecture, so that in can be used by our parser module.

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
    nbUnits = [10,10,10,10]
    nbLayers = len(nbUnits)
    use_bias = True
    epochs = 1
    sess=tf.Session()
    with sess.as_default():
        GPUidx = sess.list_devices()[2].name
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

        savePath = os.path.join(sys.path[0],"weightDir")
        plotWeight(model,use_bias)
        saveModelWeight(model,use_bias,savePath)

        print("Ended Training")
    sess.close()
    del model
    del sess
    return savePath,acc,x_test,y_test,nnAnswer