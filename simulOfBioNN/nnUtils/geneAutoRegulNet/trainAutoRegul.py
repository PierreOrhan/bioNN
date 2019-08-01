"""
    Testing capacities of a genetic network based on auto regulation and cooperation with promoters.
"""

from simulOfBioNN.nnUtils.plotUtils import displayEmbeddingHeat,plotWeight
from simulOfBioNN.nnUtils.dataUtils import loadMnist
import tensorflow as tf

from simulOfBioNN.nnUtils.geneAutoRegulNet.autoRegulLayer import autoRegulLayer


import os
import numpy as np
from simulOfBioNN.nnUtils.neurVorConcSet import VoronoiSet
import matplotlib.pyplot as plt

def trainWithChemTemplateNN(savePath):

    x_train,x_test,y_train,y_test,x_test_noise=loadMnist(rescaleFactor=2,fashion=False,size=None,mean=0,var=0.01,path="../../../Data/mnist")

    x_test = np.reshape(x_test,(x_test.shape[0],(x_test.shape[1]*x_test.shape[2]))).astype(dtype=np.float32)
    x_train = np.reshape(x_train,(x_train.shape[0],(x_train.shape[1]*x_train.shape[2]))).astype(dtype=np.float32)

    nbUnits = [100,100]
    sparsities = [0.9,0.9]

    assert len(nbUnits)==len(sparsities)

    epochs = 100
    usingSoftmax = True

    model2 = tf.keras.Sequential()
    for idx,n in enumerate(nbUnits):
        model2.add(autoRegulLayer(units=n,biasValue=np.zeros(n)+1.,fractionZero=sparsities[idx],rate = 1.,activation=tf.keras.activations.relu))
    if usingSoftmax:
        model2.add(tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax))
    model2.compile(optimizer=tf.optimizers.Adam(),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy']
                   )
    model2.build(input_shape=(None,x_train.shape[-1]))
    model2.fit(x_train[:], y_train[:],epochs=epochs,verbose=True,validation_data=(x_test,y_test))


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