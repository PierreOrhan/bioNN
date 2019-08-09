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
    sparsities = [0.5,0.5]

    assert len(nbUnits)==len(sparsities)

    epochs = 100
    usingSoftmax = True

    model2 = tf.keras.Sequential()
    for idx,n in enumerate(nbUnits):
        model2.add(autoRegulLayer(units=n,biasValue=np.zeros(n)+1.,fractionZero=sparsities[idx],rate = 10., rateInhib= 10.,activation=tf.keras.activations.relu))
    if usingSoftmax:
        model2.add(autoRegulLayer(units=10,biasValue=np.zeros(n)+1.,fractionZero=0,rate = 10., rateInhib= 10.,activation=tf.keras.activations.softmax))
    model2.compile(optimizer=tf.optimizers.Adam(),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy']
                   )
    model2.build(input_shape=(None,x_train.shape[-1]))
    model2.fit(x_train[:], y_train[:],epochs=epochs,verbose=True,validation_data=(x_test,y_test))

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class animator:

    def __init__(self,size,epochs = 10,nbUnits=[100,100],sparsities=[0.5,0.5],frames=10, use_mnist=True):
        self.size = size
        self.nbUnits = nbUnits
        self.sparsities = sparsities
        self.biasList = np.logspace(-2,2,frames)
        self.use_mnist = use_mnist
        if not self.use_mnist:
            self.barycenters=np.log(np.array([[5*10**(-6),10**(-4)],[10**(-5),5*10**(-6)],[10**(-4),10**(-4)]])/(8*10**-7))
            set=VoronoiSet(self.barycenters)
            x_train,y_train=set.generate(10000)
            x_train = np.asarray(x_train,dtype=np.float32)
            x_test, y_test=set.generate(1000)
            x_test = np.asarray(x_test,dtype=np.float32)
        else:
            x_train,x_test,y_train,y_test,x_test_noise=loadMnist(rescaleFactor=2,fashion=False,size=None,mean=0,var=0.01,path="../../../Data/mnist")
            x_test = np.reshape(x_test,(x_test.shape[0],(x_test.shape[1]*x_test.shape[2]))).astype(dtype=np.float32)
            x_train = np.reshape(x_train,(x_train.shape[0],(x_train.shape[1]*x_train.shape[2]))).astype(dtype=np.float32)

        self.epoch =epochs
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.fig, self.ax = plt.subplots(figsize=(19.2,10.8), dpi=100)

        self.norm = clr.Normalize(vmin=0., vmax=1.)
        cmap = plt.get_cmap("Oranges")
        self.sm = plt.cm.ScalarMappable(cmap=cmap, norm=self.norm )
        self.sm .set_array([])
        cbar = self.ax.figure.colorbar(self.sm ,ax=self.ax,norm=self.norm )
        cbar.ax.set_ylabel("scores",fontsize="xx-large")
        cbar.ax.tick_params(labelsize="xx-large")
        self.cmap = cmap

    def testSomeActivation(self,i,plots):

        ax = self.ax
        ax.clear()

        epochs = self.epoch
        usingLog = True
        usingSoftmax = True

        x_train =  self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test

        # print(y_test)
        # colors = ["r","g","b"]
        # for idx,x in enumerate(x_test):
        #     plt.scatter(x[0],x[1],c=colors[y_test[idx]])
        # for b in barycenters:
        #     plt.scatter(b[0],b[1],c="m",marker="x")
        # plt.show()

        size = self.size

        nbUnits = self.nbUnits
        sparsities = self.sparsities

        beta_list = np.logspace(-2,2,size)
        alpha_list = np.logspace(-2,2,size)

        scores = np.zeros((size,size))
        from tqdm import  tqdm
        # argsList=[]

        for idx1,beta in tqdm(enumerate(beta_list)):
            for idx2,alpha in enumerate(alpha_list):
                # argsList +=[[b,a,barycenters,usingSoftmax,x_train,y_train,x_test,y_test,epochs]]
                # def mylogActivation(x):
                #     return tf.math.log(a * tf.math.exp(x)/(b + tf.math.exp(x)))
                s=0
                repeat = 1
                for e in range(repeat):
                    model2 = tf.keras.Sequential()
                    for idx,n in enumerate(nbUnits):
                        model2.add(autoRegulLayer(units=n,biasValue=np.zeros(n)+self.biasList[i],fractionZero=sparsities[idx],rate = float(alpha), rateInhib= float(beta),activation=tf.keras.activations.relu))
                    if usingSoftmax:
                        if self.use_mnist:
                            model2.add(autoRegulLayer(units=10,biasValue=np.zeros(n)+self.biasList[i],fractionZero=0,rate = float(alpha), rateInhib= float(beta),activation=tf.keras.activations.softmax))
                        else:
                            model2.add(autoRegulLayer(units=len(self.barycenters),biasValue=np.zeros(n)+self.biasList[i],fractionZero=0,rate = float(alpha), rateInhib= float(beta),activation=tf.keras.activations.softmax))
                    model2.compile(optimizer=tf.optimizers.Adam(),
                                   #loss=tf.keras.losses.BinaryCrossentropy(),
                                   loss='sparse_categorical_crossentropy',
                                   #loss = tf.keras.losses.MeanSquaredError(),
                                   metrics=['accuracy']
                                   #metrics=[tf.keras.metrics.MeanSquaredError()]
                                   )
                    model2.build(input_shape=(None,x_train.shape[-1]))
                    print("starting fit")
                    model2.fit(x_train[:], y_train[:],epochs=epochs,verbose=True)
                    print("ended fit")
                    answer = np.argmax(model2.predict(x_test[:]),axis=1)
                    y_test = np.reshape(y_test,(y_test.shape[0]))
                    del model2
                    s += np.sum(np.where(np.equal(answer,y_test),1,0))/y_test.shape[0]
                scores[idx1,idx2] = s/repeat

        import pandas as pd
        df =pd.DataFrame(scores)
        if self.use_mnist:
            df.to_csv("resultsMNIST"+str(i))
        else:
            df.to_csv("results"+str(i))

        ax.imshow(scores,cmap=self.cmap,norm = self.norm)
        ax.set_xticks(np.arange(len(alpha_list)))
        ax.set_yticks(np.arange(len(beta_list)))
        ax.set_xticklabels([round(a,3) for a in alpha_list])
        ax.set_yticklabels([round(b,3) for b in beta_list])
        ax.tick_params(labelsize="xx-large")
        ax.set_xlabel("Alpha",fontsize="xx-large")
        ax.set_ylabel("Beta",fontsize="xx-large")
        ax.set_title("Scores function of alpha and beta, for bias = "+str(round(self.biasList[i],3)),fontsize="xx-large")#fontdict={"fontsize":"xx-large"}
        #ax.show()
        if self.use_mnist:
            self.fig.savefig("resultsMNIST"+str(i)+".png")
        else:
            self.fig.savefig("results"+str(i)+".png")

        return [self.ax]


from matplotlib.animation import PillowWriter
class LoopingPillowWriter(PillowWriter):
    def finish(self):
        self._frames[0].save(
            self._outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)

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
    # frames = 4
    # use_mnist = True
    # my_anim = animator(size = 5,nbUnits=[1000,1000],sparsities=[0.5,0.5], epochs = 3 , frames=frames,use_mnist=use_mnist)
    #
    # # fig,ax = plt.subplots(figsize=(19.2,10.8), dpi=100)
    # ani = animation.FuncAnimation(my_anim.fig, my_anim.testSomeActivation , fargs=[my_anim.ax], frames=frames, repeat=True, interval=20, blit=False)
    # if use_mnist:
    #     ani.save('animationScoreVSratesAndBiasMNIST.gif',writer=LoopingPillowWriter(fps=1))
    # else:
    #     ani.save('animationScoreVSratesAndBias.gif',writer=LoopingPillowWriter(fps=1))

