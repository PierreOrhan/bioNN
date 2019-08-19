"""
    In this file we produce a little animation to show the influence of rescaling E0 and the template initial concentration relatively to the network size.
"""

from simulOfBioNN.nnUtils.dataUtils import loadMnist
from simulOfBioNN.parseUtils.parser import sparseParser,read_file,generateTemplateNeuralNetwork
from simulOfBioNN.odeUtils.systemEquation import setToUnits
from simulOfBioNN.nnUtils.chemTemplateNN.chemTemplateNNModel import chemTemplateNNModel
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import numpy as np


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

class animator:

    def __init__(self,model,x_train,frames):
        self.model = model
        self.x_train = x_train
        self.rangeForRescale = 1 + np.arange(0,self.model.rescaleFactor,self.model.rescaleFactor/frames)

    def animate(self,i):

        plt.cla()
        forcedRescaleFactor = self.rangeForRescale[i%self.rangeForRescale.shape[0]]
        self.model.force_rescale(forcedRescaleFactor)
        print("computing cps at equilibrium")
        import time
        t = time.time()
        cps = self.model.obtainCp(tf.convert_to_tensor(self.x_train[:10],dtype=tf.float32))
        print("ended computing of cps in ",time.time()-t)
        plt.scatter(range(cps.shape[0]),cps[:,0],c="b")
        plt.yscale("log")
        plt.ylim(1,10**7)
        plt.title("competition, mnist dataset, with rescale: "+str(forcedRescaleFactor))
        print(i)

    def mean(self,i):
        forcedRescaleFactor = self.rangeForRescale[i%self.rangeForRescale.shape[0]]
        self.model.force_rescale(forcedRescaleFactor)
        cps = self.model.obtainCp(tf.convert_to_tensor(self.x_train[:10],dtype=tf.float32))

        modelValidity = self.model.inhibTempInitC/self.model.layerList[0].E0

        return np.mean(cps),modelValidity,cps




def drawInformation(savePath,frames=400):

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
    nbUnits = [100,30,10]
    sparsities = [0.5,0.5,0.5]
    use_bias = False
    epochs = 10
    my_batchsize = 32

    x_train = x_train/C0
    x_test = x_test/C0

    device_name = tf.test.gpu_device_name()
    if not tf.test.is_gpu_available():
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    forcedRescaleFactor = 1

    model = chemTemplateNNModel(nbUnits=nbUnits,sparsities=sparsities,reactionConstants= constantList, enzymeInitC=enzymeInit, activTempInitC=activInit,
                                inhibTempInitC=inhibInit, randomConstantParameter=None)
    print("model is running eagerly: "+str(model.run_eagerly))
    # model.run_eagerly=True
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.build(input_shape=(None,x_train.shape[-1]))

    print("creating animation:")

    my_anim = animator(model,x_train,frames)

    # fig,ax = plt.subplots(figsize=(19.2,10.8), dpi=100)
    # ani = animation.FuncAnimation(fig, my_anim.animate , frames=frames, repeat=True, interval=20, blit=False)
    # ani.save('zoomFULL.gif',writer=LoopingPillowWriter(fps=40))


    fig2,ax = plt.subplots(figsize=(19.2,10.8), dpi=100)
    mean=[]
    cps=[]
    modelValidity=[]
    from tqdm import tqdm
    for i in tqdm(np.arange(0,frames)):
        a,b,c = my_anim.mean(i)
        mean +=[a]
        modelValidity+=[b]
        cps+=[c]
    ax.plot(my_anim.rangeForRescale[:len(mean)],mean, c="b", label="cps mean")
    ax.set_xlabel("rescale factor",fontsize="xx-large")
    ax.set_ylabel("cp mean over 10 first images of mnist",fontsize="xx-large")
    ax.set_yscale("log")
    ax.tick_params(labelsize="xx-large")
    ax2 = ax.twinx()
    ax2.plot(my_anim.rangeForRescale[:len(mean)],modelValidity, c="r" , label="modelValidity")
    ax2.set_yscale("log")
    ax2.set_ylabel('templateInhibConcentration divived by E0',fontsize="xx-large")
    ax2.tick_params(labelsize="xx-large")

    fig2.legend()
    fig2.show()
    fig2.savefig("competitionAndValidtywrtRescaleFactorFULL")

    fig,ax = plt.subplots(figsize=(19.2,10.8), dpi=100)
    cps=np.array(cps)
    cmap = plt.get_cmap('tab20',cps.shape[1])
    import matplotlib.colors as clr
    norm = clr.Normalize(vmin=0, vmax=cps.shape[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm,ax=ax,norm=norm)
    cbar.ax.set_ylabel("idx of image",fontsize="xx-large")
    cbar.ax.tick_params(labelsize="xx-large")
    for e in range(cps.shape[1]):
        ax.plot(my_anim.rangeForRescale[:cps.shape[0]],cps[:,e],c=cmap(e))
    ax.set_yscale("log")
    ax.set_ylabel("computed cps",fontsize="xx-large")
    ax.set_xlabel("rescale factor",fontsize="xx-large")
    ax.tick_params(labelsize="xx-large")
    fig.show()
    fig.savefig("competitionwrtRescaleFactorFULL")

def drawCourbs(savePath,frames=400):

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
    nbUnits = [100,30,10]
    sparsities = [0.5,0.5,0.5]

    x_train = x_train/C0
    x_test = x_test/C0

    model = chemTemplateNNModel(nbUnits=nbUnits,sparsities=sparsities,reactionConstants= constantList, enzymeInitC=enzymeInit, activTempInitC=activInit,
                                inhibTempInitC=inhibInit, randomConstantParameter=None)
    print("model is running eagerly: "+str(model.run_eagerly))
    # model.run_eagerly=True
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.build(input_shape=(None,x_train.shape[-1]))

    print("creating animation:")

    my_anim = animator(model,x_train,frames)

    fig,ax = plt.subplots(figsize=(19.2,10.8), dpi=100)
    ax.plot(my_anim.rangeForRescale,np.power(my_anim.rangeForRescale,0.5)*model.enzymeInitC,c="r",label="enzyme rescaled")
    ax.plot(my_anim.rangeForRescale,np.array([model.inhibTempInitC]*my_anim.rangeForRescale.shape[0]),c="b",label="template are not rescaled")
    ax.set_xlabel("rescale factor",fontsize="xx-large")
    ax.set_ylabel("rescaled value",fontsize="xx-large")
    #ax.set_yscale("log")
    ax.tick_params(labelsize="xx-large")
    fig.legend()
    fig.show()
    fig.savefig("EnzymeAndtemplateCourbs")

from matplotlib.animation import PillowWriter

class LoopingPillowWriter(PillowWriter):
    def finish(self):
        self._frames[0].save(
            self._outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)


if __name__ == '__main__':
    #tf.debugging.set_log_device_placement(True)
    import sys
    p1 = os.path.join(sys.path[0],"..")
    p3 = os.path.join(p1,"trainingWithChemicalNN")
    if not os.path.exists(p3):
        os.makedirs(p3)
    drawInformation(p3,frames=400)
    #drawCourbs(p3,frames=400)
