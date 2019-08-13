"""
    Utilitarian for data management
"""

#from umap import UMAP
import numpy as np
import skimage.transform as skt
from skimage.util import random_noise
from tqdm import tqdm
from tensorflow.python.keras.datasets import fashion_mnist,mnist
from simulOfBioNN.nnUtils.plotUtils import imageplot

import os

import pandas


def computeUmap(x_train,x_test,y_train,n_components=100,n_neighbors=5,limitData=None):
    """
        Compute the uniform manifold approximation (UMAP)... Can be use to reduce the dimension of a dataset.
        This can be seen as "cheating", we just use it in test.
    :param x_train: original dataset
    :param x_test: original dataset
    :param y_train: original dataset
    :param n_components: int, see UMAP documentation
    :param n_neighbors: int, see UMAP documentation
    :param limitData: int, to restrict the number of elements in the training set
    :return: x_train,y_train,x_test but projected on a smaller dimension after UMAP is ran.
    """
    if(limitData==None):
        limitData=x_train.shape[0]

    umap=UMAP(n_neighbors=5,n_components=100,verbose=True)

    print("computing UMAP")
    x_train=umap.fit_transform(np.reshape(x_train[0:limitData],(limitData,x_train.shape[1]*x_train.shape[2])))
    for i in range(x_train.shape[1]):
        x_train[:,i]=x_train[:,i]/max(x_train[:,i])

    print("ended umap fit, starting test fitting")
    x_test=umap.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2])))
    for i in range(x_test.shape[1]):
        x_test[:,i]=x_test[:,i]/max(x_test[:,i])

    y_train=y_train[:limitData]

    return x_train,y_train,x_test


def _loadMnist(rescaleFactor=2,fashion=False,size=None,mean=0,var=0.01,path=""):
    '''
        Download the mnist data set
        observation: the input resolution is 8 bits (uint8) but tensorflow only computes with float values, so the set is scaled to [0-1]
    '''
    if(fashion):
        my_mnist = fashion_mnist
    else:
        my_mnist = mnist
    (x_train, y_train),(x_test, y_test) = my_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if(size==None):
        size=x_train.shape[0]
    x_train_out=np.empty((x_train.shape[0],int(x_train.shape[1] / rescaleFactor), int(x_train.shape[2] / rescaleFactor)))
    x_test_out=np.empty((x_test.shape[0],int(x_test.shape[1] / rescaleFactor), int(x_test.shape[2] / rescaleFactor)))
    if(rescaleFactor!=1):
        for idx,img in tqdm(enumerate(x_train[:size])):
            x_train_out[idx]=skt.resize(img,(int(img.shape[0]/rescaleFactor), int(img.shape[1]/rescaleFactor)),anti_aliasing=True)
        for idx,img in tqdm(enumerate(x_test)):
            x_test_out[idx]=skt.resize(img,(int(img.shape[0] / rescaleFactor), int(img.shape[1] / rescaleFactor)),anti_aliasing=True)

    x_test_noise=noiseAddition(x_test_out,mean=mean,var=var)
    imageplot((x_train_out[0], x_train[0]), names=["Downscaled Image by " + str(rescaleFactor), "Initial image"], fileName="downscaled_expl_" + str(rescaleFactor), path=path)
    imageplot((x_test_noise[0], x_test_out[0]), names=["Noised Image by mean" + str(mean) + " var " + str(var), "Initial image"], fileName="noise_" + str(rescaleFactor) + "_mean_" + str(mean) + "_var_" + str(var), path=path)
    return x_train_out,x_test_out,y_train[:size],y_test,x_test_noise

def noiseAddition(x_test,mean,var):
    """
        We increase the difficulty of test by adding random gaussian noises to our test set
    :param x_test: nd_array: test set to modify
    :param amount: proportion in [0,1] of noises to add to our test set
    :return: nd_array: the noisy test set
    """
    x_test_out=np.empty(x_test.shape)
    for idx,img in enumerate(x_test):
        x_test_out[idx]=random_noise(img,mean=mean,var=var)
    return x_test_out


def downloadmnist(rescaleFactor=2,fashion=False,size=None,mean=0,var=0.01,path="Data/mnist/2"):
    """
        Download the mnist of fashion-mnist data set and save them in the path repo
    :param rescaleFactor: int, The rescale factor on the data set
    :param fashion: use fashion-mnist.
    :param size: maximal size, default to None: keep all the data
    :param mean: mean for noise
    :param var: variance for noise
    :param path: string, default to "Data/mnist/2"
    :return:
    """
    x_train,x_test,y_train,y_test,x_test_noise=_loadMnist(rescaleFactor=rescaleFactor,fashion=fashion,size=size,mean=mean,var=var,path=path)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
    x_test_noise = np.reshape(x_test_noise,(x_test_noise.shape[0],x_test_noise.shape[1]*x_test_noise.shape[2]))
    df=pandas.DataFrame(x_train)
    df.to_csv(path+"x_train.csv")
    df=pandas.DataFrame(x_test)
    df.to_csv(path+"x_test.csv")
    df=pandas.DataFrame(y_train)
    df.to_csv(path+"y_train.csv")
    df=pandas.DataFrame(y_test)
    df.to_csv(path+"y_test.csv")
    df=pandas.DataFrame(x_test_noise)
    df.to_csv(path+"x_test_noise.csv")

def loadMnist(rescaleFactor=2,fashion=False,size=None,mean=0,var=0.01,path="Data/mnist"):
    """
        Load Mnist without downloading it.
    :param rescaleFactor: int, The rescale factor on the data set
    :param fashion: use fashion-mnist.
    :param size: maximal size, default to None: keep all the data
    :param mean: mean for noise
    :param var: variance for noise
    :param path: string, path to the directory where is situated the original dataset.
    :return: x_train,x_test,y_train,y_test,x_test_noise: the different data set.
    """
    if not str(rescaleFactor) in os.listdir(path) or not "x_train.csv" in os.listdir(path+"/"+str(rescaleFactor)):
        if not str(rescaleFactor) in os.listdir(path):
            os.makedirs(path+"/"+str(rescaleFactor)+"/")
        downloadmnist(rescaleFactor=rescaleFactor,fashion=fashion,size=None,mean=0,var=0.01,path=path+"/"+str(rescaleFactor)+"/")
    path=path+"/"+str(rescaleFactor)+"/"
    df=pandas.read_csv(path+"x_train.csv")
    x_train=df.values[:,1:]
    x_train = np.reshape(x_train,(x_train.shape[0],int(x_train.shape[1]**(0.5)),int(x_train.shape[1]**(0.5))))
    df=pandas.read_csv(path+"x_test.csv")
    x_test=df.values[:,1:]
    x_test = np.reshape(x_test,(x_test.shape[0],int(x_test.shape[1]**(0.5)),int(x_test.shape[1]**(0.5))))
    df=pandas.read_csv(path+"y_train.csv")
    y_train=df.values[:,1:]
    df=pandas.read_csv(path+"y_test.csv")
    y_test=df.values[:,1:]
    df=pandas.read_csv(path+"x_test_noise.csv")
    x_test_noise=df.values[:,1:]
    x_test_noise = np.reshape(x_test_noise,(x_test_noise.shape[0],int(x_test_noise.shape[1]**(0.5)),int(x_test_noise.shape[1]**(0.5))))
    return x_train,x_test,y_train,y_test,x_test_noise
