"""
    Utilitarian for plot management
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage

def displayEmbeddingHeat(X,precision,name):
    '''
        Creates the heat map of an embedding
    :param X:
    :param precision:
    :return:
    '''
    fig, ax = plt.subplots()
    HistMat=np.zeros((X.shape[1],int(1/precision)))
    labels=[]
    for idx in range(X.shape[1]):
        HistMat[idx]=np.histogram(X[:,idx], bins=int(1/precision))[0]
        labels=np.histogram(X[:,idx], bins=int(1/precision))[1]
    im= ax.imshow(HistMat,aspect='auto',cmap='YlGn')
    ax.set_xticks(np.arange(int(1/precision)))
    ax.set_xticklabels([str(round(l,2)) for l in labels])
    ax.set_title(name)
    cbar = ax.figure.colorbar(im, ax=ax)
    fig.tight_layout()

import matplotlib.colors as clr
def weightHeat(W,xnames,ynames):
    fig=plt.figure(figsize=(19.2,10.8), dpi=100)
    for idx,w in enumerate(W):
        ax=fig.add_subplot(2,int(len(W)/2)+len(W)%2,idx+1)
        #w=w/np.max(w)
        ax.imshow(w,aspect='auto',cmap='bwr')
        ax.set_title("Weights of layer "+str(idx+1))
        norm = clr.Normalize(vmin=-1.0, vmax=1.0)
        sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, norm=norm)
        ax.set_xlabel(xnames[idx])
        ax.set_ylabel(ynames[idx])
    fig.tight_layout()
    plt.show()

def plotWeight(model,use_bias):
    weights=model.get_weights()
    W=[]
    xnames = []
    ynames = []
    for i in range(len(weights)):
        if(use_bias):
            if(not i%2==0):
                W += [np.zeros((weights[i-1].shape[0]+1,weights[i-1].shape[1]))]
                W[-1][1:] = weights[i-1]
                W[-1][0] = weights[i]
                xnames += ["toward outputs neurons"]
                ynames += ["coming from inputs neurons"]
        else:
            W+=[weights[i-1]]
            xnames += ["toward outputs neurons"]
            ynames += ["coming from inputs neurons"]
    weightHeat(W,xnames,ynames)



def imageplot(imgs,names=None,fileName=None,path=""):
    if(names!=None):
        assert len(names)==len(imgs)
    fig, axes = plt.subplots(nrows=len(imgs), ncols=1)
    ax = axes.ravel()
    for idx,img in enumerate(imgs):
        ax[idx].imshow(img,cmap="gray")
        if(names!=None):
            ax[idx].set_title(names[idx])
    fig.tight_layout()
    fig.savefig(path+fileName+".png")

def multiTestPlot(resultArray,weights,layers,nbWeights,path=""):
    fig = plt.figure(figsize=(19.2,10.8), dpi=100)
    figsOfWeight =plt.figure(figsize=(19.2,10.8), dpi=100)
    nbWnotAt0s=[]
    for idx,res in enumerate(resultArray):
        nbWnotAt0=[]
        for idx2,w in enumerate(weights[idx]):
            ax2 = figsOfWeight.add_subplot(resultArray.shape[0],resultArray.shape[1],idx2*resultArray.shape[0]+idx+1)
            W=[]
            for e,mat in enumerate(w):
                if(e%2==0): # Working with weights and not bias
                    flat=mat.flatten()
                    W=np.concatenate((W,flat))
            Wno0=np.delete(W, np.where(W == 0), axis=0)
            nbWnotAt0+=[len(Wno0)]
            mymax=np.max(Wno0)
            mymin=np.min(Wno0)
            if(mymax>0):
                absMax=max(abs(mymin),abs(mymax))
                step=absMax/100
                bin=np.concatenate((np.arange(0,mymin-step,-step)[::-1],np.arange(0,mymax+step,step)))
            else:
                step=(mymax-mymin)/200
                bin=np.arange(mymin-step,mymax+step,step)
            ax2.hist(Wno0,bins=bin)
            ax2.set_xlabel(str(int(Wno0.shape[0]))+"/"+str(nbWeights[idx2])+'/'+str(W.shape[0]-Wno0.shape[0]))
            ax2.set_ylabel('n_layer='+str(layers[idx]))
            #'n_w_not_0='+str(int(Wno0.shape[0]))+" n_nd="+str(nbWeights[idx2])+'n_w_at_0='+str(W.shape[0]-Wno0.shape[0])
        ax = fig.add_subplot(resultArray.shape[0],1,idx+1)
        ax.scatter(nbWnotAt0,res[:,1])
        xy=[[w,r] for w,r in zip(nbWnotAt0,res[:,1])]
        xy_text=[str(s1)+"/"+str(round(s2,4)) for s1,s2 in xy]
        for t,p in zip(xy_text,xy):
            ax.annotate(t,p)
        ax.set_xlabel('number of weights')
        ax.set_ylabel("acc")
        ax.set_title("with "+str(layers[idx])+" layers")
        nbWnotAt0s+=[nbWnotAt0]
    fig.tight_layout()
    figsOfWeight.tight_layout()
    fig.savefig(path+"accuracy.png")
    figsOfWeight.savefig(path+"weights.png")
    # plt.show()
    del fig
    del figsOfWeight
    return nbWnotAt0s

def onlyAccTestPlot(resultArray,layers,path="",nbWnotAt0s=None):
    fig = plt.figure(figsize=(19.2,10.8), dpi=100)
    for idx,res in enumerate(resultArray):
        ax = fig.add_subplot(resultArray.shape[0],1,idx+1)
        ax.scatter(nbWnotAt0s[idx],res[:,1])
        xy=[[w,r] for w,r in zip(nbWnotAt0s[idx],res[:,1])]
        xy_text=[str(s1)+"/"+str(round(s2,4)) for s1,s2 in xy]
        for t,p in zip(xy_text,xy):
            ax.annotate(t,p)
        ax.set_xlabel('number of weights')
        ax.set_ylabel("acc")
        ax.set_title("with "+str(layers[idx])+" layers")
    fig.tight_layout()
    fig.savefig(path+"accuracy.png")
    # plt.show()
    del fig