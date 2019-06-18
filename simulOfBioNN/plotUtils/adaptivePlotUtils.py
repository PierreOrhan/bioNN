import os,sys
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.colors as clr

from scipy.spatial import cKDTree

def plotEvolution(time, experiment_name, nameDic, X, wishToDisp=[""], displaySeparate=False, displayOther=True):
    """
        Function to plot the evolution against the time of the concentration of some species.
    :param time: 1d-array, containing the time.
    :param experiment_name: result path will be of the form: resultPath=experiment_name+"/resultData/"+"separate"+"/"
                                                        and  resultPath=experiment_name+"/resultData/"
    :param nameDic: dictionnary with the name of species as key, and position in the array as values.
    :param X: 2d-array, axis0: time, axis1: species as described by namedic.
    :param wishToDisp: name of species one wish to disp individually
    :param displaySeparate: if we want to Display the individual species in wishToDisp
    :param displayOther: if we want to display all intermediate and all main species in a separate fashion.
    :return:
    """
    experiment_name=os.path.join(sys.path[0],experiment_name)
    resultPath=os.path.join(experiment_name,"resultData/separate/")
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    fulltxt=nameDic.keys()
    selected={}
    for t in fulltxt:
        selected[t]=0
    for w in wishToDisp:
        selected[w]=1

    c=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    if(displaySeparate):
        figs=[plt.figure(figsize=(19.2,10.8), dpi=100) for _ in wishToDisp]
        figIdx=0
        for idx,k in enumerate(nameDic.keys()):
            if(selected[k]):
                ax=figs[figIdx].add_subplot(1,1,1)
                #ax.set_yscale('log')
                ax.plot(time,X[:,nameDic[k]],c="b")
                ax.set_title(k)
                figIdx+=1
        for idx,fig in enumerate(figs):
            fig.savefig(os.path.join(resultPath,str(wishToDisp[idx])+".png"))
        for fig in figs:
            plt.close(fig)
    if(displayOther):
        try:
            assert X.shape[1]==len(list(nameDic.keys()))
        except:
            raise Exception("please, provide an array with the concentration for all species at all time step!")
        txt_inter={}
        for t in fulltxt:
            txt_inter[t]=0#we wish to plot only intermediate species: that is species of name larger than 3
        for t in fulltxt:
            if(len(t)>=3):
                txt_inter[t]=1
        resultPath=os.path.join(experiment_name,"resultData/")
        figIntermediate = plt.figure(figsize=(19.2,10.8), dpi=100)
        axIntermediate = figIntermediate.add_subplot(1,1,1)
        for idx,k in enumerate(nameDic.keys()):
            if(txt_inter[k]):
                axIntermediate.plot(time,X[:,idx],label=k)
        axIntermediate.legend()
        figIntermediate.savefig(os.path.join(resultPath,"intermediateSpecies0.png"))
        plt.close(figIntermediate)

        txt_inter={}
        for t in fulltxt:
            txt_inter[t]=0#we wish to plot only intermediate species: that is species of name larger than 3
        for t in fulltxt:
            if(len(t)<3):
                txt_inter[t]=1
        figMain = plt.figure(figsize=(19.2,10.8), dpi=100)
        axMain = figMain.add_subplot(1,1,1)
        for idx,k in enumerate(nameDic.keys()):
            if(txt_inter[k]):
                axMain.plot(time,X[:,idx],label=k)
        axMain.legend()
        figMain.savefig(os.path.join(resultPath,"mainSpecies0.png"))
        plt.close(figMain)

def colorDiagram(X,Y,Z,nameX,nameY,nameZ,figname,colorBarVal=None,equiPotential=True,lineToKeep=None,uselog=False):
    """
        Creates a 2d-plot with heat for Z.
    :param X: 1d-array, X axis
    :param Y: 1d-array, Y data
    :param Z: 1d-array, Z data
    :param nameX: str, name for X
    :param nameY: str, name for Y
    :param nameZ: str, name for Z
    :param figname: str, name for saving
    :param colorBarVal: tuple of size2, if using a predefined range for the colorbar,
                        default to None: use min and max value of Z
    :param equiPotential: if using equipotential
    :param lineToKeep: Indicate line to use in case some line are not to be used, for example for computation error.
                        Default to None: we use all line, except line where a nan value can be observed
    :param uselog: if the plot is in log with respect to the Z axis.
    :return: lintoKeep: line to keep.
    """
    Z=np.copy(Z)
    Y=Y[::-1]
    for x in range(Z.shape[0]):
        Z[x]=Z[x][::-1]

    ##  We delete every line where a nan appear, and indicate the skipped line
    if(not lineToKeep):
        X2=[]
        Z2=[]
        lintoKeep=[]
        for idx in range(X.shape[0]):
            if((Z[idx,:]*0==np.zeros(Z.shape[1])).all()): #not a nan
                X2+=[X[idx]]
                Z2+=[Z[idx,:]]
                lintoKeep+=[idx]
            else:
                print(str(idx)+" nan detected for input X:"+str(X[idx]))
        Z=np.array(Z2)
        X=np.array(X2)
    else:
        lintoKeep=lineToKeep
        Z=np.array(Z[lineToKeep])
        X=np.array(X[lineToKeep])

    if(uselog):
        Z=np.log(Z)

    fig, ax = plt.subplots(figsize=(19.2,10.8), dpi=100)

    cmap = plt.get_cmap('jet',Z.shape[0]*Z.shape[1])

    im= ax.imshow(np.transpose(Z),aspect='auto',cmap=cmap)

    xTicks = np.arange(0,X.shape[0],int(X.shape[0]/min(10,X.shape[0])))
    ax.set_xticks(xTicks)
    ax.set_xticklabels([str('{:.2e}'.format(x)) for x in X[xTicks]])


    yTicks = np.arange(0,Y.shape[0],int(Y.shape[0]/min(20,Y.shape[0])))
    ax.set_yticks(yTicks)
    ax.set_yticklabels([str('{:.2e}'.format(y)) for y in Y[yTicks]])

    ax.set_xlabel(nameX,fontsize="xx-large")
    ax.set_ylabel(nameY,fontsize="xx-large")
    ax.tick_params(labelsize="xx-large")
    if(uselog):
        if not colorBarVal:
            norm = clr.LogNorm(vmin=np.min(Z), vmax=np.max(Z))
        else:
            norm=clr.LogNorm(vmin=colorBarVal[0],vmax=colorBarVal[1])
    else:
        if not colorBarVal:
            norm = clr.Normalize(vmin=np.min(Z), vmax=np.max(Z))
        else:
            norm=clr.Normalize(vmin=colorBarVal[0],vmax=colorBarVal[1])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = ax.figure.colorbar(sm,ax=ax,norm=norm)
    cbar.ax.set_ylabel(nameZ,fontsize="xx-large")
    cbar.ax.tick_params(labelsize="xx-large")


    # EQUIPOTENTIAL:
    if(equiPotential):
        myrange =np.arange(np.min(Z),np.max(Z)+(np.max(Z)-np.min(Z))/10,(np.max(Z)-np.min(Z))/10)
        rangeTree = cKDTree(np.reshape(np.stack([myrange]),(myrange.shape[0],1)))
        lines={}
        for r in myrange:
            lines[r]=[]
        # lines=np.zeros((len(myrange),Z.shape[0],Z.shape[1],2))

        for x in range(X.shape[0]):
            for y in range(Y.shape[0]):
                dist,position = rangeTree.query([Z[x,y]])
                if(dist<Z[x,y]/50):
                    lines[myrange[position]] += [[x,y]]

        for l in lines.keys():
            data=np.array(lines[l])
            if(len(data)>0):
                plt.plot(data[:,0],data[:,1],label='{:.2e}'.format(l)+" +/- "+'{:.3e}'.format(l/50),c="w")
        plt.legend()

    fig.tight_layout()
    figpath=os.path.join(sys.path[0],figname)
    fig.savefig(figpath)
    plt.close(fig)


    return lintoKeep

def neuronPlot(X1,X2,output,figname,figname2):
    """
        Similar to color diagram.
        Plot X1=f(output) with X1  as color and X1=f(output) with X2 as color
    :param X1: 1d-array
    :param X2: 1d-array
    :param output: 1d-array
    :param figname: name for X1=f(output) with X2 as color
    :param figname2: name for X1=f(output) with X1
    :return:
    """
    #output  vs X1:
    fig,ax=plt.subplots(figsize=(19.2,10.8), dpi=100)
    cmap = plt.get_cmap('jet',X2.shape[0])
    ##time to plot:
    for idx in range(output.shape[1]):
        X=X1
        Y=output[:,idx]
        ax.plot(X,Y,c=cmap(idx)) #,label=str(X2[idx-1])
        ax.tick_params(labelsize="xx-large")

    norm = clr.Normalize(vmin=X2[0],vmax=X2[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    xtickObs=np.arange(0,X2.shape[0],int(X2.shape[0]/min(5,X2.shape[0])))
    xticks=X2[xtickObs]

    cbar = ax.figure.colorbar(sm,ticks=xticks)
    cbar.ax.set_ylabel("Initial concentration of X2",fontsize="xx-large")
    cbar.ax.tick_params(labelsize="xx-large")
    ax.set_xlabel("Initial concentration of X1",fontsize="xx-large")
    ax.set_ylabel("equilibrium concentration of the output rescaled unit",fontsize="xx-large")
    ax.tick_params(labelsize="xx-large")
    plt.show()
    figpath=os.path.join(sys.path[0],figname)
    fig.savefig(figpath)
    plt.close(fig)

    #output  vs X2:
    fig,ax=plt.subplots(figsize=(19.2,10.8), dpi=100)
    cmap = plt.get_cmap('jet',X1.shape[0])
    ##time to plot:
    for idx in range(output.shape[0]):
        X=X2
        Y=output[idx,:]
        ax.plot(X,Y,c=cmap(idx)) #,label=str(X2[idx-1])
        ax.tick_params(labelsize="xx-large")

    norm = clr.Normalize(vmin=X1[0],vmax=X1[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # xtickObs=np.arange(0,X1.shape[0],int(X1.shape[0]/min(5,X1.shape[0])))
    # xticks=X1[xtickObs]
    xticks=[0,20,40,60,80,100]
    cbar = ax.figure.colorbar(sm,ticks=xticks)
    cbar.ax.set_ylabel("Initial concentration of X1",fontsize="xx-large")
    cbar.ax.tick_params(labelsize="x-large")
    ax.set_xlabel("Initial concentration of X2",fontsize="xx-large")
    ax.set_ylabel("Equilibrium concentration of the output rescaled unit",fontsize="xx-large")
    ax.tick_params(labelsize="xx-large")
    plt.show()
    figpath=os.path.join(sys.path[0],figname2)
    fig.savefig(figpath)
    plt.close(fig)


def fitComparePlot(X1,X2,output,fitOutput,courbs,figname,figname2):
    """
        Plot on the same graph:
         the evolution of output for X1,X2 (colorbar)
         the evolution of fitoutput against X1,X2 (colorbar)
    :param X1: simulated input X1
    :param X2: simulated input X2
    :param output: simulated f(X1,X2) at equilibrium
    :param fitOutput: fitted f(fitX1,fitX2) at equilibrium
    :return:
    """
    #output  vs X1:
    fig,ax=plt.subplots(figsize=(19.2,10.8), dpi=100)
    cmap = plt.get_cmap('jet',X2.shape[0])
    ##time to plot:
    for idx in range(output.shape[1]):
        if idx in courbs:
            ax.plot(X1, output[:,idx], c=cmap(idx),label="simulated for X2="+str(round(X2[idx],4)))
            ax.plot(X1, fitOutput[:,idx], c=cmap(idx), linestyle="dashed",label="fitted for X2="+str(round(X2[idx],4)))
            ax.tick_params(labelsize="xx-large")

    norm = clr.Normalize(vmin=X2[0],vmax=X2[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    xtickObs=np.arange(0,X2.shape[0],int(X2.shape[0]/min(5,X2.shape[0])))
    xticks=X2[xtickObs]

    cbar = ax.figure.colorbar(sm,ticks=xticks)
    cbar.ax.set_ylabel("Initial concentration of X2",fontsize="xx-large")
    cbar.ax.tick_params(labelsize="xx-large")
    ax.set_xlabel("Initial concentration of X1",fontsize="xx-large")
    ax.set_ylabel("equilibrium concentration of the output rescaled unit",fontsize="xx-large")
    ax.tick_params(labelsize="xx-large")
    plt.legend()
    plt.show()
    figpath=os.path.join(sys.path[0],figname)
    fig.savefig(figpath)
    plt.close(fig)
    #output  vs X2:
    fig,ax=plt.subplots(figsize=(19.2,10.8), dpi=100)
    cmap = plt.get_cmap('jet',X1.shape[0])
    ##time to plot:
    for idx in range(output.shape[0]):
        if idx in courbs:
            ax.plot(X2, output[idx,:], c=cmap(idx),label="simulated for X1="+str(round(X1[idx],4)))
            ax.plot(X2, fitOutput[idx,:], c=cmap(idx), linestyle="dashed",label="fitted for X1="+str(round(X1[idx],4)))
            ax.tick_params(labelsize="xx-large")

    norm = clr.Normalize(vmin=X1[0],vmax=X1[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # xtickObs=np.arange(0,X1.shape[0],int(X1.shape[0]/min(5,X1.shape[0])))
    # xticks=X1[xtickObs]
    xticks=[0,20,40,60,80,100]
    cbar = ax.figure.colorbar(sm,ticks=xticks)
    cbar.ax.set_ylabel("Initial concentration of X1",fontsize="xx-large")
    cbar.ax.tick_params(labelsize="x-large")
    ax.set_xlabel("Initial concentration of X2",fontsize="xx-large")
    ax.set_ylabel("Equilibrium concentration of the output rescaled unit",fontsize="xx-large")
    ax.tick_params(labelsize="xx-large")
    plt.legend()
    plt.show()
    figpath=os.path.join(sys.path[0],figname2)
    fig.savefig(figpath)
    plt.close(fig)

def multipleComparePlot(X1,X2,outputs,courbs,figname,figname2,lineStyle=None,outputsNames=None):
    """
        Plot on the same graph:
            the evolution of outputs for X1, X2 (X2 in colorbar). Outputs is a 3d-array, with different strategies on the first axis.
    :param X1: simulated input X1
    :param X2: simulated input X2
    :param outputs: 3d-array of simulated f(X1,X2) at equilibrium for different strategy. Strategies are differentiated by the first axis.
    :param courbs: idx of the courbs with respect to the second input to display: outputs[:,:,idx] will be displayed for idx in courbs
    :param fitOutput: fitted f(fitX1,fitX2) at equilibrium
    :return:
    """
    if(lineStyle==None):
        try:
            assert len(outputs)<4
        except:
            print("Can manage at most 3 courbs if lineStyle is not given")
            raise

    if(lineStyle==None):
        lineStyle = ["-","--","-."]
    if(outputsNames==None):
        outputsNames = [str(idx) for idx in range(len(outputs))]

    #output  vs X1:
    fig,ax=plt.subplots(figsize=(19.2,10.8), dpi=100)
    cmap = plt.get_cmap('jet',X2.shape[0])
    ##time to plot:
    for idx in range(outputs.shape[2]):
        if idx in courbs:
            for idx2 in range(len(outputs)):
                ax.plot(X1, outputs[idx2,:,idx], c=cmap(idx), linestyle=lineStyle[idx2], label=outputsNames[idx2]+" for X2="+str(round(X2[idx],4)))
                ax.tick_params(labelsize="xx-large")

    norm = clr.Normalize(vmin=X2[0],vmax=X2[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    xtickObs=np.arange(0,X2.shape[0],int(X2.shape[0]/min(5,X2.shape[0])))
    xticks=X2[xtickObs]

    cbar = ax.figure.colorbar(sm,ticks=xticks)
    cbar.ax.set_ylabel("Initial concentration of X2",fontsize="xx-large")
    cbar.ax.tick_params(labelsize="xx-large")
    ax.set_xlabel("Initial concentration of X1",fontsize="xx-large")
    ax.set_ylabel("equilibrium concentration of the output rescaled unit",fontsize="xx-large")
    ax.tick_params(labelsize="xx-large")
    plt.legend()
    plt.show()
    figpath=os.path.join(sys.path[0],figname)
    fig.savefig(figpath)
    plt.close(fig)
    #output  vs X2:
    fig,ax=plt.subplots(figsize=(19.2,10.8), dpi=100)
    cmap = plt.get_cmap('jet',X1.shape[0])
    ##time to plot:
    for idx in range(outputs.shape[1]):
        if idx in courbs:
            for idx2 in range(len(outputs)):
                ax.plot(X2, outputs[idx2,idx,:], c=cmap(idx), linestyle=lineStyle[idx2], label=outputsNames[idx2]+" for X1="+str(round(X1[idx],4)))
                ax.tick_params(labelsize="xx-large")

    norm = clr.Normalize(vmin=X1[0],vmax=X1[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # xtickObs=np.arange(0,X1.shape[0],int(X1.shape[0]/min(5,X1.shape[0])))
    # xticks=X1[xtickObs]
    xticks=[0,20,40,60,80,100]
    cbar = ax.figure.colorbar(sm,ticks=xticks)
    cbar.ax.set_ylabel("Initial concentration of X1",fontsize="xx-large")
    cbar.ax.tick_params(labelsize="x-large")
    ax.set_xlabel("Initial concentration of X2",fontsize="xx-large")
    ax.set_ylabel("Equilibrium concentration of the output rescaled unit",fontsize="xx-large")
    ax.tick_params(labelsize="xx-large")
    plt.legend()
    plt.show()
    figpath=os.path.join(sys.path[0],figname2)
    fig.savefig(figpath)
    plt.close(fig)

def timeCompare(dic,figname,xticks):
    """
        Plot time took by an experiment ran on different architectures
    :param dic: A dictionary with experiment names as: architecture_experiment.
    :param figname: where to store the figure
    :param xticks: ticks for the x axis
    :return:
    """
    fig,ax=plt.subplots(figsize=(19.2,10.8), dpi=100)

    architecturesList=[]
    for k in dic.keys(): #init of the dic
        if not k.split("_")[0] in architecturesList:
            architecturesList += [k.split("_")[0]]

    times=[[] for _ in architecturesList]
    for k in dic.keys():
        times[architecturesList.index(k.split("_")[0])]+=[dic[k]] ##the dictionary should be sorted...
    maxLen=0
    for t in times:
        if maxLen<len(t):
            maxLen=len(t)

    ind = np.arange(0,maxLen)
    for idx,e in enumerate(times):
        ax.plot(np.array(ind,dtype=np.float),e,label=architecturesList[idx])
    ax.set_xticks(ind)
    ax.set_xticklabels(xticks)
    ax.set_xlabel("number of module",fontsize="xx-large")
    ax.set_ylabel("Time",fontsize="xx-large")
    ax.tick_params(labelsize="xx-large")
    plt.legend()
    plt.show()
    figpath=os.path.join(sys.path[0],figname)
    fig.savefig(figpath)
    plt.close(fig)

def timeSinglePlot(dic,figname,xticks):
    """
        Plot histogram of time took by an experiment ran on one architectures
    :param dic: A dictionary with experiment names as: architecture_experiment.
    :param figname: where to store the figure
    :param xticks: list of str, label sticks for the x axis
    :return:
    """
    fig,ax=plt.subplots(figsize=(19.2,10.8), dpi=100)

    architecturesList=[]
    for k in dic.keys(): #init of the dic
        if not k.split("_")[0] in architecturesList:
            architecturesList += [k.split("_")[0]]
    try:
        assert len(architecturesList)==1
    except:
        print("more than one architecture...")
        raise
    times=[]
    for k in dic.keys():
        times+=[dic[k]]
    figpath=os.path.join(sys.path[0],figname)
    df = pandas.DataFrame(times)
    df.to_csv(figpath.split(".")[0]+".csv")

    ind = np.arange(0,len(times))
    ax.plot(np.array(ind,dtype=np.float),np.array(times,dtype=np.float),label=architecturesList[0])
    ax.set_xticks(ind)
    ax.set_xticklabels(xticks)
    ax.set_xlabel("number of module",fontsize="xx-large")
    ax.set_ylabel("Time",fontsize="xx-large")
    ax.tick_params(labelsize="xx-large")
    plt.legend()
    plt.show()

    fig.savefig(figpath)
    plt.close(fig)

def compareEvolutionPlot(time, experiment_name, nameDic, Xs, wishToDisp=[""], displaySeparate=False, displayOther=True, experimentNames=None):
    """
        Function to plot the evolution against the time of the concentration of some species.
        :param time: 1d-array, containing the time.
        :param experiment_name: result path will be of the form: resultPath=experiment_name+"/resultData/"+"separate"+"/"
                                                            and  resultPath=experiment_name+"/resultData/"
        :param nameDic: dictionnary with the name of species as key, and position in the array as values.
        :param X: 3d-array, axis0: time, axis1: species as described by namedic, axis3: experiments
        :param wishToDisp: name of species one wish to disp individually
        :param displaySeparate: if we want to Display the individual species in wishToDisp
        :param displayOther: if we want to display all intermediate and all main species in a separate fashion.
        :param experimentNames: name for each experiment
    :return:
        """
    experiment_name=os.path.join(sys.path[0],experiment_name)
    resultPath=os.path.join(experiment_name,"resultData/separate/")
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    fulltxt=nameDic.keys()
    selected={}
    for t in fulltxt:
        selected[t]=0
    for w in wishToDisp:
        selected[w]=1

    if experimentNames == None:
        experimentNames = range(Xs.shape[2])

    c=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    if(displaySeparate):
        figs=[plt.figure(figsize=(19.2,10.8), dpi=100) for _ in wishToDisp]
        figIdx=0
        for idx,k in enumerate(nameDic.keys()):
            if(selected[k]):
                ax=figs[figIdx].add_subplot(1,1,1)
                #ax.set_yscale('log')
                for e in range(Xs.shape[2]):
                    ax.plot(time,Xs[:,nameDic[k],e],c=c[e],label=experimentNames[e])
                ax.set_title(k)
                ax.set_xlabel("time",fontsize="xx-large")
                ax.set_ylabel("concentration",fontsize="xx-large")
                ax.tick_params(labelsize="xx-large")
                ax.legend()
                figIdx+=1
        for idx,fig in enumerate(figs):
            fig.savefig(os.path.join(resultPath,str(wishToDisp[idx])+".png"))
        for fig in figs:
            plt.close(fig)
    if(displayOther):
        try:
            assert Xs.shape[0]==len(list(nameDic.keys()))
        except:
            raise Exception("please, provide an array with the concentration for all species at all time step!")
        txt_inter={}
        for t in fulltxt:
            txt_inter[t]=0#we wish to plot only intermediate species: that is species of name larger than 3
        for t in fulltxt:
            if(len(t)>=3):
                txt_inter[t]=1
        resultPath=os.path.join(experiment_name,"resultData/")
        figInter = plt.figure(figsize=(19.2,10.8), dpi=100)
        ax = figInter.add_sublpot(1,1,1)
        for idx,k in enumerate(nameDic.keys()):
            if(txt_inter[k]):
                for e in range(Xs.shape[2]):
                    ax.plot(time,Xs[:,nameDic[k],e],c=c[e],label=experimentNames[e])
                ax.set_xlabel("time",fontsize="xx-large")
                ax.set_ylabel("concentration",fontsize="xx-large")
                ax.tick_params(labelsize="xx-large")
                ax.set_title(k)
        ax.legend()
        figInter.savefig(os.path.join(resultPath,"intermediateSpecies0.png"))
        plt.close(figInter)
        txt_inter={}
        for t in fulltxt:
            txt_inter[t]=0#we wish to plot only intermediate species: that is species of name larger than 3
        for t in fulltxt:
            if(len(t)<3):
                txt_inter[t]=1
        figMain = plt.figure(figsize=(19.2,10.8), dpi=100)
        ax = figMain.add_subplot(1,1,1)
        for idx,k in enumerate(nameDic.keys()):
            if(txt_inter[k]):
                for e in range(Xs.shape[2]):
                    ax.plot(time,Xs[:,nameDic[k],e],c=c[e],label=experimentNames[e])
                ax.set_title(k)
                ax.set_xlabel("time",fontsize="xx-large")
                ax.set_ylabel("concentration",fontsize="xx-large")
                ax.tick_params(labelsize="xx-large")
        ax.legend()
        figMain.savefig(os.path.join(resultPath,"mainSpecies0.png"))
        plt.close(figMain)