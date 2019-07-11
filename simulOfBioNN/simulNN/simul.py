from simulOfBioNN.simulNN.tensorflowTraining import train
from simulOfBioNN.simulNN.launcher import launch
import numpy as np
import pandas
import os,sys
from tqdm import tqdm

if __name__ == '__main__':
    savePath=os.path.join(sys.path[0],"training")
    weightDir,acc,x_test,y_test,nnAnswer=train(savePath)
    #save the training data set and answer
    df=pandas.DataFrame(y_test)
    df.to_csv("ytest.csv")
    df=pandas.DataFrame(nnAnswer)
    df.to_csv("nnAnswer.csv")
    x_test_save = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
    df=pandas.DataFrame(x_test_save)
    df.to_csv("x_test.csv")
    y_test2 = np.transpose(y_test)[0]

    # We now need to rescale in a logarithm space our inputs
        # We can either: use a logarithmic range from 10**(-8) (corresponding to 0, to 10**(-4) corresponding to 1)
        # OR discritize even more the input values.
    # We begin with the logarithmic range:
    if(np.max(x_test)<=1):
        x_test = np.array(x_test*255,dtype=np.int)
    else:
        x_test = np.array(x_test,dtype=np.int)
    unique = list(np.sort(np.unique(x_test)))
    myLogSpace = np.logspace(-8,-4,len(unique))
    inputsConcentrations = myLogSpace[x_test]
    inputsConcentrations = np.reshape(inputsConcentrations,(inputsConcentrations.shape[0],(inputsConcentrations.shape[1]*inputsConcentrations.shape[2])))

    nbrExample = 100
    useEndo=False
    layerInit = 10**(-13) #initial concentation value for species in layers
    initValue = 10**(-13) #initial concentration value for all species.
    enzymeInit = 5*10**(-7)
    endoInit = 10**(-5) #only used if useEndo == True
    activInit =  10**(-4)
    inhibInit =  10**(-4)



    if useEndo:
        launch(inputsConcentrations[:nbrExample],y_test2[:nbrExample],nnAnswer[:nbrExample],weightDir,simulateMethod = "ODE",layerInit=layerInit,
               enzymeInit=enzymeInit,inhibInit=inhibInit,activInit=activInit,endoInit=endoInit,chemicalModel="templateModel")
    else:
        launch(inputsConcentrations[:nbrExample],y_test2[:nbrExample],nnAnswer[:nbrExample],weightDir,simulateMethod = "fixPoint",layerInit=layerInit,
               enzymeInit=enzymeInit,inhibInit=inhibInit,activInit=activInit,endoInit=None,chemicalModel="templateModel")

