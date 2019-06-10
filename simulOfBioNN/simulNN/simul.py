from simulOfBioNN.simulNN.tensorflowTraining import train
from simulOfBioNN.simulNN.launcher import launch
import numpy as np
import pandas


if __name__ == '__main__':
    weightDir,acc,x_test,y_test,nnAnswer=train()
    #save the training data set and answer
    df=pandas.DataFrame(y_test)
    df.to_csv("ytest.csv")
    df=pandas.DataFrame(nnAnswer)
    df.to_csv("nnAnswer.csv")
    x_test_save = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
    df=pandas.DataFrame(x_test_save)
    df.to_csv("x_test.csv")
    y_test2 = np.transpose(y_test)[0]


    if(np.max(x_test) > 1):
        x_test = (x_test/255)*10**(-6) + 10**(-8)
    else:
        x_test = (x_test)*10**(-6) + 10**(-8)


    launch(x_test_save[:100],y_test2[:100],nnAnswer[:100],weightDir,layerInit = 10**(-8),enzymeInit = 10**(-6))

