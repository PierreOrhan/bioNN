from simulOfBioNN.simulNN.tensorflowTraining import train
from simulOfBioNN.simulNN.launcher import load
from simulOfBioNN.parseUtils.parserForLassie import convertToLassieInput
from simulOfBioNN.parseUtils.parser import read_file,sparseParser,generateNeuralNetwork
from simulOfBioNN.odeUtils.utils import obtainSpeciesArray
from simulOfBioNN.odeUtils.systemEquation import setToUnits
from simulOfBioNN.plotUtils.adaptivePlotUtils import plotEvolution

import numpy as np
import os,sys
import subprocess
import pandas


if __name__ == '__main__':
    use_tensorflow = False
    #params
    directory_name="weightDir"
    layerInit=10**(-8)
    enzymeInit=10**(-6)
    leak=10**(-13)
    directory_for_network = os.path.join(directory_name,"Simul")
    args = ("","ONE")
    if(use_tensorflow):
        #tensorflow training
        weightDir,acc,x_test,y_test,nnAnswer=train()
        #conversion of output from tensorflow:
        y_test2 = np.transpose(y_test)[0]
        if(np.max(x_test) > 1):
            x_test = (x_test/255)*10**(-6) + 10**(-8)
        else:
            x_test = (x_test)*10**(-6) + 10**(-8)
        x_test_save = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
        inputsArray=x_test_save[:100]
        # y= y_test2[:100]
        # resultArray = nnAnswer[:100]

        #load mask from tensorflow's weights:
        _,masks = load(directory_name, directory_for_network)
    else:
        masks = np.array([[[1,-1]]])
        generateNeuralNetwork(directory_for_network,masks)

        if(args[1] == "FULL"):
            X1=np.arange(10**(-7),10**(-5),10**(-7))
            X1=np.concatenate((X1,np.arange( 10 ** (-5), 0.8*10 ** (-4), 10 ** (-6))))
            X2=np.arange( 10 ** (-8), 10 ** (-6), 10 ** (-8))
        elif(args[1]=="ONE"):
            X1 = np.array([10**(-5)])
            X2 = np.array([10**(-7)])
        else:
            X1=np.array([10**(-6),10**(-2)]) #,10**(-5),10**(-4)
            X2=np.array([10**(-7),10**(-5)])
        x_test=[]
        for x1 in X1:
            for x2 in X2:
                assert int(masks.shape[-1]/2) == masks.shape[-1]/2
                input = []
                for _ in range(int(masks.shape[-1]/2)): #mask.shape[-1] is exactly the number of output node here
                    input +=[x1,x2]
                x_test+=[input]
        x_test_save = np.array(x_test)
        inputsArray = np.array([x_test_save[0]])

    initialization_dic={}
    outputList=[]
    for layer in range(1,len(masks)):
        for node in range(masks[layer].shape[0]):
            initialization_dic["X_"+str(layer)+"_"+str(node)] = layerInit
            if(layer == len(masks)-1):
                outputList+=["X_"+str(layer)+"_"+str(node)]
    initialization_dic["E"] = enzymeInit
    initialization_dic["E2"] = enzymeInit

    parsedEquation,constants,nameDic=read_file(directory_for_network+"/equations.txt",directory_for_network+"/constants.txt")
    KarrayA,stochio,maskA,maskComplementary = sparseParser(parsedEquation,constants)
    KarrayA,T0,C0,kDic=setToUnits(constants,KarrayA,stochio)
    print("Initialisation constant: time:"+str(T0)+" concentration:"+str(C0))
    speciesArray = obtainSpeciesArray(inputsArray,nameDic,leak,initialization_dic,C0)
    time=np.array([0,10,100,1000])
    time=np.arange(0,1000,0.1)
    coLeak = leak/C0
    directory_for_lassie = os.path.join(directory_for_network,"LassieInput")

    convertToLassieInput(directory_for_lassie,parsedEquation,constants,nameDic,time,speciesArray[0])
    path_to_lassie_ex = '../../../LASSIE2/lassie'
    directory_for_lassie_outputdir = str(directory_for_lassie)
    command=[os.path.join(sys.path[0],path_to_lassie_ex), directory_for_lassie, directory_for_lassie_outputdir]
    print("launching "+command[0]+" "+command[1]+" "+command[2])
    subprocess.run(command,check=True)

    solution_path=os.path.join(sys.path[0], os.path.join(directory_for_lassie_outputdir, "output/Solution"))
    print("opening solution: "+solution_path)
    solution = pandas.read_csv(solution_path, sep='\t',header=None)
    sol = solution.values[:,1:-1] # In the solution, the first value is the time...
    plotEvolution(time, directory_for_lassie_outputdir, nameDic, sol, displayOther=True)
