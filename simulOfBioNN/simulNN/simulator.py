"""
    Tools to run simulations in parallel on the CPU, for sparse matrix.
    Here we proceed to the simulation of a graph, against some inputs.
    Equations are solved in parallel on the CPU

"""

import numpy as np
from simulOfBioNN.parseUtils.parser import read_file, sparseParser, parse
from simulOfBioNN.parseUtils.parserForLassie import convertToLassieInput
from simulOfBioNN.odeUtils.systemEquation import setToUnits,fPythonSparse
from simulOfBioNN.odeUtils.utils import saveAttribute,findRightNumberProcessus,obtainSpeciesArray,obtainCopyArgs,obtainOutputArray,obtainCopyArgsLassie,rescaleInputConcentration

from scipy.integrate import odeint

import pandas
from time import time as tm
import multiprocessing,subprocess
import os,sys


def scipyOdeSolverForMultiProcess(X):
    """
        Solve mutiple time the ODE integration using odeint from scipy.
    :param X: tuple containing speciesArray,time,df,functionArgs,outputArgs
            speciesArray: 2d-array with each row as the initialization for one run of the integration
            time: time step for output
            df: function used to compute the derivative
            functionArgs: additional args to give to the function
                    KarrayA: 3d-matrix with the reaction constant and stoichio coefficient
                    maskA: mask
                    maskComplementary: 1-mask
                    coLeak: the small leak used for stability, added at the end of each computation of the derivative for every species
            outputDic: dictionnary, args used for different output mode.
                        Should contain the following mandatory field:
                            "mode": a list indicating the different mode

                        Possible modes:
                            "verbose": display starting and finishing message
                                        outputDic should then contain the field:
                                            "idx": idx of the subprocess
                            "time":
                                    saving of time.
                            "ouputEqui":
                                    save of the last value reached by the integrator
                                        outputdDic should then contain the field:
                                            "outputDic": name of species to record
                                            "nameDic": link name to position
                                            "output": array to store the results
                            "outputPlot":
                                    save all value reached on time steps.
                                        outputdDic should then contain the field:
                                            "outputDic": name of species to record
                                            "nameDic": link name to position
                                            "outputPlot": array to store the results
    :return:Depending if the mode is present in outputDic["mode"]:
            output: for each run (column)m, for each species in outputDic (row), the final value reached.
            outputPlot: for each each run (column)m, for each species in outputDic (row), all reached values.
            avgTime: avgTime for the asked run
            The position is the same as the position of the key in outputDic["mode"]
    """

    speciesArray,time,df,functionArgs,outputDic = X

    if "display" in outputDic["mode"]:
        print("starting "+str(outputDic["idx"]))
    if "outputEqui" in outputDic["mode"]:
        output = outputDic["output"]
        nameDic = outputDic["nameDic"]
    if "outputPlot" in outputDic["mode"]:
        outputPlot = outputDic["outputPlot"]
        nameDic = outputDic["nameDic"]
    avgTime = 0
    for idx,species in enumerate(speciesArray):
        t0=tm()
        X2,_=odeint(df,species,time,args=functionArgs,full_output=True,rtol=1e-6,atol=1e-12)
        timeTook = tm()-t0
        avgTime += timeTook
        if "verbose" in outputDic["mode"]:
            print(str(idx)+" on "+str(len(speciesArray))+" for "+str(outputDic["idx"])+" in "+str(timeTook))
        if "outputEqui" in outputDic["mode"] or "outputPlot" in outputDic["mode"]:
            for idxOut,k in enumerate(outputDic["outputList"]):
                if "outputEqui" in outputDic["mode"]:
                    output[idxOut,idx]=X2[-1,nameDic[k]]
                if "outputPlot" in outputDic["mode"]:
                    outputPlot[idxOut,idx,:]=X2[:,nameDic[k]]
    results=[0 for _ in range(len(outputDic["mode"]))]
    if("outputEqui" in outputDic["mode"]):
        results[outputDic["mode"].index("outputEqui")] = output
    if("outputPlot" in outputDic["mode"]):
        results[outputDic["mode"].index("outputPlot")] = outputPlot
    if "time" in outputDic["mode"]:
        results[outputDic["mode"].index("time")] = avgTime/len(speciesArray)
    return tuple(results)

def lassieGPUsolverMultiProcess(X):
    """
        Multiple lassie instance launch on parallel processus.
    :param X: speciesArray,time, directory_for_network,parsedEquation,constants,nameDic,outputDic
    :return:
    """
    speciesArray,time, directory_for_network,parsedEquation,constants,coLeak,nameDic,path_to_lassie_ex,outputDic = X
    if "display" in outputDic["mode"]:
        print("starting "+str(outputDic["idx"]))
    if "outputEqui" in outputDic["mode"]:
        output = outputDic["output"]
        nameDic = outputDic["nameDic"]
    if "outputPlot" in outputDic["mode"]:
        outputPlot = outputDic["outputPlot"]
        nameDic = outputDic["nameDic"]
    avgTime = 0
    for idx,species in enumerate(speciesArray):
        t0=tm()
        #We create the directory of input for LASSIE:
        #TODO : change for a better communication.
        directory_for_lassie = os.path.join(os.path.join(directory_for_network,"LassieInput"),str(idx))
        convertToLassieInput(directory_for_lassie,parsedEquation,constants,nameDic,time,species,leak=coLeak)
        directory_for_lassie_outputdir = directory_for_lassie
        command=[os.path.join(sys.path[0],path_to_lassie_ex), directory_for_lassie, directory_for_lassie_outputdir]
        print("launching "+command[0]+" "+command[1]+" "+command[2])
        subprocess.run(command,check=True)

        solution_path=os.path.join(sys.path[0], os.path.join(directory_for_lassie_outputdir, "output/Solution"))
        print("opening solution: "+solution_path)
        solution = pandas.read_csv(solution_path, sep='\t',header=None)
        sol = solution.values[:,1:-1] # In the solution, the first value is the time...

        timeTook = tm()-t0
        avgTime += timeTook
        if "verbose" in outputDic["mode"]:
            print(str(idx)+" on "+str(len(speciesArray))+" for "+str(outputDic["idx"])+" in "+str(timeTook))
        if "outputEqui" in outputDic["mode"] or "outputPlot" in outputDic["mode"]:
            for idxOut,k in enumerate(outputDic["outputList"]):
                if "outputEqui" in outputDic["mode"]:
                    output[idxOut,idx]=sol[-1,nameDic[k]]
                if "outputPlot" in outputDic["mode"]:
                    outputPlot[idxOut,idx,:]=sol[:,nameDic[k]]
    results=[0 for _ in range(len(outputDic["mode"]))]
    if("outputEqui" in outputDic["mode"]):
        results[outputDic["mode"].index("outputEqui")] = output
    if("outputPlot" in outputDic["mode"]):
        results[outputDic["mode"].index("outputPlot")] = outputPlot
    if "time" in outputDic["mode"]:
        results[outputDic["mode"].index("time")] = avgTime/len(speciesArray)

    return results

def executeSimulation(funcForSolver, directory_for_network, inputsArray, initializationDic=None, outputList=None,
                      leak=10 ** (-13), endTime=1000, sparse=False, modes=["verbose","time", "outputPlot", "outputEqui"],
                      timeStep=0.1, initValue=10**(-13)):
    """
        Execute the simulation of the system saved under the directory_for_network directory.
        InputsArray contain the values for the input species.
    :param directory_for_network: directory path, where the files equations.txt and constants.txt may be found.
    :param inputsArray: The test concentrations, a t * n array where t is the number of test and n the number of node in the first layer.
    :param initializationDic: can contain initialization values for some species. If none, or the species don't appear in its key, then its value is set at leak.
    :param outputList: list or string, species we would like to see as outputs, if default (None), then will find the species of the last layer.
                                      if string and value is "nameDic" or "all", we will give all species taking part in the reaction (usefull for debug)
    :param leak: float, small leak to add at each time step at the concentration of all species
    :param endTime: final time
    :param sparse: if sparse
    :param modes: modes for outputs
    :param timeStep: float, value of time steps to use in integration
    :param initValue: initial concentration value to give to all species
    :return:
            A result tuple depending on the modes.
    """

    parsedEquation,constants,nameDic=read_file(directory_for_network + "/equations.txt", directory_for_network + "/constants.txt")
    if sparse:
        KarrayA,stochio,maskA,maskComplementary = sparseParser(parsedEquation,constants)
    else:
        KarrayA,stochio,maskA,maskComplementary = parse(parsedEquation,constants)
    KarrayA,T0,C0,constants=setToUnits(constants,KarrayA,stochio)
    print("Initialisation constant: time:"+str(T0)+" concentration:"+str(C0))

    speciesArray = obtainSpeciesArray(inputsArray,nameDic,initValue,initializationDic,C0)
    speciesArray,rescaleFactor = rescaleInputConcentration(speciesArray,nameDic=nameDic)

    time=np.arange(0,endTime,timeStep)
    derivativeLeak = leak

    ##SAVE EXPERIMENT PARAMETERS:
    attributesDic = {}
    attributesDic["rescaleFactor"] = rescaleFactor
    attributesDic["leak"] = leak
    attributesDic["T0"] = T0
    attributesDic["C0"] = C0
    attributesDic["endTime"] = endTime
    attributesDic["time_step"] = timeStep
    for k in initializationDic.keys():
        attributesDic[k] = speciesArray[0,nameDic[k]]
    for idx,cste in enumerate(constants):
        attributesDic["Constant for reaction "+str(idx)] = cste
    experiment_path=saveAttribute(directory_for_network, attributesDic)

    shapeP=speciesArray.shape[0]

    #let us assign the right number of task in each process
    num_workers = multiprocessing.cpu_count()-1
    idxList = findRightNumberProcessus(shapeP,num_workers)

    #let us find the species of the last layer in case:
    if outputList is None:
        outputList = obtainOutputArray(nameDic)
    elif type(outputList)==str:
        if outputList=="nameDic" or outputList=="all":
            outputList=list(nameDic.keys())
        else:
            raise Exception("asked outputList is not taken into account.")
    t=tm()
    print("=======================Starting simulation===================")
    if(hasattr(funcForSolver,"__call__")):
        copyArgs = obtainCopyArgs(modes,idxList,outputList,time,funcForSolver,speciesArray,KarrayA,stochio,maskA,maskComplementary,derivativeLeak,nameDic)
        with multiprocessing.get_context("spawn").Pool(processes= len(idxList[:-1])) as pool:
            myoutputs = pool.map(scipyOdeSolverForMultiProcess, copyArgs)
        pool.close()
        pool.join()
    else:
        assert type(funcForSolver)==str
        copyArgs = obtainCopyArgsLassie(modes,idxList,outputList,time,directory_for_network,parsedEquation,constants,derivativeLeak,nameDic,speciesArray,funcForSolver)
        with multiprocessing.get_context("spawn").Pool(processes= len(idxList[:-1])) as pool:
            myoutputs = pool.map(lassieGPUsolverMultiProcess, copyArgs)
        pool.close()
        pool.join()
    print("Finished computing, closing pool")
    timeResults={}
    timeResults[directory_for_network + "_wholeRun"]= tm() - t

    if("outputEqui" in modes):
        outputArray=np.zeros((len(outputList), shapeP))
    if("outputPlot" in modes):
        outputArrayPlot=np.zeros((len(outputList), shapeP, time.shape[0]))
    times = []
    for idx,m in enumerate(myoutputs):
        if("outputEqui" in modes):
            try:
                outputArray[:,idxList[idx]:idxList[idx+1]] = m[modes.index("outputEqui")]
            except:
                raise Exception("error")
        if("outputPlot" in modes):
            outputArrayPlot[:,idxList[idx]:idxList[idx+1]] = m[modes.index("outputPlot")]
        if("time" in modes):
            times += [m[modes.index("time")]]
    if("time" in modes):
        timeResults[directory_for_network + "_singleRunAvg"] = np.sum(times) / len(times)

    # Let us save our result:
    savedFiles = ["false_result.csv","output_equilibrium.csv","output_full.csv"]
    for k in nameDic.keys():
        savedFiles += [k+".csv"]
    for p in savedFiles:
        if(os._exists(os.path.join(experiment_path, p))):
            print("Allready exists: renaming older")
            os.rename(os.path.join(experiment_path,p),os.path.join(experiment_path,p.split(".")[0]+"Old."+p.split(".")[1]))
    if("outputEqui" in modes):
        df=pandas.DataFrame(outputArray)
        df.to_csv(os.path.join(experiment_path, "output_equilibrium.csv"))
    elif("outputPlot" in modes):
        assert len(outputArrayPlot == len(outputList))
        for idx,species in enumerate(outputList):
            df=pandas.DataFrame(outputArrayPlot[idx])
            df.to_csv(os.path.join(experiment_path, "output_full_"+str(species)+".csv"))

    results=[0 for _ in range(len(modes))]
    if("outputEqui" in modes):
        results[modes.index("outputEqui")]= outputArray
    if("outputPlot" in modes):
        results[modes.index("outputPlot")]= outputArrayPlot
    if "time" in modes:
        results[modes.index("time")]=timeResults

    if("outputPlot" in modes): #sometimes we need the nameDic
        results+=[nameDic]
    return tuple(results)



