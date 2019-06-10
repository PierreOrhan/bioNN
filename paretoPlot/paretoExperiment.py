"""
    Script file to run an experiment
"""

from paretoPlot.paretoConfigUtils import _getConfig
from paretoPlot.paretoTrainTools import train,paretoPlot

def run():
    # 0: large plot
    # 1: small plot
    Batches,colors,colors2,colors3,Sparsity,NbUnits,Layers = _getConfig(0)
    epochs = 1
    repeat = 1
    use_bias = False
    gpuIdx = None
    listOfRescale = [1,2,4]
    Initial_Result_Path= "resultPareto/"
    train(listOfRescale,Batches,Sparsity,NbUnits,Initial_Result_Path,epochs,repeat,use_bias)
    paretoPlot(listOfRescale,Batches,colors,colors3,Initial_Result_Path)

if __name__ == '__main__':
    run()