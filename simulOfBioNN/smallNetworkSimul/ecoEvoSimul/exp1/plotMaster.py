import numpy as np
import pandas
from simulOfBioNN.plotUtils.adaptivePlotUtils import colorDiagram
import matplotlib.pyplot as plt

dfoutput=pandas.read_csv("equilibriumTest.csv")
output=dfoutput.values[:,1:]
dfA2onA1=pandas.read_csv("a2on1Test.csv")
A2onA1=np.reshape(dfA2onA1.values[:,1:],(dfA2onA1.values[:,1:].shape[0]))
dfC=pandas.read_csv("cTest.csv")
C=np.reshape(dfC.values[:,1:],(dfC.values[:,1:].shape[0]))

colorDiagram(A2onA1,C,output,"Ratio A2/A1","Concentration of cooperative species, arbitrary unit",'Ratio: Y2/Y1',"diagrammTest.png")
plt.show()