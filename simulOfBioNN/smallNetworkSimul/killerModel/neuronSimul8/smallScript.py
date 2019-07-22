
import os,sys
import pandas
import numpy as np
from simulOfBioNN.plotUtils.adaptivePlotUtils import colorDiagram,neuronPlot,fitComparePlot
import matplotlib.pyplot as plt

experiment_path=os.path.join(sys.path[0],"")

df=pandas.read_csv(os.path.join(experiment_path, "neural_equilibrium.csv"))
output = df.values[:,1:]
df=pandas.read_csv(os.path.join(experiment_path, "neural_X1.csv"))
X1 = np.transpose(df.values[:,1:])[0]
df=pandas.read_csv(os.path.join(experiment_path, "neural_X2.csv"))
X2 = np.transpose(df.values[:,1:])[0]

C0 = 8.086075400626399e-07
#separate from bad values
X1=X1[:164]/C0
output=output[:164]
X2=X2[:164]/C0

colorDiagram(X1,X2,output,"Concentration of X1","Concentration of X2","Equilibrium concentration of the output",figname=os.path.join(experiment_path, "neuralDiagramm2.png"),equiPotential=False)
# neuronPlot(X1/(8.086075400626399e-07),X2/(8.086075400626399e-07),output,figname=os.path.join(experiment_path, "activation2.png"))
neuronPlot(X1,X2,output,figname=os.path.join(experiment_path, "activationX1.png"),figname2=os.path.join(experiment_path, "activationX2.png"))
# Plotting time:
colorDiagram(X1,X2,np.exp(output),"Concentration of X1","Concentration of X2","Equilibrium concentration of the output",figname=os.path.join(experiment_path, "neuralDiagramm2log.png"),equiPotential=False)
# neuronPlot(X1/(8.086075400626399e-07),X2/(8.086075400626399e-07),output,figname=os.path.join(experiment_path, "activation2.png"))
neuronPlot(X1,X2,np.exp(output),figname=os.path.join(experiment_path, "activationX1log.png"),figname2=os.path.join(experiment_path, "activationX2log.png"))
## Let us plot the frontier:
coords=[]
Xfrontier=[]
X2frontier=[]
for y in range(output.shape[1]):
    for x in range(output.shape[0]-1):
        if(output[x,y]<1 and 2<output[x+1,y]): #we have a discontinuity --> we are on the frontier
            coords+=[[X1[x],X2[y]]]
            Xfrontier+=[[X1[x],output[x+1,y]]]
            X2frontier+=[[X2[y], output[x + 1, y]]]
            break

from scipy.optimize import curve_fit

def frontierFit(x1,a,b):
    return a*np.sqrt(x1)-b

def computeR2(X1,X2,a,b):
    return 1-np.sum(np.power(X2 - frontierFit(X1,a,b),2))/np.sum(np.power(X2-np.average(X2),2))


p0=[0.1,0]
coords=np.array(coords)
popt,pcov = curve_fit(frontierFit,coords[:,0],coords[:,1],p0)
R2=computeR2(coords[:,0],coords[:,1],popt[0],popt[1])
print("solved frontier, the estimated covariance is : "+str(np.sqrt(np.diag(pcov)))+" and result:"+str(popt)+" and R2="+str(R2))

fig=plt.figure()
plt.plot(coords[:,0],coords[:,1],label="True courb")
plt.plot(coords[:,0],frontierFit(coords[:,0],popt[0],popt[1]),label="fitted courb with a="+str(round(popt[0],6))+" and b="+str(round(popt[1],6)))
plt.legend()
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("the frontier")
# plt.show()


def frontierFitForOut(x2,K):
    return K*x2**2/(1+x2)

Xfrontier=np.array(Xfrontier)
X2frontier=np.array(X2frontier)
K0=np.array([1])
popt2,pcov2 = curve_fit(frontierFitForOut,X2frontier[:,0],np.exp(X2frontier[:,1]),K0)
print("solved frontier, the estimated covariance is : "+str(np.sqrt(np.diag(pcov2)))+" and result:"+str(popt2))
#
# fig=plt.figure()
# plt.plot(X2frontier[:, 0], np.exp(X2frontier[:, 1]), label="frontier in X2")
# plt.plot(X2frontier[:,0],frontierFitForOut(X2frontier[:,0],popt2[0]),label="fitted courb with K="+str(round(popt[0],6)))
# plt.xlabel("X2")
# plt.ylabel("output")
# plt.legend()
# plt.title("the frontier")
# # plt.show()
#
# fig=plt.figure()
# plt.plot(Xfrontier[:, 0], np.exp(Xfrontier[:, 1]), label="frontier in X1")
# plt.plot(Xfrontier[:,0],frontierFitForOut(frontierFit(coords[:,0],popt[0],popt[1]),popt2[0]),label="fitted frontier in X1 with the 2 fit")
# plt.xlabel("X")
# plt.ylabel("output")
# plt.legend()
# plt.title("the frontier")
# plt.show()

X=[]
for x1 in X1:
    for x2 in X2:
        X+=[[x1,x2]]
X=np.array(X)

def booleanFrontier(x):
    x1 = x[:,0]
    x2 = x[:,1]
    Xres =  np.where((popt[0] * np.sqrt(x1) - popt[1] - x2>0 ),
                     1.0, 0.0)
    return Xres

outputBoolFront=booleanFrontier(X)
outputBoolFront2 = np.zeros(output.shape)
e=0
for idx,x in enumerate(X1):
    for idx2,x2 in enumerate(X2):
        outputBoolFront2[idx,idx2]=outputBoolFront[e]
        e=e+1

# colorDiagram(X1,X2,outputBoolFront2,"Concentration of X1","Concentration of X2","Equilibrium concentration of the output",figname=os.path.join(experiment_path, "testDiagFrontier.png"),equiPotential=False)
# neuronPlot(X1,X2,outputBoolFront2,figname=os.path.join(experiment_path, "testfrontierX1.png"),figname2=os.path.join(experiment_path, "testfrontierX2.png"))

def hypothetiquef(x,a1,K1,K2,K3,K4):
    x1 = x[:,0]
    x2 = x[:,1]
    Xres =  np.where(popt[0] * np.sqrt(x1) - popt[1] - x2>0,
                     np.log(a1 * np.power(x1,2) / (np.power(x1+K1,2))) + np.log(K3 * x2 + np.exp(-K2*x2/(x1 + K4))), 0)
    return Xres

a1 = 10
K2 = 1
K1 = 1
K3 = 1
K4 = 1

#goodResult= [7.13814690e+04 1.06181304e+02 7.74264755e+01 2.45383488e-18]

x0 = (a1,K1,K2 ,K3,K4)
output2 = np.reshape(output,X.shape[0])
popt3,pcov3 = curve_fit(hypothetiquef,X,output2,x0,bounds=(0,np.inf))
print("solved frontier, the estimated covariance is : "+str(np.sqrt(np.diag(pcov3)))+" and result:"+str(popt3))

fitOutput = np.zeros(output.shape)

for idx,x in enumerate(X1):
    test= np.zeros(X2.shape[0]) + x
    test = np.stack([test,X2],axis=1)
    fitOutput[idx,:]=hypothetiquef(test,popt3[0],popt3[1],popt3[2],popt3[3],popt3[4])

colorDiagram(X1,X2,output,"Concentration of X1","Concentration of X2","Equilibrium concentration of the output",figname=os.path.join(experiment_path, "testDiagramm2.png"),equiPotential=False)
neuronPlot(X1,X2,output,figname=os.path.join(experiment_path, "testActivationX1.png"),figname2=os.path.join(experiment_path, "testActivationX2.png"))
# neuronPlot(X1,X2,np.exp(output),figname=os.path.join(experiment_path, "testActivationX1exp.png"),figname2=os.path.join(experiment_path, "testActivationX2exp.png"))
courbs=[0,int(fitOutput.shape[1]/2),fitOutput.shape[1]-1,int(fitOutput.shape[1]/3),int(2*fitOutput.shape[1]/3)]
fitComparePlot(X1,X2,output,fitOutput,courbs,
               figname=os.path.join(experiment_path, "fitComparisonX1.png"),
               figname2=os.path.join(experiment_path, "fitComparisonX2.png"))

