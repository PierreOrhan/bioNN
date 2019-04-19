import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
    In this file we simulate the differential equation of our model.
"""

def fYj(EYA2j,EYAj,EYTj,Tj,Yj,E,Aj,K,j):
    """
    We compute the derivative of Yj with respect to concentration of other species in solution
    :param EYA2j: array, elements are concentration of EYA2ij
    :param EYAj: array elements are concentration of EYAij
    :param EYTj: array elements are concentration of EYTij
    :param Tj: concentration of Tj
    :param Yj: concentration of Yj
    :param E: concentration of E
    :param Aj: array, elements concentration of Aij
    :param K the dictionnary for constants, here we use k1,k2,k1n,k2n
    :return: compute element by element product and return the sum of the elements of the array, which is dYj/dt
    """
    k1=K["k1"][j]
    k1m=K["k1m"][j]
    k2n=K["k2n"][j]
    k5=K["k5"][j]
    k5m=K["k5m"][j]
    kd=K["kd"][j]
    return np.sum(k1m*EYAj-k1*Yj*E*Aj+2*k2n*EYA2j-k5*Tj*Yj*E+k5m*EYTj)-kd*Yj

#In the following we compute variation for
def fEYAj(EYAj,Aj,Yj,E,K,j):
    """
    Computes for each EYAij the derivative of its concentration
    :param EYAj:
    :param Aj:
    :param Yj:
    :param E:
    :return: an array with the updated concentration
    """
    k2=K["k2"][j]
    k1=K["k1"][j]
    k1m=K["k1m"][j]
    return k1*E*Yj*Aj-k1m*EYAj-k2*EYAj

def fEYA2j(EYA2j,YAj,E2,K,j):
    k2n=K["k2n"][j]
    k1n=K["k1n"][j]
    k1mn=K["k1mn"][j]
    return k1n*E2*YAj-k1mn*EYA2j-k2n*EYA2j

def fEMj(Mj,E,EMj,K,j):
    k4=K["k4"][j]
    k3=K["k3"][j]
    k3m=K["k3m"][j]
    return k3*E*Mj-k3m*EMj-k4*EMj

def fEMTj(MTj,E2,EMTj,K,j):
    k4n=K["k4n"][j]
    k3n=K["k3n"][j]
    k3mn=K["k3mn"][j]
    return k3n*E2*MTj-k3mn*EMTj-k4n*EMTj

def fEYTj(EYTj,Tj,Yj,E,K,j):
    k6=K["k6"][j]
    k5=K["k5"][j]
    k5m=K["k5m"][j]
    return k5*E*Tj*Yj-k5m*EYTj-k6*EYTj

def fMTj(MTj,E2,EMj,EMTj,K,j):
    k4=K["k4"][j]
    k3n=K["k3n"][j]
    k3nm=K["k3mn"][j]
    return k4*EMj-k3n*E2*MTj+k3nm*EMTj

def fYAj(EYAj,E2,YAj,EYA2j,K,j):
    k1n=K["k1n"][j]
    k1mn=K["k1mn"][j]
    k2=K["k2"][j]
    return k2*EYAj-k1n*E2*YAj+k1mn*EYA2j

def fTYj(EYTj,TYj,Tj,Ydj,K,j):
    k6=K["k6"][j]
    k7m=K["k7m"][j]
    k7=K["k7"][j]
    return k6*EYTj-k7*TYj+k7m*Tj*Ydj

def fYdj(TYj,Tj,Ydj,K,j):
    k7m=K["k7m"][j]
    k7=K["k7"][j]
    return k7*TYj-k7m*Tj*Ydj

def fTj(EYTj,Yj,E,Tj,EMTj,K,j):
    k5=K["k5"][j]
    k5m=K["k5m"][j]
    k4n=K["k4n"][j]
    return k4n*EMTj+k5m*EYTj-k5*E*Yj*Tj

def fAj(Aj,Yj,E,EYAj,EYA2j,K,j):
    k1=K["k1"][j]
    k1m=K["k1m"][j]
    k2n=K["k2n"][j]
    return -k1*Yj*E*Aj+k1m*EYAj+k2n*EYA2j

def fMj(Mj,E,EMj,EMTj,K,j):
    k3=K["k3"][j]
    k3m=K["k3m"][j]
    k4n=K["k4n"][j]
    return -k3*Mj*E+k3m*EMj+k4n*EMTj

def gE(Et,EYA,EM,EYT):
    """
        We use conservation of mass to obtain directly E
        EYA: array with all different EYAij
        EYA2: array with all different EYA2ij
    """
    return Et-np.sum(EYA)-np.sum(EM)-np.sum(EYT)
def gE2(E2t,EYA2,EMT):
    return E2t-np.sum(EYA2)-np.sum(EMT)
def gX(x,EM,EMT,EYA,EYA2):
    """
    update of one inputs using conservation of mass
    :param x:
    :param EM:
    :param EMT:
    :return:
    """
    return x-np.sum(EM)-np.sum(EMT)-np.sum(EYA)-np.sum(EYA2)


def update(E,E2,Yj,Aj,Mj,EYAj,YAj,EYA2j,EMj,EMTj,MTj,EYTj,TYj,Ydj,Tj,K,j):
    """
    Update all species only involved in the creation of Yj, as well as Yj.
    :var: The inputs are previous values of every species.
    :return:
    """
    nYj = fYj(EYA2j,EYAj,EYTj,Tj,Yj,E,Aj,K,j)
    nEYAj = fEYAj(EYAj,Aj,Yj,E,K,j)
    nYAj = fYAj(EYAj,E2,YAj,EYA2j,K,j)
    nEYA2j = fEYA2j(EYA2j,YAj,E2,K,j)
    nEMj = fEMj(Mj,E,EMj,K,j)
    nEMTj = fEMTj(MTj,E2,EMTj,K,j)
    nMTj = fMTj(MTj,E2,EMj,EMTj,K,j)
    nEYTj = fEYTj(EYTj,Tj,Yj,E,K,j)
    nTYj = fTYj(EYTj,TYj,Tj,Ydj,K,j)
    nYdj = fYdj(TYj,Tj,Ydj,K,j)
    nTj = fTj(EYTj,Yj,E,Tj,EMTj,K,j)
    return nYj,nEYAj,nYAj,nEYA2j,nEMj,nEMTj,nMTj,nEYTj,nTYj,nYdj,nTj

class bioLayer:
    def __init__(self,inputs,K,E,E2,Y,mask=np.array([[0,0,1,0],[1,-1,0,0],[-1,1,-1,1],[0,0,0,0]])):
        #   Creation of the network:
        self.mask=mask  #indicates how each input influences outputs, 1: activator, 0:not link, -1:deshinibitor
                        #The line corresponds to each output.
        #   Initialization:
        self.K=K
        self.maskActivator=np.where(mask>=0,mask,np.zeros(mask.shape))
        self.maskInhibitor=np.where(mask<=0,-1*mask,np.zeros(mask.shape))
        self.inputs=inputs
        self.A=inputs*self.maskActivator
        self.AT=np.copy(self.A)
        self.M=inputs*self.maskInhibitor
        self.MT=np.copy(self.M)
        self.ET=np.copy(E)
        self.E=E
        self.E2T=np.copy(E2)
        self.E2=E2
        self.Y=np.copy(Y)
        #compute shape:
        shapeActiv=[]
        shapeInhib=[]
        self.EYA, self.YA, self.EYA2 = [np.zeros(self.maskActivator.shape) for _ in range(3)]
        self.EM,self.EMT,self.MT,self.EYT,self.TY,self.Yd,self.T=[np.zeros(self.maskActivator.shape) for _ in range(7)]

    def f(self):
        nE = gE(self.ET,self.EYA,self.EM,self.EYT)
        nE2 = gE2(self.E2T,self.EYA2,self.EMT)
        nX=np.empty(self.inputs.shape)
        for idx,x in enumerate(self.inputs):
            # accumulate species where Ati has a role:
            EYA,EYA2,EM,EMT=[],[],[],[]
            for j,m in enumerate(self.mask[:,idx]):
                if(m==1):
                    EYA+=[self.EYA[j][idx]]
                    EYA2+=[self.EYA2[j][idx]]
                elif(m==-1):
                    EM+=[self.EM[j][idx]]
                    EMT+=[self.EMT[j][idx]]
            nX[idx]=gX(x,EM,EMT,EYA,EYA2)
        nInter=[]
        for j,X in enumerate(zip(self.Y,self.A,self.M,self.EYA,self.YA,self.EYA2,self.EM,self.EMT,self.MT,self.EYT,self.TY,self.Yd,self.T)):
            Yj,Aj,Mj,EYAj,YAj,EYA2j,EMj,EMTj,MTj,EYTj,TYj,Ydj,Tj=X
            nYj,nEYAj,nYAj,nEYA2j,nEMj,nEMTj,nMTj,nEYTj,nTYj,nYdj,nTj=update(self.E,self.E2,Yj,Aj,Mj,EYAj,YAj,EYA2j,EMj,EMTj,MTj,EYTj,TYj,Ydj,Tj,self.K,j)
            nInter+=[[nYj,nEYAj,nYAj,nEYA2j,nEMj,nEMTj,nMTj,nEYTj,nTYj,nYdj,nTj]]
        nInter=np.array(nInter)
        result=[np.stack(nInter[:,0])]
        result+=[nX*self.maskActivator]
        result+=[nX*self.maskInhibitor]
        result+=[np.stack(nInter[:,i]) for i in range(1,nInter.shape[1])]
        return result

    def step(self,dt):
        # we use runge kutta strategy:
        oldy=[np.copy(self.Y),np.copy(self.A),np.copy(self.M),np.copy(self.EYA),np.copy(self.YA),np.copy(self.EYA2),np.copy(self.EM),np.copy(self.EMT),np.copy(self.MT),np.copy(self.EYT),np.copy(self.TY),np.copy(self.Yd),np.copy(self.T)]
        #computes f(t,y)
        f1=self.f()
        #updates y=oldy+dt/2*f(t,y)
        self.Y,self.A,self.M,self.EYA,self.YA,self.EYA2,self.EM,self.EMT,self.MT,self.EYT,self.TY,self.Yd,self.T=[oldy[idx]+dt/2*f for idx,f in enumerate(f1)]
        f2=self.f()
        self.Y,self.A,self.M,self.EYA,self.YA,self.EYA2,self.EM,self.EMT,self.MT,self.EYT,self.TY,self.Yd,self.T=[oldy[idx]+dt/2*f for idx,f in enumerate(f2)]
        f3=self.f()
        self.Y,self.A,self.M,self.EYA,self.YA,self.EYA2,self.EM,self.EMT,self.MT,self.EYT,self.TY,self.Yd,self.T=[oldy[idx]+dt*f for idx,f in enumerate(f3)]
        f4=self.f()
        gradY=f1[0]+2*f2[0]+2*f3[0]+f4[0]
        self.Y,self.A,self.M,self.EYA,self.YA,self.EYA2,self.EM,self.EMT,self.MT,self.EYT,self.TY,self.Yd,self.T=[oldy[idx]+dt/6*(f+2*f2[idx]+2*f3[idx]+f4[idx]) for idx,f in enumerate(f1)]

        return self.Y,self.A,self.M,self.EYA,self.YA,self.EYA2,self.EM,self.EMT,self.MT,self.EYT,self.TY,self.Yd,self.T,gradY

    def codedConvergence(self):
        k2=K["k2"]
        kA=K["k1"]/(K["k1m"]+k2)
        k4=K["k4"]
        kM=K["k3"]/(K["k3m"]+k4)
        kd=K["kd"]
        sumA=np.sum(k2*kA*self.A,axis=1)
        sumAllM=np.sum(kM*np.max(self.M,axis=1))
        sumM=np.sum(k4*kM*self.M,axis=1)
        alpha=np.sum(kA*self.A,axis=1)
        Et=self.ET

        for j,y in enumerate(self.Y):
            if(sumA[j]-kd[j]/Et*(sumAllM+1)<=0):
                self.Y[j]=0
            else:
                self.Y[j]=1
        k6=K["k6"]
        beta=np.sum(np.max((k4/k6+1)*kM*self.M,axis=1))
        mu=np.sum(np.where(self.Y>0,Et/kd*(sumA-sumM)*alpha,np.zeros(self.Y.shape)))
        c=((beta+1)+((beta+1)**2+4*mu))/2
        for j,y in enumerate(self.Y):
            if(y>0):
                self.Y[j]=(sumA[j]-sumM[j])*Et/(kd[j]*c)
        return self.Y

def getK(inputsize):
    K={}
    #association constants:
    K["k1"]=[26*10**6 for _ in range(inputsize)]
    K["k1m"]=[3 for _ in range(inputsize)]
    K["k1n"]=[26*10**(-2) for _ in range(inputsize)]
    K["k1mn"]=[3 for _ in range(inputsize)]
    #enzymatic constants
    K["k2"]=[17 for _ in range(inputsize)] # reaction rate for the process of polymerization
    K["k2n"]=[3 for _ in range(inputsize)]

    #association constants
    K["k3"]=[26*10**(-2) for _ in range(inputsize)]
    K["k3m"]=[3 for _ in range(inputsize)]
    K["k3mn"]=[2 for _ in range(inputsize)]
    K["k3n"]=[26*10**(-2) for _ in range(inputsize)]
    #enzymatic constants
    K["k4"]=[17 for _ in range(inputsize)]
    K["k4n"]=[3 for _ in range(inputsize)]

    #association constants
    K["k5"]=[26*10**6 for _ in range(inputsize)]
    K["k5m"]=[3 for _ in range(inputsize)]
    #enzymatic constants
    K["k6"]=[17 for _ in range(inputsize)]

    #association constants
    K["k7"]=[0.03 for _ in range(inputsize)]
    K["k7m"]=[26*10**8 for _ in range(inputsize)]

    K["kd"]=[0.32 for _ in range(inputsize)]

    for k in K.keys():
        K[k]=np.array(K[k])
    return K



Y=np.array([10**(-15),10**(-15)])
inputs=np.array([10**(-9),10**(-8)])

K=getK(inputs.shape[0])

E=15
E2=15
mask=np.array([[1,0],[0,1]])


layer=bioLayer(inputs,K,E,E2,Y,mask)



step=0.01
T=np.arange(0,100,step)
X=[]

for t in tqdm(T):
    result=layer.step(step)
    X+=[result[:len(result)-1]]
    grad=result[-1]
    Y=result[0]
    if(np.max(np.abs(grad)*step-0.001*Y)<0):
        print("reached convergence")
        T=np.arange(0,t+step,step)
        break

txt=["Y","A","M","EYA","YA","EYA2","EM","EMT","MT","EYT","TY","Yd","T"]
selected={}
for t in txt:
    selected[t]=0
wishToDisp=["Y","EYA","EM","Yd","TY"]
for w in wishToDisp:
    selected[w]=1

figs=[plt.figure() for _ in wishToDisp]
c=["r","b","g"]

figIdx=0
for idx,t in enumerate(txt):
    if(selected[t]==1):
        R=[]
        for x in X:
            R+=[x[idx]]
        R=np.array(R)
        if(len(R.shape)<=2):
            ax=figs[figIdx].add_subplot(1,1,1)
            for j in range(R.shape[1]):
                ax.plot(T,R[:,j],c=c[j])
                ax.set_title(t)
        else:
            for i in range(R.shape[2]):
                ax2=figs[figIdx].add_subplot(1,R.shape[2],i+1)
                for j in range(R.shape[1]):
                    ax2.plot(T,R[:,j,i],c=c[j])
                    ax2.set_title(t)
        figIdx+=1
plt.show()

Y=np.array([10**(-15),10**(-15)])
layer=bioLayer(inputs,K,E,E2,Y,mask)
print(layer.codedConvergence())

print("wait")



# Y,A,M,EYA,YA,EYA2,EM,EMT,MT,EYT,TY,Yd,T






