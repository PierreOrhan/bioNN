import matplotlib.pyplot as plt
import numpy as np




def f(k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kd,TA,TI,E0,A,I):
    k1M = k1/(k1n+k2)
    k5M = k5/(k5n+k6)
    k3M = k3/(k3n+k4)
    Cactiv = k2*k1M*TA*E0
    CInhib = k6*k5M*k4*k3M*TI*E0*E0
    Kactiv = k1M*TA
    Kinhib = k3M*TI
    print("Cactiv value is :"+str(Cactiv))
    print("Cinhib value is :"+str(CInhib))
    print("kd is :"+str(kd))

    cp =kd*(1 + Kactiv*(A+100*10**(-6)) + Kinhib*(I+100*10**(-6)))
    return Cactiv*A/(cp + CInhib*I/cp), cp ,  CInhib*I


k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kd=(26*10**12,3,17,26*10**12,3,17,26*10**12,3,17,0.32)


A=np.array([10**-8,10**-6,10**-5,10**-4])
I = np.logspace(-13,-4,100)
TA = 10**(-6)
TI = 10**(-6)
E0 = 10**(-5)

k1M = k1/(k1n+k2)
k5M = k5/(k5n+k6)
k3M = k3/(k3n+k4)
Cactiv = k2*k1M*TA*E0
CInhib = k6*k5M*k4*k3M*TI*E0*E0
Kactiv = k1M*TA
Kinhib = k3M*TI
print("Cactiv value is :"+str(Cactiv))
print("Cinhib value is :"+str(CInhib))
print("Kactiv value is :"+str(Kactiv))
print("Kinhib value is :"+str(Kinhib))

for a in A:
    Out, cp , K = f(k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kd,TA,TI,E0,a,I)
    plt.plot(K/np.power(cp,2),Out,label=str(a))
plt.xscale('log')
plt.legend()
plt.xlabel("inhibitors/(cp**2)",fontsize="xx-large")
plt.ylabel("output",fontsize="xx-large")
plt.tick_params(labelsize="xx-large")
plt.show()

# plt.plot(I,K/np.power(cp,2))
# plt.xscale('log')
# plt.xlabel("Xinhib",fontsize="xx-large")
# plt.ylabel("inhibitors/(cp**2)",fontsize="xx-large")
# plt.tick_params(labelsize="xx-large")
# plt.show()

A=np.logspace(-8,-6,100)
I = np.array([10**-8,10**-6,10**-5,10**-4])
for i in I:
    Out, cp , K = f(k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kd,TA,TI,E0,A,i)
    plt.plot(A,Out,label="Xinhib="+str(i))
#plt.xscale('log')
plt.legend()
plt.xlabel("activator",fontsize="xx-large")
plt.ylabel("output",fontsize="xx-large")
plt.tick_params(labelsize="xx-large")
plt.show()

# plt.plot(I,K/np.power(cp,2))
# plt.xscale('log')
# plt.xlabel("Xinhib",fontsize="xx-large")
# plt.ylabel("inhibitors/(cp**2)",fontsize="xx-large")
# plt.tick_params(labelsize="xx-large")
# plt.show()