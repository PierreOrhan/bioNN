import numpy as np
from scipy.optimize import minimize,root,brentq
import time


class pythonSolver():
    def __init__(self,masks,k1,k1n,k2,k3,k3n,k4,k5,k5n,k6,kdI,kdT,TA,TI,E0):
        self.k1 = [np.zeros(m.shape)+k1 for m in masks]
        self.k1n = [np.zeros(m.shape)+k1n for m in masks]
        self.k2 = [np.zeros(m.shape)+k2 for m in masks]
        self.k3 = [np.zeros(m.shape)+k3 for m in masks]
        self.k3n = [np.zeros(m.shape)+k3n for m in masks]
        self.k4 = [np.zeros(m.shape)+k4 for m in masks]
        self.k5 = [np.zeros(m.shape)+k5 for m in masks]
        self.k5n = [np.zeros(m.shape)+k5n for m in masks]
        self.k6 = [np.zeros(m.shape)+k6 for m in masks]
        self.kdT = [np.zeros(m.shape)+kdT for m in masks]
        self.kdI = [np.zeros(m.shape)+kdI for m in masks]

        self.TA0 = [np.zeros(m.shape)+TA for m in masks]
        self.TI0 = [np.zeros(m.shape)+TI for m in masks]
        self.k1M = [self.k1[l]/(self.k1n[l]+self.k2[l]) for l in range(len(self.k1))] #(matrix operations)
        self.k3M = [self.k3[l]/(self.k3n[l]+self.k4[l]) for l in range(len(self.k3))]
        self.k5M = [self.k5[l]/(self.k5n[l]+self.k6[l]) for l in range(len(self.k5))]
        self.Kactiv0 = [self.k1M[l]*self.TA0[l] for l in range(len(self.k1M))] # element to element product
        self.Kinhib0 = [self.k3M[l]*self.TI0[l] for l in range(len(self.k3M))]
        self.Cactiv0 = [self.k2[l]*self.k1M[l]*self.TA0[l]*E0 for l in range(len(self.k1M))]
        self.Cinhib0 =[self.k6[l]*self.k5M[l]*self.k4[l]*self.k3M[l]*self.TI0[l]*E0*E0 for l in range(len(self.k3M))]

        self.E0 = E0
        self.masks = masks

        self.cstList = [self.k1,self.k1n,self.k2,self.k3,self.k3n,self.k4,self.k5,self.k5n,self.k6,self.kdI,self.kdT,self.TA0,self.E0,
                        self.k1M,self.Cactiv0,self.Cinhib0,self.k5M,self.k3M]
        self.cstListName = ["self.k1","self.k1n","self.k2","self.k3","self.k3n","self.k4","self.k5","self.k5n","self.k6","self.kdI","self.kdT","self.TA0","self.E0",
                            "self.k1M","self.Cactiv","self.Cinhib","self.k5M","self.k3M"]


    def obtainBornSup(self,X0):
        return self._obtainBornSup(self.k6,self.kdT,self.kdI,self.Kactiv0,self.Kinhib0,self.Cactiv0,self.Cinhib0,self.E0,X0,self.masks)

    def cpEquilibriumFunc(self,cp,X0):
        return self._cpEquilibriumFunc(cp,self.k6,self.kdT,self.kdI,self.Kactiv0,self.Kinhib0,self.Cactiv0,self.Cinhib0,self.E0,X0,self.masks)

    def allEquilibriumFunc(self,cps,X0):
        return self._allEquilibriumFunc(cps,self.k6,self.kdT,self.kdI,self.Kactiv0,self.Kinhib0,self.Cactiv0,self.Cinhib0,self.E0,X0,self.masks,self.k1M,self.k3M)

    def _obtainBornSup(self,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks):
        """
            Given approximate only for the enzyme competition term (cp), we compute the next approximate using G.
            The real value for the competitions term verify cp = G(cp,initialConditions)
        :param cp: float, competition over tempalte.
        :param E0: float, initial concentration in the enzyme
        :param X0: nbrInputs array, contains initial value for the inputs
        :param masks: masks giving the network topology.
        :return:
        """
        max_cp = 1
        olderX = [np.zeros(m.shape[1]) for m in masks]
        for layeridx,layer in enumerate(masks):
            layerEq = np.zeros(layer.shape[1])
            if(layeridx==0):
                for inpIdx in range(layer.shape[1]):
                    #compute of Kactivs,Kinhibs;
                    Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                    Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                    #compute of "weights": sum of kactivs and kinhibs
                    w_inpIdx = np.sum(Kactivs)+np.sum(Kinhibs)
                    max_cp += w_inpIdx*X0[inpIdx]
                    # saving values
                    layerEq[inpIdx] = X0[inpIdx]
                olderX[layeridx] = layerEq
            else:
                for inpIdx in range(layer.shape[1]):
                    #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                    #Terms for the previous layers
                    CactivsOld = np.where(masks[layeridx-1][inpIdx,:]>0,Cactiv0[layeridx-1][inpIdx],0)
                    CinhibsOld = np.where(masks[layeridx-1][inpIdx,:]<0,Cinhib0[layeridx-1][inpIdx],0)
                    Inhib = np.sum(CinhibsOld*olderX[layeridx-1]/kdT[layeridx-1][inpIdx])
                    #computing of new equilibrium
                    x_eq = np.sum(CactivsOld*olderX[layeridx-1]/kdT[layeridx-1][inpIdx])
                    layerEq[inpIdx] = x_eq
                    #compute of Kactivs,Kinhibs, for the current layer:
                    Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                    Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                    #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                    firstComplex = np.sum(np.where(layer[:,inpIdx]>0,Kactivs*x_eq,np.where(layer[:,inpIdx]<0,Kinhibs*x_eq,0)))
                    #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                    Inhib2 = np.sum(CinhibsOld*olderX[layeridx-1]/(kdT[layeridx-1][inpIdx]*k6[layeridx-1][inpIdx]))
                    max_cp += firstComplex + Inhib2/E0*x_eq
                olderX[layeridx] = layerEq
        #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layers
        for outputsIdx in range(masks[-1].shape[0]):
            Cinhibs = np.where(masks[-1][outputsIdx,:]<0,Cinhib0[-1][outputsIdx],0)
            Cactivs = np.where(masks[-1][outputsIdx,:]>0,Cactiv0[-1][outputsIdx],0)
            x_eq = np.sum(Cactivs*olderX[-1]/(kdI[-1][outputsIdx]))
            Inhib2 = np.sum(Cinhibs*olderX[-1]/(kdT[-1][outputsIdx]*k6[-1][outputsIdx]))
            max_cp += Inhib2/E0*x_eq
        return max_cp
    def _cpEquilibriumFunc(self,cp,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks):
        """
            Given approximate only for the enzyme competition term (cp), we compute the next approximate using G.
            The real value for the competitions term verify cp = G(cp,initialConditions)
        :param cp: float, competition over tempalte.
        :param E0: float, initial concentration in the enzyme
        :param X0: nbrInputs array, contains initial value for the inputs
        :param masks: masks giving the network topology.
        :return:
        """
        new_cp = 1
        olderX = [np.zeros(m.shape[1]) for m in masks]
        for layeridx,layer in enumerate(masks):
            layerEq = np.zeros(layer.shape[1])
            if(layeridx==0):

                for inpIdx in range(layer.shape[1]):
                    if self.k1M[layeridx][inpIdx,inpIdx]*E0*self.TA0[layeridx][inpIdx,inpIdx]/cp == 0:
                        #case where there is no polynomial equation
                        new_input = X0[inpIdx]
                        new_cp += 0
                    else:
                        bOnA = X0[inpIdx]-self.TA0[layeridx][inpIdx,inpIdx]- cp/(self.k1M[layeridx][inpIdx,inpIdx]*E0)
                        # BE CAREFUL, HERE ONE SHOULD USE THE GREATER ROOT of the polynomial (only positive)
                        new_input = 1/2*(bOnA + (bOnA**2+4*cp*X0[inpIdx]/(self.k1M[layeridx][inpIdx,inpIdx]*E0))**0.5)
                        # update for fixed point:
                        new_cp += self.k1M[layeridx][inpIdx,inpIdx]*new_input*self.TA0[layeridx][inpIdx,inpIdx]/(1+self.k1M[layeridx][inpIdx,inpIdx]*new_input*E0/cp)
                    layerEq[inpIdx] = new_input
                olderX[layeridx] = layerEq

            else:
                for inpIdx in range(layer.shape[1]):
                    #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                    #Terms for the previous layers
                    CactivsOld = np.where(masks[layeridx-1][inpIdx,:]>0,Cactiv0[layeridx-1][inpIdx]/(1+self.k1M[layeridx-1][inpIdx]*olderX[layeridx-1]*self.E0/cp),0)
                    CinhibsOld = np.where(masks[layeridx-1][inpIdx,:]<0,Cinhib0[layeridx-1][inpIdx]/(1+self.k3M[layeridx-1][inpIdx]*olderX[layeridx-1]*self.E0/cp),0)
                    Inhib = np.sum(CinhibsOld*olderX[layeridx-1]/kdT[layeridx-1][inpIdx])
                    #computing of new equilibrium
                    x_eq = np.sum(CactivsOld*olderX[layeridx-1]/(kdI[layeridx-1][inpIdx]*cp+Inhib/cp))
                    # print(np.sum(CactivsOld*olderX[layeridx-1])/(kdI[layeridx-1][inpIdx,0]*cp+Inhib/cp))
                    # print(x_eq)
                    # assert np.sum(CactivsOld*olderX[layeridx-1])/(kdI[layeridx-1][inpIdx,0]*cp+Inhib/cp) == x_eq

                    layerEq[inpIdx] = x_eq


                    #compute of Kactivs,Kinhibs, for the current layer:
                    Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0)
                    Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                    #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                    firstComplex = np.sum(np.where(layer[:,inpIdx]>0,Kactivs*x_eq/(1+self.k1M[layeridx][:,inpIdx]*x_eq*self.E0/cp),np.where(layer[:,inpIdx]<0,Kinhibs*x_eq/(1+self.k3M[layeridx][:,inpIdx]*x_eq*self.E0/cp),0)))
                    if layeridx>1:
                    #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                        Inhib2 = np.sum(CinhibsOld*olderX[layeridx-1]/(kdT[layeridx-1][inpIdx]*k6[layeridx-1][inpIdx]))
                        new_cp += Inhib2/(E0*cp)*x_eq
                    new_cp += firstComplex

                olderX[layeridx] = layerEq

        #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layer
        for outputsIdx in range(masks[-1].shape[0]):
            Cactivs = np.where(masks[-1][outputsIdx,:]>0,Cactiv0[-1][outputsIdx]/(1+self.k1M[-1][outputsIdx]*olderX[-1]*self.E0/cp),0)
            Cinhibs = np.where(masks[-1][outputsIdx,:]<0,Cinhib0[-1][outputsIdx]/(1+self.k3M[-1][outputsIdx]*olderX[-1]*self.E0/cp),0)

            Inhib = np.sum(Cinhibs*olderX[-1]/kdT[-1][outputsIdx])
            x_eq = np.sum(Cactivs*olderX[-1]/(kdI[-1][outputsIdx]*cp+Inhib/cp))
            Inhib2 = np.sum(Cinhibs*olderX[-1]/(kdT[-1][outputsIdx]*k6[-1][outputsIdx]))
            new_cp += Inhib2/(E0*cp)*x_eq


        return cp - new_cp
    def _allEquilibriumFunc(self,cps,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,E0,X0,masks,k1M,k3M):
        """
            Given approximate for the different competitions term (in the vector cps), we compute the next approximate using G.
            The real value for the competitions term verify cps = G(cps,initialConditions)
        :param cps: 1+nbrTemplate array. In the first one we store the competition on the enzyme.
                                         In the other the competition on the template species.
        :param E0: float, initial concentration in the enzyme
        :param X0: nbrInputs array, contains initial value for the inputs
        :param masks: masks giving the network topology.
        :return:
        """
        # We move from an array to a list of 2d-array for the competition over each template:

        cp=cps[0]
        cpt = [np.reshape(cps[(l*m.shape[0]*m.shape[1]+1):((l+1)*m.shape[0]*m.shape[1]+1)],(m.shape[0],m.shape[1])) for l,m in enumerate(masks)]
        new_cpt = [np.zeros(l.shape)+1 for l in cpt]
        new_cp = 1

        olderX = [np.zeros(m.shape[1]) for m in masks]
        for layeridx,layer in enumerate(masks):
            layerEq = np.zeros(layer.shape[1])
            if(layeridx==0):
                for inpIdx in range(layer.shape[1]):
                    #compute of Kactivs,Kinhibs;
                    Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                    Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                    #compute of "weights": sum of kactivs and kinhibs
                    w_inpIdx = np.sum(Kactivs)+np.sum(Kinhibs)
                    x_eq = X0[inpIdx]/(1+E0*w_inpIdx/cp)
                    # update for fixed point:
                    new_cp += w_inpIdx*x_eq
                    # saving values
                    layerEq[inpIdx] = x_eq
                new_cpt[layeridx] = np.where(layer>0,(1+k1M[layeridx]*E0*layerEq/cp),np.where(layer<0,(1+k3M[layeridx]*E0*layerEq/cp),1))
                olderX[layeridx] = layerEq
            else:
                for inpIdx in range(layer.shape[1]):

                    #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                    #Terms for the previous layers
                    CactivsOld = np.where(masks[layeridx-1][inpIdx,:]>0,Cactiv0[layeridx-1][inpIdx],0)
                    CinhibsOld = np.where(masks[layeridx-1][inpIdx,:]<0,Cinhib0[layeridx-1][inpIdx],0)
                    Inhib = np.sum(CinhibsOld*olderX[layeridx-1]/kdT[layeridx-1][inpIdx])
                    #computing of new equilibrium
                    x_eq = np.sum(CactivsOld*olderX[layeridx-1]/(kdI[layeridx-1][inpIdx]*cp+Inhib/cp))
                    layerEq[inpIdx] = x_eq

                    #compute of Kactivs,Kinhibs, for the current layer:
                    Kactivs = np.where(layer[:,inpIdx]>0,Kactiv0[layeridx][:,inpIdx],0) #This is also a matrix element wise multiplication
                    Kinhibs = np.where(layer[:,inpIdx]<0,Kinhib0[layeridx][:,inpIdx],0)
                    #Adding, to the competition over enzyme, the complex formed in this layer by this input.
                    firstComplex = np.sum(np.where(layer[:,inpIdx]>0,Kactivs*x_eq,np.where(layer[:,inpIdx]<0,Kinhibs*x_eq,0)))
                    #We must also add the effect of pseudoTempalte enzymatic complex in the previous layers which can't be computed previously because we missed x_eq
                    Inhib2 = np.sum(CinhibsOld*olderX[layeridx-1]/(kdT[layeridx-1][inpIdx]*k6[layeridx-1][inpIdx]))
                    new_cp += firstComplex + Inhib2/(E0*cp)*x_eq
                new_cpt[layeridx] = np.where(layer>0,(1+k1M[layeridx]*E0*layerEq/cp),np.where(layer<0,(1+k3M[layeridx]*E0*layerEq/cp),1))
                olderX[layeridx] = layerEq
        #Finally we must add the effect of pseudoTemplate enzymatic complex in the last layers
        for outputsIdx in range(masks[-1].shape[0]):
            Cinhibs = np.where(masks[-1][outputsIdx,:]<0,Cinhib0[-1][outputsIdx]/cpt[-1][outputsIdx],0)
            Cactivs = np.where(masks[-1][outputsIdx,:]>0,Cactiv0[-1][outputsIdx]/cpt[-1][outputsIdx],0)

            Inhib = np.sum(Cinhibs*olderX[-1]/kdT[-1][outputsIdx])
            x_eq = np.sum(Cactivs*olderX[-1]/(kdI[-1][outputsIdx]*cp+Inhib/cp))
            Inhib2 = np.sum(Cinhibs*olderX[-1]/(kdT[-1][outputsIdx]*k6[-1][outputsIdx]))
            new_cp += Inhib2/(E0*cp)*x_eq

        #we know that the root should be larger than 1 so we artificially add a root barrier at 1.
        diff_cps = np.zeros(cps.shape)
        if new_cp>=1:
            diff_cps[0] = cp - new_cp
        else:
            diff_cps[0] = 10**3
        for l,m in enumerate(masks):
            layer_new_cpt = np.reshape(new_cpt[l],(m.shape[0]*m.shape[1]))
            layer_old_cpt = cps[(l*m.shape[0]*m.shape[1]+1):((l+1)*m.shape[0]*m.shape[1]+1)]
            diff_cps[(l*m.shape[0]*m.shape[1]+1):((l+1)*m.shape[0]*m.shape[1]+1)] = layer_old_cpt - np.where(layer_new_cpt<1,layer_old_cpt + 10**3, layer_new_cpt)
        return diff_cps
    def for_scalar(self,cps,X0):
        return np.sum(np.abs(self.allEquilibriumFunc(cps,X0)))
    def computeCPonly(self,X0,fittedValue=None):
        """
             This function computes the competition's value by solving a fixed point equation.
             It is based on the most simplest chemical model for the template model: no endonuclease; polymerase and nickase are considered together.
         :param k1, and the others k are the reactions constants
         :param X: array with initial concentration of the first layer.
         :param masks:
         :param fittedValue: value obtained with an analytical model that is an upper bound on the real value. if not provided we use 10**6
         :return:
         """

        if fittedValue is None:
            cp0max=self.obtainBornSup(X0)# we start from an upper bound
        else:
            cp0max=fittedValue
        t0=time.time()
        if self.cpEquilibriumFunc(1,X0) == 0:
            return 1

        computedCp,r = brentq(self.cpEquilibriumFunc,1,cp0max,args=(X0),full_output=True)
        #print("Ended brentq methods in "+str(time.time()-t0)+" with "+str(r.iterations)+" steps")
        return computedCp
    def computeCP(self,X0,initValue=None):
        """
            This function computes the competition's value by solving a fixed point equation.
            It is based on the most simplest chemical model for the template model: no endonuclease; polymerase and nickase are considered together.
        :param X0: array with initial concentration of the first layer.
        :param initValue: initial value for cp, the competition over the enzymes. Can be computed with brentq method by fixing the competition over template...
        :return:
        """

        if initValue is None:
            cps0min=np.concatenate(([1],np.zeros(np.sum([m.shape[0]*m.shape[1] for m in self.masks]))+1)) # we start from an inferior bound
        else:
            cps0min=np.concatenate(([initValue],np.zeros(np.sum([m.shape[0]*m.shape[1] for m in self.masks]))+1))
        t0=time.time()
        #computedCp = minimize(for_scalar,cps0min,args=(k2,k4,k6,kdT,kdI,Kactiv0,Kinhib0,Cactiv0,Cinhib0,TA0,TI0,E0,X0,masks),method="L-BFGS-B",bounds=[(1,None) for _ in cps0min])
        computedCp = root(self.allEquilibriumFunc,cps0min,args=(X0))
        print("ended root methods in "+str(time.time()-t0))
        if not computedCp["success"]:
            print(computedCp["message"])
        return computedCp["x"]


    def computeEquilibriumValue(self,cp,X0,observed=None,verbose=False):
        return self._computeEquilibriumValue(cp,self.kdT,self.kdI,self.Cactiv0,self.Cinhib0,self.E0,X0,self.masks,observed=observed,verbose=verbose)
    def _computeEquilibriumValue(self,cp,kdT,kdI,Cactiv0,Cinhib0,E0,X0,masks,observed=None,verbose=False):
        """
            Given cp, compute the equilibrium values for all nodes in solutions.
        :param cps: Value for the competitions over enzyme and template obtained by fixed point strategy.
        :param observed: tuple, default to None. If provided, we only give the value for the species at the position observed.
        :return:
        """
        # We move from an array to a list of 2d-array for the competition over each template:
        olderX = [np.zeros(m.shape[1]) for m in masks]
        for layeridx,layer in enumerate(masks):
            layerEq = np.zeros(layer.shape[1])
            if(layeridx==0):
                if verbose:
                    print(" input to python:",X0," with cp: ",cp)
                for inpIdx in range(layer.shape[1]):
                    # The first layer is a one to one non-linearity leading to an equation of the second order:
                    if self.k1M[layeridx][inpIdx,inpIdx]*E0*self.TA0[layeridx][inpIdx,inpIdx]/cp == 0:
                        x_eq = X0[inpIdx]
                    else:
                        bOnA = X0[inpIdx]-self.TA0[layeridx][inpIdx,inpIdx]- cp/(self.k1M[layeridx][inpIdx,inpIdx]*self.E0)
                        if verbose:
                            print("bOnA python",bOnA)
                            print("second part",4*cp*X0[inpIdx]/(self.k1M[layeridx][inpIdx,inpIdx]*self.E0))
                        x_eq = 1/2*(bOnA  + (bOnA**2+4*cp*X0[inpIdx]/(self.k1M[layeridx][inpIdx,inpIdx]*self.E0))**0.5)
                    # saving values
                    layerEq[inpIdx] = x_eq
                olderX[layeridx] = layerEq
                if verbose:
                    print("python first layer:",layerEq)
            else:
                for inpIdx in range(layer.shape[1]):

                    #compute of Cactivs,Cinhibs, the denominator marks the template's variation from equilibrium
                    #Terms for the previous layers
                    CactivsOld = np.where(masks[layeridx-1][inpIdx,:]>0,Cactiv0[layeridx-1][inpIdx]/(1+self.k1M[layeridx-1][inpIdx]*olderX[layeridx-1]*self.E0/cp),0)
                    CinhibsOld = np.where(masks[layeridx-1][inpIdx,:]<0,Cinhib0[layeridx-1][inpIdx]/(1+self.k3M[layeridx-1][inpIdx]*olderX[layeridx-1]*self.E0/cp),0)
                    Inhib = np.sum(CinhibsOld*olderX[layeridx-1]/kdT[layeridx-1][inpIdx])
                    #computing of new equilibrium
                    x_eq = np.sum(CactivsOld*olderX[layeridx-1]/(kdI[layeridx-1][inpIdx]*cp+Inhib/cp))
                    if verbose:
                        print("numerateur: ",np.sum(CactivsOld*olderX[layeridx-1]), " denominateur: ",np.sum(kdI[layeridx-1][inpIdx]*cp+Inhib/cp))
                    layerEq[inpIdx] = x_eq
                    #compute of Kactivs,Kinhibs, for the current layer:
                olderX[layeridx] = layerEq
        #Compute equilibrium for last layers:
        layerEq = np.zeros(masks[-1].shape[0])
        for outputIdx in range(masks[-1].shape[0]):

            CactivsOld = np.where(masks[-1][outputIdx,:]>0,Cactiv0[-1][outputIdx]/(1+self.k1M[-1][outputIdx]*olderX[-1]*self.E0/cp),0)
            CinhibsOld = np.where(masks[-1][outputIdx,:]<0,Cinhib0[-1][outputIdx]/(1+self.k3M[-1][outputIdx]*olderX[-1]*self.E0/cp),0)
            Inhib = np.sum(CinhibsOld*olderX[-1]/kdT[-1][outputIdx])
            #computing of new equilibrium
            x_eq = np.sum(CactivsOld*olderX[-1]/(kdI[-1][outputIdx]*cp+Inhib/cp))
            layerEq[outputIdx]=x_eq
        olderX += [layerEq]

        if observed is not None:
            try:
                assert len(observed)==2
            except:
                raise Exception("please provide a tuple of size two for observed indicating the value to use.")
            return olderX[observed[0]][observed[1]]
        return olderX

    def print_constant(self):
        for idx,c in enumerate(self.cstList):
            if(type(c)==list):
                if len(c[0].shape)==2:
                    print(self.cstList[idx][0][0,0],self.cstListName[idx]," dim 2")
                elif len(c[0].shape)==1:
                    print(self.cstList[idx][0][0],self.cstListName[idx]," dim 1")
                else:
                    print(self.cstList[idx][0],self.cstListName[idx]," dim 0")
            else:
                print(self.cstList[idx],self.cstListName[idx]," cste")