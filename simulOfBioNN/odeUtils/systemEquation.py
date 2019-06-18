from numba import float64,guvectorize,void,float32
import numpy as np
import sparse


@guvectorize([void(float64[::1],float64,float64[:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64,float64[::1])],'(n),(),(a,n),(a,n,n),(a,n,n),(a,n,n),()->(n)',nopython=True)
def f(speciesA, t, KarrayA,inputStochio, maskA, maskComplementary,derivLeak,output):
    """
        Compute the temporal derivative of each species.
        Please refer to the following presentation for clear explanation of the strategy of this function:
            https://docs.google.com/presentation/d/1ZayLqDJUJofIbhZDxJycxC7s8eSLk1yoRpdWa8xXkjM/edit?usp=sharing
        Using numba.
        Numba cannot manage to do product of concentration.
            Therefore we are restricted to using a log-exponential strategy
            We also take advantage of this strategy for the stoechiometry
        derivLeak: a small value to add to the derivative, account for tiny leak
    :param speciesA:
    :param t:
    :param KarrayA:
    :param inputStochio:
    :param maskA:
    :param maskComplementary:
    :param derivLeak:
    :param output:
    :return:
    """
    #In case of a species of concentration zero, we cannot us the logarithm strategy => We set the concentration to 1 but delete the wrongly computed intermediate afterward with lineSuppressor mask.
    if(np.prod(speciesA)==0):
        lineSuppressor=np.zeros(KarrayA.shape)+1
        lineZero=np.zeros(KarrayA.shape)
        lineOne=np.zeros(KarrayA.shape)+1
        mask2=np.zeros(maskComplementary.shape) #maskComplementary should remain constant...
        mask2[:,:,:]=maskComplementary[:,:,:]
        for a in range(speciesA.shape[0]):
            if(speciesA[a]==0):#pbm in the logarithm
                mask2[:,:,a]=1
                lineSuppressor[:,:]=np.where(lineSuppressor[:,:]-maskA[:,:,a]<=0,lineZero,lineOne) #multiple species could have a null concentration in a reaction -> clip to 0 the mask
        prod= np.exp(np.sum(np.log(maskA * speciesA + mask2) * inputStochio, axis=2)) * KarrayA * lineSuppressor
    else:
        prod= np.exp(np.sum(np.log(maskA * speciesA + maskComplementary) * inputStochio, axis=2)) * KarrayA
    output[:]=np.sum(prod,axis=0)+derivLeak


#In case of memory trouble, one might want to reduce the matrix type at float32
@guvectorize([void(float32[::1],float32,float32[:,:],float32[:,:,:],float32[:,:,:],float32[:,:,:],float32,float32[::1])],'(n),(),(a,n),(a,n,n),(a,n,n),(a,n,n),()->(n)',nopython=True)
def f32(speciesA, t, KarrayA,inputStochio, maskA, maskComplementary,derivLeak,output):
    """
        Implementation of f for float32
    """
    #In case of a species of concentration zero, we cannot us the logarithm strategy => We set the concentration to 1 but delete the wrongly computed intermediate afterward with lineSuppressor mask.
    if(np.prod(speciesA)==0):
        lineSuppressor=np.zeros(KarrayA.shape,dtype=np.float32)+1
        lineZero=np.zeros(KarrayA.shape,dtype=np.float32)
        lineOne=np.zeros(KarrayA.shape,dtype=np.float32)+1
        mask2=np.zeros(maskComplementary.shape,dtype=np.float32) #maskComplementary should remain constant...
        mask2[:,:,:]=maskComplementary[:,:,:]
        for a in range(speciesA.shape[0]):
            if(speciesA[a]==0):#pbm in the logarithm
                mask2[:,:,a]=1
                lineSuppressor[:,:]=np.where(lineSuppressor[:,:]-maskA[:,:,a]<=0,lineZero,lineOne) #multiple species could have a null concentration in a reaction -> clip to 0 the mask
        prod= np.exp(np.sum(np.log(maskA * speciesA + mask2) * inputStochio, axis=2)) * KarrayA * lineSuppressor
    else:
        prod= np.exp(np.sum(np.log(maskA * speciesA + maskComplementary) * inputStochio, axis=2)) * KarrayA
    output[:]=np.sum(prod,axis=0)+derivLeak


"""
    To improve the speed, one can leverage the fact that no species should see its concentration goes to zero during the experiment.
    
@guvectorize([void(float64[::1],float64,float64[:,:],float64[:,:,:],float64[:,:,:], float64[:,:],float64[:,:,:],float64[::1])],'(n),(),(a,n),(a,n,n),(a,n,n),(a,n),(a,n,n)->(n)',nopython=True)
def securef(speciesA, t, KarrayA,inputStochio, maskA, lineSuppressor, mask2 , output):
    prod= np.exp(np.sum(np.log(maskA * speciesA + mask2) * inputStochio, axis=2)) * KarrayA * lineSuppressor
    output[:]=np.sum(prod,axis=0)
"""

def fLambda(KarrayA,stochio,maskA,maskComplementary,derivLeak=0):
    return lambda t,speciesA: f(speciesA,t,KarrayA,stochio,maskA,maskComplementary,derivLeak)

@guvectorize([void(float64[::1],float64,float64[:,:,:],float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:,:],float64,float64[:,::1])],'(n),(),(n,a,n),(n,a,n,n),(n,a,n,n),(n,a,n,n),()->(n,n)')
def jacobianCompute(speciesA,t,outputKarrays,outputStochios,outputMasks,maskComplementarys,derivLeak,output):
    for i in range(speciesA.shape[0]):
        output[i,:] = f(speciesA,t,outputKarrays[i],outputStochios[i],outputMasks[i],maskComplementarys[i],derivLeak)

def jacobian(outputKarrays,outputStochios,outputMasks,maskComplementarys,derivLeak=0):
    return lambda speciesA,t,a,b,c,d,e: jacobianCompute(speciesA,t,outputKarrays,outputStochios,outputMasks,maskComplementarys,derivLeak)


def obtainJacobianMasks(karrayA,inputStochio,maskA):
    """
        It is really easy to derive the jacobian from the way we wrote our system it down
        The original function can be applied by just changing the masks
    """
    outputStochios = np.zeros([inputStochio.shape[1]]+list(inputStochio.shape))
    outputStochios[:] = inputStochio
    outputKarrays = np.zeros([karrayA.shape[1]]+list(karrayA.shape))
    outputKarrays[:] = karrayA
    outputMasks = np.zeros([maskA.shape[1]]+list(maskA.shape))
    outputMasks[:] = maskA

    for idxJacobSpecies in range(maskA.shape[1]):
        #Computation of the mask of partial derivative with respect to the idxJacobSpecies-th species
        for m in range(maskA.shape[0]):
            for idxSpecies in range(maskA.shape[1]):
                if(maskA[m,idxSpecies,idxJacobSpecies]>0): #In the equation for update of idx-th species from the m-th reaction, the species we compute the jacobian for is present (idxJacobSpecies-th)
                    oldStochio = inputStochio[m,idxSpecies,idxSpecies]
                    if(oldStochio>1):
                        outputStochios[idxJacobSpecies,m,idxSpecies,idxJacobSpecies] = oldStochio-1
                        outputKarrays[idxJacobSpecies,m,idxSpecies] = karrayA[m,idxSpecies] * oldStochio
                    else:
                        outputMasks[idxJacobSpecies,m,idxSpecies,idxJacobSpecies] = 0
                else: ##In the equation for update of idx-th species from the m-th reaction, the species we compute the jacobian for is not present  ==> delete intermediate
                    outputMasks[idxJacobSpecies,m,idxSpecies,:] = 0
                    outputStochios[idxJacobSpecies,m,idxSpecies,:] = 1
                    outputKarrays[idxJacobSpecies,m,idxSpecies] = 0
    outputMasksComplementarys = 1 - outputMasks

    return outputKarrays,outputStochios,outputMasks,outputMasksComplementarys

##Creation tools:

def setToUnits(constants, KarrayA, inputStochio):
    """
        This function parametrize the system in simpler range of work.
        The three parameters should be the output of the parse function.
    :param constants: a d-array with the reaction constant for the equation
    :param KarrayA: The array with the reactions
    :param inputStochio: The stochiometry
    :return: karrayA,T0,C0,kdic
            karrayA: A new KarrayA with modified values.
            T0: value used to rescale with regard to time.
            C0: value used to rescale with regard to concentrations.
            constants: modified d-array with the reaction constant for the equation
    """
    constantStochiodic={}
    #Let us retrieve the number of species acting as input for each equation and its stochiometry
    for idxe in range(KarrayA.shape[0]):
        if(type(KarrayA) == sparse.COO):
            #Fast solution:
            intermediateToTest = np.transpose(np.argwhere(KarrayA[idxe]<0))[0]
            nbSpSt = np.sum(np.max(inputStochio[idxe,:,intermediateToTest],axis=0))
        else:
            # Naive solution:
            nbSpSt=0
            for y in range(KarrayA.shape[1]):
                if(KarrayA[idxe,y]<0): #if it is an input!
                    nbSpSt+=np.max(inputStochio[idxe,:,y])
        constantStochiodic[idxe]=[constants[idxe], nbSpSt]

    #now we extract the maximal reaction constant of time unit and of time*concentraion**2 unit

    ## NEED AND UPDATE TO TAKE INTO ACCOUNT ALL POSSIBILITIES
    #Note: this analysis tool only look in reactions with 3 ,1 inputs. => should at least be extended to take into account reactions with 2 inputs...
    # Nonetheless it remains corrects for reactions with 3,2,1 inputs

    kmaxT=0
    kmaxC=0
    for k in constantStochiodic.keys():
        if(constantStochiodic[k][1]==1 and constantStochiodic[k][0]>kmaxT):
            kmaxT=constantStochiodic[k][0]
        elif(constantStochiodic[k][1]==3 and constantStochiodic[k][0]>kmaxC):
            kmaxC=constantStochiodic[k][0]
    T0=1/kmaxT
    C0=(kmaxC*T0)**(-0.5)

    ## We work differently in the case sparse (the sparse library doesn't provide item assignment) and dense
    if(type(KarrayA) is sparse.COO):
        # karrayA is a 2D sparse matrix
        lilMat = KarrayA.tocsr().tolil()
        for k in constantStochiodic.keys():
            if(constantStochiodic[k][1]==1):
                lilMat[k]=lilMat[k]*T0
                constants[k]= constants[k] * T0
            elif(constantStochiodic[k][1]==2):
                lilMat[k]=lilMat[k]*T0*C0
                constants[k]= constants[k] * T0 * C0
            elif(constantStochiodic[k][1]==3):
                lilMat[k]=lilMat[k]*T0*C0*C0
                constants[k]= constants[k] * T0 * C0 * C0
        KarrayA = sparse.COO.from_scipy_sparse(lilMat)
    else:
        for k in constantStochiodic.keys():
            if(constantStochiodic[k][1]==1):
                KarrayA[k]=KarrayA[k]*T0
                constants[k]= constants[k] * T0
            elif(constantStochiodic[k][1]==2):
                KarrayA[k]=KarrayA[k]*T0*C0
                constants[k]= constants[k] * T0 * C0
            elif(constantStochiodic[k][1]==3):
                KarrayA[k]=KarrayA[k]*T0*C0*C0
                constants[k]= constants[k] * T0 * C0 * C0
    return KarrayA, T0, C0, constants



##Code for non-accelerated solution:
def fPython(speciesA, t, KarrayA,inputStochio, maskA, maskComplementary,derivLeak):
    """

    :param speciesA:
    :param t:
    :param KarrayA:
    :param inputStochio:
    :param maskA:
    :param maskComplementary:
    :param derivLeak:
    :return:
    """
    #In case of a species of concentration zero, we cannot us the logarithm strategy => We set the concentration to 1 but delete the wrongly computed intermediate afterward with lineSuppressor mask.
    output=np.zeros(speciesA.shape[0])
    if(np.prod(speciesA)==0):
        lineSuppressor=np.zeros(KarrayA.shape)+1
        lineZero=np.zeros(KarrayA.shape)
        lineOne=np.zeros(KarrayA.shape)+1
        mask2=np.zeros(maskComplementary.shape) #maskComplementary should remain constant...
        mask2[:,:,:]=maskComplementary[:,:,:]
        for a in range(speciesA.shape[0]):
            if(speciesA[a]==0):#pbm in the logarithm
                mask2[:,:,a]=1
                lineSuppressor[:,:]=np.where(lineSuppressor[:,:]-maskA[:,:,a]<=0,lineZero,lineOne) #multiple species could have a null concentration in a reaction -> clip to 0 the mask
        prod= np.exp(np.sum(np.log(maskA * speciesA + mask2) * inputStochio, axis=2)) * KarrayA * lineSuppressor
    else:
        prod= np.exp(np.sum(np.log(maskA * speciesA + maskComplementary) * inputStochio, axis=2)) * KarrayA
    output[:]=np.sum(prod,axis=0)+derivLeak
    return output

def fPythonSparse(speciesA, t, KarrayA,inputStochio, maskA, maskComplementary ,derivLeak):
    """
        Computation of the derivative for each species for a system parametrized by sparse matrix.
        We use the sparse module. Indeed Numba can't manage sparsity.
        In the current implementation, speciesA should not contains null elements.
        derivLeak: a small value to add to the derivative, account for tiny leak
    """
    output = np.zeros(speciesA.shape[0])
    species = np.where(speciesA>0,speciesA,1)
    #simpler version:
    prod= np.exp(np.sum(maskA * np.log(species) * inputStochio, axis=2)) * KarrayA
    #other version:
    ## In the case of a sparse matrix, the log applies to all data, and to the fill_value
    # ==> we must be in a case where the fill value is 1
    #    and this can be done by adding the complementary sparse mask, previously created by a 1 - maskA which is an operation implemented in the sparse library
    #       ==> the log operation + addition (longest among the 2) of the mask takes between 0.04 and 0.07 for 1000 species
    #    we also found an hack, by astuciously changing the fill_value we can make the log be well applied without having to add the mask complementary
    #       ==> should be faster but in practice we don't see much improvements.
    # We must be careful of the case where some species become null and are still considered as data! ==> Next update?
    # intermediates = maskA * speciesA
    # intermediates.fill_value = np.array([1.])
    # prod = np.exp(np.sum(np.log(intermediates) * inputStochio, axis=2)) * KarrayA

    output[:]=np.sum(prod,axis=0).data+derivLeak

    if(abs(t-1)<10**(-8)):
        print(t)
    return output

