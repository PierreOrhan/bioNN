import os
import numpy as np
import time as tm
import pandas

def convertToLassieInput(inputDir,equations,constants,nameDic,time,
                         initialization,atol=None,be_step=None,newton_iter=None,
                         newton_tol=None,stiffness_tol=None,rkf_step=None,cs_vector=None,
                         leak=None):
    """
        Convert from our format to Lassies's format of inputs
        Note: we observe that Lassie's behavior is different from the one pretended in the paper.
              it appeared that the main reason for switching to stiff is the presence of concentration going to zero.
                This is a bit strange considering that concentration go to 0 late in the integration...

    :param inputDir: directory at which the file should be created
    :param equations: equations: a d*n array where d is the number of equations (one sided : -> ) and n the number of species
                      The syntax should be the following, an array with the stochiometric coefficient of the species: [alpha,-beta] for alpha*A->beta*B for example
                      Negative stochiometric coeff represent product when positive represent input.
    :param constants: a d array, the reaction constant for the equation
    :param nameDic: a dictionnary with the name of species as keys and their position as values.
    :param time: array with the time steps to collect.
    :param initialization: either an array or a dic with the initialization for each species.
    :param atol: if not None, tolerance for rkf, default to 10**-12
    :param be_step: if not None, initial step for backward_euler, default to 0.01
                    Please consider that this is only use when dt<isStiff
                    Then dt is reset to be_step for use in the bdf...
    :param newton_iter: if not None, maximum number of newton iteration, default to 10000
                        From our experiment, most of the time only 2 iterations are made for the method to succeed.
    :param newton_tol: if not None, tolerance for the newton resolution.
                        From X0 the newton method iterates Xn until J(a)(Xn+1-Xn)=-g(Xn)=0
                        The criteria is on |Xn+1 - Xn|^2 < newton_tol
    :param stiffness_tol: if not None, tolerance for considering to switch do bdf default to 10**-6,
                          if time step below isStiff: consider the system stiff.
    :param rkf_step: initial step size for rkf, default to 10**-3.
    :param cs_vector: Vector of chemical species (idx) to be sampled, None let LASSIE create it.
    :param leak: float, an additional leak to add to the derivative of the species at each ODE step. Should be very small.
                 Such leak stabilizes the solving and corresponds more to the biology. If None: default to 0.
    :return:
    """
    print("Starting LASSIE parsing")
    if equations.shape[0] > 65536 or equations.shape[1] > 65536:
        raise Exception("LASSIE CAN'T SIMULATE MODEL LARGER THAN 65536 equations or species, because of the short type")

    if not os.path.exists(inputDir):
        os.makedirs(inputDir)

    t =tm.time()
    # ==== mandatory file for LASSIE: =====

    #generate left_side file
    #Left_side represents the A matrix, the matrix for coefficient for input
    #Lassie asks for '\t' as a separation character
    A = np.array(np.where(equations>0,equations,0),dtype=np.int)
    np.savetxt(os.path.join(inputDir,"left_side"),A,fmt='%i',delimiter="\t")

    #generate right_side file:
    #right side represents the B matrix, the matrix for coefficient for output
    #Lassie asks for '\t' as a separation character
    B = np.array(np.where(equations<0,-1*equations,0),dtype=np.int)
    np.savetxt(os.path.join(inputDir,"right_side"),B,fmt='%i',delimiter="\t")

    print("Wrote matrix")
    #generate c_vector file:
    #column with the coefficient for each reaction
    np.savetxt(os.path.join(inputDir,"c_vector"),constants,delimiter="\n")

    #generate t_vector file:
    #column with the time steps to collect
    np.savetxt(os.path.join(inputDir,"t_vector"),time,fmt='%f',delimiter="\n")

    #generate M_0 file:
    #line with the initial concentration for every species
    print("Starting conversion")
    if(type(initialization)==dict):
        initArray = np.zeros(A.shape[1])
        for k in nameDic.keys():
            initArray[nameDic[k]] = initialization[k]
    else:
        initArray=initialization
    np.savetxt(os.path.join(inputDir,"M_0"),[initArray],delimiter="\t")
    print("wrote conversion")
    # ==== not-mandatory file for LASSIE: =====
    #generate alphabet:
    with open(os.path.join(inputDir,"alphabet"),'w') as file:
        for n in list(nameDic.keys()):
            file.write(n+"\t")

    #generate atol_vector:
    if atol is not None:
        np.savetxt(os.path.join(inputDir,"atol_vector"),np.array([atol for _ in range(equations.shape[0])]),delimiter="\n")

    #generate modelkind:
    with open(os.path.join(inputDir,"modelkind"),'w') as file:
        file.write("deterministic")

    #generate be_step file:
    if be_step is not None:
        with open(os.path.join(inputDir,"be_step"),'w') as file:
            file.write(str(be_step)+"\n")

    #generate newton_iter
    if newton_iter is not None:
        with open(os.path.join(inputDir,"newton_iter"),'w') as file:
            file.write(str(newton_iter)+"\n")

    #generate newton_tol:
    if newton_tol is not None:
        with open(os.path.join(inputDir,"newton_tol"),'w') as file:
            file.write(str(newton_tol)+"\n")

    #generate stiffness_tol:
    if stiffness_tol is not None:
        with open(os.path.join(inputDir,"stiffness_tol"),'w') as file:
            file.write(str(stiffness_tol)+"\n")

    #generate rkf_step:
    if stiffness_tol is not None:
        with open(os.path.join(inputDir,"rkf_step"),'w') as file:
            file.write(str(rkf_step)+"\n")

    #generate cs_vector:
    if cs_vector is not None:
        np.savetxt(os.path.join(inputDir,"cs_vector"),cs_vector,delimiter="\n")

    #generate leak:
    if leak is not None:
        with open(os.path.join(inputDir,"leak"),'w') as file:
            file.write(str(leak)+"\n")


    print("LASSIE WRITING TIME:" + str(tm.time()-t))


def updateLassieConcentrationInput(inputDir,initialization,nameDic):
    """
    Convert from our format to Lassies's format of inputs
        Note: we observe that Lassie's behavior is different from the one pretended in the paper.
              it appeared that the main reason for switching to stiff is the presence of concentration going to zero.
                This is a bit strange considering that concentration go to 0 late in the integration...

    :param inputDir: dir at which the file should be created
    :param initialization: either an array or a dic with the initialization for each species.
    :return:
    """
    print("Starting conversion")
    if(type(initialization)==dict):
        initArray = np.zeros(len(list(nameDic.keys())))
        for k in nameDic.keys():
            initArray[nameDic[k]] = initialization[k]
    else:
        initArray=initialization
    np.savetxt(os.path.join(inputDir,"M_0"),[initArray],delimiter="\t")
    print("wrote conversion")