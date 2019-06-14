"""
    In this file we give a function to create and model predator-prey systems using the parser.
"""
import simulOfBioNN.parseUtils.equationWriter
import simulOfBioNN.parseUtils.parser as parser
import os


def generateEcoEvoNetwork(name,activConstants=None,inhibConstants=None,endoConstant=None,erase=True):
    """
        Generate an network with an ecological competition between two species.
        For the moment a simple equation is written down, but could be easily changed to a more complex one, either by hand or by creating other features.
    :param name:
    :param activConstants:
    :param inhibConstants:
    :param endoConstant:
    :param erase:
    :return:
    """
    pathEquations = name+"/equations.txt"
    pathConstants = name+"/constants.txt"
    if(erase):
        if(os.path.exists(pathConstants)):
            with open(pathConstants,"w") as file:
                pass
        if(os.path.exists(pathEquations)):
            with open(pathEquations,"w") as file:
                pass
    if(not os.path.exists(pathEquations)):
        if(not os.path.exists(pathEquations.split("/")[0])):
            os.makedirs(pathEquations.split("/")[0])
        with open(pathEquations,"w") as file:
            pass
    if(not os.path.exists(pathConstants)):
        if(not os.path.exists(pathConstants.split("/")[0])):
            os.makedirs(pathConstants.split("/")[0])
        with open(pathConstants,"w") as file:
            pass
    namesA = ["A1","A2"]
    namesY = ["Y1","Y2"]
    namesC = ["C1","C2"]
    nameE = ["E","E2"]
    if not activConstants:
        activConstants=[[26*10**12,3,17,7*10**6,3,3] for _ in range(4)]
    if not inhibConstants:
        inhibConstants=[[26*10**12,3,17,26*10**12,3,3,26*10**12,3,17,0.03,26*10**6] for _ in range(4)]
    if not endoConstant:
        endoConstant=0.32
    constantsMask=[activConstants[0],activConstants[0],activConstants[0],activConstants[0]]
    simulOfBioNN.parseUtils.equationWriter.autocatalysisWrite(namesA[0], namesY[0], nameE[0], nameE[1], constantsMask[0], pathEquations, pathConstants)
    simulOfBioNN.parseUtils.equationWriter.autocatalysisWrite(namesA[1], namesY[1], nameE[0], nameE[1], constantsMask[1], pathEquations, pathConstants)
    simulOfBioNN.parseUtils.equationWriter.coopWrite(namesC[0], namesY[0], namesY[1], nameE[0], nameE[1], constantsMask[2], pathEquations, pathConstants)
    simulOfBioNN.parseUtils.equationWriter.coopWrite(namesC[1], namesY[1], namesY[0], nameE[0], nameE[1], constantsMask[3], pathEquations, pathConstants)
    parser.endonucleasedWrite(namesY[0], endoConstant, pathEquations, pathConstants)
    parser.endonucleasedWrite(namesY[1], endoConstant, pathEquations, pathConstants)
