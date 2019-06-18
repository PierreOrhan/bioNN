def coopWrite(nameC, nameY1, nameY2, nameE, nameE2, constants, pathEquations, pathConstants):
    """
        Autogenerate equations for the autocatalysis, and parse them in a file:
    :param nameC: Template used for cooperation
    :param nameY1: catalyzed species
    :param nameE: Polymerase name
    :param nameE2: Nickase name
    :return: add names in the txt file
    """
    for n in [nameC, nameY1, nameE, nameE2]:
        assert "&" not in n
        assert "+" not in n
        assert "-" not in n
    nameYC= nameY1 + nameC
    nameEYC= nameE + nameY1 + nameC
    nameE2YC= nameE2 + nameY1 + nameC
    equations=[
        nameC + "+" + nameY1 + "+" + nameE + "-" + nameEYC,
        nameEYC +"-" + nameC + "+" + nameY1 + "+" + nameE,
        nameEYC +"-" + nameE + "+" + nameYC,
        nameYC +"+" + nameE2 +"-" + nameE2YC,
        nameE2YC +"-" + nameE2 +"+" + nameYC,
        nameE2YC +"-" + nameE2 +"+" + nameY1 + "+" + nameY2 + "+" + nameC #Main differences with an auto-catalysis
    ]
    assert len(constants)==len(equations)
    with open(pathEquations,'a') as file:
        for e in equations:
            file.write(e+"\n")
    with open(pathConstants,'a') as file:
        for c in constants:
            file.write(str(c)+"\n")

def killingTemplateWrite(nameM,nameY,nameE,nameE2,constants,pathEquations,pathConstants):
    """
        Writer for the generation of a killer template.
        Implemented reactions are:
            M+E=EM-E+MT
            MT+E2=E2MT-E2+M+T
            Y+T+E=EYT-E+TY
            TY=T+Yd
    :param nameM: name of inhibitor
    :param nameY: name of inhibited
    :param nameE: name of polymerase
    :param nameE2: name of nickase
    :param pathEquations:
    :param pathConstants:
    :return:
    """
    for n in [nameM,nameY,nameE,nameE2]:
        assert "&" not in n
        assert "+" not in n
        assert "-" not in n
    nameT="Templ_"+nameM+nameY
    nameEM = nameE+nameM
    nameE2MT = nameE2+nameM+nameT
    nameMT = nameM+nameT
    nameEYT = nameE+nameY+nameT
    nameTY = nameT+nameY
    nameTYd=nameTY+"d"
    equations=[
        nameM+"+"+nameE+"-"+nameEM,
        nameEM+"-"+nameM+"+"+nameE,
        nameEM+"-"+nameE+"+"+nameMT,
        nameMT+"+"+nameE2+"-"+nameE2MT,
        nameE2MT+"-"+nameE2+"+"+nameMT,
        nameE2MT+"-"+nameE2+"+"+nameT+"+"+nameM, #creation of T
        nameT+"+"+nameY+"+"+nameE+"-"+nameEYT,
        nameEYT+"-"+nameT+"+"+nameY+"+"+nameE,
        nameEYT+"-"+nameE+"+"+nameTY,
        nameTY+"-"+nameT+"+"+nameTYd,
        nameT+"+"+nameTYd+"-"+nameTY
    ]

    assert len(constants)==len(equations)

    with open(pathEquations,'a') as file:
        for e in equations:
            file.write(e+"\n")
    with open(pathConstants,'a') as file:
        for c in constants:
            file.write(str(c)+"\n")

def autocatalysisWrite(nameA,nameY,nameE,nameE2,constants,pathEquations,pathConstants):
    """
        Autogenerate equations for the autocatalysis, and parse them in a file.
        The reaction are: A+Y+E = EYA -> YA
                          YA+E2 = E2YA -> E2 + 2Y +A
    :param nameA: Template
    :param nameY: catalyzed species
    :param nameE: Polymerase name
    :param nameE2: Nickase name
    :return: add names in the txt file
    """
    for n in [nameA,nameY,nameE,nameE2]:
        assert "&" not in n
        assert "+" not in n
        assert "-" not in n
    nameYA=nameY+nameA
    nameEYA=nameE+nameY+nameA
    nameE2YA=nameE2+nameY+nameA
    equations=[
        nameA+"+"+nameY+"+"+nameE+"-"+nameEYA,
        nameEYA+"-"+nameA+"+"+nameY+"+"+nameE,
        nameEYA+"-"+nameE+"+"+nameYA,
        nameYA+"+"+nameE2+"-"+nameE2YA,
        nameE2YA+"-"+nameE2+"+"+nameYA,
        nameE2YA+"-"+nameE2+"+"+"2&"+nameY+"+"+nameA
    ]

    assert len(constants)==len(equations)

    with open(pathEquations,'a') as file:
        for e in equations:
            file.write(e+"\n")
    with open(pathConstants,'a') as file:
        for c in constants:
            file.write(str(c)+"\n")

def templateActivationWrite(nameA,nameY,nameE,nameE2,constants,pathEquations,pathConstants,complexity="normal",templateName=None):
    """
        Autogenerate equations for the template activation, and parse them in a file.

        The template name, T, is wrote as Templ_output_input, so will here be Templ_Y_A, unless it is given
    :param nameA: small strand clipping on template
    :param nameY: generated species
    :param nameE: Polymerase name
    :param nameE2: Nickase name
    :param complexity: the complexity of the model:
            full:
                In this case, we use the full decomposition of the reaction, only using 2 input for each reaction.
                The reaction are: A+T = AT
                                  AT + E = EAT -> E + ATY
                                  ATY + E2 = E2ATY -> E2 + A + T + Y
            simple:
                In this case we summarize the dependence in enzymes under one enzyme name.
            The reaction are: A+T+E = EAT -> E + A + T + Y
            normal:
                The reaction are: A+ T + E = EAT -> E + ATY
                          ATY + E2 = E2ATY -> E2 + A + T + Y
    :return: add names in the txt file
    """
    for n in [nameA,nameY,nameE,nameE2]:
        assert "&" not in n
        assert "+" not in n
        assert "-" not in n

    if(complexity=="full"):
        assert len(constants)==8
        if templateName is None:
            nameT = "Templ_"+nameY+"_"+nameA
        else:
            nameT = templateName
        nameTA = nameT+nameA
        nameETA = nameE+nameTA
        nameTAY = nameTA+nameY
        nameE2TAY = nameE2+nameTAY
        equations=[
            nameA+"+"+nameT+"-"+nameTA,
            nameTA+"-"+nameA+"+"+nameT,
            nameTA+"+"+nameE+"-"+nameETA,
            nameETA+"-"+nameTA+"+"+nameE,
            nameETA+"-"+nameE+"+"+nameTAY,
            nameTAY+"+"+nameE2+"-"+nameE2TAY,
            nameE2TAY+"-"+nameTAY+"+"+nameE2,
            nameE2TAY+"-"+nameE2+"+"+nameA+"+"+nameT+"+"+nameY
        ]
    elif(complexity=="simple"):
        assert len(constants)==3
        if templateName is None:
            nameT = "Templ_"+nameY+"_"+nameA
        else:
            nameT = templateName
        nameTA = nameA+nameT
        nameETA = nameE+nameTA
        equations=[
            nameA+"+"+nameT+"+"+nameE+"-"+nameETA,
            nameETA+"-"+nameA+"+"+nameT+"+"+nameE,
            nameETA+"-"+nameE+"+"+nameA+"+"+nameT+"+"+nameY
        ]
    elif(complexity=="normal"):
        for n in [nameA,nameY,nameE,nameE2]:
            assert "&" not in n
        assert "+" not in n
        assert "-" not in n
        assert len(constants)==6
        if templateName is None:
            nameT = "Templ_"+nameY+"_"+nameA
        else:
            nameT = templateName
        nameTA = nameA+nameT
        nameETA = nameE+nameTA
        nameTAY = nameTA+nameY
        nameE2TAY = nameE2+nameTAY
        equations=[
            nameA+"+"+nameT+"+"+nameE+"-"+nameETA,
            nameETA+"-"+nameA+"+"+nameT+"+"+nameE,
            nameETA+"-"+nameE+"+"+nameTAY,
            nameTAY+"+"+nameE2+"-"+nameE2TAY,
            nameE2TAY+"-"+nameTAY+"+"+nameE2,
            nameE2TAY+"-"+nameE2+"+"+nameA+"+"+nameT+"+"+nameY
        ]
    assert len(constants)==len(equations)

    with open(pathEquations,'a') as file:
        for e in equations:
            file.write(e+"\n")
    with open(pathConstants,'a') as file:
        for c in constants:
            file.write(str(c)+"\n")

def templateInhibWrite(nameA,nameY,nameE,nameE2,constants,pathEquations,pathConstants,complexity="normal"):
    """
        Autogenerate equations for the template inhibition, and parse them in a file.
        The reaction are: The activation of a pseudo-template, with the complexity given by complexity.
                            generate: A + TYAd + E -> A + TY + TYAd
        For every different nameY that this function is called to, one should call the function templateRealInhibtionWrite which produce:
                         The clipping of Y on this pseudo-template:
                            Y+TY + E = ETYY -> E + Yd +TY
                          similarly the complexity of the implementation of the clipping will depend on the complexity parameter.
        The template name, TYAd, is wrote as Templ_output_inputd, so will here be Templ_Y_Ad
    :param nameA: small strand clipping on template
    :param nameY: generated species
    :param nameE: name for polymeraze
    :param nameE2: name for nickaze
    :param constants: Array with the value for the reaction constants, len depends on the complexity.
    :param complexity: complexity of the activation step
    :return: add names in the txt file
    """
    nameTd = "Templ_"+nameY+"_"+nameA+"d"
    nameT = "T_"+nameY
    if complexity=="simple":
        templateActivationWrite(nameA,nameT,nameE,constants[:3],pathEquations,pathConstants,complexity,templateName=nameTd)
    elif complexity=="normal":
        templateActivationWrite(nameA,nameT,nameE,nameE2,constants[:6],pathEquations,pathConstants,complexity,templateName=nameTd)
    elif complexity=="full":
        templateActivationWrite(nameA,nameT,nameE,nameE2,constants[:8],pathEquations,pathConstants,complexity,templateName=nameTd)
    else:
        raise Exception("please provide a complexity in [simple,normal,full]")

def templateRealInhibitionWrite(nameY,nameE,constants,pathEquations,pathConstants,complexity="normal"):
    """
    The clipping of Y on this pseudo-template:
                            Y+TY + E = ETYY -> E + Yd +TY
                          similarly the complexity of the implementation of the clipping will depend on the complexity parameter.
    :param nameY: species to be inhibited
    :param nameE: polymerase
    :param constants: reaction constants
    :param pathEquations:
    :param pathConstants:
    :param complexity:
    :return:
    """
    if complexity=="simple":
        constants = constants[3:]
    elif complexity=="normal":
        constants = constants[6:]
    elif complexity=="full":
        constants = constants[8:]
    nameT = "T_"+nameY
    if complexity=="simple" or complexity=="normal":
        for n in [nameY]:
            assert "&" not in n
            assert "+" not in n
            assert "-" not in n
        assert len(constants)==3
        nameYd = nameY+"d"
        nameETY = nameE+nameT+nameY
        equations=[
            nameY+"+"+nameT+"+"+nameE+"-"+nameETY,
            nameETY+"-"+nameY+"+"+nameT+"+"+nameE,
            nameETY+"-"+nameE+nameYd+nameT
        ]
    else:
        for n in [nameY]:
            assert "&" not in n
            assert "+" not in n
            assert "-" not in n
        assert len(constants)==6
        nameYd = nameY+"d"
        nameETY = nameE+nameT+nameY
        nameTY = nameT+nameY
        equations=[
            nameY+"+"+nameT+"-"+nameTY,
            nameTY+"-"+nameY+"+"+nameT,
            nameTY+"+"+nameE+"-"+nameETY,
            nameETY+"-"+nameE+"+"+nameTY,
            nameETY+"-"+nameE+nameYd+nameT
        ]
    assert len(constants)==len(equations)

    with open(pathEquations,'a') as file:
        for e in equations:
            file.write(e+"\n")
    with open(pathConstants,'a') as file:
        for c in constants:
            file.write(str(c)+"\n")