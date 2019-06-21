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
        #TODO Change this
        assert len(constants)==8
        if templateName is None:
            nameT_Y_A = "Templ_"+nameY+"_"+nameA
        else:
            nameT_Y_A = templateName
        nameT_Y_AA = nameT_Y_A+nameA
        nameETA = nameE+nameT_Y_AA
        nameT_Y_AA = nameT_Y_AA+nameY
        nameE2TA = nameE2+nameT_Y_AA
        equations=[
            nameA+"+"+nameT_Y_A+"-"+nameT_Y_AA,
            nameT_Y_AA+"-"+nameA+"+"+nameT_Y_A,
            nameT_Y_AA+"+"+nameE+"-"+nameETA,
            nameETA+"-"+nameT_Y_AA+"+"+nameE,
            nameETA+"-"+nameE+"+"+nameT_Y_AA,
            nameT_Y_AA+"+"+nameE2+"-"+nameE2TA,
            nameE2TA+"-"+nameT_Y_AA+"+"+nameE2,
            nameE2TA+"-"+nameE2+"+"+nameA+"+"+nameT_Y_A+"+"+nameY
        ]
    elif(complexity=="simple"):
        assert len(constants)==3
        if templateName is None:
            nameT_Y_A = "Templ_"+nameY+"_"+nameA
        else:
            nameT_Y_A = templateName
        nameT_Y_AA = nameA+nameT_Y_A
        nameETA = nameE+nameT_Y_AA
        equations=[
            nameA+"+"+nameT_Y_A+"+"+nameE+"-"+nameETA,
            nameETA+"-"+nameA+"+"+nameT_Y_A+"+"+nameE,
            nameETA+"-"+nameE+"+"+nameA+"+"+nameT_Y_A+"+"+nameY
        ]
    elif(complexity=="normal"):
        for n in [nameA,nameY,nameE,nameE2]:
            assert "&" not in n
        assert "+" not in n
        assert "-" not in n
        assert len(constants)==6
        if templateName is None:
            nameT_Y_A = "Templ_"+nameY+"_"+nameA
        else:
            nameT_Y_A = templateName
        nameT_Y_AA = nameT_Y_A+nameA
        nameETA = nameE+nameT_Y_AA
        nameE2TA = nameE2+nameT_Y_AA
        equations=[
            nameA+"+"+nameT_Y_A+"+"+nameE+"-"+nameETA,
            nameETA+"-"+nameA+"+"+nameT_Y_A+"+"+nameE,
            nameETA+"-"+nameE+"+"+nameT_Y_AA,
            nameT_Y_AA+"+"+nameE2+"-"+nameE2TA,
            nameE2TA+"-"+nameT_Y_AA+"+"+nameE2,
            nameE2TA+"-"+nameE2+"+"+nameA+"+"+nameT_Y_A+"+"+nameY
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
        templateActivationWrite(nameA,nameT,nameE,nameE2,constants[:3],pathEquations,pathConstants,complexity=complexity,templateName=nameTd)
    elif complexity=="normal":
        templateActivationWrite(nameA,nameT,nameE,nameE2,constants[:6],pathEquations,pathConstants,complexity=complexity,templateName=nameTd)
    elif complexity=="full":
        templateActivationWrite(nameA,nameT,nameE,nameE2,constants[:8],pathEquations,pathConstants,complexity=complexity,templateName=nameTd)
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
            nameETY+"-"+nameE+"+"+nameYd+"+"+nameT
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
            nameETY+"-"+nameE+"+"+nameYd+"+"+nameT
        ]
    assert len(constants)==len(equations)

    with open(pathEquations,'a') as file:
        for e in equations:
            file.write(e+"\n")
    with open(pathConstants,'a') as file:
        for c in constants:
            file.write(str(c)+"\n")


def endonucleaseWrite2(nameY, nameEndo, constants, pathEquations, pathConstants):
    """
        Write the equation of a template eaten by an endonuclease.
        The endonuclease bind to the target and then disolve it:
            Y + Endo = YEndo - Endo
    :param nameY: name for the species to be destroyed
    :param nameEndo: name for the endonuclease
    :param constants: reaction constants
    :param pathEquations: file at which the reaction will be appended.
    :param pathConstants: file at which the reaction will be appended.
    :return:
    """
    assert "&" not in nameY
    assert "+" not in nameY
    assert "-" not in nameY

    assert len(constants)==3

    nameEndoY = nameEndo+nameY

    equations = [
        nameY+"+"+nameEndo+"-"+nameEndoY,
        nameEndoY+"-"+nameY+"+"+nameEndo,
        nameEndoY+"-"+nameEndo
    ]

    assert len(constants)==len(equations)
    with open(pathEquations,'a') as file:
        for e in equations:
            file.write(e+"\n")
    with open(pathConstants,'a') as file:
        for c in constants:
            file.write(str(c)+"\n")


def endonucleaseWrite(nameY, constant, pathEquations, pathConstants):
    """
        Write the equation of a template eaten by an endonuclease.
        The endonuclease is considered to remain of constant concentration, equal to 1.
    :param nameY:
    :param constantName:
    :return:
    """
    assert "&" not in nameY
    assert "+" not in nameY
    assert "-" not in nameY
    equation=nameY+"-"
    with open(pathEquations,'a') as file:
        file.write(equation+"\n")
    with open(pathConstants,'a') as file:
        file.write(str(constant)+"\n")

def templateProtection(nameA, nameE, nameE2, constants, pathEquations, pathConstants, complexity="normal", nameAp=None, templateName=None):
    """
        Autogenerate equations for protection of an inputs by elongating it, and parse them in a file.

        The template name, is wrote as Templ_output_input, so will here be Templ_Ap_A, unless it is given (in templateName)
        The protected strand will be noted as the initial strand with an additional p at the end of the input name.
    :param nameA: small strand clipping on template
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
            The reaction are: A + Templ_Ap_A + E = EATempl_Ap_A- Templ_Ap_A + Ap + A
            normal:
                The reaction are: A + Templ_Ap_A + E = EATempl_Ap_A-  ATempl_Ap_A
                                  ATempl_Ap_A + E2 = E2ATempl_Ap_A - Templ_Ap_A + Ap + A +E2
                          ATY + E2 = E2ATY -> E2 + A + T + Y
    :param nameAp: string, generated species, if None will default to nameA+"p".
    :param templateName: string, the template name, is wrote as Templ_output_input, so will here be Templ_Ap_A, unless it is given (in templateName)
    :return: add names in the txt file
    """
    if nameAp is None:
        nameAp = nameA +"p"
    for n in [nameA,nameAp, nameE, nameE2]:
        assert "&" not in n
        assert "+" not in n
        assert "-" not in n
    if(complexity=="full"):
        raise Exception("to be implemented")
        #TODO: to implement
    elif(complexity=="simple"):
        assert len(constants)==3
        if templateName is None:
            nameT_Ap_A = "Templ_" + nameAp + "_" + nameA
        else:
            nameT_Ap_A = "Templ_"+nameAp+"_"+nameA

        nameEAT_Ap_A = nameE+nameA+"Templ_"+nameAp+"_"+nameA
        equations=[
            nameT_Ap_A +"+" + nameE +"+" + nameA +"-" + nameEAT_Ap_A,
            nameEAT_Ap_A + "-" + nameT_Ap_A +"+" + nameE +"+" + nameA,
            nameEAT_Ap_A + "-" + nameAp+"+"+nameE+"+"+ nameT_Ap_A +"+"+ nameA
        ]
    elif(complexity=="normal"):
        for n in [nameA, nameAp, nameE, nameE2]:
            assert "&" not in n
        assert "+" not in n
        assert "-" not in n
        assert len(constants)==6
        if templateName is None:
            nameT_Ap_A = "Templ_" + nameAp + "_" + nameA
        else:
            nameT_Ap_A = templateName
        nameAT_Ap_A = nameT_Ap_A+nameA
        nameETA = nameE+nameAT_Ap_A
        nameE2AT_Ap_A = nameE2+nameAT_Ap_A
        equations=[
            nameA +"+" + nameT_Ap_A +"+" + nameE +"-" + nameETA,
            nameETA +"-" + nameA +"+" + nameT_Ap_A +"+" + nameE,
            nameETA +"-" + nameE +"+" + nameAT_Ap_A,
            nameAT_Ap_A +"+" + nameE2 +"-" + nameE2AT_Ap_A,
            nameE2AT_Ap_A +"-" + nameAT_Ap_A +"+" + nameE2,
            nameE2AT_Ap_A +"-" + nameE2 +"+" + nameA +"+" + nameT_Ap_A +"+" + nameAp
        ]
    assert len(constants)==len(equations)

    with open(pathEquations,'a') as file:
        for e in equations:
            file.write(e+"\n")
    with open(pathConstants,'a') as file:
        for c in constants:
            file.write(str(c)+"\n")