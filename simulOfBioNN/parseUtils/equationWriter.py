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

def killingTemplateWrite(nameM,nameT,nameY,nameE,nameE2,constants,pathEquations,pathConstants):
    """
    :param nameM:
    :param nameY:
    :param nameE:
    :param nameE2:
    :param pathEquations:
    :param pathConstants:
    :return:
    """
    for n in [nameM,nameT,nameY,nameE,nameE2]:
        assert "&" not in n
        assert "+" not in n
        assert "-" not in n
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

def templateActivationWrite(nameA,nameY,nameE,nameE2,constants,pathEquations,pathConstants,complexity="normal"):
    """
        Autogenerate equations for the template activation, and parse them in a file.

        The template name, T, is wrote as Templ_inputoutput, so will here be Templ_AY
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
        nameT = "Templ_"+nameY+nameA
        nameAT = nameA+nameT
        nameEAT = nameE+nameAT
        nameATY = nameAT+nameY
        nameE2ATY = nameE2+nameATY
        equations=[
            nameA+"+"+nameT+"-"+nameAT,
            nameAT+"-"+nameA+"+"+nameT,
            nameAT+"+"+nameE+"-"+nameEAT,
            nameEAT+"-"+nameAT+"+"+nameE,
            nameEAT+"-"+nameE+"+"+nameATY,
            nameATY+"+"+nameE2+"-"+nameE2ATY,
            nameE2ATY+"-"+nameATY+"+"+nameE2,
            nameE2ATY+"-"+nameE2+"+"+nameA+"+"+nameT+"+"+nameY
        ]
    elif(complexity=="simple"):
        assert len(constants)==3
        nameT = "Templ_"+nameY+nameA
        nameAT = nameA+nameT
        nameEAT = nameE+nameAT
        equations=[
            nameA+"+"+nameT+"+"+nameE+"-"+nameEAT,
            nameEAT+"-"+nameA+"+"+nameT+"+"+nameE,
            nameEAT+"-"+nameE+"+"+nameA+"+"+nameT+"+"+nameY
        ]
    elif(complexity=="normal"):
        for n in [nameA,nameY,nameE,nameE2]:
            assert "&" not in n
        assert "+" not in n
        assert "-" not in n
        assert len(constants)==6
        nameT = "Templ_"+nameY+nameA
        nameAT = nameA+nameT
        nameEAT = nameE+nameAT
        nameATY = nameAT+nameY
        nameE2ATY = nameE2+nameATY
        equations=[
            nameA+"+"+nameT+"+"+nameE+"-"+nameEAT,
            nameEAT+"-"+nameA+"+"+nameT+"+"+nameE,
            nameEAT+"-"+nameE+"+"+nameATY,
            nameATY+"+"+nameE2+"-"+nameE2ATY,
            nameE2ATY+"-"+nameATY+"+"+nameE2,
            nameE2ATY+"-"+nameE2+"+"+nameA+"+"+nameT+"+"+nameY
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
                            generate: A + T + E -> A + Td + T
                          The clipping of Y on this pseudo-template:
                            Y+Td + E = ETdY -> E + YAd +Td (we consider the little extension to be dependent on Td, itself dependent on A.
                          similarly the complexity of the implementation of the clipping will depend on the complexity parameter.
        The template name, T, is wrote as Templ_inputoutput, so will here be Templ_AY
    :param nameA: small strand clipping on template
    :param nameY: generated species
    :param nameE: name for polymeraze
    :param nameE2: name for nickaze
    :param complexity: complexity of the activation step
    :return: add names in the txt file
    """
    nameTd = "Templ_"+nameY+nameA+"d"
    if complexity=="simple":
        templateActivationWrite(nameA,nameTd,nameE,constants[:3],pathEquations,pathConstants,complexity)
        constants=constants[3:]
    elif complexity=="normal":
        templateActivationWrite(nameA,nameTd,nameE,nameE2,constants[:6],pathEquations,pathConstants,complexity)
        constants=constants[6:]
    elif complexity=="full":
        templateActivationWrite(nameA,nameTd,nameE,nameE2,constants[:8],pathEquations,pathConstants,complexity)
        constants=constants[8:]
    else:
        raise Exception("please provide a complexity in [simple,normal,full]")
    if complexity=="simple" or complexity=="normal":
        for n in [nameY]:
            assert "&" not in n
            assert "+" not in n
            assert "-" not in n
        assert len(constants)==3
        nameYAd = nameY+nameA+"d"
        nameETdY = nameE+nameTd+nameY

        equations=[
            nameY+"+"+nameTd+"+"+nameE+"-"+nameETdY,
            nameETdY+"-"+nameY+"+"+nameTd+"+"+nameE,
            nameETdY+"-"+nameE+nameYAd+nameTd
        ]
    else:
        for n in [nameY]:
            assert "&" not in n
            assert "+" not in n
            assert "-" not in n
        assert len(constants)==6
        nameYAd = nameY+nameA+"d"
        nameETdY = nameE+nameTd+nameY
        nameTdY = nameTd+nameY

        equations=[
            nameY+"+"+nameTd+"-"+nameTdY,
            nameTdY+"-"+nameY+"+"+nameTd,
            nameTdY+"+"+nameE+"-"+nameETdY,
            nameETdY+"-"+nameE+"+"+nameTdY,
            nameETdY+"-"+nameY+"+"+nameTd+"+"+nameE,
            nameETdY+"-"+nameE+nameYAd+nameTd
        ]
    assert len(constants)==len(equations)

    with open(pathEquations,'a') as file:
        for e in equations:
            file.write(e+"\n")
    with open(pathConstants,'a') as file:
        for c in constants:
            file.write(str(c)+"\n")