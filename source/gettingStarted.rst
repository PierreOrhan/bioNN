Getting started
=========================================

Our API, simulOfBioNN is divided in 5 submodule.

.. hlist::
   * nnUtils: provide the layer with sparsity for tensorflow.
   * odeUtils: provide function for computing the derivative of the concentration arrays given the network. Using numba and sparsity package.
   * parseUtils: utilitarian to read/write the necessary masks defining one network for the ode solver. Using
   * plotUtils: all plot function using matplotlib.pyplot
   * simulNN: pipeline tools using previous module

.. epigraph::

    A last directory, smallNetworkSimul shows example of simulation of ode solving made with the previous API.

Installation

.. epigraph::

    For installation, please clone the github directory, after creating a virtual environment (we suggest a conda one) and having activated it
    one can run the following instruction to obtain all necessary files.
    Please note that you need to have a proper cuda installation linked with tensorflow to use this API.
    If you have any trouble concerning the installation, I recommend using conda. It might not find all package in casual channel so you will need to find these channel by yourself. For example for mkl, I used conda install -c intel mkl-fft, which use the intel channel

.. code-block:: [python]

    pip install requirements.txt
    conda install requirements.txt


First steps

.. epigraph::

    To train a bio-chemical algorithm, one can take inspiration on the tensorflowTraining file in simulNN.
    You can also simply use the given function, which will train a tensorflow neural network, and then test this network by solving the ODE for all test example:

.. code-block:: [python]

    from simulOfBioNN.simulNN.tensorflowTraining import train
    from simulOfBioNN.simulNN.launcher import launch
    import numpy as np
    import pandas
    weightDir,acc,x_test,y_test,nnAnswer=train()
    launch(x_test_save,y_test2,nnAnswer,weightDir,layerInit = 10**(-8),enzymeInit = 10**(-6))

.. epigraph::

    You can also only simulate the ODE solve.
    In this case we have multiple mode of output:
    Possible modes:
        "verbose":
            display starting and finishing message
        "time":
            saving of time.
        "ouputEqui":
            save of the last value reached by the integrator
        "outputPlot":
            save all value reached on time steps.

    The main function to launch simulation is executeSimulation in the simulNN.simulator module.
    A detailed example of its use can be found in smallNetworkSimul directory, or in the developmentTest, under testDifferentOdeSolver.py file
