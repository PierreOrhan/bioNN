from numpy import zeros

def _getConfig(config):
    """
        Just gives dictionary of config
    :param config: 0 : large test, 1: unit test
    :return:
    """
    Batches=["sparsity","pyramid","sparsePyramid"]
    colors={
        "sparsity":"r",
        "pyramid":"g",
        "sparsePyramid":"b"
    }
    colors2={
        1:[[255,0,0],[255,30,30],[255,60,60]],
        2:[[0,0,255],[30,30,255],[60,60,255]],
        4:[[0,255,0],[30,255,30],[60,255,60]]
    }
    colors3={
        1:"r",
        2:"g",
        4:"b"
    }
    if(config==0):
        """
         Pareto experiment:
        Having found good result against mnist and fashion mnist, we create a big training session.
        Goals:
            obtain a lot of different model to create a pareto plot: ACC v.s obtained weight
        We fix :
            rescaleFactor = 4 given an input size of 196 voxels
            epochs=5
            R=2 ==> here the repetition is to have different initial configuration for the distribution of sparsity.
        We tested the following strategy:
            batch sparsity:
                Nb units => constant : [15,30,50,70]
                Nb Layer => 4
                fractionZero => We keep high sparsity in the beginning and then reduce it:
                                [0.9,0.9,0.8,0.7]
                                [0.9,0.8,0.7,0.6]
                                [0.9,0.5,0.3,0.3]
                                [0.9,0.3,0.3,0.3]
            batch units  => We build a pyramidal strategy for the network:
                Nb units: => [70,35,10]
                             [40,10,10]
                             [50,25,10]
                             [100,50,10]
                Nb Layers: 3
                sparsity=> constant: [0.9,0.7,0.5]
            batch sparsPyramid => We build a strategy that combines the previous: the sparsity should be decreased with respect to the number of weights
                config: [0.9,0.8,0.7] [70,35,10]
                                          [40,10,10]
                                          [50,25,10]
                                          [100,50,10]
                        [0.9,0.8,0.5]  /idem
                        [0.9,0.8,0]  /idem
                        [0.9,0.5,0.3]  /idem
                        [0.9,0.5,0]  /idem
        
        """
        Sparsity={
            "sparsity":[[[0.9,0.9,0.8,0] for _ in range (4)],[[0.9,0.8,0.7,0] for _ in range (4)],[[0.9,0.5,0.3,0] for _ in range (4)],[[0.9,0.3,0.3,0] for _ in range (4)]],
            "pyramid":[[[0.9,0.9,0] for _ in range(4)],[[0.7,0.7,0] for _ in range(4)],[[0.5,0.5,0] for _ in range(4)]],
            "sparsePyramid": [[[0.9,0.8,0] for _ in range (4)],[[0.9,0.8,0] for _ in range (4)],[[0.9,0.8,0] for _ in range (4)],[[0.9,0.5,0] for _ in range (4)],[[0.9,0.5,0]for _ in range (4)]]
        }
        NbUnits={
            "sparsity":[[[15,15,15,10],[30,30,30,10],[50,50,50,10],[75,75,75,10]] for _ in range (4)],
            "pyramid":[[[70,35,10],[40,10,10],[50,25,10],[100,50,10]] for _ in range(3)],
            "sparsePyramid":[[[100,50,10],[70,35,10],[40,10,10],[50,25,10]] for _ in range(5)]
        }
        Layers={
            "sparsity":zeros(4)+4,
            "pyramid":zeros(3)+3,
            "sparsePyramid":zeros(5)+3,
        }
        return Batches,colors,colors2,colors3,Sparsity,NbUnits,Layers
    else:
        Sparsity={
            "sparsity":[[[0.9,0.9,0]]],
            "pyramid":[[[0.9,0.9,0]]],
            "sparsePyramid":[[[0.9,0.9,0]]]
        }
        NbUnits={
            "sparsity":[[[15,15,10]]],
            "pyramid":[[[15,15,10]]],
            "sparsePyramid":[[[15,15,10]]],
        }
        Layers={
            "sparsity":[3],
            "pyramid":[3],
            "sparsePyramid":[3],
        }
        return Batches,colors,colors2,colors3,Sparsity,NbUnits,Layers
