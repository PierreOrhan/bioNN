'''
    In this little experiment we define the following strategy:
        given a set of hyperplans in R^d we can obtain a dataset of input concentration from the voronoi diagram
        The neuronal network is supervisingly trained to retrieve the voronoi diagram from these concentrations.
'''
from scipy.spatial import Voronoi,voronoi_plot_2d
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np

class VoronoiSet():
    # def __init__(self,dimensions=1,equations={},spacebound=[(0,1)]):
    #     '''
    #     create our data set: converts the given equation to a set of constraints.
    #     :param dimensions: gives the space dimension
    #     :param equations: equations should be a dictionnary of elements:
    #         [equation,Xmin,Xmax], where Xmin and Xmax are the min/max value for X, defining the space of values
    #                                     equation should be a couple (w,b) representing the hyperplan definition
    #     .:param Spacebound: the space maximums and minimums
    #     '''
    #     self.dim=dimensions


    def __init__(self,inputPoints):
        '''
        Create the voronoi diagrams from some input points, which should be taken as barycenter of the each voronoi regions
        :param inputPoints: the barycenters from each voronoi regions
        '''
        try:
            assert len(inputPoints)>2
        except:
            print("please provide at least 2 points")
            raise
        self.tree=cKDTree(inputPoints)
        self.dim=len(inputPoints[0])
        if(self.dim==2):
            voronoi_plot_2d(Voronoi(inputPoints))
            plt.show()

        self.min = np.min(inputPoints,axis=0)
        self.max = np.max(inputPoints,axis=0)
        print(self.min)
        print(self.max)

    def generate(self,numberOfpoint):
        X=np.random.rand(numberOfpoint,self.dim)*self.max+self.min
        Y=self.tree.query(X)[1] #gives the location of th closest point that initially created the kd tree
        return X,Y