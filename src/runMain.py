import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# import matplotlib.path as path

#from utils import *
# from getMap import *
#from sensorModel import *
# import cplex, gurobipy
#import sys
#sys.path.append("..")
#import simulated_disparity
from simUtils import *
from utils import *
from GaussianProcess import *


def max_uncertainty(GPdata,numpoints=1):
    # return the x,y locations of the "numpoints" largest uncertainty values
    ind = np.argpartition(GPdata[3].flatten(), -numpoints)[-numpoints:]
    newpointx=GPdata[0].flatten()[ind]
    newpointy=GPdata[1].flatten()[ind]
    
    return np.array([newpointx,newpointy]).T

def max_MI(GPdata,numpoints=1):
    # TODO
    pass



# set boundary
rangeX = [-2,2]
rangeY = [-1,1]

# choose surface for simulation
surface=GaussianSurface

# initialize measurement from stereo data
meas = getSimulatedStereoMeas(GaussianSurface,rangeX=rangeX,rangeY=rangeY)


# add termination criterion
i=0

while i<10:
    # initialzie Gaussian process
    gpmodel = update_GP_het(meas)

    # Predections based on current GP estimate
    GPdata=eval_GP(gpmodel, rangeX, rangeY)
    
    #plot_belief(GPdata)

    # choose points to probe based on max uncertainty
    next_samples_points=max_uncertainty(GPdata,numpoints=2)

    # sample surface at points
    measnew= getSimulatedProbeMeas(surface, next_samples_points)

    # add new measurements to old measurements
    meas=np.append(meas,measnew,axis=0)

    i=i+1

plot_error(surface, gpmodel, rangeX, rangeY)

# plot_belief(GPdata)

# def estimate(verbose=False):
#     """
#     Input Options Description
#     """

#     # Initialize Belief -- this is akin to gettig camera estimate
#     # x,y vectors, xx,yy,z- is a matrix
#     x, y, xx, yy, z = getMap(modality=1)

#     if verbose:
#         plotBelief(xx, yy, z)

#     # add noise and normalize to get belief
#     z += 0.05 * np.random.standard_normal(z.shape)
#     if verbose:
#         plotBelief(xx, yy, z)

#     # Calculate FIM

#     # calculate EID

#     # Planning Trajectory

#     # greedy Local
#     # greedy FI

#     # tEID -- refer the paper.

#     # Collect Observations- simulate
#     # Sensor simulator
#     h = sensorHeight(z, probePos)
#     # Belief Update

#     # update information  - Fisher information?

#     # To keep all the plots made during the execution.
#     plt.show()

# if __name__ == "__main__":
#     planning(verbose=True)
