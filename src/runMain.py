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
import ErgodicPlanner

def max_uncertainty(GPdata,numpoints=1):
    # return the x,y locations of the "numpoints" largest uncertainty values
    ind = np.argpartition(GPdata[3].flatten(), -numpoints)[-numpoints:]
    newpointx=GPdata[0].flatten()[ind]
    newpointy=GPdata[1].flatten()[ind]
    
    return np.array([newpointx,newpointy]).T

def max_uncertainty_IS(GPdata,numpoints=1):
    # return the x,y locations of the "numpoints" largest uncertainty values
    GPISdat=implicitsurface(GPdata)
    ind = np.argpartition(GPISdat[2].flatten(), -numpoints)[-numpoints:]
    newpointx=GPdata[0].flatten()[ind]
    newpointy=GPdata[1].flatten()[ind]
    
    return np.array([newpointx,newpointy]).T

def max_MI(GPdata,numpoints=1):
    # TODO
    pass


##############################
# set boundary
rangeX = [-2,2]
rangeY = [-1,1]

# choose surface for simulation
surface=SixhumpcamelSurface


##############################
# Phase 1
###############################
# initialize measurement from stereo data
meas = getSimulatedStereoMeas(surface,rangeX=rangeX,rangeY=rangeY)


# add termination criterion
i=0

while i<10:
    # initialzie Gaussian process
    gpmodel = update_GP_het(meas)

    # Predections based on current GP estimate
    GPdata=eval_GP(gpmodel, rangeX, rangeY)
    
    # plot_belief(GPdata)

    # choose points to probe based on max uncertainty
    next_samples_points=max_uncertainty(GPdata,numpoints=2)

    # sample surface at points
    measnew= getSimulatedProbeMeas(surface, next_samples_points)

    # add new measurements to old measurements
    meas=np.append(meas,measnew,axis=0)

    i=i+1

plot_error(surface, gpmodel, rangeX, rangeY)

##############################
# Phase 2
###############################

