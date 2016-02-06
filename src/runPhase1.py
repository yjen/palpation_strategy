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
# import ErgodicPlanner


from Planner import *




##############################
# set boundary
rangeX = [0,50]
rangeY = [0,50]

# choose surface for simulation
#surface=SixhumpcamelSurface
surface = "image_pairs/smooth3"


##############################
# Phase 1
###############################
# initialize measurement from stereo data
meas = getSimulatedStereoMeas(surface,rangeX=rangeX,rangeY=rangeY)


# add termination criterion
i=0

while i<5:
    print "iteration = ", j

    # initialzie Gaussian process
    gpmodel = update_GP_sparse(meas)

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

