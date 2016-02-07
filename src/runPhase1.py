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
plot_data = None

while i<5:
    print "iteration = ", i

    # initialzie Gaussian process
    gpmodel = update_GP_sparse(meas, numpts=5)

    # Predections based on current GP estimate
    GPdata=eval_GP(gpmodel, rangeX, rangeY)
    
    # plot_belief(GPdata)

    # choose points to probe based on max uncertainty
    next_samples_points=max_uncertainty(GPdata,numpoints=2)

    # sample surface at points
    measnew= getSimulatedProbeMeas(surface, next_samples_points)

    # add new measurements to old measurements
    meas=np.append(meas,measnew,axis=0)

    plot_data = plot_error(surface, gpmodel, rangeX, rangeY, plot_data)
    
    i=i+1

plt.show(block=True)

##############################
# Phase 2
###############################

