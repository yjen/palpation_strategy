import sys, os
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
#import \import time
import time
from simUtils import *
from utils import *
from GaussianProcess import *
# import ErgodicPlanner


from Planner import *




##############################
# set boundary
rangeX = [0,50]
rangeY = [0,50]
# set workspace boundary

bounds=((0,50),(0,50))
# choose surface for simulation

#grid resolution: should be same for plots, ergodic stuff
gridres = 20

#initialize workspace object
workspace = Workspace(bounds,gridres)


surfacename="smooth3"
surface = "image_pairs/"+ surfacename 

planner=''
# acquisition functions:  MaxVar, UCB_GP, EI_GP, UCB_IS, EI_IS
AcFunction='MaxVar'

directory='phase1_'+surfacename+'_'+planner
if not os.path.exists(directory):
    os.makedirs(directory)
##############################
# Phase 1
###############################
# initialize measurement from stereo data
meas = getSimulatedStereoMeas(surface,workspace)


# add termination criterion
i=0
plot_data = None

while i<20:
    print "iteration = ", i

    # initialzie Gaussian process
    gpmodel = update_GP(meas)
    mean, sigma = get_moments(gpmodel, workspace.x)

    # Predections based on current GP estimate
    # GPdata=eval_GP(gpmodel, bounds)
    
    # plot_belief(GPdata)

    # choose points to probe based on max uncertainty
    if AcFunction == 'MaxVar':
        xgrid, AqcuisFunction = MaxVar_GP(gpmodel, workspace.x)

    next_samples_points = maxAcquisition(workspace, AqcuisFunction,
                                           numpoints=10)
    # sample surface at points
    measnew= getSimulatedProbeMeas(surface, workspace, next_samples_points)

    # add new measurements to old measurements
    meas=np.append(meas,measnew,axis=0)
    time.sleep(0.05)
    plt.pause(0.0001) 
    plot_data = plot_error(surface, workspace, mean, sigma, meas, directory, plot_data,iternum=i)
    
    i=i+1

plt.show(block=True)

##############################
# Phase 2
###############################

