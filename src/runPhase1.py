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

# set workspace boundary
bounds=((0,50),(0,50))
# choose surface for simulation

# grid resolution: should be same for plots, ergodic stuff
gridres = 100

# initialize workspace object
workspace = Workspace(bounds,gridres)

# select surface to simulate
surfacename="smooth7"
surface = "image_pairs/"+ surfacename 

planner=''

# acquisition functions:  MaxVar_GP, UCB_GP, EI_GP, UCB_IS, EI_IS
AcFunction=MaxVar_GP#MaxVar_plus_gradient
Acfunctionname="MaxVar_GP"

directory='phase1_'+surfacename+'_'+Acfunctionname
if not os.path.exists(directory):
    os.makedirs(directory)
##############################
# Phase 1
###############################


i=0
plot_data = None

# TODO: add termination criterion instead of counting i (when estimate stops changing)

while i<100:
    print "iteration = ", i
    if i==0:
        # initialize measurement from stereo data
        meas = getSimulatedStereoMeas(surface,workspace)
        next_samples_points =  randompoints(bounds, 10)
        meastouchonly = getSimulatedProbeMeas(surface, workspace, next_samples_points)
    else:
        # add new measurements to old measurements
        meastouchonly = np.append(meastouchonly,measnew,axis=0)
        #    add new measurements to old measurements
        meas = np.append(meas,measnew,axis=0)
   
    # update Gaussian process
    gpmodel = update_GP(meas, method='het')

    # evaluate mean, sigma on a grid
    mean, sigma = get_moments(gpmodel, workspace.x)

    # choose points to probe based on max uncertainty
    xgrid, AqcuisFunction = AcFunction(gpmodel, workspace)

    next_samples_points = maxAcquisition(workspace, AqcuisFunction,
                                           numpoints=3)
    # sample surface at points
    measnew = getSimulatedProbeMeas(surface, workspace, next_samples_points)

    # Plot everything
    time.sleep(0.0001)
    plt.pause(0.0001) 
    plot_data = plot_error(surface, workspace, mean, sigma, AqcuisFunction,  meas, directory, plot_data, projection3D=True, iternum=i)
    
    i=i+1

plt.show(block=True)



