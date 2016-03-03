#!/usr/bin/env python
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# import matplotlib.path as path

#from utils import *
# from getMap import *
#from sensorModel import *
# import cplex, gurobipy
#sys.path.append("..")
import time
from simUtils import *
from utils import *
from GaussianProcess import *
# import ErgodicPlanner
from Planner import *

##############################


def run_single_phase1_experiment(surfacename, method, disparityMeas=None, block=False, stops=0.38):
    # set workspace boundary
    bounds=((0,50),(0,50))
    # choose surface for simulation

    # grid resolution: should be same for plots, ergodic stuff
    gridres = 100

    # initialize workspace object
    workspace = Workspace(bounds,gridres)

    # select surface to simulate
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
    means = []
    sigmas = []
    sampled_points = []
    measures = []
    sigma = 1000.0


    # TODO: add termination criterion instead of counting i (when estimate stops changing)

    while np.max(sigma) > stops: #i < 10:
        print "iteration = ", i
        if i==0:
            # initialize measurement from stereo data
            if disparityMeas is None:
                disparityMeas = getSimulatedStereoMeas(surface, workspace, block)
            meas = np.copy(disparityMeas)
            next_samples_points = randompoints(bounds, 10)
            sampled_points.append(next_samples_points)
            meastouchonly = getSimulatedProbeMeas(surface, workspace, next_samples_points)
            measures.append(meastouchonly)
        else:
            # add new measurements to old measurements
            meastouchonly = np.append(meastouchonly,measnew,axis=0)
            #    add new measurements to old measurements
            meas = np.append(meas,measnew,axis=0)
       
        # update Gaussian process
        gpmodel = update_GP(meas, method='het')

        # evaluate mean, sigma on a grid
        mean, sigma = get_moments(gpmodel, workspace.x)
        means.append(np.mean(mean))
        sigmas.append(np.max(sigma))
        print np.max(sigma)

        # choose points to probe based on max uncertainty
        xgrid, AqcuisFunction = AcFunction(gpmodel, workspace)

        if method == "maxAcquisition":
            next_samples_points = maxAcquisition(workspace, AqcuisFunction,
                                                   numpoints=10)
        elif method == "random":
            next_samples_points =  randompoints(bounds, 10)
        else:
            print "ERROR: invalid method in runPhase1.py!"
            sys.exit(1)
        
        # sample surface at points
        sampled_points.append(next_samples_points)
        measnew = getSimulatedProbeMeas(surface, workspace, next_samples_points)
        measures.append(measnew)

        # Plot everything
        time.sleep(0.0001)
        plt.pause(0.0001) 
        plot_data = plot_error(surface, workspace, mean, sigma, AqcuisFunction, meastouchonly, directory, plot_data, projection3D=True, iternum=i)
        
        i=i+1

    plt.show(block=block)
    if disparityMeas is not None:
        plt.close()

    return disparityMeas, means, sigmas, sampled_points, measures, i



if __name__ == "__main__":
    run_single_phase1_experiment("smooth1", "maxAcquisition", block=True)


