#!/usr/bin/env python
import sys, os, time, IPython
import numpy as np
import matplotlib.pyplot as plt
from runPhase1 import *
from Planner import *
from simUtils import *

def run_single_phase2_simulation(phantomname, dirname, AcFunction=MaxVar_GP, control='Max', plot=False):

    bounds=((-.04,.04),(-.04,.04))

    # grid resolution: should be same for plots, ergodic stuff
    gridres = 200

    # initialize workspace object
    workspace = Workspace(bounds,gridres)

    # set level set to look for-- this should correspond to something, max FI?
    level = .8 #pick something between min/max deflection

    plot_data = None
    means = []
    sigmas = []
    acqvals = []
    sampled_points = []
    measures = []

    directory=dirname
    if not os.path.exists(directory):
        os.makedirs(directory)

    ###############
    #Initializing
    ###############
    next_samples_points = randompoints(bounds, 10) #randompoints(bounds, 100)

    # collect initial meausrements

    meas = getSimulateStiffnessMeas(phantomname, next_samples_points)

    for j in range (30): #(1,100,1)
        # print "iteration = ", j
        # collect measurements
        measnew = getSimulateStiffnessMeas(phantomname, next_samples_points)

        #   concatenate measurements to prior measurements
        meas = np.append(meas,measnew,axis=0)

        # update the GP model    
        gpmodel = update_GP(meas)

        # use GP to predict mean, sigma on a grid
        mean, sigma = get_moments(gpmodel, workspace.x)

        # evaluate selected aqcuisition function over the grid
        xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, level=level)

        acqvals.append(AqcuisFunction)
        means.append(mean)
        sigmas.append(sigma)
        measures.append(meas)

        # select next sampling points. for now, just use Mac--dMax and Erg need work.
        if control=='Max':            
            next_samples_points = maxAcquisition(workspace, AqcuisFunction,
                                                    numpoints=1)
        elif control=='dMax':

            next_samples_points = dmaxAcquisition(gpmodel, workspace, AcFunction, meas[-1][0:2],
                                                numpoints=10, level=level)
        else:
            next_samples_points=randompoints(bounds,1)
    
        # Plot everything
        if plot==True:
            time.sleep(0.0001)
            plt.pause(0.0001)  
            plot_data = plot_beliefGPIS(phantomname, workspace, mean, sigma,
                                      AqcuisFunction, meas,
                                      directory, plot_data, level=level,
                                      iternum=j, projection3D=False)

    plt.show(block=True)
    return means, sigmas, acqvals, measures, j

def save_data(arr, dirname, name):
    directory=dirname

    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(directory+'/data_'+name, 'w')
    f.write(str(arr))
    f.close()
    return

tumors = [rantumor, rantumor]              # add another model ?
stops = [[6.4, 0.01, 3.6, 0.0],
         [0.0, 0.36, 0.0, 0.0]]                 # TODO change 0.0's (variance is not monotonically decreasing)
# textures = ["_lam", "_text", "_spec", "_st"]
    # lambert, texture, specular, specular + texture

aqfunctions = [MaxVar_GP, UCB_GP, UCB_GPIS]
aqfunctionsnames = ["MaxVar_GP", "UCB_GP", "UCB_GPIS","random"]

    # acquisition functions:  MaxVar_GP, UCB_GP, EI_GP, UCB_GPIS, EI_IS, MaxVar_plus_gradient
controls =["Max"]

def run_phase2_full():
    for i, tumor in enumerate(tumors):
        for j, acq in enumerate(aqfunctions):
            for m, cont in enumerate(controls):
                # if i*len(textures) + j != 0:        # Use this to run only the nth surface
                #     continue
                its = [0.0, 0.0]
                for k in range(2): # repeat experiment 5 times
                    # disparityMeas = None
                    # for l, method in enumerate(methods):
                    start = time.time()
                    dirname = str(i) + '_' + aqfunctionsnames[j] + '_' + cont
                    means, sigmas, acqvals, measures, num_iters = run_single_phase2_simulation(tumor, dirname, AcFunction=acq, control=cont, plot=True)
                    plt.close() 

                    end = time.time()
                    time_elapsed = end - start # in seconds
                    # plot or save/record everything
                    # its[k] += num_iters / 5.0
                    print measures
                    save_data([means, sigmas, acqvals, measures, num_iters, time_elapsed], dirname, str(k))
                    print k
    return



if __name__ == "__main__":
    run_phase2_full()
    # IPython.embed()


