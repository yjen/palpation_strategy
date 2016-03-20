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
from plotscripts import *

##############################
def evalerror(surface, workspace, mean):
    # choose points to compare
    xx=workspace.xx
    yy=workspace.yy

    mean = gridreshape(mean,workspace)
    x = workspace.xlin
    y = workspace.ylin

    interp=getInterpolatedGTSurface(surface, workspace)
    # interp=getInterpolatedObservationModel(surface)

    GroundTruth = interp(x,y)
    # integrate error
    dx = workspace.bounds[0][1]/np.float(workspace.res)
    #this is the incorrect error calculcation
    # error = np.sum(np.sqrt((GroundTruth-np.squeeze(mean))**2))*dx**2
    # return error / 10e6 # error/mm^2 to error/m^2 conversion

    numPoints = len(x)*len(y)
    error = np.sqrt(np.sum((GroundTruth-np.squeeze(mean))**2)/numPoints)    
    
    return error # this is in mm

def run_single_phase1_experiment(surfacename, method, disparityMeas=None, block=False, stops=1.343, shouldPlot=True):
    # set workspace boundary
    bounds=((0,200),(0,200))
    # choose surface for simulation

    # grid resolution: should be same for plots, ergodic stuff
    gridres = 200

    # initialize workspace object
    workspace = Workspace(bounds,gridres)

    # select surface to simulate
    surface = "image_pairs/"+ surfacename

    planner=''

    # acquisition functions:  MaxVar_GP, UCB_GP, EI_GP, UCB_IS, EI_IS
    AcFunctionSet = {"maxVarGrad", "maxVar", "UCB_dGPIS", "UCB_dGPIS2", "UCB_GP"}

    if method=='maxVarGrad':
        AcFunction=MaxVar_plus_gradient
        Acfunctionname="MaxVar_plus_gradient"
    elif method=='maxVar':
        AcFunction=MaxVar_GP
        Acfunctionname="MaxVar_GP"
    elif method=='UCB_dGPIS':
        AcFunction=UCB_dGPIS
        Acfunctionname="UCB_dGPIS"
    elif method=='UCB_dGPIS2':
        AcFunction=UCB_dGPIS2
        Acfunctionname="UCB_dGPIS2"        
    elif method=='UCB_GP':
        AcFunction=UCB_GP
        Acfunctionname="UCB_GP"
    elif method=='random':
        AcFunction=MaxVar_GP #just filler
        Acfunctionname="random"
    else:
        print "ERROR: invalid acquisition method specified in runPhase1.py!"
        sys.exit(1)
    
    if shouldPlot:
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
    errors=[]
    sigma = 1000.0

    # if method in {"random","UCB_GP", "UCB_dGPIS", "UCB_dGPIS2"}:
    #     maxIters = 200
    # else:
    #     maxIters = 50
    maxIters = 100        

    while i < maxIters: # or np.max(sigma) > stops:
        print "iteration =", i
        if i==0:
            # initialize measurement from stereo data
            if disparityMeas is None:
                disparityMeas = getSimulatedStereoMeas(surface, workspace, plot=False, block=block)                

            nStereoMeas = np.shape(disparityMeas)[0]            
            meas = np.copy(disparityMeas)

            gpmodel = update_GP_ph1(meas, nStereoMeas,  method='heteroscedastic')

            #probe at some random points in first iteration
            if method in {"UCB_dGPIS","UCB_dGPIS2","UCB_GP" }:
                numpoints_begin = 10
            else:
                numpoints_begin = 1
            
            next_samples_points = randompoints(bounds, numpoints_begin)
            sampled_points.append(next_samples_points)
            meastouchonly = getSimulatedProbeMeas(surface, workspace, next_samples_points)
            meas = np.append(meas,meastouchonly,axis=0)
            measures.append(meastouchonly)
        else:
            # add new measurements to old measurements
            meastouchonly = np.append(meastouchonly,measnew,axis=0)
            #    add new measurements to old measurements
            meas = np.append(meas,measnew,axis=0)
       
        # import IPython
        # IPython.embed()

        # update Gaussian process
        if i>0:
            gpmodel = update_GP_ph1(meas, nStereoMeas,  method='heteroscedastic')

            #call update function and pass in the existing model and it will utilize only the new measurements
            # gpmodel = update_GP_ph1(meastouchonly, method='heteroscedastic', model=gpmodel)

        # evaluate mean, sigma on a grid
        mean, sigma = get_moments(gpmodel, workspace.x)
        means.append(np.mean(mean))
        sigmas.append(np.max(sigma))
        # print np.max(sigma)

        # choose points to probe based on max uncertainty
        if method in {"maxVarGrad", "maxVar"}:        
            xgrid, AqcuisFunction = AcFunction(gpmodel, workspace)
        elif method in {"UCB_dGPIS"}:
            if surfacename in {"smooth_sin1_st", "smooth_sin1_spec"}:
                xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, acquisition_par=[0.99,0.95])                
            else:
                xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, acquisition_par=[0.95,0.7])                
        elif method in {"UCB_dGPIS2"}:
            if surfacename in {"smooth_sin1_st", "smooth_sin1_spec"}:
                xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, acquisition_par=[0.98,0.9, 0.07])                
            else:
                xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, acquisition_par=[0.95,0.4, 0.37])                
        elif method in {"UCB_GP"}:
            if surfacename in {"smooth_sin1_st", "smooth_sin1_spec"}:
                xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, acquisition_par=0.95)                
            else:
                xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, acquisition_par=0.8)                

        # choose points to probe based on maxima of acquisition function
        if method in AcFunctionSet:
            next_samples_points = maxAcquisition(workspace, AqcuisFunction,
                                                   numpoints=1)
        elif method == "random":
            # can exit at i=2, have already sampled hundred points
            if i>1: 
                print "Random Acquisition Function, have already sampled",maxIters, "points" 
                break
            xgrid, AqcuisFunction = AcFunction(gpmodel, workspace)
            next_samples_points =  randompoints(bounds, maxIters)
            
        else:
            print "ERROR: invalid method in runPhase1.py!"
            sys.exit(1)
        
        # sample surface at points
        sampled_points.append(next_samples_points)
        measnew = getSimulatedProbeMeas(surface, workspace, next_samples_points)
        measures.append(measnew)
        error = evalerror(surface, workspace, mean)
        if i==0: 
            print"only disparity error:", error
        elif i==maxIters-1 or (method=="random" and i>0):
            print"Final error:", error
        errors.append(error)
                
        if (shouldPlot) and (i%10==0 or i==maxIters or method=="random"):
            # Plot everything
            time.sleep(0.0001)
            plt.pause(0.0001)
            # plot_data = plot_error(surface, workspace, mean, sigma, AqcuisFunction, meastouchonly, dirname=directory, data=plot_data, projection3D=False, iternum=i)
            # plot but not save
            plot_data = plot_error(surface, workspace, mean, sigma, AqcuisFunction, meastouchonly, dirname=None, data=plot_data, projection3D=False, iternum=i)

        i=i+1

    plt.show(block=block)
    if disparityMeas is not None:
        plt.close()

    
    return disparityMeas, means, sigmas, sampled_points, measures, errors, i


if __name__ == "__main__":

    # surfacename = "smooth_sin1_spec"
    surfacename = "smooth_sin1_text"
    #Note have to use different parameters for specularity. 
    # In that case disparity calculations fail, hence need to increase weight on the variance term to encourage exploration.

    # run_single_phase1_experiment(surfacename, method="random", block=True)
    # run_single_phase1_experiment(surfacename, method="maxVar", block=True)
    # run_single_phase1_experiment(surfacename, method="UCB_GP", block=True)
    # run_single_phase1_experiment(surfacename, method="UCB_dGPIS", block=True)
    run_single_phase1_experiment(surfacename, method="UCB_dGPIS2", block=True)
    

    # todo: stereo variance should dependd on which model we are testing on -- based on goodness of fit of disparity calc


