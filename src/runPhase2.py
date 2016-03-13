import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

from simUtils import *
from plotscripts import *

from utils import *
from GaussianProcess import *
import ErgodicPlanner
from Planner import *
# import rospy
import pickle

##############################
# Phase 2
###############################
def evalerror_ph2(tumor,workspace,mean,variance,level):
    # see: http://toblerity.org/shapely/manual.html
    boundaryestimate = getLevelSet (workspace, mean, level)
    boundaryestimateupper = getLevelSet (workspace, mean+variance, level)
    boundaryestimatelower = getLevelSet (workspace, mean-variance, level)
    #boundaryestimate=boundaryestimateupper
    GroundTruth = np.vstack((tumor,tumor[0]))
    GroundTruth = Polygon(GroundTruth)
    # print GroundTruth
    if len(boundaryestimate)>3:
        try: 
            boundaryestimate=Polygon(boundaryestimate)
            boundaryestimate=boundaryestimate.buffer(-offset)
            err=GroundTruth.symmetric_difference(boundaryestimate)
            err=err.area

        except TopologicalError:
            err=.100
            tumorleft=.100
    else:
        err=.100
        tumorleft=.100
    return err, 0

def run_single_phase2_simulation(phantomname, dirname, AcFunction=MaxVar_GP, control='Max', plot=False, smode='RecordedExp',iters=20):
    if smode=='RecordedExp': 
        getmeasurements=getRecordedExperimentalStiffnessMeas
        bounds=((.0,0.0229845803642),(.0,0.0577416388862))
    elif smode=='Exp':
        getmeasurements=getExperimentalStiffnessMeas
        from expUtils import *
        bounds=calculate_boundary("../scripts/env_registration.p")
    elif smode=='Sim':

        getmeasurements=getSimulateStiffnessMeas
        bounds=((-.04,.04),(-.04,.04))
    else: 
        print 'invalid mode!'
    
    if smode=='RecordedExp' or smode=='Exp': 
        UCB_GP_acpar=.7 # set parameters for acquiisition functions: balancing mean vs. var in prioritizing search
        UCB_GPIS_acpar=.2
        UCB_GPIS_implicit_acpar=[.2,.9]
        UCB_dGP_acpar=[7,.9]
        # GP_params= [6,.005,.0001,7] # parameters for gaussian process update
        GP_params= [14,.003,.02,63] # parameters for gaussian process update

    else:   #params for simulation
        UCB_GP_acpar=.13
        UCB_GPIS_acpar=.8
        UCB_GPIS_implicit_acpar=[4,.8]
        UCB_dGP_acpar=[2,.5]
        GP_params= [.3,.007,1e-4,52]

    if AcFunction==UCB_GPIS:
        acquisition_par=UCB_GPIS_acpar
    elif AcFunction==UCB_GP:
        acquisition_par=UCB_GP_acpar
    elif AcFunction==UCB_dGP:
         acquisition_par=UCB_dGP_acpar
    elif AcFunction==UCB_GPIS_implicitlevel:
        acquisition_par=UCB_GPIS_implicit_acpar
    else:
        acquisition_par=0
    # print acquisition_par
    # grid resolution: how finely mean, sigma, etc should be calculated
    gridres = 200

    # initialize workspace object
    workspace = Workspace(bounds,gridres)
    # print workspace.bounds
    plotSimulatedStiffnessMeas(phantomname, workspace, ypos=0, sensornoise = .05)

    # set level set to look for-- this should correspond to something, max FI?
    level = .8*(measmax-measmin)+measmin #pick something between min/max deflection
    print "level=",level
    # print level
    plot_data = None
    means = []
    sigmas = []
    acqvals = []
    healthyremoveds=[]
    tumorlefts=[]
    sampled_points = []
    measures = []

    directory=dirname
    if not os.path.exists(directory):
        os.makedirs(directory)

    ###############
    #Initializing
    ###############
    next_samples_points = randompoints(bounds, 10) #randompoints(bounds, 100)
    # next_samples_points=solve_tsp_dynamic(next_samples_points)    # collect initial meausrements
    # if mode =='RecordedExp':
    #     meas = getRecordedExperimentalStiffnessMeas(next_samples_points)
    # if mode =='Exp':
    #     meas = getExperimentalStiffnessMeas(next_samples_points)
    # else:        
    #     meas = getSimulateStiffnessMeas(phantomname, next_samples_points)
    meas = getmeasurements(next_samples_points,phantomname)

    measnew = meas
    print  'f'
    for j in range (iters): #(1,100,1)

        # print "iteration = ", j
        # collect measurements

        # update the GP model    
        gpmodel = update_GP(meas, params=GP_params)

        # use GP to predict mean, sigma on a grid
        mean, sigma = get_moments(gpmodel, workspace.x)

        # evaluate selected aqcuisition function over the griddUCB_GPIS_implicit_acpar
        xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, level=level, acquisition_par=acquisition_par)

        # save data to lists
        acqvals.append(AqcuisFunction)
        means.append(mean)
        sigmas.append(sigma)
        measures.append(meas)
        healthyremoved,tumorleft= evalerror_ph2(phantomname, workspace, mean,sigma,level)
        healthyremoveds.append(healthyremoved)
        tumorlefts.append(tumorleft)

        # select next sampling points. for now, just use Mac--dMax and Erg need work.
        bnd=getLevelSet (workspace, mean, level)

        if control=='Max':         
            
            next_samples_points = batch_optimization(gpmodel, workspace, AcFunction, 5, meas[-1][0:2], GP_params,level=level, acquisition_par=acquisition_par)
        # elif control=='dMax':
        #     next_samples_points = dmaxAcquisition(gpmodel, workspace, AcFunction, meas[-1][0:2],
        #                                         numpoints=10, level=level)
        else:
            next_samples_points=randompoints(bounds,1)
            print 'RANDOM'
        # print next_samples_points
        # Plot everything
        if plot==True:
            time.sleep(0.0001)
            plt.pause(0.0001)  
            plot_data = plot_beliefGPIS(phantomname, workspace, mean, sigma,
                                      AqcuisFunction, measnew,
                                      directory, [healthyremoveds,tumorlefts],plot_data,level=level,
                                      iternum=j, projection3D=False)
        measnew = getmeasurements(next_samples_points,phantomname)
        # print measnew.max()
        # if mode=='Exp':
        #     measnew = getExperimentalStiffnessMeas(next_samples_points)
        # if mode=='RecordedExp':
        #     measnew = getRecordedExperimentalStiffnessMeas(next_samples_points)#np.zeros(next_samples_points.shape)
        # else:
        #     measnew = getSimulateStiffnessMeas(phantomname, next_samples_points)

        #   concatenate measurements to prior measurements
        meas = np.append(meas,measnew,axis=0)

    # plt.show(block=False)
    return means, sigmas, acqvals, measures, healthyremoveds, tumorlefts, j, gpmodel


if __name__ == "__main__":
    dirname='tt'
    run_single_phase2_simulation(phantomsquareGT, dirname, AcFunction=UCB_GPIS_implicitlevel, control='Max', plot=True, smode='RecordedExp',iters=100)
    # outleft,outrem,aclabellist=run_phase2_full()
    # plot_error(outrem,outleft,aclabellist)

#TODO:
# To run Phase 2 on the robot, the function getExperimentalStiffnessMeas, 
# in Gaussian Process.py, needs to be written to command the robot and collect measurements
# from expUtils import *

# def run_single_phase2_simulation(AcFunction, dirname, control='Max', block=False, stops=0.38, plot=False):
    
#     bounds = calculate_boundary("../scripts/env_registration.p")

#     # grid resolution: should be same for plots, ergodic stuff
#     gridres = 200

#     # initialize workspace object
#     workspace = Workspace(bounds,gridres)

#     # set level set to look for-- this should correspond to something, max FI?
#     level=.8 #pick something between min/max deflection

#     # acquisition functions:  MaxVar_GP, UCB_GP, EI_GP, UCB_GPIS, EI_IS,MaxVar_plus_gradient
#     # AcFunction=UCB_GPIS
#     # Acfunctionname="UCB_GPIS"

#     plot_data = None
#     means = []
#     sigmas = []
#     acqvals = []
#     sampled_points = []
#     measures = []
#     errors=[]
#     directory = dirname #phase2_'+'_'+control+'_'+Acfunctionname

#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     ###############
#     #   Initializing
#     ###############
#     next_samples_points = randompoints(bounds, 5) 
#     # collect initial meausrements
#     meas = getExperimentalStiffnessMeas(next_samples_points)

#     for j in range (50): #(1,100,1)
#         print "iteration = ", j
#         # collect measurements
       
#         measnew = getExperimentalStiffnessMeas(next_samples_points)
#         # concatenate measurements to prior measurements

#         # import IPython; IPython.embed()
#         meas = np.append(meas,measnew,axis=0)

#         # update the GP model    
#         gpmodel = update_GP(meas)

#         # use GP to predict mean, sigma on a grid
#         mean, sigma = get_moments(gpmodel, workspace.x)
#         means.append(np.mean(mean))
#         sigmas.append(np.max(sigma))
#         # evaluate selected aqcuisition function over the grid
#         xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, level)
#         acqvals.append(AqcuisFunction)

#         # select next sampling points. for now, just use Mac--dMax and Erg need work.
#         if control=='Max':            
#             next_samples_points = maxAcquisition(workspace, AqcuisFunction,
#                                                  numpoints=3, level=level)
#         if control=='dMax':
#            next_samples_points = dmaxAcquisition(gpmodel, workspace, AcFunction, meas[-1][0:2],
#                                                numpoints=3, level=level)
#         else:
#             print 'RANDOM'
#             next_samples_points=randompoints(bounds,1)
        
#         # if next_samples_points.shape[0]>1:

#         time.sleep(0.0001)
#         plt.pause(0.0001)  

#         # Plot everything
#         plot_data = plot_beliefGPIS(phantomname,workspace,mean,sigma,
#                                     AqcuisFunction,meas,
#                                     directory,plot_data,errors,level=level,
#                                     iternum=j,projection3D=False)
#         # Save everything--this needs to be debugged
#         # prename=directory+'/'
#         # save_p2_data(prename+'mean'+str(j),mean)
#         # save_p2_data(prename+'sigma'+str(j),sigma)
#         # save_p2_data(prename+'AqcuisFunction'+str(j),AqcuisFunction)
#         # save_p2_data(prename+'meas'+str(j),meas)

#     plt.show(block=True)
#     return means, sigmas, acqvals, measures, errors, j

# # if __name__ == "__main__":
# #     planning(verbose=True)




# if __name__ == "__main__":
#     # run_single_phase2_simulation("rantumor", "maxAcquisition", block=True, plot=True,dirname='test')
#     run_single_phase2_experiment(UCB, dirname='exp', plot=True)

# ##############
# #Initializing
# ###############
# if control == 'Erg':
#     # initialize ergodic cplanner
#     xdim = 2
#     udim = 2
#     LQ = ErgodicPlanner.lqrsolver(xdim, udim, Nfourier=30, wlimit=(4.), res=gridres,
#                                 barrcost=50, contcost=.03, ergcost=1000)
#     # initialize stiffness map (uniform)
#     initpdf = ErgodicPlanner.uniform_pdf(LQ.xlist)
#     next_samples_points = ErgodicPlanner.ergoptimize(LQ, initpdf, xinit,
#                                                    maxsteps=15,plot=True)
# else:
#     next_samples_points = randompoints(bounds, 10)
    
#     # collect initial meausrements
# # meas = getSimulateStiffnessMeas(tumorpoly, next_samples_points)
#     meas = getExperimentalStiffnessMeas(next_samples_points)


# for j in range (10): #(1,100,1)
#     print "iteration = ", j
#     # collect measurements
#     # measnew = getSimulateStiffnessMeas(tumorpoly,
#     #                                    next_samples_points)
#     # to run experiment instead of simulation:
#     measnew = getExperimentalStiffnessMeas(next_samples_points)
#     # concatenate measurements to prior measurements
#     import IPython; IPython.embed()
#     meas = np.append(meas,measnew,axis=0)
    
#     # update the GP model    
#     gpmodel = update_GP(meas)

#     # use GP to predict mean, sigma on a grid
#     mean, sigma = get_moments(gpmodel, workspace.x)

#     # find mean points closest to the level set
#     boundaryestimate = getLevelSet (workspace, mean, level)
#     GPIS = implicitsurface(mean,sigma,level)

#     # evaluate selected aqcuisition function over the grid
#     xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, level)

#     # select next sampling points. for now, just use Mac--dMax and Erg need work.
#     if control=='Max':     
#         next_samples_points = maxAcquisition(workspace, AqcuisFunction,
#                                            numpoints=1)
#     elif control=='dMax':            
#         next_samples_points = dmaxAcquisition(workspace, gpmodel, AcFunction, xinit=meas[-1,0:2],
#                                            numpoints=5)
#     elif control=='Erg':
#         next_samples_points = ergAcquisition(workspace, AqcuisFunction,
#                                              LQ, xinit=meas[-1,0:2])
#     else:
#         print 'RANDOM'
#         next_samples_points=randompoints(bounds,1)
        
#     time.sleep(0.0001)
#     plt.pause(0.0001)  

#     # Plot everything
#     plot_data = plot_beliefGPIS(tumorpoly,workspace,mean,sigma,
#                                 GPIS,AqcuisFunction,meas,
#                                 directory,plot_data,level=level,
#                                 iternum=j)

# plt.show(block=True)

# # if __name__ == "__main__":
# #     planning(verbose=True)




