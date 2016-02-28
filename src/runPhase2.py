import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

from simUtils import *

from utils import *
from GaussianProcess import *
import ErgodicPlanner
from Planner import *
# import rospy
import pickle

##############################
# Phase 2
###############################
#TODO:
# To run Phase 2 on the robot, the function getExperimentalStiffnessMeas, 
# in Gaussian Process.py, needs to be written to command the robot and collect measurements

Sim=True
phantomnamed='rantumor'
phantomname = rantumor

if Sim==True:
    # set workspace boundary
    bounds=((-.04,.04),(-.04,.04))
else: #experiment
    bounds = calculate_boundary("../scripts/env_registration.p")

    from expUtils import *

print(bounds)

# grid resolution: should be same for plots, ergodic stuff
gridres = 200

# initialize workspace object
workspace = Workspace(bounds,gridres)

# set level set to look for-- this should correspond to something, max FI?
level=.5 #pick something between min/max deflection

# control choices: Max or Erg or dMax. Erg and dMax are still in development.
control='Max'

# acquisition functions:  MaxVar_GP, UCB_GP, EI_GP, UCB_GPIS, EI_IS,MaxVar_plus_gradient
AcFunction=UCB_GPIS
Acfunctionname="UCB_GPIS"

plot_data = None

directory='phase2_'+phantomnamed+'_'+control+'_'+Acfunctionname

if not os.path.exists(directory):
    os.makedirs(directory)

###############
#Initializing
###############
next_samples_points = randompoints(bounds, 100)

# collect initial meausrements

if Sim==True:
    meas = getSimulateStiffnessMeas(phantomname, next_samples_points)
else:
    meas = getExperimentalStiffnessMeas(next_samples_points)

for j in range (30): #(1,100,1)
    print "iteration = ", j
    # collect measurements
    if Sim==True:
        measnew = getSimulateStiffnessMeas(phantomname, next_samples_points)
    else:
        # to run experiment instead of simulation:
        measnew = getExperimentalStiffnessMeas(next_samples_points)
    # concatenate measurements to prior measurements

    # import IPython; IPython.embed()
    meas = np.append(meas,measnew,axis=0)

    # update the GP model    
    gpmodel = update_GP(meas)

    # use GP to predict mean, sigma on a grid
    mean, sigma = get_moments(gpmodel, workspace.x)

    # evaluate selected aqcuisition function over the grid
    xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, level)

    # select next sampling points. for now, just use Mac--dMax and Erg need work.
    if control=='Max':            
       next_samples_points = maxAcquisition(workspace, AqcuisFunction,
                                           numpoints=3)
    else:
        print 'RANDOM'
        next_samples_points=randompoints(bounds,1)
    
    # if next_samples_points.shape[0]>1:

    time.sleep(0.0001)
    plt.pause(0.0001)  

    # Plot everything
    plot_data = plot_beliefGPIS(phantomname,workspace,mean,sigma,
                                AqcuisFunction,meas,
                                directory,plot_data,level=level,
                                iternum=j,projection3D=False)
    # Save everything--this needs to be debugged
    # prename=directory+'/'
    # save_p2_data(prename+'mean'+str(j),mean)
    # save_p2_data(prename+'sigma'+str(j),sigma)
    # save_p2_data(prename+'AqcuisFunction'+str(j),AqcuisFunction)
    # save_p2_data(prename+'meas'+str(j),meas)

plt.show(block=True)

# if __name__ == "__main__":
#     planning(verbose=True)




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




