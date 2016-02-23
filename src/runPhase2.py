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
import rospy
import pickle

##############################
# Phase 2
###############################
#TODO:
# To run Phase 2 on the robot, the function getExperimentalStiffnessMeas, 
# in Gaussian Process.py, needs to be written to command the robot and collect measurements


def calculate_boundary(filename):
    data_dict = None
    try:
        data_dict = pickle.load(open(filename, "rb"))
    except Exception as e:
        print "Exception: ", e
        rospy.logerror("Error: %s", e)
        rospy.logerror("Failed to load saved tissue registration.")

    # compute tissue frame
    nw = data_dict['nw']
    ne = data_dict['ne']
    sw = data_dict['sw']
    nw_position = np.hstack(np.array(nw.position))
    ne_position = np.hstack(np.array(ne.position))
    sw_position = np.hstack(np.array(sw.position))
    u = sw_position - nw_position
    v = ne_position - nw_position
    tissue_length = np.linalg.norm(u)
    tissue_width = np.linalg.norm(v)
    return ((0, tissue_length), (0, tissue_width))


# set workspace boundary
bounds = calculate_boundary("../scripts/env_registration.p")
print(bounds)

# grid resolution: should be same for plots, ergodic stuff
gridres = 200

# initialize workspace object
workspace = Workspace(bounds,gridres)

# set level set to look for-- this should correspond to something, max FI?
level=2000 #pick something between min/max deflection

# initialize probe state--only needed for ergodic
xinit=np.array([0.01,.011])
            
# choose inclusion shape for simulation: there are several saved
# surface funcions in simUtils
polyname='square'
tumorpoly = rantumor

# control choices: Max or Erg or dMax. Erg and dMax are still in development.
control='Max'

# acquisition functions:  MaxVar_GP, UCB_GP, EI_GP, UCB_GPIS, EI_IS
AcFunction=UCB_GPIS
Acfunctionname="UCB_GPIS"

plot_data = None

directory='phase2_'+polyname+'_'+control+'_'+Acfunctionname
if not os.path.exists(directory):
    os.makedirs(directory)

###############
#Initializing
###############
if control == 'Erg':
    # initialize ergodic cplanner
    xdim = 2
    udim = 2
    LQ = ErgodicPlanner.lqrsolver(xdim, udim, Nfourier=30, wlimit=(4.), res=gridres,
                                barrcost=50, contcost=.03, ergcost=1000)
    # initialize stiffness map (uniform)
    initpdf = ErgodicPlanner.uniform_pdf(LQ.xlist)
    next_samples_points = ErgodicPlanner.ergoptimize(LQ, initpdf, xinit,
                                                   maxsteps=15,plot=True)
else:
    next_samples_points = randompoints(bounds, 10)
    
    # collect initial meausrements
# meas = getSimulateStiffnessMeas(tumorpoly, next_samples_points)
    meas = getExperimentalStiffnessMeas(next_samples_points)


for j in range (10): #(1,100,1)
    print "iteration = ", j
    # collect measurements
    # measnew = getSimulateStiffnessMeas(tumorpoly,
    #                                    next_samples_points)
    # to run experiment instead of simulation:
    measnew = getExperimentalStiffnessMeas(next_samples_points)
    # concatenate measurements to prior measurements
    import IPython; IPython.embed()
    meas = np.append(meas,measnew,axis=0)
    
    # update the GP model    
    gpmodel = update_GP(meas)

    # use GP to predict mean, sigma on a grid
    mean, sigma = get_moments(gpmodel, workspace.x)

    # find mean points closest to the level set
    boundaryestimate = getLevelSet (workspace, mean, level)
    GPIS = implicitsurface(mean,sigma,level)

    # evaluate selected aqcuisition function over the grid
    xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, level)

    # select next sampling points. for now, just use Mac--dMax and Erg need work.
    if control=='Max':     
        next_samples_points = maxAcquisition(workspace, AqcuisFunction,
                                           numpoints=1)
    elif control=='dMax':            
        next_samples_points = dmaxAcquisition(workspace, gpmodel, AcFunction, xinit=meas[-1,0:2],
                                           numpoints=5)
    elif control=='Erg':
        next_samples_points = ergAcquisition(workspace, AqcuisFunction,
                                             LQ, xinit=meas[-1,0:2])
    else:
        print 'RANDOM'
        next_samples_points=randompoints(bounds,1)
        
    time.sleep(0.0001)
    plt.pause(0.0001)  

    # Plot everything
    plot_data = plot_beliefGPIS(tumorpoly,workspace,mean,sigma,
                                GPIS,AqcuisFunction,meas,
                                directory,plot_data,level=level,
                                iternum=j)

plt.show(block=True)

# if __name__ == "__main__":
#     planning(verbose=True)




