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

##############################
# Phase 2
###############################

# set workspace boundary
bounds = ((0,4),(0,4))

#grid resolution: should be same for plots, ergodic stuff
gridres = 200

#initialize workspace object
workspace = Workspace(bounds,gridres)

# set level set to look for-- this should correspond to somehting, max FI?
level=.5

# initialize probe state--only needed for ergodic
xinit=np.array([0.01,.011])
            
# choose inclusion shape for simulation: there are a bunch of saved
# surface funcions in simUtils
polyname='square'
tumorpoly = rantumor

# control choices: Max or Erg or dMax. Erg and dMax are still in development.
control='Max'

# acquisition functions:  MaxVar_GP, UCB_GP, EI_GP, UCB_IS, EI_IS
AcFunction=MaxVar_GP
Acfunctionname="MaxVar_GP"
# planner='RM'

plot_data = None

directory='phase2_'+polyname+'_'+control+'_'+Acfunctionname
if not os.path.exists(directory):
    os.makedirs(directory)

###############
#Initializingfffffffff
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
meas = getSimulateStiffnessMeas(tumorpoly, next_samples_points)

for j in range (1,100,1):
    print "iteration = ", j
    # concatenate measurements to prior measurements
    # collect measurements
    measnew = getSimulateStiffnessMeas(tumorpoly,
                                       next_samples_points)
    
    meas = np.append(meas,measnew,axis=0)
        
    gpmodel = update_GP(meas)

    mean, sigma = get_moments(gpmodel, workspace.x)
    boundaryestimate = getLevelSet (workspace, mean, level)
    GPIS = implicitsurface(mean,sigma,level)
    # if AcFunction == 'AcFunction':
    xgrid, AqcuisFunction = AcFunction(gpmodel, workspace.x, level)
    # elif AcFunction == 'UCB_GP':
    #     xgrid, AqcuisFunction = UCB_GP(gpmodel, workspace.x, acquisition_par=.4)
    # elif AcFunction == 'UCB_GPIS':
    #     xgrid, AqcuisFunction = UCB_GPIS(gpmodel, workspace.x, level, acquisition_par=1)
    # elif AcFunction == 'EI_GP':
    #     xgrid, AqcuisFunction = EI_GP(gpmodel, workspace.x, acquisition_par=0)
    # elif AcFunction == 'EI_GPIS':
    # #     xgrid, AqcuisFunction = EI_GPIS(gpmodel, workspace.x, level, acquisition_par=0)

    # else:
    #     pass
    if control=='Max':     
        print control       
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
        
    time.sleep(0.05)
    plt.pause(0.0001)    
    plot_data = plot_beliefGPIS(tumorpoly,workspace,mean,sigma,
                                GPIS,AqcuisFunction,meas,
                                directory,plot_data,level=level,
                                iternum=j)

plt.show(block=True)

# if __name__ == "__main__":
#     planning(verbose=True)




