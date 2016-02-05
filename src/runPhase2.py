import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from simUtils import *
from utils import *
from GaussianProcess import *
# import ErgodicPlanner
from Planner import *

##############################
# Phase 2
###############################

# set boundary
rangeX = [-2,2]
rangeY = [-2,2]

# choose inclusion shape for simulation: there are a bunch of saved
# surface funcions in simUtils
tumorpoly = squaretumor

##############################
# ignore  for now
###############################
# initialize planner
# xdim=2
# udim=2
#LQ=ErgodicPlanner.lqrsolver(xdim,udim, Nfourier=10, res=100,barrcost=50,contcost=.1,ergcost=10)

# initialize probe state
# xinit=np.array([0.01,.0])
# U0=np.array([0,0])

# initialize stiffness map (uniform)
#pdf=uniform_pdf(LQ.xlist)

# fig = plt.figure(figsize=(16, 4))

# plt.ion()
# plt.show()

for j in range (1,20,1):
    print "iteration = ", j
    # ignore for now: for testing alternate planning strategy
    # j=1
    # traj=ErgodicPlanner.ergoptimize(LQ,pdf,
    # xinit,control_init=U0,maxsteps=20)
    # if j>1:
    #    trajtotal=np.concatenate((trajtotal,traj),axis=0)
    # else:
    #    trajtotal=traj
    # choose points to probe based on max uncertainty
    if j==1:
        # initialize with a few probes at predefined points (could be
        # random?)
        next_samples_points = np.array([[-1,-1],[-1,1],[0,0],
                                        [1,-1],[1,1]])
        # collect initial meausrements
        meas = getSimulateStiffnessMeas(tumorpoly,
                                        next_samples_points)
    else:
        # choose samples based on some sort of uncertainty measure--this
        # will vary between experiments
        next_samples_points = max_uncertainty_joint(GPdata,
                                                    numpoints=1)
        # collect measurements
        measnew = getSimulateStiffnessMeas(tumorpoly,
                                           next_samples_points)
        # concatenate measurements to prior measurements
        meas = np.append(meas,measnew,axis=0)
        
    gpmodel=update_GP(meas)

    # Predections based on current GP estimate
    GPdata=eval_GP(gpmodel, rangeX, rangeY,res=200)
    
    GPISdat=implicitsurface(GPdata)
    # plot_beliefGPIS(fig,testpoly,GPdata,GPISdat,meas)
    time.sleep(0.05)

plot_beliefGPIS(tumorpoly,GPdata,GPISdat,meas)

# if __name__ == "__main__":
#     planning(verbose=True)

# plot_beliefGPIS(testpoly,GPdata,GPISdat,meas)
