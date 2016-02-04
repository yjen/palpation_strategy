import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# import matplotlib.path as path

#from utils import *
# from getMap import *
#from sensorModel import *
# import cplex, gurobipy
#import sys
#sys.path.append("..")
#import simulated_disparity
from simUtils import *
from utils import *
from GaussianProcess import *
# import ErgodicPlanner
from Planner import *

# def max_uncertainty(GPdata,numpoints=1):
#     # return the x,y locations of the "numpoints" largest uncertainty values
#     ind = np.argpartition(GPdata[3].flatten(), -numpoints)[-numpoints:]
#     newpointx=GPdata[0].flatten()[ind]
#     newpointy=GPdata[1].flatten()[ind]
    
#     return np.array([newpointx,newpointy]).T

# def max_uncertainty_IS(GPdata,numpoints=1):
#     # return the x,y locations of the "numpoints" largest uncertainty values
#     GPISdat=implicitsurface(GPdata)
#     ind = np.argpartition(GPISdat[2].flatten(), -numpoints)[-numpoints:]
#     newpointx=GPdata[0].flatten()[ind]
#     newpointy=GPdata[1].flatten()[ind]
    
#     return np.array([newpointx,newpointy]).T



##############################
# set boundary
rangeX = [-2,2]
rangeY = [-2,2]

# choose surface for simulation

##############################
# Phase 2
###############################

# choose surface for simulation
surface=gaussian_tumor

testpoly=np.array([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]])

#surface=SixhumpcamelSurface

# initialize planner
xdim=2
udim=2
#LQ=ErgodicPlanner.lqrsolver(xdim,udim, Nfourier=10, res=100,barrcost=50,contcost=.1,ergcost=10)

# initialize probe state
xinit=np.array([0.01,.0])
U0=np.array([0,0])

# initialize stiffness map (uniform)
#pdf=uniform_pdf(LQ.xlist)

fig = plt.figure(figsize=(16, 4))

plt.ion()
plt.show()

for j in range (1,20,1):
    print "iteration = ", j
#j=1
    #traj=ErgodicPlanner.ergoptimize(LQ,pdf,
    #xinit,control_init=U0,maxsteps=20)
    #if j>1:
    #    trajtotal=np.concatenate((trajtotal,traj),axis=0)
    #else:
    #    trajtotal=traj
    # choose points to probe based on max uncertainty
    if j>1:
        next_samples_points=max_uncertainty_joint(GPdata,numpoints=1)
        measnew=getSimulateStiffnessMeas(testpoly,
                                          next_samples_points)
        meas=np.append(meas,measnew,axis=0)
    else:
        next_samples_points=np.array([[-1,-1],[-1,1],[0,0],
                                      [1,-1],[1,1]])
        meas=getSimulateStiffnessMeas(testpoly,
                                          next_samples_points)

    gpmodel=update_GP(meas)

    # Predections based on current GP estimate
    GPdata=eval_GP(gpmodel, rangeX, rangeY,res=200)
    GPISdat=implicitsurface(GPdata)
    # plot_beliefGPIS(fig,testpoly,GPdata,GPISdat,meas)
    time.sleep(0.05)

plot_beliefGPIS(testpoly,GPdata,GPISdat,meas)

# if __name__ == "__main__":
#     planning(verbose=True)




# plot_beliefGPIS(testpoly,GPdata,GPISdat,meas)
