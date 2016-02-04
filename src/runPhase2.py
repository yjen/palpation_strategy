import sys
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
#import simulated_disparity
from simUtils import *
from utils import *
from GaussianProcess import *
# import ErgodicPlanner
from planner import *



##############################
# set boundary
rangeX = [-2,2]
rangeY = [-1,1]

# choose surface for simulation

##############################
# Phase 2
###############################

# choose surface for simulation
surface=gaussian_tumor
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

for j in range (1,30,1):
#j=1
    #traj=ErgodicPlanner.ergoptimize(LQ,pdf, xinit,control_init=U0,maxsteps=20)
    #if j>1:
    #    trajtotal=np.concatenate((trajtotal,traj),axis=0)
    #else:
    #    trajtotal=traj
    # choose points to probe based on max uncertainty
    if j>1:
        next_samples_points=max_uncertainty_IS(GPdata,numpoints=4)
        measnew=getSimulateStiffnessMeas(surface,
                                          next_samples_points)
        meas=np.append(meas,measnew,axis=0)
    else:
        next_samples_points=np.array([[-1,-1],[-1,1],[0,0],[1,-1],[1,1]])
        meas=getSimulateStiffnessMeas(surface,
                                          next_samples_points)
        

    gpmodel=update_GP(meas)

    # Predections based on current GP estimate
    GPdata=eval_GP(gpmodel, rangeX, rangeY,res=200)
   # GPISdat=implicitsurface(GPdata)
    


# if __name__ == "__main__":
#     planning(verbose=True)


def plot_beliefGPIS(surface,GPdata):
    # parse locations, measurements, noise from data
    thresh=.4
    GPISdat=implicitsurface(GPdata,thresh=thresh)

    #gp data
    xx=GPdata[0]
    yy=GPdata[1]
    mean=GPdata[2]
    variance=GPdata[3]

    #GPIS data
    xx=GPISdat[0]
    yy=GPISdat[1]
    GPISvar=GPISdat[2]

    # evaluate surface ground truth:
    GroundTruth = surface(xx,yy)
    
    fig = plt.figure(figsize=(16, 4))

    # plot the mean
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(xx, yy, mean, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
    ax.set_title("GP Mean")
    ax.set_xlabel('x')
    ax.set_ylabel('y')    
    # plot the uncertainty
    ax1 = fig.add_subplot(132, projection='3d')
    lim=1
    cs1=ax1.plot_surface(xx, yy, variance, rstride=1, cstride=1,
                         cmap=cm.Greys, linewidth=0, antialiased=False)
    ax1.set_title("GP Uncertainty")  
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    # plot the uncertainty
    # ax2 = fig.add_subplot(133, projection='3d')
    # lim=1
    # cs2=ax2.plot_surface(xx, yy, np.abs(GPISvar), rstride=1, cstride=1,
    #                      cmap=cm.Greys, linewidth=0, antialiased=False)
    # ax1.set_title("GP Uncertainty")  
    ax2 = fig.add_subplot(133)
    
    ax2.set_title("GPIS")
    #cs = pl.contour(xx, yy, mean, [thresh], colors='k', linestyles='dashdot')
    ax2.contour(xx, yy, GPdata[2], [.4], colors='r', linestyles='dashdot')
    ax2.contour(xx, yy, GroundTruth, [.4], colors='g', linestyles='dashdot')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')    #phi = stats.distributions.norm.pdf
    #GPISv=np.flipud(phi(GPdata[2],loc=.4,scale=( v+GPdata[3])))
    #GPISv=GPISv/GPISv.max()
    cs2=ax2.imshow(GPISvar, cmap=cm.Greys,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max())
    )
    norm = plt.matplotlib.colors.Normalize(vmin=0., vmax=GPISvar.max())
    cb2 = plt.colorbar(cs2, norm=norm)
    cb2.set_label('${\\rm \mathbb{P}}\left[\widehat{G}(\mathbf{x}) = 0\\right]$')# # Define
    plt.show()


plot_beliefGPIS(surface,GPdata)
