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
def evalerror_ph2(tumor,workspace,mean,variance,level,tiltlev=0,offset=offset):

    if tiltlev>0:
        level=.5*(mean.max()-mean.min())+mean.min()
    
    GroundTruth = Polygon(tumor)
    print offset
    print level
    offset=0
    # see: http://toblerity.org/shapely/manual.html
    boundaryestimate = getLevelSet (workspace, mean, level, allpoly=True)
    offsetboundary=[]

    for i in range(0,boundaryestimate.shape[0]):
        bnd = boundaryestimate[i]
        if bnd.shape[0]>3:
            # data[4].plot(bnd[:,0], bnd[:,1], '-',color='k',
            #            linewidth=1, solid_capstyle='round', zorder=2)
            # for bn in bnd:
            bnd=Polygon(bnd)
            bnd=bnd.buffer(-offset)
            try:
                bnd=np.array(bnd.exterior.coords)
                # data[4].plot(bnd.T[0], bnd.T[1], '-',color='r',
                #     linewidth=1, solid_capstyle='round', zorder=2)
            except AttributeError:
                 bnd=[]
            offsetboundary.append(bnd)

    arealist=[0]
    bnlist=[0]
    err=0
    area=0
    print len(offsetboundary)
    if len(offsetboundary)>0:
        
        for b in offsetboundary:
            if len(b)>3:
                bn=Polygon(b)
                arealoc=bn.area
                if arealoc>area:
                    area=arealoc
                    c1 = GroundTruth.symmetric_difference(bn)
                    if c1.geom_type == 'Polygon':
                        err=c1.area
                         
                    elif c1.geom_type == 'MultiPolygon':
                        err1=0
                        for p in c1:
                            err1=err1+p.area
                        err=err1

    error=err
    if area==0 and err==0:
        error=GroundTruth.area#(workspace.bounds[0][1]-workspace.bounds[0][0])*(workspace.bounds[1][1]-workspace.bounds[1][0])

    return 10000.0*error

def run_single_phase2_simulation(phantomname, dirname, AcFunction=MaxVar_GP, 
    control='Max', noiselev=.05,tiltlev=0,plot=False, smode='RecordedExp',iters=20):
    if smode=='RecordedExp': 
        getmeasurements=getRecordedExperimentalStiffnessMeas
        bounds=((.0,0.0229845803642),(.0,0.0577416388862))
    elif smode=='Exp':
        from expUtils import *
        getmeasurements=getExperimentalStiffnessMeas
        bounds=calculate_boundary("../scripts/env_registration.p")
    elif smode=='Sim':
        getmeasurements=getSimulateStiffnessMeas
        bounds=((.0,0.025),(.0,0.05))

    else: 
        print 'invalid mode!'
    
    # set level set to look for-- this should correspond to something, max FI?
    levelrel=.5
    level = levelrel*(measmax-measmin)+measmin #pick something between min/max deflection

    if smode=='RecordedExp' or smode=='Exp': 
        UCB_GP_acpar=.7 # set parameters for acquiisition functions: balancing mean vs. var in prioritizing search
        UCB_GPIS_acpar=.2

        UCB_GPIS_implicit_acpar=[.8,.2]
        UCB_dGP_acpar=[1.1,.2]
        UCB_dGP_acpar=[.5,.7]

        GP_params= [14,.003,.002,63] # parameters for gaussian process update

    else:   #params for simulation
        UCB_GP_acpar=.5
        UCB_GPIS_acpar=.5
        UCB_GPIS_implicit_acpar=[.7,.5]
        UCB_dGP_acpar=[.5,.7]
        GP_params= [25,.0033,.004,52]
        # GP_params= [.22,.0033,.004,52]

    if AcFunction==UCB_GPIS:
        acquisition_par=UCB_GPIS_acpar
    elif AcFunction==UCB_GP:
        acquisition_par=UCB_GP_acpar
    elif AcFunction==UCB_dGPIS:
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

    plot_data = None
    means = []
    sigmas = []
    acqvals = []
    errors=[]
    sampled_points = []
    measures = []

    directory=dirname
    if not os.path.exists(directory):
        os.makedirs(directory)

    ###############
    #Initializing
    ###############
    next_samples_points = randompoints(bounds, 5) #randompoints(bounds, 100)

    tiltlev=np.arctan2(tiltlev*(measmax-measmin),workspace.bounds[1][1])

    meas = getmeasurements(next_samples_points,phantomname,noiselev=noiselev,tiltlev=tiltlev)

    measnew = meas
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
        error= evalerror_ph2(phantomname, workspace, mean,sigma,.5,tiltlev)
        errors.append(error)

        print 'error=',error
        # select next sampling points. for now, just use Mac--dMax and Erg need work.
        bnd=getLevelSet (workspace, mean, level)

        if control=='Max':         
            
            next_samples_points = batch_optimization(gpmodel, workspace, AcFunction, 10, 
                meas[-1][0:2], GP_params,level=level, acquisition_par=acquisition_par)
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
                                      directory, errors,plot_data,level=level,tiltlev=tiltlev,
                                      iternum=j, projection3D=False)
        measnew = getmeasurements(next_samples_points,phantomname,noiselev=noiselev,tiltlev=tiltlev)

        meas = np.append(meas,measnew,axis=0)

    # plt.show(block=False)
    return means, sigmas, acqvals, measures, errors, j, gpmodel


if __name__ == "__main__":
    dirname='tests'
    run_single_phase2_simulation(horseshoe, dirname, AcFunction=UCB_dGPIS, control='Max', plot=True, tiltlev=1,smode='RecordedExp',iters=20)
    # outleft,outrem,aclabellist=run_phase2_full()
    # plot_error(outrem,outleft,aclabellist)






