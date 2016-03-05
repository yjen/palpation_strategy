#!/usr/bin/env python
import sys, os, time, IPython
sys.path.append('../scripts')

import numpy as np
import matplotlib.pyplot as plt
from runPhase1 import *
from Planner import *
from simUtils import *
from model_fit import *
# from figures import SIZE, BLUE, GRAY
# from shapely.geometry import Point
# from descartes import PolygonPatch
NUM_EXPERIMENTS = 1

# data1 = pickle.load(open("../scripts/saved_palpation_data/single_row_raster_100x.p", "rb"))
# # data2 = pickle.load(open("probe_data_L2R.p", "rb"))
# xdata,zdata=get_stiffness_data(data1)
# model=fit_measmodel(xdata,zdata)
# mod=plot_model(data1,model,scale=True)
# print mod
def run_single_phase2_simulation(phantomname, dirname, AcFunction=MaxVar_GP, control='Max', plot=False):

    bounds=((-.04,.04),(-.04,.04))

    # grid resolution: should be same for plots, ergodic stuff
    gridres = 200

    # initialize workspace object
    workspace = Workspace(bounds,gridres)

    # plotSimulatedStiffnessMeas(phantomname, workspace, ypos=0, sensornoise = .01)

    # set level set to look for-- this should correspond to something, max FI?
    level = .9 #pick something between min/max deflection

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

    # collect initial meausrements

    meas = getSimulateStiffnessMeas(phantomname, next_samples_points)

    for j in range (10): #(1,100,1)
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
        
        xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, level=level,acquisition_par=.4 )

        acqvals.append(AqcuisFunction)
        means.append(mean)
        sigmas.append(sigma)
        measures.append(meas)
        healthyremoved,tumorleft= evalerror(phantomname, workspace, mean,level)
        healthyremoveds.append(healthyremoved)
        tumorlefts.append(tumorleft)

        # select next sampling points. for now, just use Mac--dMax and Erg need work.
        bnd=getLevelSet (workspace, mean, level)
        if len(bnd)==0:
            next_samples_points = randompoints(bounds, 1)
        elif control=='Max':            
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
    return means, sigmas, acqvals, measures, errors, errors1, j

def evalerror(tumor,workspace,mean,level):
    # see: http://toblerity.org/shapely/manual.html
    boundaryestimate = getLevelSet (workspace, mean, level)
    GroundTruth = np.vstack((tumor,tumor[0]))
    GroundTruth=Polygon(GroundTruth)
    if len(boundaryestimate)>0:

        boundaryestimate=Polygon(boundaryestimate)
        healthyremoved=boundaryestimate.difference(GroundTruth) # mislabeled data ()
        #boundaryestimate.difference(GroundTruth) #mislabeled as tumor--extra that would be removed
        #

        tumorleft=GroundTruth.difference(boundaryestimate) # mislbaled as not-tumor--would be missed and should be cut out
        #correct=boundaryestimate.intersection(GroundTruth) #correctly labeled as tumor
        healthyremoved=healthyremoved.area
        tumorleft=tumorleft.area
        print 'healthyremoved',healthyremoved
        print 'healthyremoved',tumorleft
    else:
        healthyremoved=100
        tumorleft=100
    return healthyremoved,tumorleft

def save_data(arr, dirname, name):
    directory=dirname

    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(directory+'/data_'+name, 'w')
    f.write(str(np.array(arr).tolist()))
    f.close()
    return

tumors = [rantumor]#, rantumor]              # add another model ?
stops = [[6.4, 0.01, 3.6, 0.0],
         [0.0, 0.36, 0.0, 0.0]]                 # TODO change 0.0's (variance is not monotonically decreasing)
# textures = ["_lam", "_text", "_spec", "_st"]
    # lambert, texture, specular, specular + texture

# acquisition functions:  MaxVar_GP, UCB_GP, EI_GP, UCB_GPIS, EI_IS, MaxVar_plus_gradient
aqfunctions = [MaxVar_GP, UCB_GP, UCB_GPIS]
aqfunctionsnames = ["MaxVar_GP", "UCB_GP", "UCB_GPIS"]#, "random"]

controls =["Max"]

def save_table(table, name):
    f = open(name + ".csv", 'wb')
    # header
    f.write(",,MaxVar_GP,UCB_GB,UCB_GBIS\n")
    # data in table
    # f.write("flat,lam,{},{}\n".format(table[0][0], table[0][1]))
    f.write(",tumor1: healthy tissue removed,{},{},{}\n".format(table[0][0], table[0][1],table[0][2]))
    f.write(",tumor1: tumor left behind,{},{},{}\n".format(table[1][0], table[1][1],table[1][2]))

    f.write(",tumor2: healthy tissue removed,{},{},{}\n".format(table[2][0], table[2][1],table[2][2]))
    f.write(",tumor2: tumor left behind,{},{},{}\n".format(table[3][0], table[3][1],table[3][2]))

    # f.write(",st,{},{}\n".format(table[0][0], table[0][1]))

    # f.write("S,lam,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",text,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",spec,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",st,{},{}\n".format(table[1][0], table[1][1]))

    f.close()
    return





def run_phase2_full():
    iter_table = np.zeros((len(tumors), len(aqfunctions)+len(controls)))
    error_table = np.zeros((len(tumors)*2, len(aqfunctions)+len(controls)))
    # error_table1 = np.zeros((2, len(aqfunctions)+len(controls)))

    for i, tumor in enumerate(tumors):
        for j, acq in enumerate(aqfunctions):
            for m, cont in enumerate(controls):
                # if i*len(textures) + j != 0:        # Use this to run only the nth surface
                #     continue
                its = [0.0, 0.0]
                for k in range(NUM_EXPERIMENTS): # repeat experiment 5 times
                    # disparityMeas = None
                    # for l, method in enumerate(methods):
                    start = time.time()
                    dirname = str(i) + '_' + aqfunctionsnames[j] + '_' + cont
                    print dirname
                    means, sigmas, acqvals, measures, healthyremoved,tumorleft, num_iters = run_single_phase2_simulation(tumor, dirname, AcFunction=acq, control=cont, plot=True)
                    plt.close() 
                    end = time.time()
                    time_elapsed = end - start # in seconds
                    # plot or save/record everything
                    # its[k] += num_iters / 5.0
                    save_data([means, sigmas, acqvals, measures, healthyremoved, tumorleft, num_iters], dirname, str(k))
                    iter_table[i][j + m]+= num_iters / float(NUM_EXPERIMENTS)
                    error_table[i][j+ m]+= healthyremoved[-1] / float(NUM_EXPERIMENTS)
                    error_table[i+len(tumors)][j + m]+= tumorleft[-1] / float(NUM_EXPERIMENTS)
                    print k

                save_table(iter_table, "phase2_iterations")
                save_table(error_table, "phase2_errors")
                # save_table(error_table1, "phase2_errors1")

    return



if __name__ == "__main__":
    run_phase2_full()
    # IPython.embed()


