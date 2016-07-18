#!/usr/bin/env python
import sys, os, time, IPython
sys.path.append('../scripts')
import matplotlib.pyplot as plt
from runPhase2 import *
from model_fit import *
from plotscripts import *
import time

###############################################
# Configure Experimentes
###############################################

# how many times to run each experiment
NUM_EXPERIMENTS = 5
# simulated tumors to try
tumors = [simCircle,horseshoe]      
# acquisition functions to compare        
aqfunctions = [MaxVar_GP,UCB_GP,UCB_GPIS,UCB_GPIS_implicitlevel,UCB_dGPIS] #MaxVar_plus_gradient
aqfunctionsnames = ["MaxVar_GP","UCB_GP","UCB_GPIS","UCB_GPIS_implicitlevel","UCB_dGPIS" ]#, "random"]"MaxVar_plus_gradient"
controls =["Max"]

# Levels of Gaussian measurement noise to test
noisedefault=.01 # set default to lowest noise level
noiseerrors= float((measmax-measmin))*np.array([.01,.05,.1,.25])

# Levels of measurement bias to test
tiltdefault=.0 # set default to no tilt
tilterrors = [0,.05,.5,1]



def save_table(table, name):
    # Save results table from set of experiments to a csv
    f = open(name + ".csv", 'wb')
    # header
    f.write(",,MaxVar_GP,UCB_GB,UCB_GBIS,UCB_GPIS_implicitlevel,UCB_dGPIS,\n") #MaxVar_plus_gradient
    # data in table
    # f.write("flat,lam,{},{}\n".format(table[0][0], table[0][1]))
    f.write(",tumor1: iterations,{},{},{}\n".format(table[0][0], table[0][1],table[0][2],table[0][3],table[0][4]))#,table[0][5]))
    f.write(",tumor1: healthy tissue removed,{},{},{}\n".format(table[1][0], table[1][1],table[1][2],table[1][3],table[1][4]))
    f.write(",tumor1: tumor left behind,{},{},{}\n".format(table[2][0], table[2][1],table[2][2],table[2][3],table[2][4]))#,table[2][5]))

    f.write(",tumor2: iterations,{},{},{}\n".format(table[3][0], table[3][1],table[3][2],table[3][3],table[3][4]))#,table[3][5]))
    f.write(",tumor2: healthy tissue removed,{},{},{}\n".format(table[4][0], table[4][1],table[4][2],table[4][3],table[4][4]))#,table[4][5]))
    f.write(",tumor2: tumor left behind,{},{},{}\n".format(table[5][0], table[5][1],table[5][2],table[5][3],table[5][4]))#,table[5][5]))

    f.close()
    return

def save_data(arr, fname, aclabellist, noiselevels):
    # save raw data
    arr=np.array(arr).tolist()
    arr=[arr,aclabellist,noiselevels]
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    f = open('data_'+fname, 'w')
    f.write(str(np.array(arr).tolist()))
    f.close()
    return


def run_phase2_experiments(vary_tilt=False):
    # This script runs all of the experiments, saves plots and data at each iteration for each experimental configuration
    tumorerrlist=[]
    tumortimelist=[]
    timelist=[]

    # vary either the noise or bias levels
    if vary_tilt==True:
        modelerrors=tilterrors
    else:
        modelerrors=noiseerrors
    error_table = np.zeros((len(tumors)*len(modelerrors),len(aqfunctions)))

    for i, tumor in enumerate(tumors):
        acqerrlist=[]
        acqtimelist=[]
        aclabellist=[]
        for j, acq in enumerate(aqfunctions):
            noiseerrlist=[]
            noisetimelist=[]
            noiselabellist=[]  
            for m, modelerr in enumerate(modelerrors):
                its = [0.0, 0.0]
                experrlist=[]
                exptimelist=[]
                for k in range(NUM_EXPERIMENTS): # repeat experiment 5 times
                    start = time.time()
                    if vary_tilt==True:
                        tiltlev=modelerr
                        noiselev=noisedefault
                    else:
                        tiltlev=tiltdefault
                        noiselev=modelerr
                    dirname = str(i) + '_' + aqfunctionsnames[j]+'_noise_'+str(noiselev)+'_tilt_'+str(tiltlev)+'_exp_'+str(k)
                    print dirname
                    means, sigmas, acqvals, measures, error, num_iters, gpmodel = run_single_phase2_simulation(
                        tumor, dirname, AcFunction=acq, noiselev=noiselev, tiltlev=tiltlev, plot=True, smode='Sim', iters=10)
                    plt.close() 
                    end = time.time()
                    time_elapsed = end - start # in seconds

                    error_table[i+i*(len(modelerrors)-1)+m][j]+= error[-1] / float(NUM_EXPERIMENTS)
                    experrlist.append(error)  
                    exptimelist.append(time_elapsed)  
                    print k

                print error_table
                noiseerrlist.append(experrlist)
                noisetimelist.append(exptimelist)
                noiselabellist.append(dirname)
            acqerrlist.append(noiseerrlist)
            acqtimelist.append(noisetimelist)
            aclabellist.append(noiselabellist)

        tumorerrlist.append(acqerrlist)
        tumortimelist.append(acqtimelist)
        timestr = time.strftime("%Y-%m-%d-%H:%M:%S")
        fname='ph2error'+'_tumor'+'_'+timestr
        np.save(fname,np.array([tumorerrlist,tumortimelist,aqfunctionsnames,modelerrors]))
    return np.array(tumorerrlist),np.array(tumortimelist),np.array(aclabellist),np.array(modelerrors),fname



if __name__ == "__main__":
    dirname='tt'
    vary_tilt=False

    if vary_tilt==False:
        noisetype='measurement noise'
    else:
        noisetype='measurement bias'
    #run_single_phase2_simulation(simCircle, dirname, AcFunction=UCB_GP, control='Max', plot=True, smode='Sim',iters=20)
    errorlist,timelist,aclabellist,modelerrors,fname=run_phase2_experiments(vary_tilt=vary_tilt)
    plot_ph2_error(fname,errorlist,aclabellist,aqfunctionsnames,modelerrors,noisetype)
    make_error_table(fname,noisetype)

