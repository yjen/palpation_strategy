#!/usr/bin/env python
import sys, os, time, IPython
sys.path.append('../scripts')
# from shapely.topology import *
# import numpy as np
import matplotlib.pyplot as plt
from runPhase2 import *
# from Planner import *
# from simUtils import *
from model_fit import *
from plotscripts import *
import time

def save_table(table, name):
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

    # f.write(",st,{},{}\n".format(table[0][0], table[0][1]))

    # f.write("S,lam,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",text,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",spec,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",st,{},{}\n".format(table[1][0], table[1][1]))

    f.close()
    return

def save_data(arr, fname, aclabellist, noiselevels):
    # directory=dirname
    arr=np.array(arr).tolist()
    #aclabellist=np.array(aclabellist.flatten()).tolist()
    arr=[arr,aclabellist,noiselevels]
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    f = open('data_'+fname, 'w')
    f.write(str(np.array(arr).tolist()))
    f.close()
    return

NUM_EXPERIMENTS = 1

tumors = [simCircle,horseshoe]              # add another model ?
aqfunctions = [MaxVar_GP,UCB_GP,UCB_GPIS,UCB_GPIS_implicitlevel,UCB_dGPIS]#MaxVar_plus_gradient
aqfunctionsnames = ["MaxVar_GP","UCB_GP","UCB_GPIS","UCB_GPIS_implicitlevel","UCB_dGPIS" ]#, "random"]"MaxVar_plus_gradient"
controls =["Max"]


# to test gaussian noise level
noiseerrors= [.01,.05,.1,.5]
tiltdefault=.0
# to test noise bias level
tilterrors = [0,.05,.5,1]

noisedefault=.01

def run_phase2_full(vary_tilt=False):
    tumorerrlist=[]
    tumortimelist=[]
    timelist=[]
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

                    # save_data([means, sigmas, acqvals, measures, error, num_iters], dirname, aqfunctionsnames,str(k))
                    # error_table[i+0*i+0][j+m]+= num_iters / float(NUM_EXPERIMENTS)
                    # error_table[i+i*2+2][j+ m]+= time_elapsed[-1] / float(NUM_EXPERIMENTS)
                    error_table[i+i*(len(modelerrors)-1)+m][j]+= error[-1] / float(NUM_EXPERIMENTS)
                    experrlist.append(error)  
                    exptimelist.append(time_elapsed)  
                    print k

                print error_table
                # save_table(iter_table, "phase2_iterations")
                # save_table(error_table, "phase2_errors")
                # save_table(error_table1, "phase2_errors1")
                noiseerrlist.append(experrlist)
                noisetimelist.append(exptimelist)
                noiselabellist.append(dirname)
            acqerrlist.append(noiseerrlist)
            acqtimelist.append(noisetimelist)
            aclabellist.append(noiselabellist)
        # plot_error(errors_per_method, "phase1_error_exp"+str(k), surf+text)

        tumorerrlist.append(acqerrlist)
        tumortimelist.append(acqtimelist)
        timestr = time.strftime("%Y-%m-%d-%H:%M:%S")
        fname='ph2error'+'_tumor'+'_'+timestr
        np.save(fname,np.array([tumorerrlist,tumortimelist,aqfunctionsnames,modelerrors]))
    return np.array(tumorerrlist),np.array(tumortimelist),np.array(aclabellist),np.array(modelerrors),fname



if __name__ == "__main__":
    dirname='tt'
    vary_tilt=True

    if vary_tilt==False:
        noisetype='measurement noise'
    else:
        noisetype='measurement bias'
    #run_single_phase2_simulation(simCircle, dirname, AcFunction=UCB_GP, control='Max', plot=True, smode='Sim',iters=20)
    errorlist,timelist,aclabellist,modelerrors,fname=run_phase2_full(vary_tilt=vary_tilt)
    plot_ph2_error(fname,errorlist,aclabellist,aqfunctionsnames,modelerrors,noisetype)
    make_error_table(fname,noisetype)

