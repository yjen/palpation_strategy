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
# from figures import SIZE, BLUE, GRAY
# from shapely.geometry import Point
# from descartes import PolygonPatch

# data1 = pickle.load(open("../scripts/saved_palpation_data/single_row_raster_100x.p", "rb"))
# # data2 = pickle.load(open("probe_data_L2R.p", "rb"))
# xdata,zdata=get_stiffness_data(data1)
# model=fit_measmodel(xdata,zdata)
# mod=plot_model(data1,model,scale=True)
# print mod



def save_table(table, name):
    f = open(name + ".csv", 'wb')
    # header
    f.write(",,UCB_GPIS_implicitlevel,UCB_GBIS,UCB_GB,MaxVar_GP\n") #MaxVar_plus_gradient
    # data in table
    # f.write("flat,lam,{},{}\n".format(table[0][0], table[0][1]))
    f.write(",tumor1: iterations,{},{},{}\n".format(table[0][0], table[0][1],table[0][2],table[0][3]))#,table[0][4]))
    f.write(",tumor1: healthy tissue removed,{},{},{}\n".format(table[1][0], table[1][1],table[1][2],table[1][3]))#,table[1][4]))
    f.write(",tumor1: tumor left behind,{},{},{}\n".format(table[2][0], table[2][1],table[2][2],table[2][3]))#,table[2][4]))

    f.write(",tumor2: iterations,{},{},{}\n".format(table[3][0], table[3][1],table[3][2],table[3][3]))#,table[3][4]))
    f.write(",tumor2: healthy tissue removed,{},{},{}\n".format(table[4][0], table[4][1],table[4][2],table[4][3]))#,table[4][4]))
    f.write(",tumor2: tumor left behind,{},{},{}\n".format(table[5][0], table[5][1],table[5][2],table[5][3]))#,table[5][4]))

    # f.write(",st,{},{}\n".format(table[0][0], table[0][1]))

    # f.write("S,lam,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",text,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",spec,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",st,{},{}\n".format(table[1][0], table[1][1]))

    f.close()
    return

def save_data(arr, dirname, name):
    directory=dirname

    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(directory+'/data_'+name, 'w')
    f.write(str(np.array(arr).tolist()))
    f.close()
    return

NUM_EXPERIMENTS = 5


tumors = [squaretumor,rantumor]              # add another model ?
stops = [[6.4, 0.01, 3.6, 0.0],
         [0.0, 0.36, 0.0, 0.0]]                 # TODO change 0.0's (variance is not monotonically decreasing)
# textures = ["_lam", "_text", "_spec", "_st"]
    # lambert, texture, specular, specular + texture

# acquisition functions:  MaxVar_GP, UCB_GP, EI_GP, UCB_GPIS, EI_IS, MaxVar_plus_gradient
# MaxVar_plus_gradient(model, workspace, level=0, x=None, acquisition_par=0,numpoints=1)
aqfunctions = [UCB_GPIS_implicitlevel,UCB_GPIS,UCB_GP,MaxVar_GP]#MaxVar_plus_gradient
aqfunctionsnames = ["UCB_GPIS_implicitlevel","UCB_GPIS","UCB_GP", "MaxVar_GP"]#, "random"]"MaxVar_plus_gradient"

controls =["Max"]



def run_phase2_full():

    # iter_table = np.zeros((len(aqfunctions)+len(controls),len(tumors)))
    error_table = np.zeros((len(tumors)*3,len(aqfunctions)+len(controls)))
    # print iter_table.shapetumor
    # error_table1 = np.zeros((2, len(aqfunctions)+len(controls)))
    tumorerrlistleft=[]
    tumorerrlistremoved=[]
    for i, tumor in enumerate(tumors):
        acqerrlistleft=[]
        acqerrlistremoved=[]
        aclabellist=[]
        for j, acq in enumerate(aqfunctions):
            conterrlistleft=[]
            conterrlistremoved=[]
            contlabellist=[]  
            for m, cont in enumerate(controls):
                # if i*len(textures) + j != 0:        # Use this to run only the nth surface
                #     continue
                its = [0.0, 0.0]
                # errors_per_method = []
                experrlistleft=[]
                experrlistremoved=[]
                for k in range(NUM_EXPERIMENTS): # repeat experiment 5 times
                    # disparityMeas = None
                    # for l, method in enumerate(methods):
                    start = time.time()
                    dirname = str(i) + '_' + aqfunctionsnames[j] + '_' + cont

                    means, sigmas, acqvals, measures, healthyremoved, tumorleft, num_iters, gpmodel = run_single_phase2_simulation(
                        tumor, dirname, AcFunction=acq, control=cont, plot=True, mode='RecordedExp')
                    plt.close() 
                    end = time.time()
                    time_elapsed = end - start # in seconds
                    # plot or save/record everything
                    # its[k] += num_iters / 5.0
                    save_data([means, sigmas, acqvals, measures, healthyremoved, tumorleft, num_iters], dirname, str(k))
                    # iter_table[j + m][i]+= num_iters / float(NUM_EXPERIMENTS)
                    # 3=num of errors in table
                    error_table[i+i*2+0][j+m]+= num_iters / float(NUM_EXPERIMENTS)
                    error_table[i+i*2+1][j+ m]+= healthyremoved[-1] / float(NUM_EXPERIMENTS)
                    error_table[i+i*2+2][j+ m]+= tumorleft[-1] / float(NUM_EXPERIMENTS)
                    experrlistleft.append(tumorleft)  
                    experrlistremoved.append(healthyremoved)  
                    print k

                print error_table
                # save_table(iter_table, "phase2_iterations")
                # save_table(error_table, "phase2_errors")
                # save_table(error_table1, "phase2_errors1")
                conterrlistleft.append(experrlistleft)
                conterrlistremoved.append(experrlistremoved)
                contlabellist.append(dirname)
            acqerrlistleft.append(conterrlistleft)
            acqerrlistremoved.append(conterrlistremoved)
            aclabellist.append(contlabellist)
        # plot_error(errors_per_method, "phase1_error_exp"+str(k), surf+text)

        tumorerrlistleft.append(acqerrlistleft)
        tumorerrlistremoved.append(acqerrlistremoved)
    return np.array(tumorerrlistleft),np.array(tumorerrlistremoved),np.array(aclabellist)



if __name__ == "__main__":
    dirname='tt'
    run_single_phase2_simulation(rantumor, dirname, AcFunction=UCB_GPIS, control='Max', plot=True, smode='Sim',iters=20)
    # outleft,outrem,aclabellist=run_phase2_full()
    # plot_ph2_error(outrem,outleft,aclabellist)

