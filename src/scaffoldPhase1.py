#!/usr/bin/env python
import sys, os, time, IPython
import numpy as np
import matplotlib.pyplot as plt
from runPhase1 import *


NUM_EXPERIMENTS = 1

surfaces = ["flat", "smooth_sin1"]              # add another model ?
stops = [[1.343, 0.01, 3.6, 0.0],
         [0.0, 0.36, 0.0, 0.0]]                 # TODO change 0.0's (variance is not monotonically decreasing)
textures = ["_lam", "_text", "_spec", "_st"]
    # lambert, texture, specular, specular + texture
methods = ["random", "maxVar","maxVarGrad"]


def save_data(arr, name, surface_name):
    f = open("image_pairs/"+surface_name+'/'+name, 'w')
    f.write(str(np.array(arr).tolist()))
    f.close()
    return

def save_table(table, name):
    f = open(name + ".csv", 'wb')
    # header
    f.write(",,Random Probings,Max Acquisition\n")
    # data in table
    f.write("flat,lam,{},{},{}\n".format(*table[0]))
    f.write(",text,{},{},{}\n".format(*table[1]))
    f.write(",spec,{},{},{}\n".format(*table[2]))
    f.write(",st,{},{},{}\n".format(*table[3]))

    f.write("S,lam,{},{},{}\n".format(*table[4]))
    f.write(",text,{},{},{}\n".format(*table[5]))
    f.write(",spec,{},{},{}\n".format(*table[6]))
    f.write(",st,{},{},{}\n".format(*table[7]))

    f.close()
    return

def plot_error(errors, name, surface_name):
    plt.plot(errors)
    plt.xlabel("Iterations")
    plt.ylabel("RMS Error")
    plt.title("Integrated Error between Estimate and Ground Truth - Phase 1")
    plt.savefig("image_pairs/"+surface_name+'/'+name)
    plt.close()
    return

def run_phase1_full():
    iter_table = np.zeros((len(surfaces)*len(textures), len(methods)))
    error_table = np.zeros((len(surfaces)*len(textures), len(methods)))
    for i, surf in enumerate(surfaces):
        for j, text in enumerate(textures):
            # if i*len(textures) + j != 0:        # Use this to run only the nth surface
            #     continue
            for k in range(NUM_EXPERIMENTS): # repeat experiment number of times
                disparityMeas = None
                for l, method in enumerate(methods):
                    start = time.time()
                    disparityMeas, means, sigmas, sampled_points, measures, errors, num_iters = run_single_phase1_experiment(surf+text, method, disparityMeas, False, stops[i][j])
                    end = time.time()
                    time_elapsed = end - start # in seconds
                    # plot or save/record everything

                    iter_table[i*len(textures) + j, l] += num_iters / float(NUM_EXPERIMENTS)
                    error_table[i*len(textures) + j, l] += errors[-1] / float(NUM_EXPERIMENTS)
                    save_data([means, sigmas, sampled_points, measures, errors, num_iters, time_elapsed], 'data_'+method+"_exp"+str(k), surf+text)
                    save_table(iter_table, "phase1_iterations")
                    save_table(error_table, "phase1_errors")
                    plot_error(errors, "phase1_error_"+method+"_exp"+str(k), surf+text)
    return



if __name__ == "__main__":
    run_phase1_full()
    # IPython.embed()


