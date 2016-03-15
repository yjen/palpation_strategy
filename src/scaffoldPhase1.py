#!/usr/bin/env python
import sys, os, time, IPython
import numpy as np
import matplotlib.pyplot as plt
from runPhase1 import *
from plotscripts import *

NUM_EXPERIMENTS = 5

surfaces = ["smooth_sin1","flat"]              # add another model ?
stops = [[1.343, 1.343, 1.343, 1.343],
         [1.343, 1.343, 1.343, 1.343]]
textures = [ "_text", "_spec", "_st","_lam"]
    # lambert, texture, specular, specular + texture
methods = ["random", "maxVarGrad", "maxVar", "UCB_dGPIS", "UCB_dGPIS2", "UCB_GP"]


def save_data(arr, name, surface_name):
    f = open("image_pairs/"+surface_name+'/'+name, 'w')
    f.write(str(np.array(arr).tolist()))
    f.close()
    return

def save_table(table, name):
    formatter = ",{}"*len(methods) + "\n"
    f = open(name + ".csv", 'wb')
    # header
    f.write(",,Random Probings,Max Variance,Max Variance Gradient\n")
    # data in table
    f.write(("flat,lam"+formatter).format(*table[0]))
    f.write((",text"+formatter).format(*table[1]))
    f.write((",spec"+formatter).format(*table[2]))
    f.write((",st"+formatter).format(*table[3]))

    f.write(("S,lam"+formatter).format(*table[4]))
    f.write((",text"+formatter).format(*table[5]))
    f.write((",spec"+formatter).format(*table[6]))
    f.write((",st"+formatter).format(*table[7]))

    f.close()
    return

def plot_error(errors, name, surface_name):
    errors = np.array(errors)
    errors = np.mean(errors, axis=0)
    for e in errors:
        plt.plot(e)
    plt.xlabel("Iterations")
    # plt.xlim(0, 30)
    plt.ylabel("RMS Error")
    plt.title("Integrated Error b/w Estimate and GT - Phase 1 - "+str(NUM_EXPERIMENTS)+" Experiments")
    plt.legend(methods, loc='upper right')
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
            errors_per_experiment = []
            for k in range(NUM_EXPERIMENTS): # repeat experiment number of times
                disparityMeas = None
                errors_per_method = []
                for l, method in enumerate(methods):
                    print "Running " + surf+text + " "+ method + ":"
                    start = time.time()
                    disparityMeas, means, sigmas, sampled_points, measures, errors, num_iters = run_single_phase1_experiment(surf+text, method, disparityMeas, False, stops[i][j], shouldPlot=False)
                    end = time.time()
                    time_elapsed = end - start # in seconds
                    # plot or save/record everything

                    iter_table[i*len(textures) + j, l] += num_iters / float(NUM_EXPERIMENTS)
                    error_table[i*len(textures) + j, l] += errors[-1] / float(NUM_EXPERIMENTS)
                    save_data([means, sigmas, sampled_points, measures, errors, num_iters, time_elapsed], 'data_'+method+"_exp"+str(k), surf+text)
                    save_table(iter_table, "phase1_iterations")
                    save_table(error_table, "phase1_errors")
                    errors_per_method.append(errors)
                errors_per_experiment.append(errors_per_method)
            plot_error(errors_per_experiment, "phase1_error", surf+text)
    return



if __name__ == "__main__":
    run_phase1_full()
    # IPython.embed()


