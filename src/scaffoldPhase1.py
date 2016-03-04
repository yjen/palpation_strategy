#!/usr/bin/env python
import sys, os, time, IPython
import numpy as np
import matplotlib.pyplot as plt
from runPhase1 import *


NUM_EXPERIMENTS = 1

surfaces = ["flat", "smooth_sin1"]              # add another model ?
stops = [[6.4, 0.01, 3.6, 0.0],
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
    f.write("flat,lam,{},{}\n".format(table[0][0], table[0][1]))
    f.write(",text,{},{}\n".format(table[0][0], table[0][1]))
    f.write(",spec,{},{}\n".format(table[0][0], table[0][1]))
    f.write(",st,{},{}\n".format(table[0][0], table[0][1]))

    f.write("S,lam,{},{}\n".format(table[1][0], table[1][1]))
    f.write(",text,{},{}\n".format(table[1][0], table[1][1]))
    f.write(",spec,{},{}\n".format(table[1][0], table[1][1]))
    f.write(",st,{},{}\n".format(table[1][0], table[1][1]))

    f.close()
    return

def run_phase1_full():
    iter_table = np.zeros((8, 2))
    error_table = np.zeros((8, 2))
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

                    iter_table[i*len(textures) + j][l] += num_iters / float(NUM_EXPERIMENTS)
                    error_table[i*len(textures) + j][l] += errors[-1] / float(NUM_EXPERIMENTS)
                    save_data([means, sigmas, sampled_points, measures, errors, num_iters, time_elapsed], '_exp'+str(l), surf+text)
                    save_table(iter_table, "phase1_iterations")
                    save_table(error_table, "phase1_errors")
    return



if __name__ == "__main__":
    run_phase1_full()
    # IPython.embed()


