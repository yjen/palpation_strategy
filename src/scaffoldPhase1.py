#!/usr/bin/env python
import sys, os, time, IPython
import numpy as np
import matplotlib.pyplot as plt
from runPhase1 import *


surfaces = ["flat", "smooth_sin1"]              # add another model ?
stops = [[6.4, 0.01, 3.6, 0.0],
         [0.0, 0.36, 0.0, 0.0]]                 # TODO change 0.0's (variance is not monotonically decreasing)
textures = ["_lam", "_text", "_spec", "_st"]
    # lambert, texture, specular, specular + texture
methods = ["random", "maxAcquisition"]


def save_data(arr, name, surface_name):
    f = open("image_pairs/"+surface_name+'/'+name, 'w')
    f.write(str(arr))
    f.close()
    return

def run_phase1_full():
    for i, surf in enumerate(surfaces):
        for j, text in enumerate(textures):
            # if i*len(textures) + j != 0:        # Use this to run only the nth surface
            #     continue
            its = [0.0, 0.0]
            for k in range(5): # repeat experiment 5 times
                disparityMeas = None
                for l, method in enumerate(methods):
                    start = time.time()
                    disparityMeas, means, sigmas, sampled_points, measures, num_iters = run_single_phase1_experiment(surf+text, method, disparityMeas, False, stops[i][j])
                    end = time.time()
                    time_elapsed = end - start # in seconds
                    # plot or save/record everything
                    its[l] += num_iters / 5.0
                    save_data([means, sigmas, sampled_points, measures, num_iters, time_elapsed], '_exp'+str(l), surf+text)
            print its
    return



if __name__ == "__main__":
    run_phase1_full()
    # IPython.embed()



"""
termination condition
save data
fill out chart
create image pairs of models
"""
