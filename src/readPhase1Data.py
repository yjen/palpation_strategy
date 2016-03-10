#!/usr/bin/env python
import sys, os, time, IPython
import numpy as np
import matplotlib.pyplot as plt

NUM_EXPERIMENTS = 1

surfaces = ["flat", "smooth_sin1"]
textures = ["_lam", "_text", "_spec", "_st"]
    # lambert, texture, specular, specular + texture
methods = ["random", "maxVar","maxVarGrad"]


def save_data(arr, name, surface_name):
    f = open("image_pairs/"+surface_name+'/'+name, 'w')
    f.write(str(np.array(arr).tolist()))
    f.close()
    return

def get_data(name, surface_name):
    f = open("image_pairs/"+surface_name+'/'+name, 'r')
    string = f.read()
    array = np.array;
    data = np.array(eval(string))
    f.close()
    return data

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
    for e in errors:
        plt.plot(e)
    plt.xlabel("Iterations")
    # plt.xlim(0, 30)
    plt.ylabel("RMS Error")
    plt.title("Integrated Error between Estimate and Ground Truth - Phase 1")
    plt.legend(methods, loc='upper right')
    plt.savefig("image_pairs/"+surface_name+'/'+name)
    plt.close()
    return

def read_errors_from_data():
    error_table = np.zeros((len(surfaces)*len(textures), len(methods)))
    for i, surf in enumerate(surfaces):
        for j, text in enumerate(textures):
            for k in range(NUM_EXPERIMENTS): # repeat experiment number of times
                disparityMeas = None
                errors_per_method = []
                for l, method in enumerate(methods):
                    data = get_data('data_'+method+"_exp"+str(k), surf+text)
                    means, sigmas, sampled_points, measures, errors, num_iters, time_elapsed = data
                    error_table[i*len(textures) + j, l] += errors[-1] / float(NUM_EXPERIMENTS)
                    errors_per_method.append(errors)
                plot_error(errors_per_method, "phase1_error_exp"+str(k), surf+text)
                # return
    save_table(error_table, "phase1_errors")
    return



if __name__ == "__main__":
    read_errors_from_data()
    # IPython.embed()


