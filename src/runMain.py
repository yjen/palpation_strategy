import sys
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.path as path

from utils import *
from getMap import getMap
from sensorModel import *
# import cplex, gurobipy


def planning(verbose=False):
    """
    Input Options Description
    """

    # Initialize Belief -- this is akin to gettig camera estimate
    # x,y vectors, xx,yy,z- is a matrix
    x, y, xx, yy, z = getMap(modality=1)

    if verbose:
        plotBelief(xx, yy, z)

    # add noise and normalize to get belief
    z += 0.05 * np.random.standard_normal(z.shape)
    if verbose:
        plotBelief(xx, yy, z)

    # Calculate FIM

    # calculate EID

    # Planning Trajectory

    # greedy Local
    # greedy FI

    # tEID -- refer the paper.

    # Collect Observations- simulate
    # Sensor simulator
    h = sensorHeight(z, probePos)
    # Belief Update

    # update information  - Fisher information?

    # To keep all the plots made during the execution.
    plt.show()

if __name__ == "__main__":
    planning(verbose=True)
