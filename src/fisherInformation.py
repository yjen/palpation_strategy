import numpy as np
from utils import *


def fisherHeight(height, probePos, gridSize):
    """
    """
    # params = Params()
    # sizeX = gridSize[0]
    # sizeY = gridSize[1]


def EID(probePos, pMap):
    # sigSensor controls the dropoff
    # sigSensor = sqrt(2) * spatialRes
    # normalization constant for the palpation probe
    # eta = 1
    # eps = 1e-5  # results in using truncated gaussians

    # integrate
    for i in np.linspace(minHeight, maxHeight, num=100):
        pass


def getProbeValue(posToEstimate, probePos):
    z = eta * exp(-pow(2 * (sigSensor**2), -1) * ((H_actual - H_probePose)**2))
    return z


def probHeight(h):
    meanH = 0  # what should this value be?
    sigmaH = 1
    z = pow(2 * np.pi * (sigmaH ** 2), -0.5) * \
        exp(-pow(2 * (sigSensor**2), -1) * ((h - meanH)**2))
    return z


def fisherPalpation(tumor_location, probePos, gridSize):
    pass
