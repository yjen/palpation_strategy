import sys
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.path as path

from utils import *
from getMap import getMap 
# import cplex, gurobipy

def planning(verbose=False):
	"""
	Input Options Description
	"""

	# Initialize Belief -- this is akin to gettig camera estimate
	#x,y vectors, xx,yy,z- is a matrix
	x,y,xx,yy,z = getMap() 

	if verbose:	
		plotBelief (xx,yy,z)		

	# Planning Trajectory
	

	# Collect Observations. 

	# Observation Model

	# Sensor model --is there a difference

	# Belief Update

	# update information  - Fisher information?


if __name__ == "__main__":
	 planning(verbose=True)