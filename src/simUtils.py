import numpy as np 
from getMap import getMap 


def getActualHeight (pos, modality=0):
	"""
	Get actual surface height at a point 'pos'
	"""
	x,y,xx,yy,z = getMap(modality) 
	h = z (pos[0], pos[1])
	return h
