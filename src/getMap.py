import numpy as np
from IPython import embed

def getMap (rangeX = [-5,5], rangeY = [-5,5] , gridSize = 100 ,type=1):
	"""
	Type = {1(sinusoid depression), 2 (2 sinusoid depressions)}
	"""
	x = np.linspace(rangeX[0], rangeX[1], num = gridSize)
	y = np.linspace(rangeY[0], rangeY[1], num = gridSize)
	
	sizeX = rangeX[1] - rangeX[0]
	sizeY = rangeY[1] - rangeY[0]

	xx, yy = np.meshgrid(x, y)

	if type ==1:        
		# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)  
		# embed()
		z = -np.exp (-(xx**2 + yy**2)/max(sizeX, sizeY)) #centered around zero

	elif type == 2:
		# z = -np.exp (-(xx**2 + yy**2)/max(sizeX, sizeY)) #centered around zero
		pass

	return x,y,xx,yy,z
