import numpy as np
from IPython import embed

def getMap (rangeX = [-5,5], rangeY = [-5,5], gridSize = 100, modality=2):
	"""
	modality = {0(flat),1(unimodal depression), 2 (bimodal depressions)}
	"""
	x = np.linspace(rangeX[0], rangeX[1], num = gridSize)
	y = np.linspace(rangeY[0], rangeY[1], num = gridSize)
	
	sizeX = rangeX[1] - rangeX[0]
	sizeY = rangeY[1] - rangeY[0]

	xx, yy = np.meshgrid(x, y)

	if modality == 0:
		# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)  
		# embed()
		z = np.zeros_like(xx)

	elif modality ==1:        
		# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)  
		# embed()
		z = -np.exp (-(xx**2 + yy**2)/max(sizeX, sizeY)) #centered around zero

	elif modality == 2:
		minDim = min(sizeX, sizeY)
		z = -(np.exp (-((xx-0.25*minDim)**2 + (yy-0.25*minDim)**2)/max(sizeX, sizeY)) \
			+ np.exp (-((xx+0.25*minDim)**2 + (yy+0.25*minDim)**2)/max(sizeX, sizeY))) 

	elif modality > 2:
		print "modality can only be {1,2}"
		return

	return x,y,xx,yy,z
