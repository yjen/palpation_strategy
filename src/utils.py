import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.path as path
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm #colormap

def plotBelief (xx,yy,z):
	#xx,yy -- matrix obtained from meshgrid
	# z -- matrix of values		
	fig = plt.figure()	
	ax = fig.gca(projection='3d')

	ax.plot_surface(xx, yy, z, rstride=5, cstride=5, alpha=0.4)
	cset = ax.contourf(xx, yy, z, zdir='z', offset=-2, cmap=cm.coolwarm)
	cset = ax.contourf(xx, yy, z, zdir='x', offset=-5, cmap=cm.coolwarm)
	cset = ax.contourf(xx, yy, z, zdir='y', offset=5, cmap=cm.coolwarm)

	ax.set_xlabel('X')
	ax.set_xlim(-5, 5)
	ax.set_ylabel('Y')
	ax.set_ylim(-5, 5)
	ax.set_zlabel('Z')
	ax.set_zlim(-2, 1)

	plt.show(block=False)	
	# plt.draw()

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# # simulates probe & tumor. questionable
# def get_probe_data(self):
# 	pos = self.calculate_tip_pos()
# 	if pos[1] <= 0:
# 		if abs(0.5 - pos[0]) <= 0.125:
# 			#print(4096 -
# 			 abs(0.25 - pos[0]) * 4000)
# 			#return 4096 - abs(0.25 - pos[0]) * 4000
# 			return 4096 * gaussian(pos[0], 0.5, 0.1) + 64 * random.gauss(0, 6)  
# 	return 64 * random.gauss(0, 6)

class Params():
	def __init__(self):
		self.gridSize = [100,100]
		self.minHeight = -1
		self.maxHeight = 1
		print "Initializing Parameters"						

	def getParams(self):
		return self

	def setParams(self):
		pass

########################## New--added by Lauren

def stereo_pad(x,y,z,rangeX,rangeY):
        # pad the stereo measurements by a fixed amount. Necessary to avoid weird undertainty at the edges of the region when using Gaussian Processes
        percentpad=.1

        padbyX = percentpad*(rangeX[1]-rangeX[0])
        padbyY = percentpad*(rangeY[1]-rangeY[0])

        # how many extra points to add in each direction
        gridSize=20
        numpads=np.int(gridSize*percentpad)

        # pad grid arrays

        x=np.pad(x,(numpads,numpads),mode='linear_ramp',
                 end_values=(rangeX[0]-padbyX,rangeX[1]+padbyX))
        y=np.pad(y,(numpads,numpads),mode='linear_ramp',
                 end_values=(rangeY[0]-padbyY,rangeY[1]+padbyY))
        xx, yy = np.meshgrid(x, y)

        z=np.pad(z,numpads,mode='edge')
        # plt.scatter(xx, yy, c=z)
        # plt.show()
        return xx,yy,z

########################## Plot Scripts


