import numpy as np 
# from getMap import getMap 
import numpy as np
import GPyOpt
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import *

def getActualHeight (pos, modality=0):
	"""
	Get actual surface height at a point 'pos'
	"""
	x,y,xx,yy,z = getMap(modality) 
	h = z (pos[0], pos[1])
	return h

def GaussianSurface(xx, yy):
        """
        test function for simulation: Gaussian Surface
        standin for input from stero simulations
        
        xx,yy: test points to evaluate function
	"""
        mu = [0,0]
        mu2 =[.2,.2]
        var = [.5,.7]
      
        z= np.exp(-((xx - mu[0])**2/( 2*var[0]**2)) -
              ((yy - mu[1])**2/(2*var[1]**2))) + \
              np.exp(-((xx - mu[0])**2/( 2*var[0]**2)) -
                     ((yy - mu[1])**2/(2*var[1]**2))) + \
                     np.exp(-((xx - mu2[0])**2/( 2*var[0]**2)) -
                            ((yy - mu2[0])**2/(2*var[1]**2)))

        #noise=np.random.randn(z.flatten().shape[0],1)*noisevar
        z = z#+noise.reshape(z.shape)

        return z

def SixhumpcamelSurface(xx,yy):
        """
        test function for simulation: simulated surface from GPyopt
        standin for input from stero simulations
        
        xx,yy: test points to evaluate function
	"""
        sixhumpcamel = GPyOpt.fmodels.experiments2d.sixhumpcamel().f
        
        # function takes in list of (x,y) pairs:
        xgrid = np.vstack([xx.flatten(), yy.flatten()]).T

        # evaluate function
        z = sixhumpcamel(xgrid)

        # add noise
        z = z#+np.random.randn(xgrid.shape[0],1)*noise

        # reshape
        z = z.reshape(xx.shape)

        return z

def SimulateStereoMeas(surface, rangeX,
                       rangeY, noisevar=.01,
                       gridSize = 20,
                       plot = True):
	"""
	simulate measurements from stereo depth mapping for the test functions above
        
        inputs:
           *surface: a function defining a test surface
           *rangeX, rangeY: boundaries of the regions
           *gridSize: resolution for simulated stereo measurements

        outputs:
           *xx,yy, z, matrices

        This functions would be replaced by experiment

	"""
	x = np.linspace(rangeX[0], rangeX[1], num = gridSize)
	y = np.linspace(rangeY[0], rangeY[1], num = gridSize)
	
	sizeX = rangeX[1] - rangeX[0]
	sizeY = rangeY[1] - rangeY[0]

	xx, yy = np.meshgrid(x, y)

        z = surface(xx,yy)
        z = z+np.random.randn(z.shape[0],1)*noisevar

        xx, yy, z = stereo_pad(x,y,z,rangeX,rangeY)

        if plot==True:
            # plot the surface from disparity
            fig = plt.figure(figsize=(4, 4))
            ax = fig.gca(projection='3d')
            ax.plot_surface(xx, yy, z, rstride=1, cstride=1,
                            cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_title("Depth from Disparity")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

	return xx, yy, z

def SimulateProbeMeas(surface, sample_locations):
	"""
        Simulate measurements from palpation (tapping mode) for the test functions above
        inputs:	
           *surface: a function defining a test surface
           *locations: list of points [[x1,y1],[x2,y2]]
        outputs:
           *xx,yy, z, matrices

        This functions would be replaced by experiment
        """

        # unpack
        xx, yy = sample_locations.T

        # this is a simulated measurement, add noise
        sensornoise = .001
        
        z = surface(xx,yy)
        z = z + sensornoise*np.random.randn(z.shape[0])

	return xx, yy, z
