import numpy as np 
# from getMap import getMap 
import numpy as np
import GPyOpt
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import *

from matplotlib import _cntr as cntr #to get polygon of getLevelSet
from shapely.geometry import asShape, Point, Polygon #to calculate point-polygon distances



#######################################
# Curved surface functions for simulation: don't delete, but these
# should be replaced by some sort of output from Maya for the full
# simulation pipeline
#######################################

def SimulateStereoMeas(surface, rangeX, rangeY, sensornoise=.01, gridSize = 20):
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
    z = z + np.random.randn(z.shape[0],1)*sensornoise

    xx, yy, z = stereo_pad(x,y,z,rangeX,rangeY)
    return xx, yy, z

def SimulateProbeMeas(surface, sample_locations, sensornoise = .001):
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

    # this is a simulated measurement-- add noise!
    z = surface(xx,yy)
    z = z + sensornoise*np.random.randn(z.shape[0])

    return xx, yy, z

def SimulateStiffnessMeas(poly, sample_locations, sensornoise = .001):
    """Simulate measurements from palpation (tapping mode) for the test
    functions above inputs: *surface: a function defining a test surface
    *locations: list of points [[x1,y1],[x2,y2]] outputs: *xx,yy, z,
    matrices

    This functions would be replaced by experiment

    """
    # unpack
    xx, yy = sample_locations.T

    # this is a simulated measurement, add noise
    
    z = makeMeasurement_LS(sample_locations, poly)
    z = z + sensornoise*np.random.randn(z.shape[0])

    return xx, yy, z

#######################################
# LM question: this function was already in here--not sure if it does anything?
#######################################

def getActualHeight (pos, modality=0):
    """
    Get actual surface height at a point 'pos'
    """
    x,y,xx,yy,z = getMap(modality) 
    h = z (pos[0], pos[1])
    return h

#######################################
# Curved surface functions for simulating phase1: don't delete, but these
# should be replaced by some sort of output from Maya for the full
# simulation pipeline
#######################################

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
    z = z

    return z

def gaussian_tumor(xx,yy):
    mu =[.1,.1]
    var = [.3,.3]
  
    z= np.exp(-((xx - mu[0])**2/( 2*var[0]**2)) -
           ((yy - mu[1])**2/(2*var[1]**2))) 
       
    return z

def uniform_pdf(X):
    return np.ones(X.shape)

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

#######################################
# polygon test functions for simulating phase2: 
#######################################
squaretumor=np.array([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]])


#######################################
# Functions for simulating deflection measurements
#######################################

def sigmoid(dist, alpha=50, a=0, b=1, c=-0.1):
    """  
    a, b: base and max readings of the probe with and without tumor
    dist = xProbe-xEdge
    xProbe: 
    xEdge: the center of sigmoid
    alpha: slope  
    Output:
    y = a + (b-a)/(1+ exp(-alpha*(xProbe-xEdge)))
    """
    y = a + np.divide((b-a),(1+ np.exp(-alpha*(dist-c))))  
    return y


def getLevelSet (Pts, z, level):
    """
    Input: 
        Pts: x, y - 2-d numpy array N-by-d
        Z : values at each of the points 1-D numpy array length N
        level: find the polygon at level

    output:
        poly: ordered list of points on the polygon 2-d array
    """
    x = Pts[:,0]
    y = Pts[:,1]
    c = cntr.Cntr(x, y, z)

    res = c.trace(level)
    nseg = len(res) // 2
    segments, codes = res[:nseg], res[nseg:]
    poly = segments[0]

    return poly


def makeMeasurement_LS(xProbe, boundaryEstimate):
    """
    makeMeasurement_LS : make measurement with Level Set
    xProbe: desired point of measurement
    boundaryEstimate: current estimate of the boundary -- list of lists
    """
    # convert list of lists to list of tuples
    boundaryEst_tuples = [tuple(p) for p in boundaryEstimate]
    
    #get distance of xProbe form polygon
    sizeProbe = xProbe.shape[0]
    z = []
    for p in range(sizeProbe):
        dist= Point(tuple(xProbe[p])).distance(Polygon(boundaryEst_tuples))
        point_in_poly =  Point(tuple(xProbe[p])).within(Polygon(boundaryEst_tuples))
        #create signed distance function
        if point_in_poly:
            dist = dist
        else:
            dist = -dist

        z.append(sigmoid(dist))
    #calculate the sigmoidal measurement value due to this distance
    #z = sigmoid(dist)
    
    # return measurement value z
    return np.array(z)

# debug plotting
# x = np.arange(-10, 10, 0.2)
# sig = sigmoid(x,0)
# plt.plot(x, sig, linewidth=3.0)

def probeMeasure(xProbe, Pts, z, level):
    """
    input: 
        xProbe: desired point(s) of measurement
        Pts: x, y - 2-d numpy array N-by-d
        Z : values at each of the points 1-D numpy array length N
        level: find the polygon at level
    """
    poly = getLevelSet(Pts, z, level)
    z = makeMeasurement_LS(xProbe, poly)
    return z
