import numpy as np 
# from getMap import getMap 
import numpy as np
import GPyOpt
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import *
from scipy import interpolate
from matplotlib import _cntr as cntr #to get polygon of getLevelSet
from shapely.geometry import asShape, Point, Polygon #to calculate point-polygon distances
from scipy.optimize import curve_fit
import pylab

from simulated_disparity import getStereoDepthMap, getObservationModel,getInterpolatedObservationModel



#######################################
# Curved surface functions for simulation: don't delete, but these
# should be replaced by some sort of output from Maya for the full
# simulation pipeline
#######################################
IMG_SIZE = 50

#######################################
# polygon test functions for simulating phase2: 
#######################################
squaretumor=np.array([[1.25,1.25],[2.75,1.25],[2.75,2.75],[1.25,2.75]])
thintumor=np.array([[2.25,0.25],[2.75,2.25],[2.75,2.75],[2.25,2.75]])
rantumor=.02*np.array([[2.25,0.75],[3.25,1.25],[2.75,2.25],[2.75,2.75],[2.25,2.75],[2.,1.25]])-.04

# def interp_function(image, workspace):
#     # creating interpolation functions
#     x = np.array(range(image.shape[0]))
#     y = np.array(range(image.shape[1]))
#     if (rangeX is not None and rangeY is not None):
#         return RGI((rangeX, rangeY), image, bounds_error=False, fill_value=0)
#     return RGI((x, y), image, bounds_error=False, fill_value=0)

# def getInterpolatedObservationModel(planeName):
#     model = getObservationModel(planeName)
#     if model is None:
#         return None
#     rangeX = np.array(range(IMG_SIZE))
#     rangeY = np.array(range(IMG_SIZE))
#     return interp_function(model, rangeX, rangeY)

def getInterpolatedGTSurface(surface, workspace):
    z = getObservationModel(surface)
    res = z.shape[0]
    x = np.linspace(workspace.bounds[0][0], workspace.bounds[0][1], num = res)
    y = np.linspace(workspace.bounds[1][0], workspace.bounds[1][1], num = res)
    f = interpolate.interp2d(x, y, z, kind='cubic')
    # f=getInterpolatedObservationModel(surface)

    return f

def getInterpolatedStereoMeas(surface, workspace):
    z = getStereoDepthMap(surface)[5:45,5:45]
    z = np.pad(z,((5,5),(5,5)),mode='edge')
    z[z<0]=0
    z[z>20]=20
    res = z.shape[0]
    x = np.linspace(workspace.bounds[0][0], workspace.bounds[0][1], num = res)
    y = np.linspace(workspace.bounds[1][0], workspace.bounds[1][1], num = res)
    f = interpolate.interp2d(x, y, z, kind='cubic')
    return f

def SimulateStereoMeas(surface, workspace, sensornoise=.001, subsample=True, numstereo=10):
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
    x = np.linspace(workspace.bounds[0][0], workspace.bounds[0][1], num = numstereo)
    y = np.linspace(workspace.bounds[1][0], workspace.bounds[1][1], num = numstereo)
    
    # sizeX = rangeX[1] - rangeX[0]
    # sizeY = rangeY[1] - rangeY[0]

    # xx, yy = np.meshgrid(x, y)

    # z = surface(xx,yy)

    if subsample==False:
        z = getStereoDepthMap(surface)[:40,:40]
        z = np.pad(z,((0,10),(0,10)),mode='edge')
    else:
        interpf = getInterpolatedStereoMeas(surface,workspace)
        # xx, yy = np.meshgrid(x, y)
        z = interpf(x,y)
        # z = interp(np.array(
        #     [xx.flatten(),yy.flatten()]).T).flat
    z = z #+ np.random.randn(z.shape[0],1)*sensornoise

    xx, yy, z = stereo_pad(x,y,z,workspace.bounds[0],workspace.bounds[1])

    return xx, yy, z


def getSimulatedStereoMeas(surface, workspace, plot = True):
    """
    wrapper function for SimulateStereoMeas
    hetero. GP model requires defining the variance for each measurement 
    standard stationary kernel doesn't need this

    should fix these functions so they're not necessary by default...
    """
    xx, yy, z = SimulateStereoMeas(surface, workspace)

    # we assume Gaussian measurement noise:
    sigma_g = .1
    focalplane=workspace.bounds[1][1]/2.0
    # noise component due to curvature:
    # finite differencing
    #xgrid = np.vstack([xx.flatten(), yy.flatten()]).T
    grad = np.gradient(z)
    dx,dy = grad
    sigma_fd = np.sqrt(dx**2+dy**2)
    
    sigma_fd[np.isinf(sigma_fd)]=0

    # todo: noise due to  offset uncertainty
    sigma_offset=(yy-focalplane)**2
    # weighted total noise for measurements
    sigma_total = sigma_g + 0*sigma_fd  + .001*sigma_offset

    if plot==True:
        # plot the surface from disparity
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, z, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
        ax.set_title("Depth from Disparity")
        ax.set_zlim3d(0,20)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return np.array([xx.flatten(), yy.flatten(),
                     z.flatten(),
                     sigma_total.flatten()]).T




def SimulateProbeMeas(surface, workspace, sample_locations, sensornoise = .00001):
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
    interp = getInterpolatedGTSurface(surface, workspace)
    z = interp(sample_locations[0][0],sample_locations[0][1]) 
    for pt in sample_locations[1:]:
        tmp = interp(pt[0],pt[1]) 
        z=np.vstack((z,tmp))

    z = z.flatten() + sensornoise*np.random.randn(z.shape[0])
    z = np.array(z)

    return xx, yy, z

def getSimulatedProbeMeas(surface, workspace, sample_points):
    """
    wrapper function for SimulateProbeMeas
    hetero. GP model requires defining the variance for each measurement 
    standard stationary kernel doesn't need this
    """
    xx,yy,z = SimulateProbeMeas(surface, workspace, sample_points)
    # we assume Gaussian measurement noise:
    noise=.000001
    sigma_t = np.full(z.shape, noise)

    return np.array([xx, yy,
                     z,
                     sigma_t]).T

def SimulateStiffnessMeas(poly, sample_locations, sensornoise = .01):
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

def fit_measmodel():
    xdata = np.array([0.0,   1.0,  3.0,  4.3,  7.0,   8.0,   8.5, 10.0,  
    12.0, 14.0])
    ydata = np.array([0.11, 0.12, 0.14, 0.21, 0.83,  1.45,   1.78,  1.9, 
    1.98, 2.02])

    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    print "Fit:"
    print "x0 =", popt[0]
    print "k  =", popt[1]
    print "a  =", popt[2]
    print "c  =", popt[3]
    print "Asymptotes are", popt[3], "and", popt[3] + popt[2]

    x = np.linspace(-1, 15, 50)
    y = sigmoid(x, *popt)


    pylab.plot(xdata, ydata, 'o', label='data')
    pylab.plot(x,y, label='fit')
    pylab.ylim(0, 2.05)
    pylab.legend(loc='upper left')
    pylab.grid(True)
    pylab.show()

def plotSimulatedStiffnessMeas(poly, workspace, ypos=None, sensornoise = .03):
    if ypos==None:
       ypos=(workspace.bounds[1][1]-workspace.bounds[1][0])/2.0+workspace.bounds[1][0]
       print ypos
    x = np.arange(workspace.bounds[0][0], workspace.bounds[0][1], 1/float(10000))
    y = np.zeros(x.shape)+ypos
    sample_locations = np.array([x,y]).T

    meas = SimulateStiffnessMeas(poly, sample_locations, sensornoise=sensornoise)
    meas = meas[2]
    print sample_locations
    print meas
    plt.plot(x.flatten(), meas.flatten(), linewidth=3.0)
    plt.show()

def getSimulateStiffnessMeas(surface, sample_points):
    """wrapper function for SimulateProbeMeas hetero. GP model requires
    defining the variance for each measurement standard stationary
    kernel doesn't need this

    """
    xx,yy,z = SimulateStiffnessMeas(surface, sample_points)

    # we assume Gaussian measurement noise:
    noise=.001
    sigma_t = np.full(z.shape, noise)
    return np.array([xx, yy,
                     z,
                     sigma_t]).T

#######################################
# LM question: this function was already in here--not sure if it does anything?
#######################################

# def getActualHeight (pos, modality=0):
#     """
#     Get actual surface height at a point 'pos'
#     """
#     x,y,xx,yy,z = getMap(modality) 
#     h = z (pos[0], pos[1])
#     return h

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
# Functions for simulating deflection measurements
#######################################

def sigmoid(dist, alpha=1000, a=0.0, b=1.0, c=-0.004):
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


# def getLevelSet (Pts, z, level):
#     """
#     Input: 
#         Pts: x, y - 2-d numpy array N-by-d
#         Z : values at each of the points 1-D numpy array length N
#         level: find the polygon at level

#     output:
#         poly: ordered list of points on the polygon 2-d array
#     """
#     x = Pts[:,0]
#     y = Pts[:,1]
#     c = cntr.Cntr(x, y, z)

#     res = c.trace(level)
#     nseg = len(res) // 2
#     segments, codes = res[:nseg], res[nseg:]
#     poly = segments[0]

#     return poly
def getLevelSet (workspace, mean, level):
    """
    Input: 
        Pts: x, y - 2-d numpy array N-by-d
        Z : values at each of the points 1-D numpy array length N
        level: find the polygon at level

    output:
        poly: ordered list of points on the polygon 2-d array
    """
    x = workspace.xx #GPdata[0]
    y = workspace.yy# GPdata[1]
    z = gridreshape(mean,workspace) #GPdata[2]
    c = cntr.Cntr(x, y, z)

    res = c.trace(level)
    if len(res)>0:
        nseg = len(res) // 2
        segments, codes = res[:nseg], res[nseg:]
        poly = segments[0]
    else:
        poly=[]
    return np.array(poly)

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
        point_in_poly =  Point(tuple(xProbe[p])).within(
            Polygon(boundaryEst_tuples))
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
