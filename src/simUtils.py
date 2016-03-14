import numpy as np 
import sys

sys.path.append('../scripts')

# from getMap import getMap 
import GPyOpt
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import *
from scipy import interpolate
from matplotlib import _cntr as cntr #to get polygon of getLevelSet
from shapely.geometry import asShape, Point, Polygon, LineString, MultiPoint#to calculate point-polygon distances
from shapely.ops import cascaded_union
import numpy as np

from simulated_disparity import getStereoDepthMap, getObservationModel, getInterpolatedObservationModel

from descartes import PolygonPatch


#######################################
# Curved surface functions for simulation: don't delete, but these
# should be replaced by some sort of output from Maya for the full
# simulation pipeline
#######################################
IMG_SIZE = 50
offset=.002
measmin=7
measmax=9
# measmin=9
# measmax=7
#######################################
# polygon test functions for simulating phase2: 
#######################################
squaretumor=np.array([[-.01,-.01],[.02,-.01],[.02,.02],[-.01,.02]])
# quaretumor=Polygon(quaretumor)
thintumor=np.array([[2.25,0.25],[2.75,2.25],[2.75,2.75],[2.25,2.75]])
rantumor=.02*np.array([[2.25,0.75],[3.25,1.25],[2.75,2.25],[2.75,2.75],[2.25,2.75],[2.,1.25]])-.04

phantomsquareGT=np.array([[.001,.019],[.02,.019],[.02,.03],[.001,.03]])

# make circular tumor
rad=.0125/2.
loc=[0.01,.03]
simCircle = Point(loc[0],loc[1]).buffer(rad)
simCircle = np.array(simCircle.exterior.coords)
rad=.0125
loc=[0.0229845803642/2.0,.035]
expCircle = Point(loc[0],loc[1]).buffer(rad)
expCircle = np.array(expCircle.exterior.coords)

# create horseshow
rad=.007
loc=[0.0229845803642/2.0-.002,.015]
circle = Point(loc[0],loc[1]).buffer(rad)
circle = np.array(circle.exterior.coords)
semicircle=circle[circle[:,0]>=loc[0]]
semi=semicircle[semicircle[:,1].argsort()]
line = LineString(semi)#LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
horseshoe = LineString(semi).buffer(.0025)
horseshoe = np.array(horseshoe.exterior.coords)


othertumor = LineString(.008*np.array([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)]))
othertumor = othertumor.buffer(.005)
othertumor = np.array(othertumor.exterior.coords)

def polybuff(tum,minus=False):
    if minus==True:
        offs=-offset
    else:
        offs=offset
    tum=Polygon(tum)
    tum=tum.buffer(offs)
    return np.array(tum.exterior.coords)


def getInterpolatedGTSurface(surface, workspace):
    z = getObservationModel(surface)
    res = z.shape[0]
    x = np.linspace(workspace.bounds[0][0], workspace.bounds[0][1], num = res)
    y = np.linspace(workspace.bounds[1][0], workspace.bounds[1][1], num = res)
    f = interpolate.interp2d(x, y, z, kind='cubic')

    return f

def getInterpolatedStereoMeas(surface, workspace):
    #z = getStereoDepthMap(surface)[5:45,5:45]
    #z = np.pad(z,((5,5),(5,5)),mode='edge')
    z = getStereoDepthMap(surface)
    #z = np.pad(z,((5,5),(5,5)),mode='edge')
        
    z[z<5]= 5
    z[z>30]=5 #if dont know its baseline

    #delete measurements outside range
    # z = z[z<=30]
    # z = z[z>=5]

    # import IPython
    # IPython.embed()

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


def getSimulatedStereoMeas(surface, workspace, plot=False, block=False):
    """
    wrapper function for SimulateStereoMeas
    hetero. GP model requires defining the variance for each measurement 
    standard stationary kernel doesn't need this

    should fix these functions so they're not necessary by default...
    """
    xx, yy, z = SimulateStereoMeas(surface, workspace)
    
    # todo: noise due to  offset uncertainty    
    focalplane=(workspace.bounds[1][1]-workspace.bounds[1][0])/2.0
    # we subtract yy from focal plane as an estimate for looking at it obliquely
    sigma_offset=(yy-focalplane)

    sigma_offset = sigma_offset.ravel()/np.max(sigma_offset) #normalize
    
    # noise component due to curvature:
    # finite differencing
    #xgrid = np.vstack([xx.flatten(), yy.flatten()]).T
    grad = np.gradient(z)
    dx,dy = grad
    sigma_fd = np.sqrt(dx**2+dy**2)
    
    sigma_fd[np.isinf(sigma_fd)]=0 

    sigma_fd = sigma_fd.ravel()/np.max(sigma_fd)  #normalize

    # we assume Gaussian measurement noise:
    sigma_g = 0.05

    # weighted total variance for measurements
    sigma_total = sigma_g + sigma_fd  + 0.2*sigma_offset
    # sigma_total = sigma_g + 0*sigma_fd*sigma_offset


    if plot==True:
        # plot the surface from disparity
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, z, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
        ax.set_title("Depth from Disparity")
        # ax.set_zlim3d(0,100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show(block=block)

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
    noise=0.2 #assuming there is a 0.2 variance in height in mm
    sigma_t = np.full(z.shape, noise)

    return np.array([xx, yy,
                     z,
                     sigma_t]).T

def SimulateStiffnessMeas(poly, sample_locations, noiselev = .05):
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
    z = z + noiselev*np.random.randn(z.shape[0])

    return xx, yy, z

def plotSimulatedStiffnessMeas(poly, workspace, xpos=None, sensornoise = .03):
    if xpos==None:
       xpos=(workspace.bounds[0][1]-workspace.bounds[0][0])/2.0+workspace.bounds[0][0]

    y = np.arange(workspace.bounds[1][0], workspace.bounds[1][1], 1/float(1000))
    x = np.zeros(y.shape)+xpos
    sample_locations = np.array([x,y]).T

    meas = SimulateStiffnessMeas(poly, sample_locations, sensornoise=sensornoise)
    meas = meas[2]

    plt.plot(y.flatten(), meas.flatten(), linewidth=3.0)
    plt.show()

def getSimulateStiffnessMeas(sample_points,surface,noiselev):
    """wrapper function for SimulateProbeMeas hetero. GP model requires
    defining the variance for each measurement standard stationary
    kernel doesn't need this

    """
    xx,yy,z = SimulateStiffnessMeas(surface, sample_points,noiselev)

    # we assume Gaussian measurement noise:
    noise=.05
    sigma_t = np.full(z.shape, noise)
    return np.array([xx, yy,
                     z,
                     sigma_t]).T

def getRecordedExperimentalStiffnessMeas(sample_points,surface=None,noiselev=None):
    filename = '../scripts/dense_grid.p'
    data_dict = pickle.load(open(filename, "rb"))
    data = np.array(data_dict['data'])
    x, y, z = data[:,0], data[:,1], data[:,2]

    buffx=.3*(x.max()-x.min())

    from scipy.ndimage.filters import gaussian_filter
    z = gaussian_filter(z.reshape(21,41), sigma=1)
    z = z.reshape((21*41,))

    from scipy.interpolate import Rbf
    rbfi = Rbf(x, y, z)

    stiffnesses = np.array([rbfi(a[0], a[1]) for a in sample_points])
    stiffnesses=stiffnesses/1000.0

    stiffnesses[sample_points[:,0]>x.max()-buffx]=z.mean()/1000.0
    stiffnesses[sample_points[:,0]<x.min()+buffx]=z.mean()/1000.0
    # print z[x>x.min()+buffx]
    output = np.zeros((len(sample_points), 3))
    output[:,:2] = sample_points
    output[:,2] = stiffnesses
    return output

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
#def sigmoid(dist, alpha=1014, a=0.0, b=1.0):

def sigmoid(dist, alpha=1400, a=0.0, b=1.0):
    """  
    a, b: base and max readings of the probe with and without tumor
    dist = xProbe-xEdge
    xProbe: 
    xEdge: the center of sigmoid
    alpha: slope  
    Output:
    y = a + (b-a)/(1+ exp(-alpha*(xProbe-xEdge)))
    """
    a = measmin
    b = measmax
    c = -offset
    y = a + np.divide((b-a),(1+ np.exp(-alpha*(dist-c))))  
    return y

def getLevelSet (workspace, mean, level, allpoly=False):
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
        # poly = segments[0]
        if allpoly==True:
            poly=segments
            for pol in poly:
                #close polygons
                pol = np.vstack((pol,pol[0]))
        else:
            poly=[segments[0]]
    else:
        if allpoly==True:
            poly=[[]]
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

    return np.array(z)


def probeMeasure(xProbe, Pts, z, level):
    """
    input: 
        xProbe: desired point(s) of measurement
        Pts: x, y - 2-d numpy array N-by-d
        Z : values at each of the points 1-D numpy array length N
        level: find the polygon at level
    """
    poly = getMapLevelSet(Pts, z, level)
    z = makeMeasurement_LS(xProbe, poly)
    return z
