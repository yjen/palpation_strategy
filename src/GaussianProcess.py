import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# import matplotlib.path as path
import GPy
#import GPyOpt
#from utils import *
#from getMap import *
#from sensorModel import *
# import cplex, gurobipy
#import sys
#sys.path.append("..")
#import simulated_disparity
from simUtils import *
from utils import *


        
def update_GP_het(measurements,plot_GP=True,method='nonhet'):
    """
    Update the GP using heteroskedactic noise model
    Inputs: data=[x position, y position, measurement, measurement noise]
    """
    sensornoise = .00001

    # parse locations, measurements, noise from data
    X = measurements[:,0:2]
    Y = np.array([measurements[:,2]]).T

    # set up the Gaussian Process

    if method=="het":
        # use heteroskedactic kernel
        noise = np.array([measurements[:,3]]).T

        kern = GPy.kern.MLP(input_dim=2) + GPy.kern.Bias(1)
        m = GPy.models.GPHeteroscedasticRegression(X,Y,kern)
        m['.*het_Gauss.variance'] = abs(noise)
        m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it
    else:
        # use stationary kernel
        kern = GPy.kern.RBF(input_dim=2 #variance=1., lengthscale=.05
        ) + GPy.kern.White(2)#GPy.kern.Bias(1)
        m = GPy.models.GPRegression(X,Y,kern)
    m.optimize()

    return m

def eval_GP(m, rangeX, rangeY, res=100):
    """
    Update the GP using heteroskedactic noise model
    Inputs: data=[x position, y position, measurement, measurement noise]
    """

    # parse locations, measurements, noise from data
   
    xx, yy = np.meshgrid(np.linspace(rangeX[0], rangeX[1], res),
                  np.linspace(rangeY[0],  rangeY[1], res))
    xgrid = np.vstack([xx.flatten(), yy.flatten()]).T
    
    z_pred, sigma = m._raw_predict(xgrid)
    z_pred = z_pred.reshape(xx.shape)
    sigma = sigma.reshape(xx.shape)

    return [xx, yy, z_pred, sigma]


def getSimulatedStereoMeas(surface, focalplane=0, rangeX = [-2,2], rangeY = [-1,1], modality=3):
    """
    wrapper function for SimulateStereoMeas
    hetero. GP model requires defining the variance for each measurement 
    standard stationary kernel doesn't need this

    should fix these functions so they're not necessary by default...
    """

    xx, yy, z = SimulateStereoMeas(surface, rangeX, rangeY)

    # we assume Gaussian measurement noise:
    sigma_g = .001

    # noise component due to curvature:
    # finite differencing
    #xgrid = np.vstack([xx.flatten(), yy.flatten()]).T
    grad = np.gradient(z)
    dx,dy = grad
    sigma_fd = 1/(dx**2+dy**2)
    sigma_fd[np.isinf(sigma_fd)]=0

    # todo: noise due to  offset uncertainty
    sigma_offset=(yy-focalplane)**2
    # weighted total noise for measurements
    sigma_total = sigma_g + 0*sigma_fd  + .5*sigma_offset

    return np.array([xx.flatten(), yy.flatten(),
                     z.flatten(),
                     sigma_total.flatten()]).T

def getSimulatedProbeMeas(surface, sample_points,modality=3):
    """
    wrapper function for SimulateProbeMeas
    hetero. GP model requires defining the variance for each measurement 
    standard stationary kernel doesn't need this
    """
    xx,yy,z = SimulateProbeMeas(surface, sample_points)
    # we assume Gaussian measurement noise:
    noise=.001
    sigma_t = np.full(z.shape, noise)

    return np.array([xx, yy,
                     z,
                     sigma_t]).T

# def simulate_touch_measurement(surface,sample_locations):
# 	"""
#         Simulate measurements from palpation (tapping mode) for the test functions above
#         inputs:	"""
#         print sample_locations

#         # where to probe
#         xx, yy = sample_locations.T

#         # this is a simulated measurement, add noise
#         sensornoise = .001
        
#         newmeas = surface(xx,yy) + sensornoise*np.random.randn(xx.shape[0])

# 	return xx,yy,newmeas

########################## Plot Scripts

def plot_error(surface, GP, rangeX,rangeY, gridSize=100):
    # choose points to compare

    x = np.linspace(rangeX[0], rangeX[1], num = gridSize)
    y = np.linspace(rangeY[0], rangeY[1], num = gridSize)

    xx, yy = np.meshgrid(x, y)

    # evaluate surface ground truth:
    GroundTruth = surface(xx,yy)

    # evaluate the Gaussian Process mean at the same points
    EstimateMean = eval_GP(GP, rangeX, rangeY, res=gridSize)[2]

    # evaluate the RMSerror
    error =np.sqrt((GroundTruth-EstimateMean)**2)
    fig = plt.figure(figsize=(9, 3))

    # plot the ground truth
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(xx, yy, GroundTruth, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
    ax.set_title("Ground Truth")
        
    # plot the estimate
    ax1 = fig.add_subplot(132, projection='3d')
    cs1=ax1.plot_surface(xx, yy, EstimateMean, rstride=1, cstride=1,
                         cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.set_title("Estimate Mean")  

    # plot the error
    ax1 = fig.add_subplot(133, projection='3d')
    cs1=ax1.plot_surface(xx, yy, error, rstride=1, cstride=1,
                         cmap=cm.Greys, linewidth=0, antialiased=False)
    ax1.set_title("Error")  

    plt.show()

def plot_belief(GPdata):
    # parse locations, measurements, noise from data
    xx=GPdata[0]
    yy=GPdata[1]
    mean=GPdata[2]
    
    variance=GPdata[3]
    
    fig = plt.figure(figsize=(16, 4))

    # plot the mean
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(xx, yy, mean, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
    ax.set_title("GP Mean")
        
    # plot the uncertainty
    ax1 = fig.add_subplot(122, projection='3d')
    lim=1
    cs1=ax1.plot_surface(xx, yy, variance, rstride=1, cstride=1,
                         cmap=cm.Greys, linewidth=0, antialiased=False)
    ax1.set_title("GP Uncertainty")  

    plt.colorbar(cs1)
    plt.show()
