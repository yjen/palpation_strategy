import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import pyplot as pl

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
from scipy import stats


        
def update_GP_het(measurements,method='nonhet'):
    """
    GP for phase1:
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

def update_GP(measurements):
    """
    GP for phase2:
    Inputs: data=[x position, y position, measurement, measurement noise]
    TODO: maybe combine with updateGP above
    """
    sensornoise=.00001

    # parse locations, measurements, noise from data
    X = measurements[:,0:2]
    Y = np.array([measurements[:,2]]).T

    kern = GPy.kern.Matern52(2,ARD=True) +\
           GPy.kern.White(2)+GPy.kern.Bias(1)

    m = GPy.models.GPRegression(X,Y,kern)
    m.optimize(messages=True,max_f_eval = 1000)
    # xgrid = np.vstack([self.x1.reshape(self.x1.size),
    #                    self.x2.reshape(self.x2.size)]).T
    # y_pred=m.predict(self.xgrid)[0]
    # y_pred=y_pred.reshape(self.x1.shape)
    # sigma=m.predict(self.xgrid)[1]
    # sigma=sigma.reshape(self.x1.shape)
    return m

def implicitsurface(GPdata,thresh=.4):
    """
    not sure bout this one...
    """
    xx=GPdata[0]

    yy=GPdata[1]

    mean=GPdata[2]
    sigma=GPdata[3]
    phi = stats.distributions.norm.pdf
    GPIS=np.flipud(phi(mean,loc=thresh,scale=(.1+sigma)))
    GPIS=GPIS/GPIS.max()
    return [xx, yy, GPIS] 


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

def getSimulatedProbeMeas(surface, sample_points):
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

def getSimulateStiffnessMeas(surface, sample_points):
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

# def plot_errorIS(surface, GP, rangeX,rangeY, gridSize=100):
#     # choose points to compare

#     x = np.linspace(rangeX[0], rangeX[1], num = gridSize)
#     y = np.linspace(rangeY[0], rangeY[1], num = gridSize)

#     xx, yy = np.meshgrid(x, y)

#     # evaluate surface ground truth:
#     GroundTruth = surface(xx,yy)

#     # evaluate the Gaussian Process mean at the same points
#     EstimateMean = eval_GP(GP, rangeX, rangeY, res=gridSize)[2]

#     # evaluate the RMSerror
#     error =np.sqrt((GroundTruth-EstimateMean)**2)
#     fig = plt.figure(figsize=(9, 3))

#     # plot the ground truth
#     ax = fig.add_subplot(131, projection='3d')
#     ax.plot_surface(xx, yy, GroundTruth, rstride=1, cstride=1,
#                     cmap=cm.coolwarm, linewidth=0,
#                     antialiased=False)
#     ax.set_title("Ground Truth")
        
#     # plot the estimate
#     ax1 = fig.add_subplot(132, projection='3d')
#     cs1=ax1.plot_surface(xx, yy, EstimateMean, rstride=1, cstride=1,
#                          cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     ax1.set_title("Estimate Mean")  

#     # plot the error
#     ax1 = fig.add_subplot(133, projection='3d')
#     cs1=ax1.plot_surface(xx, yy, error, rstride=1, cstride=1,
#                          cmap=cm.Greys, linewidth=0, antialiased=False)
#     ax1.set_title("Error")
#         ax2.contour(xx, yy, GPdata[2], [.4], colors='r', linestyles='dashdot')

#     plt.show()

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

def plot_beliefGPIS(GPdata):
    # parse locations, measurements, noise from data
    GPISdat=implicitsurface(GPdata)

    #gp data
    xx=GPdata[0]
    yy=GPdata[1]
    mean=GPdata[2]
    variance=GPdata[3]

    #GPIS data
    xx=GPISdat[0]
    yy=GPISdat[1]
    GPISvar=GPISdat[2]
    
    fig = plt.figure(figsize=(16, 4))

    # plot the mean
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(xx, yy, mean, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
    ax.set_title("GP Mean")
        
    # plot the uncertainty
    ax1 = fig.add_subplot(132, projection='3d')
    lim=1
    cs1=ax1.plot_surface(xx, yy, variance, rstride=1, cstride=1,
                         cmap=cm.Greys, linewidth=0, antialiased=False)
    ax1.set_title("GP Uncertainty")  

    # plot the uncertainty
    # ax2 = fig.add_subplot(133, projection='3d')
    # lim=1
    # cs2=ax2.plot_surface(xx, yy, np.abs(GPISvar), rstride=1, cstride=1,
    #                      cmap=cm.Greys, linewidth=0, antialiased=False)
    # ax1.set_title("GP Uncertainty")  
    ax2 = fig.add_subplot(133)
    cs2=ax2.imshow(GPISvar, cmap=cm.Greys,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max())
    )
    ax2.set_title("GPIS")
    #cs = pl.contour(xx, yy, mean, [thresh], colors='k', linestyles='dashdot')
    cs = pl.contour(xx, yy, GPISvar, [.2], colors='b',
                    linestyles='solid')
    pl.clabel(cs, fontsize=11)

    cs = pl.contour(xx, yy, GPISvar, [0.5], colors='k',
                    linestyles='dashed')
    pl.clabel(cs, fontsize=11)
    
    norm = plt.matplotlib.colors.Normalize(vmin=0., vmax=GPISvar.max())
    cb2 = plt.colorbar(cs2, norm=norm)
    cb2.set_label('${\\rm \mathbb{P}}\left[\widehat{G}(\mathbf{x}) = 0\\right]$')# # Define
    plt.show()



