import sys
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib import pyplot as pl

# import matplotlib.path as path
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
# from simulated_disparity import getObservationModel


        
# def update_GP_het(measurements,method='nonhet'):
#     """
#     GP for phase1:
#     Update the GP using heteroskedactic noise model
#     Inputs: data=[x position, y position, measurement, measurement noise]
#     """
#     sensornoise = .01

#     # parse locations, measurements, noise from data
#     X = measurements[:,0:2]
#     Y = np.array([measurements[:,2]]).T

#     # set up the Gaussian Process

#     if method=="het":
#         # use heteroskedactic kernel
#         noise = np.array([measurements[:,3]]).T

#         kern = GPy.kern.RBF(2) + GPy.kern.Fixed(2, GPy.util.linalg.tdot(noise))
#         m = GPy.models.GPHeteroscedasticRegression(X,Y,kern)
#         m = GPy.models.GPRegression(X,Y,kern)

#         # m['.*het_Gauss.variance'] = abs(noise)
#         # m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it
#     else:
#         # use stationary kernel
#         kern = GPy.kern.RBF(2 #variance=1., lengthscale=.05
#         ) #GPy.kern.Bias(1)
#         m = GPy.models.GPRegression(X,Y,kern)
#     m.optimize()

#     return m

def update_GP(measurements,method='nonhet'):
    """
    GP for phase2:
    Inputs: data=[x position, y position, measurement, measurement noise]
    TODO: maybe combine with updateGP above
    """
    sensornoise=.01

    # parse locations, measurements, noise from data
    X = measurements[:,0:2]
    Y = np.array([measurements[:,2]]).T
    if method=="het":
        # use heteroskedactic kernel
        noise = np.array([measurements[:,3]]).T

        kern = GPy.kern.RBF(2) 
        m = GPy.models.GPHeteroscedasticRegression(X,Y,kern)
        # m = GPy.models.GPRegression(X,Y,kern)
        m['.*het_Gauss.variance'] = abs(noise)
        m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it

        # m['.*het_Gauss.variance'] = abs(noise)
        # m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it
    else:
        kern = GPy.kern.RBF(2)+ GPy.kern.White(2)
        m = GPy.models.GPRegression(X,Y,kern)
        # m.optimize_restarts(num_restarts = 10)
    m.optimize()
    # xgrid = np.vstack([self.x1.reshape(self.x1.size),
    #                    self.x2.reshape(self.x2.size)]).T
    # y_pred=m.predict(self.xgrid)[0]
    # y_pred=y_pred.reshape(self.x1.shape)
    # sigma=m.predict(self.xgrid)[1]
    # sigma=sigma.reshape(self.x1.shape)
    return m


def update_GP_sparse(measurements,numpts=10):
    """
    GP for phase2:
    Inputs: data=[x position, y position, measurement, measurement noise]
    TODO: maybe combine with updateGP above
    """
    sensornoise=.00001

    # parse locations, measurements, noise from data
    X = measurements[:,0:2]
    Y = np.array([measurements[:,2]]).T

    # kern = GPy.kern.Matern52(2,ARD=True) +\
    #        GPy.kern.White(2)
    kern = GPy.kern.RBF(2)

    #subsample range in the x direction
    subx=np.linspace(X.T[0].min(),X.T[0].max(),numpts)
    suby=np.linspace(X.T[1].min(),X.T[1].max(),numpts)

    subxx,subyy=np.meshgrid(subx,suby)
    #subsample in y
    Z = np.array([subxx.flatten(),subyy.flatten()]).T
    m = GPy.models.SparseGPRegression(X,Y,Z=Z)
    m.optimize('bfgs')
    # xgrid = np.vstack([self.x1.reshape(self.x1.size),
    #                    self.x2.reshape(self.x2.size)]).T
    # y_pred=m.predict(self.xgrid)[0]
    # y_pred=y_pred.reshape(self.x1.shape)
    # sigma=m.predict(self.xgrid)[1]
    # sigma=sigma.reshape(self.x1.shape)
    return m

def implicitsurface(mean,sigma,level):
    """
    not sure bout this one...
    """
    #xx=GPdata[0]

    #yy=GPdata[1]

    #mean=GPdata[2]
    #sigma=GPdata[3]
    phi = stats.distributions.norm.pdf
    GPIS=phi(mean,loc=level,scale=(sigma))
    GPIS=GPIS/GPIS.max()
    return  GPIS


def eval_GP(m, bounds, res=100):
    """
    evaluate the GP on a grid
    """
    rangeX=bounds[0]
    rangeY=bounds[1]
    # parse locations, measurements, noise from data
   
    xx, yy = np.meshgrid(np.linspace(rangeX[0], rangeX[1], res),
                  np.linspace(rangeY[0],  rangeY[1], res))
    xgrid = np.vstack([xx.flatten(), yy.flatten()]).T
    
    z_pred, sigma = m._raw_predict(xgrid)
    z_pred = z_pred.reshape(xx.shape)
    sigma = sigma.reshape(xx.shape)

    return [xx, yy, z_pred, sigma]

def predict_GP(m, pts):
    """
    evaluate GP at specific points
    """
    z_pred, sigma = m._raw_predict(pts)
    # z_pred = z_pred.reshape(xx.shape)
    # sigma = sigma.reshape(xx.shape)

    return [pts, z_pred, sigma]


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


########################## Plot Scripts
def plot_error(surface, workspace, mean, sigma, aq, meas, dirname, data=None,iternum=0, projection3D=False, plotmeas=True):
    # choose points to compare
    xx=workspace.xx
    yy=workspace.yy

    mean = gridreshape(mean,workspace)
    sigma = gridreshape(sigma,workspace)
    x = workspace.xlin
    y = workspace.ylin
    aq = gridreshape(aq,workspace)

    # xx, yy = np.meshgrid(x, y)

    interp=getInterpolatedGTSurface(surface, workspace)
    # interp=getInterpolatedObservationModel(surface)

    GroundTruth = interp(x,y)

    # GroundTruth=GroundTruth.reshape(xx.shape)
    # evaluate the Gaussian Process mean at the same points

    # evaluate the RMSerror
    error =np.sqrt((GroundTruth-np.squeeze(mean))**2)

    if data is None:
        fig = plt.figure(figsize=(20, 4))
        if projection3D==True:
            ax1 = fig.add_subplot(151, projection='3d')
            ax2 = fig.add_subplot(152, projection='3d')
            # ax3 = fig.add_subplot(153, projection='3d')
            # ax4 = fig.add_subplot(154, projection='3d')
            # ax5 = fig.add_subplot(155, projection='3d')
        else:
            ax1 = fig.add_subplot(151)
            ax2 = fig.add_subplot(152)
        ax3 = fig.add_subplot(153)
        ax4 = fig.add_subplot(154)
        ax5 = fig.add_subplot(155)
        fig.canvas.draw()
        fig.canvas.draw()
        plt.show(block=False)
        data = [fig, ax1, ax2, ax3, ax4,ax5]
    data[1].clear()
    data[2].clear()
    data[3].clear()
    data[4].clear()
    data[5].clear()

    # plot the ground truth
    if projection3D==True:
        data[1].plot_surface(xx, yy, GroundTruth.reshape(workspace.res,workspace.res), rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
        data[1].set_zlim3d(0,20)
    else:
        data[1].imshow(np.flipud(GroundTruth), cmap=cm.coolwarm,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
        if plotmeas==True:
            data[1].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=20,
                        cmap=cm.coolwarm)
    
    data[1].set_title("Ground Truth")
        
    # plot the estimate
    if projection3D==True:
        data[2].plot_surface(xx, yy, mean.reshape(workspace.res,workspace.res), rstride=1, cstride=1,
                         cmap=cm.coolwarm, linewidth=0, antialiased=False)
        data[2].set_zlim3d(0,20)
    else:
        data[2].imshow(np.flipud(mean), cmap=cm.coolwarm,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
        if plotmeas==True:
            data[2].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=20,
                        cmap=cm.coolwarm)
    data[2].set_title("Estimate Mean")

    # plot the variance
    # if projection3D==True:
       # data[3].plot_surface(xx, yy, sigma.reshape(workspace.res,workspace.res), rstride=1, cstride=1,
                         # cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # else:
    data[3].imshow(np.flipud(sigma.reshape(workspace.res,workspace.res)), cmap=cm.coolwarm,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
    if plotmeas==True:
        data[3].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=20,
                    cmap=cm.coolwarm)
    data[3].set_title("Estimate Variance")

    # plot the aquis function
    # if projection3D==True:
        # data[4].plot_surface(xx, yy, aq.reshape(workspace.res,workspace.res), rstride=1, cstride=1,
                         # cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # else:
    data[4].imshow(np.flipud(aq.reshape(workspace.res,workspace.res)), cmap=cm.coolwarm,
                   extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
    if plotmeas==True:
        data[4].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=20,
                        cmap=cm.coolwarm)
    data[4].set_title("Estimate Variance")

    # plot the error
    # if projection3D==True:
        # data[5].plot_surface(xx, yy, error.reshape(workspace.res,workspace.res), rstride=1, cstride=1,
                           # cmap=cm.Greys, linewidth=0, antialiased=False)
    # else:
    data[5].imshow(np.flipud(error.reshape(workspace.res,workspace.res)), cmap=cm.coolwarm,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
    data[5].set_title("Error from GT")
    
    data[0].canvas.draw()
    data[0].savefig(dirname + '/' + str(iternum) + ".pdf" ,bbox_inches='tight')

    return data


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

def plot_beliefGPIS(poly,workspace,mean,variance,GPIS,aq,meas,dirname,data=None, iternum=0, level=.4, projection3D=False):
    # parse locations, measurements, noise from data
    # gp data
    xx=workspace.xx
    yy=workspace.yy
    mean=gridreshape(mean,workspace)
    variance=gridreshape(variance,workspace)
    aq=gridreshape(aq,workspace)
    GPIS=gridreshape(GPIS,workspace)

    # for plotting, add first point to end
    GroundTruth = np.vstack((poly,poly[0]))
    if data is None:
        fig = plt.figure(figsize=(16, 4))
        if projection3D==True:
            ax1 = fig.add_subplot(141, projection='3d')
            ax2 = fig.add_subplot(142, projection='3d')
        else:
            ax1 = fig.add_subplot(141)
            ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)

        fig.canvas.draw()
        plt.show(block=False)
        data = [fig, ax1, ax2, ax3, ax4]
    data[1].clear()
    data[2].clear()
    data[3].clear()
    data[4].clear()

    # plot the mean
    if projection3D==True:
        data[1].plot_surface(xx, yy, mean, rstride=1, cstride=1,
                             cmap=cm.coolwarm, linewidth=0,
                             antialiased=False)

    else:
        data[1].imshow(np.flipud(mean), cmap=cm.coolwarm,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
        data[1].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=20,
                        cmap=cm.coolwarm)
    data[1].set_title("Data and GP Mean: Stiffness map")
    data[1].set_xlabel('x')
    data[1].set_ylabel('y')
    
    # plot the uncertainty
    if projection3D==True:
        #lim=1
        cs1=data[2].plot_surface(xx, yy, variance, rstride=1, cstride=1,
                                 cmap=cm.Greys, linewidth=0,
                                 antialiased=False)
    else:
        data[2].imshow(np.flipud(variance), cmap=cm.Greys,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
        data[2].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=20,
                        cmap=cm.coolwarm)
    
    data[2].set_title("GP Uncertainty: Stiffness map")  
    data[2].set_xlabel('x')
    data[2].set_ylabel('y')

    data[3].set_title("acquisition function")
    cs=data[3].imshow(np.flipud(aq), cmap=cm.jet, extent=(xx.min(),
                                                            xx.max(),
                                                            yy.min(),yy.max())
    )
    data[3].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=20,
                    cmap=cm.coolwarm)
    
    data[4].set_title("GPIS")

    cs1 = data[4].contour(xx, yy, mean, [level], colors='r',
                        linewidth=1, linestyles='dashdot')
    data[4].plot(GroundTruth.T[0], GroundTruth.T[1], '-.',color='g',
                 linewidth=1, solid_capstyle='round', zorder=2)

    data[4].legend( loc='upper right' )
    data[4].set_xlabel('x')
    data[4].set_ylabel('y')  
    cs2=data[4].imshow(np.flipud(GPIS), cmap=cm.Greys,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max()))
    norm = plt.matplotlib.colors.Normalize(vmin=0., vmax=GPIS.max())
    if (len(data) > 5):
        data[5].remove()
        data = data[:5]
    # cb2 = plt.colorbar(cs2, norm=norm)
    # cb2.set_label('${\\rm \mathbb{P}}\left[\widehat{G}(\mathbf{x}) = 0\\right]$')# # Define
    # data.append(cb2)
    
    data[0].canvas.draw()
    data[0].savefig(dirname + '/' + str(iternum) + ".pdf" ,bbox_inches='tight')
    return data

