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
from shapely.geometry import Point, Polygon #to calculate point-polygon distances
from descartes import PolygonPatch

# from simulated_disparity import getObservationModel

def update_GP_ph1(measurements,method='nonhet'):
    """
    GP for phase2:
    Inputs: data=[x position, y position, measurement, measurement noise]
    TODO: maybe combine with updateGP above
    """
    sensornoise=.01

    # parse locations, measurements, noise from data
    X = measurements[:,0:2]
    Y = np.array([measurements[:,2]]).T
    # Y=Y/10000.0
    if method=="het":
        # use heteroskedactic kernel
        noise = np.array([measurements[:,3]]).T
        var = .001 # variance
        theta = 4 # lengthscale
        kern = GPy.kern.RBF(2, variance=var,lengthscale=theta) 
        m = GPy.models.GPHeteroscedasticRegression(X,Y,kern)
        # m = GPy.models.GPRegression(X,Y,kern)
        m['.*het_Gauss.variance'] = abs(noise)
        m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it
    else:
        # var = 100 # variance
        var = 25
        theta = 20 #engthscale
        kern = GPy.kern.RBF(2, variance=var,lengthscale=theta)
        m = GPy.models.GPRegression(X,Y,kern)
        m.Gaussian_noise.fix(.4)
        #m.optimize_restarts(num_restarts = 5)
    m.optimize()
    # print m
    # xgrid = np.vstack([self.x1.reshape(self.x1.size),
    #                    self.x2.reshape(self.x2.size)]).T
    # y_pred=m.predict(self.xgrid)[0]
    # y_pred=y_pred.reshape(self.x1.shape)
    # sigma=m.predict(self.xgrid)[1]
    # sigma=sigma.reshape(self.x1.shape)
    return m

def update_GP(measurements,method='nonhet',params=[1,.006,1e-5,0]):
    """
    GP for phase2:
    Inputs: data=[x position, y position, measurement, measurement noise]
    TODO: maybe combine with updateGP above
    """
    sensornoise=.01

    # parse locations, measurements, noise from data
    X = measurements[:,0:2]
    Y = np.array([measurements[:,2]]).T
    # Y=Y/10000.0
    if method=="het":
        # use heteroskedactic kernel
        noise = np.array([measurements[:,3]]).T
        var = 1. # variance
        theta = 4 # lengthscale
        kern = GPy.kern.RBF(2, var, theta) 
        m = GPy.models.GPHeteroscedasticRegression(X,Y,kern)
        # m = GPy.models.GPRegression(X,Y,kern)
        m['.*het_Gauss.variance'] = abs(noise)
        m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it
    else:
        var = params[0] # variance
        theta = params[1] # lengthscale
        noise = params[2]
        kern = GPy.kern.RBF(2,variance=var,lengthscale=theta)+GPy.kern.Bias(2,variance=params[3])
        m = GPy.models.GPRegression(X,Y,kern)
        # m.Gaussian_noise.fix(noise)
    #     m.optimize_restarts(num_restarts = 10)
    # m.optimize()
    # print m
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
    phi = stats.distributions.norm.pdf
    GPIS=phi(mean,loc=level,scale=(sigma))
    GPIS=GPIS/GPIS.max()
    return  GPIS


def predict_GP(m, pts):
    """
    evaluate GP at specific points
    """
    z_pred, sigma = m._raw_predict(pts)

    return [pts, z_pred, sigma]

def calcCurveErr(workspace,poly,mean,sigma,level):
    # see: http://toblerity.org/shapely/manual.html
    boundaryestimate = getLevelSet (workspace, mean, level)
    # GroundTruth = np.vstack((poly,poly[0]))
    GroundTruth=Polygon(GroundTruth)
    boundaryestimate=Polygon(boundaryestimate)

    mislabeled=boundaryestimate.symmetric_difference(GroundTruth) # mislabeled data ()
    boundaryestimate.difference(GroundTruth) #mislabeled as tumor--extra that would be removed
    GroundTruth.difference(boundaryestimate) # mislbaled as not-tumor--would be missed and should be cut out
    correct=boundaryestimate.intersection(GroundTruth) #correctly labeled as tumor
    return correct.area, mislabeled.area
    



# def ploterr(a,b,workspace):
#     fig = pyplot.figure(1, figsize=SIZE, dpi=90)

#     # a = Point(1, 1).buffer(1.5)
#     # b = Point(2, 1).buffer(1.5)

#     # 1
#     ax = fig.add_subplot(121)

#     patch1 = PolygonPatch(a, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
#     ax.add_patch(patch1)
#     patch2 = PolygonPatch(b, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
#     ax.add_patch(patch2)
#     c = a.intersection(b)
#     patchc = PolygonPatch(c, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
#     ax.add_patch(patchc)

#     ax.set_title('a.intersection(b)')

#     xrange = [workspace.bounds[0][0], workspace.bounds[0][1]]
#     yrange = [workspace.bounds[1][0], workspace.bounds[1][1]]
#     ax.set_xlim(*xrange)
#     ax.set_xticks(range(*xrange) + [xrange[-1]])
#     ax.set_ylim(*yrange)
#     ax.set_yticks(range(*yrange) + [yrange[-1]])
#     ax.set_aspect(1)

#     #2
#     ax = fig.add_subplot(122)

#     patch1 = PolygonPatch(a, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
#     ax.add_patch(patch1)
#     patch2 = PolygonPatch(b, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
#     ax.add_patch(patch2)
#     c = a.symmetric_difference(b)

#     if c.geom_type == 'Polygon':
#         patchc = PolygonPatch(c, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
#         ax.add_patch(patchc)
#     elif c.geom_type == 'MultiPolygon':
#         for p in c:
#             patchp = PolygonPatch(p, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
#             ax.add_patch(patchp)

#     ax.set_title('a.symmetric_difference(b)')

#     xrange = [-1, 4]
#     yrange = [-1, 3]
#     ax.set_xlim(*xrange)
#     ax.set_xticks(range(*xrange) + [xrange[-1]])
#     ax.set_ylim(*yrange)
#     ax.set_yticks(range(*yrange) + [yrange[-1]])
#     ax.set_aspect(1)

#     pyplot.show()
# def eval_GP(m, bounds, res=100):
#     """
#     evaluate the GP on a grid
#     """
#     rangeX=bounds[0]
#     rangeY=bounds[1]
#     # parse locations, measurements, noise from data
   
#     xx, yy = np.meshgrid(np.linspace(rangeX[0], rangeX[1], res),
#                   np.linspace(rangeY[0],  rangeY[1], res))
#     xgrid = np.vstack([xx.flatten(), yy.flatten()]).T
    
#     z_pred, sigma = m._raw_predict(xgrid)
#     z_pred = z_pred.reshape(xx.shape)
#     sigma = sigma.reshape(xx.shape)

#     return [xx, yy, z_pred, sigma]
