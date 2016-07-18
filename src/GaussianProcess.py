import sys
import numpy as np
from simUtils import *
from utils import *
from scipy import stats
from shapely.geometry import Point, Polygon #to calculate point-polygon distances
from descartes import PolygonPatch


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
        m['.*het_Gauss.variance'] = abs(noise)
        m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it
    else:
        var = params[0] # variance
        theta = params[1] # lengthscale
        noise = params[2]
        kern = GPy.kern.RBF(2,variance=var,lengthscale=theta)+GPy.kern.Bias(2,variance=params[3])
        # kern = GPy.kern.RBF(2)

        m = GPy.models.GPRegression(X,Y,kern,normalizer=True)
        # m.Gaussian_noise.fix(noise)
    #     m.optimize_restarts(num_restarts = 10)
    # m.optimize()
    return m


def update_GP_sparse(measurements,numpts=10):
    """
    Needs testing!
    GP for phase2, using sparse measurements:
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


def update_GP_ph1(measurements, nStereoMeas, method='heteroscedastic'):
    """
    Needs Testing! 
    GP for phase1:
    Inputs: 
        measurements=[x position, y position, measurement, measurement noise]
        nStereoMeas: number of stereo points; the rest are assumed to be probe measurements
        method: "heteroscedastic" -- stereo and probe measurements have different noise values
                "homoscedastic" -- stereo and probe measurements have same noise values
    TODO: maybe combine with updateGP above
    """
    sensornoise=.01

    # parse locations, measurements, noise from data
    X = measurements[:,0:2]
    Y = np.array([measurements[:,2]]).T
    # Y=Y/10000.0
    if method=="heteroscedastic":
        # use heteroskedactic kernel
        noise = np.array([measurements[:,3]]).T
        var = 10 # variance 0.001
        theta = 50  # lengthscale 4
        kern = GPy.kern.RBF(2, variance=var,lengthscale=theta)

        #define two types of noise
        nProbeMeas = len(noise)-nStereoMeas
        noiseModel = np.concatenate((np.zeros((nStereoMeas,), dtype=int), np.ones((nProbeMeas,), dtype=int)), axis=0)                

        #define heteroscedastic noise. 
        Y_meta = {'output_index':noiseModel[:,None]}
        #the following two lines are only for reference
        # likelihood = GPy.likelihoods.HeteroscedasticGaussian(Y_metadata)
        # m.predict(np.array([[0.3]]),Y_metadata={'output_index':np.zeros((1,1))[:,None].astype(int)})

        m = GPy.models.GPHeteroscedasticRegression(X,Y,kern, Y_metadata=Y_meta)
        # m = GPy.models.GPHeteroscedasticRegression(X,Y,kern)
        m['.*het_Gauss.variance'] = abs(noise) #specify observation of the noise. 
        # m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it        

    elif method=="homoscedastic":
        # var = 100 # variance
        var = 25
        theta = 20 #lengthscale
        kern = GPy.kern.RBF(2, variance=var,lengthscale=theta)
        m = GPy.models.GPRegression(X,Y,kern)
        m.Gaussian_noise.fix(.4)
        #m.optimize_restarts(num_restarts = 5)
        m.optimize()
    else:
        assert (method=="homoscedastic")or(method=="heteroscedastic")

    # m.optimize()

    return m


def implicitsurface(mean,sigma,level):
    """
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
    



