import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# import matplotlib.path as path

#from utils import *
# from getMap import *
#from sensorModel import *
# import cplex, gurobipy
#import sys
#sys.path.append("..")
#import simulated_disparity
from simUtils import *
from scipy.stats import norm

from utils import *
from GaussianProcess import *
import ErgodicPlanner
from itertools import combinations



# def max_uncertainty(GPdata,numpoints=1):
#     # return the x,y locations of the "numpoints" largest uncertainty values
#     ind = np.argpartition(GPdata[3].flatten(),
#                           -numpoints)[-numpoints:]
#     newpointx=GPdata[0].flatten()[ind]
#     newpointy=GPdata[1].flatten()[ind]
    
#     return np.array([newpointx,newpointy]).T

# def max_uncertainty_IS(GPdata,numpoints=1):
#     # return the x,y locations of the "numpoints" largest uncertainty values
#     GPISdat=implicitsurface(GPdata)
#     ind = np.argpartition(GPISdat[2].flatten(),
#                           -numpoints)[-numpoints:]
#     newpointx=GPdata[0].flatten()[ind]
#     newpointy=GPdata[1].flatten()[ind]
    
#     return np.array([newpointx,newpointy]).T

# def ergodic(LQ,GPdata,xinit,thresh,numpoints=1):
#     # return the x,y locations of the "numpoints" largest uncertainty values
#     GPISdat=implicitsurface(GPdata,thresh)
#     pdf=GPISdat[2]
#     newpts=ErgodicPlanner.ergoptimize(LQ,pdf,
#                                xinit,
#                                 maxsteps=15)
    
#     return newpts

##########################
#Acuisition functions
##########################
def MaxVar_plus_gradient(model, workspace , level=0, acquisition_par=0):
    """
    choose next sample points based on maximizing prior variance
    """
    #x = multigrid(bounds, res)
    mean, sigma = get_moments(model, workspace.x)     
    meansq = mean.reshape(workspace.res,workspace.res)

    grad = np.gradient(meansq)
    dx,dy = grad
    fd = np.sqrt((dx**2+dy**2))
    fd=fd/np.max(fd)
    fd[np.isinf(fd)]=0

    buffx=.02*workspace.bounds[0][1]
    buffy=.02*workspace.bounds[1][1]

    sigma[workspace.x[:,0]<buffx]=0
    sigma[workspace.x[:,1]<buffy]=0
    sigma[workspace.x[:,0]>workspace.bounds[0][1]-buffx]=0
    sigma[workspace.x[:,1]>workspace.bounds[1][1]-buffy]=0
    f_acqu = sigma.flatten()+.35*fd.flatten()
    f_acqu=np.array([f_acqu]).T
    return workspace.x, f_acqu  # note: returns negative value for posterior minimization

def MaxVar_GP(model, workspace , level=0, acquisition_par=0):
    """
    choose next sample points based on maximizing prior variance
    """
    x=workspace.x
    #x = multigrid(bounds, res)
    mean, sigma = get_moments(model, x)     
    f_acqu = sigma
    return x, f_acqu  # note: returns negative value for posterior minimization

def UCB_GP(model, workspace, level=0, acquisition_par=.4 ):
    """
    Upper Confidence Band
    """
    x=workspace.x
    #x = multigrid(bounds, res)
    mean, sigma = get_moments(model, x)     
    f_acqu = acquisition_par * (mean) +  sigma
    return x, f_acqu  # note: returns negative value for posterior 

def dmaxAcquisition(workspace, model, acfun, xinit=[.2,.3], numpoints=1, level=0):
    """
    Selects numpoints number of points that are maximal from the list of AcquisitionFunctionVals
    """
    dx=.1
    x=[xinit]
    for n in range(numpoints):
        _, current_ac = acfun(model, x[n], level=level)
        testpts=x[n]+np.array([[dx,0],[-dx,0],[-dx,-dx],[0,dx],[dx,dx],[dx,-dx],[-dx,dx],[0,-dx]])
        allpts=np.vstack((x[n],testpts))
        allpts, new_acqu = MaxVar_GP(model, allpts)

        grad=new_acqu-current_ac  
        i=0
        ind = np.argpartition(new_acqu.flatten(), -1)[-1-i]
        newpt = allpts[ind]
        while (newpt[0]>workspace.bounds[0][1] or newpt[0]<workspace.bounds[0][0] or 
                newpt[1]>workspace.bounds[1][1] or newpt[1]<workspace.bounds[1][0]):
            i = i+1
            ind = np.argpartition(new_acqu.flatten(), -1)[-1-i]
            newpt = allpts[ind]
        x.append(newpt)
    return np.array(x[1:])
# def d_UCB_GP(model,x,acquisition_par=1):
#     """
#     Derivative of the Upper Confidence Band
#     """
#     dmdx, dsdx = get_d_moments(model, x)
#     df_acqu = acquisition_par * dmdx +  dsdx
#     return df_acqu

def UCB_GPIS(model, workspace, level, acquisition_par=0 ):
    """
    Upper Confidence Band
    """
    x=workspace.x
    #x = multigrid(bounds, res)
    mean, sigma = get_moments(model, x)   
    sdf=abs(mean-level)
    f_acqu = - sdf +  acquisition_par*sigma
    return x, f_acqu+abs(min(f_acqu))  # note: returns negative value for posterior minimization



# def d_UCB_GPIS(model,x,acquisition_par=1):
#     """
#     Derivative of the Upper Confidence Band
#     """
#     dmdx, dsdx = get_d_moments(model, x)
#     df_acqu = acquisition_par * dmdx +  dsdx
#     return df_acqu

def EI_GP(model, workspace, level=0, acquisition_par = 0 ):
    """
    Expected Improvement
    """
    x=workspace.x
    mean, sigma = get_moments(model, x)     
    fmax = max(model.predict(model.X)[0])
    phi, Phi, _ = get_quantiles(fmax, mean, sigma, acquisition_par=acquisition_par)    
    f_acqu = (-fmax + mean - acquisition_par) * Phi + sigma * phi
    return x, f_acqu  # note: returns negative value for posterior minimization 

def EI_GPIS(model, workspace,  level, acquisition_par =0):
    """
    Expected Improvement
    """
    x=workspace.x
    mean, sigma = get_moments(model, x)     
    sdf=abs(mean-level)
    fmin = min(abs(model.predict(model.X)[0]-level))
    phi, Phi, _ = get_quantiles(fmin, sdf, sigma, acquisition_par=acquisition_par)    
    f_acqu = (-sdf+fmin + acquisition_par) * Phi + sigma * phi
    return x, f_acqu  # note: returns negative value for posterior minimization 

# def UCBISacquisition_function(model, boundaryestimate, level, numpoints=1):
#     """
#     uncertainty along the surface estimate (this only works with the max ac for now...)
#     """
#     mean, sigma = get_moments(model, boundaryestimate, level)  
    
#     return boundaryestimate, sigma 

# def MLacquisition_function(model, x, level, acquisition_par=0):
#     """
#     Marginal Likelihood
#     marginal curve likelihood
#     P(f(x) = 0).
#     """
#     mean, sigma, _ = get_moments(model, x, level) 
#     f_acqui = np.zeros(x.shape) 
#     phi = stats.distributions.norm.pdf
#     GPIS = phi(mean,loc = level, scale = (sigma))
#     GPIS = GPIS/GPIS.max()
#     return  x, GPIS

def length(x,y):
    val = np.sqrt((x-y).dot(x-y))
    return val#np.sqrt(np.sum((x-y)*(x-y)))

def solve_tsp_dynamic(points):
    #calc all lengths
    all_distances = [[length(x,y) for y in points] for x in points]
    #initial value - just distance from 0 to every other point + keep the track of edges
    A = {(frozenset([0, idx+1]), idx+1): (dist, [0,idx+1]) for idx,dist in enumerate(all_distances[0][1:])}
    cnt = len(points)
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0} for C in combinations(range(1, cnt), m)]:
            for j in S - {0}:
                B[(S, j)] = min( [(A[(S-{j},k)][0] + all_distances[k][j], A[(S-{j},k)][1] + [j]) for k in S if k != 0 and k!=j])  #this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
        A = B
    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
    indices = res[1]
    return points[indices]

def maxAcquisition(workspace, AcquisitionFunctionVals, numpoints=1):
    """
    Selects numpoints number of points that are maximal from the list of AcquisitionFunctionVals
    """
    ind = np.argpartition(AcquisitionFunctionVals.flatten(), -numpoints)[-numpoints:]
    newpts = workspace.x[ind]
    return newpts



def ergAcquisition(workspace, AcquisitionFunctionVals, LQ, xinit=[.2,.3], T=10, StepstoReplan=20):
    """
    Selects numpoints number of points that are maximal from the list of AcquisitionFunctionVals
    inputs:
        xinit: starting point for ergodic trajectory optimization
        T : time horizon for ergodic trajectory optimization
        dt : time discretization for ergodic trajectory optimization
        StepstoReplan: number of steps to take/measurements to take before replanning (receding horizon)
    """
    pdf = AcquisitionFunctionVals.reshape(workspace.res,workspace.res)
    newpts = ErgodicPlanner.ergoptimize(LQ,
                                      pdf,
                                      xinit,
                                      maxsteps=25,plot=True)
    #just use the first half of trajectory
    #newlen=int(newpts.shape[0]/2)
    return newpts #newpts[0::newlen]
    
def randompoints(bounds, numpoints=1):
    rangeX = bounds[0]
    rangeY = bounds[1]
    return np.array([np.random.uniform(rangeX[0],
                                       rangeX[1],
                                       numpoints),
                     np.random.uniform(rangeY[0],
                                       rangeY[1],
                                       numpoints)]).T



