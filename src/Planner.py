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



def UCB_dGP(model, workspace, level=0, x=None, acquisition_par=[0,0],numpoints=1):
    """
    choose next sample points based on maximizing prior variance
    """
    #x = multigrid(bounds, res)
    mean, sigma = get_moments(model, workspace.x)     
    fd= gradfd(mean,workspace)
    fd=fd/np.max(fd)
    fd[np.isinf(fd)]=0

    buffx=.05*workspace.bounds[0][1]
    buffy=.05*workspace.bounds[1][1]

    fd=np.array([fd.flatten()]).T
    fd=fd/fd.mean()
    # lev=(fd.max()-fd.min())+fd.min()
    # print lev
    # print mean.shape
    # print fd.shape

    # implev=acquisition_par[1]*lev
    # bound=getLevelSet (workspace, fd, implev)
    #fd=fd.flatten()

    # f_acqu = 1*sigma.flatten()+acquisition_par*fd.flatten()
    # f_acqu=np.array([f_acqu]).T
    # print bound.shape[0]
    # if bound.shape[0]>0:
        # sdf=abs(fd-level)
        # sdf=-mean
        # print -sdf.max()
        # print sigma.mean()
    f_acqu = fd +  acquisition_par[0]*sigma
    f_acqu=abs(f_acqu)+abs(min(f_acqu)) 
    f_acqu[workspace.x[:,0]<buffx]=0
    f_acqu[workspace.x[:,1]<buffy]=0
    f_acqu[workspace.x[:,0]>workspace.bounds[0][1]-buffx]=0
    f_acqu[workspace.x[:,1]>workspace.bounds[1][1]-buffy]=0
    # else: 
    #     f_acqu=sigma

    return workspace.x, f_acqu  # note: returns negative value for posterior minimization

def MaxVar_GP(model, workspace, level=0,x=None, acquisition_par=0):
    """
    choose next sample points based on maximizing prior variance
    """
    if x==None:
        x=workspace.x
    #x = multigrid(bounds, res)
    mean, sigma = get_moments(model, x)   
    sigmasq = sigma.reshape(workspace.res,workspace.res)
  
    sigmasq[:,0:10]=0
    sigmasq[0:10,:]=0
    sigmasq[:,-10:]=0
    sigmasq[-10:,:]=0

    f_acqu = sigmasq.flatten()
    return x, f_acqu  # note: returns negative value for posterior minimization

def UCB_GP(model, workspace, level=0, x=None, acquisition_par=.8  ):
    """
    .8 for ph 1
    Upper Confidence Band
    """
    # print acquisition_par
    # print level
    if x==None:
        x=workspace.x
    #x = multigrid(bounds, res)
    mean, sigma = get_moments(model, x)     
    # print 'mean=', mean.max()
    # print 'sigma=',sigma.max()
    f_acqu = acquisition_par * (mean) +  sigma
    return x, f_acqu  # note: returns negative value for posterior 

def dmaxAcquisition(model, workspace, acfun, xinit=[.2,.3], numpoints=1, level=0):
    """
    Selects numpoints number of points that are maximal from the list of AcquisitionFunctionVals
    """
    dx=.1
    x=[xinit]
    for n in range(numpoints):
        _, current_ac = acfun(model, workspace, x=x[n], level=level)
        testpts=x[n]+np.array([[dx,0],[-dx,0],[-dx,-dx],[0,dx],[dx,dx],[dx,-dx],[-dx,dx],[0,-dx]])
        allpts=np.vstack((x[n],testpts))
        allpts, new_acqu = acfun(model, workspace, x= allpts,level=level)

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

def UCB_GPIS(model, workspace, level=0, x=None, acquisition_par=.1 ):
    """
    Upper Confidence Band
    """
    x=workspace.x
    #x = multigrid(bounds, res)
    # print 'ac=',acquisition_par
    mean, sigma = get_moments(model, x)  
    bound=getLevelSet (workspace, mean, level)
    # bound=bound.flatten()
    # print 'bnd=', bound
    if bound.shape[0]>0:
        sdf=abs(mean-level)
        f_acqu = - sdf +  acquisition_par*sigma
        f_acqu=f_acqu+abs(min(f_acqu)) 
    else: 
        f_acqu=sigma

    return x, f_acqu  # note: returns negative value for posterior minimization

def UCB_GPIS_implicitlevel(model, workspace, level=0, x=None, acquisition_par=[.1,.5]):
    """
    Upper Confidence Band
    """
    x=workspace.x
    
    #x = multigrid(bounds, res)
    # print 'run'
    # print acquisition_par
    mean, sigma = get_moments(model, x)  
    lev=(mean.max()-mean.min())+mean.min()
    # print lev
    implev=acquisition_par[1]*lev
    bound=getLevelSet (workspace, mean, implev)
    # bound=bound.flatten(axis=0)
    if bound.shape[0]>0:
        sdf=abs(mean-level)
        # print -sdf.max()
        # print sigma.mean()
        f_acqu = - sdf +  acquisition_par[0]*sigma
        f_acqu=f_acqu+abs(min(f_acqu)) 
    else: 
        f_acqu=sigma

    return x, f_acqu  # note: returns negative value for posterior minimization


# def d_UCB_GPIS(model,x,acquisition_par=1):
#     """
#     Derivative of the Upper Confidence Band
#     """
#     dmdx, dsdx = get_d_moments(model, x)
#     df_acqu = acquisition_par * dmdx +  dsdx
#     return df_acqu

def EI_GP(model, workspace, level=0, x=None, acquisition_par = 0 ):
    """
    Expected Improvement
    """
    if x==None:
        x=workspace.x
    mean, sigma = get_moments(model, x)     
    fmax = max(model.predict(model.X)[0])
    phi, Phi, _ = get_quantiles(fmax, mean, sigma, acquisition_par=acquisition_par)    
    f_acqu = (-fmax + mean - acquisition_par) * Phi + sigma * phi
    return x, f_acqu  # note: returns negative value for posterior minimization 

def EI_GPIS(model, workspace,  level=0, x=None, acquisition_par =0):
    """
    Expected Improvement
    """
    if x==None:
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
    # print newpts
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

    ## ---- Predictive batch optimization
def batch_optimization(model,workspace,aqfunction, n_inbatch, xinit, GP_params, level=0, acquisition_par=.1):   
    '''
    Computes batch optimization using the predictive mean to obtain new batch elements
    :param acquisition: acquisition function in which the batch selection is based
    :param d_acquisition: gradient of the acquisition
    :param bounds: the box constrains of the optimization
    :param acqu_optimize_restarts: the number of restarts in the optimization of the surrogate
    :param acqu_optimize_method: the method to optimize the acquisition function
    :param model: the GP model based on the current samples
    :param n_inbatch: the number of samples to collect
    '''
    model_copy = model.copy()
    X = model_copy.X 
    Y = model_copy.Y
    input_dim = X.shape[1] 
    kernel = model_copy.kern    

    # Optimization of the first element in the batch
    xgrid,AcquisitionFunctionVals = aqfunction(model, workspace, level=level, x=None, acquisition_par=acquisition_par)
    X_new = maxAcquisition(workspace, AcquisitionFunctionVals, numpoints=1)
    #X_new = optimize_acquisition(acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None)
    X_batch = reshape(X_new,input_dim)

    if n_inbatch>1:
        
        k = 1
        while k < n_inbatch:
            X = np.vstack((X, reshape(X_new,input_dim))) 
            Y = np.vstack((Y, model.predict(reshape(X_new, input_dim))[0]))
            model = update_GP(np.hstack((X, Y)),params=GP_params)
            xgrid,AcquisitionFunctionVals = aqfunction(model, workspace, level=level, x=None, acquisition_par=acquisition_par )
            X_new = maxAcquisition(workspace, AcquisitionFunctionVals, numpoints=1)
            X_batch = np.vstack((X_batch,X_new))
            k+=1 
        X_batch=np.vstack((xinit,X_batch))
        X_batch=solve_tsp_dynamic(X_batch)
    # if interpolate==True:
    #     x = np.linspace(0, 10, num=11, endpoint=True)
    #     y = np.cos(-x**2/9.0)
    #     f = interp1d(x, y)
    #     f2 = interp1d(x, y, kind='cubic')
    return X_batch
    
def randompoints(bounds, numpoints=1):
    rangeX = bounds[0]
    rangeY = bounds[1]
    return np.array([np.random.uniform(rangeX[0],
                                       rangeX[1],
                                       numpoints),
                     np.random.uniform(rangeY[0],
                                       rangeY[1],
                                       numpoints)]).T



