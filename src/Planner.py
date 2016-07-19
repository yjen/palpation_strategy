import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from simUtils import *
from scipy.stats import norm
from utils import *
from GaussianProcess import *
import ErgodicPlanner
from itertools import combinations


#######################################################################
#Acquisition Functions
#######################################################################


def UCB_dGPIS(model, workspace, level=0, x=None, acquisition_par=[0,0], numpoints=1):
    '''
    Choose next sample points based on maximizing prior variance + gradient of the mean
    :model: Model from GP
    :acquisition_par weights for acquisition function: level set, variance, mean
    '''
    mean, sigma = get_moments(model, workspace.x)  
    sigma=sigma/sigma.max()   
    fd= gradfd(mean,workspace)
    fd=fd/np.max(fd)
    fd[np.isinf(fd)]=0

    fd=np.array([fd.flatten()]).T
    fd=fd/fd.max()
    implev=acquisition_par[0]*(fd.max()-fd.min())+fd.min()

    bound=getLevelSet (workspace, fd, implev)
    sdf=abs(fd-implev)
    sdf=sdf/sdf.max()

    f_acqu =  acquisition_par[1]*sigma - (1-acquisition_par[1])*sdf

    buffx=.1*(workspace.bounds[0][1]-workspace.bounds[0][0])
    buffy=.1*(workspace.bounds[1][1]-workspace.bounds[1][0])
    f_acqu[workspace.x[:,0]<workspace.bounds[0][0]+buffx]=f_acqu.mean()
    f_acqu[workspace.x[:,1]<workspace.bounds[1][0]+buffy]=f_acqu.mean()
    f_acqu[workspace.x[:,0]>workspace.bounds[0][1]-buffx]=f_acqu.mean()
    f_acqu[workspace.x[:,1]>workspace.bounds[1][1]-buffy]=f_acqu.mean()

    return workspace.x, f_acqu  # note: returns negative value for posterior minimization


def MaxVar_GP(model, workspace, level=0,x=None, acquisition_par=0):
    """
    Choose next sample points based on maximizing prior variance
    :model: Model from GP
    :acquisition_par: None for this function
    """
    if x==None:
        x=workspace.x
    #x = multigrid(bounds, res)
    mean, sigma = get_moments(model, x)   
    sigma=sigma/sigma.max()
    sigmasq = sigma.reshape(workspace.res,workspace.res)
  
    sigmasq[:,0:10]=0
    sigmasq[0:10,:]=0
    sigmasq[:,-10:]=0
    sigmasq[-10:,:]=0

    f_acqu = sigmasq.flatten()

    return x, f_acqu  # note: returns negative value for posterior minimization

def UCB_GP(model, workspace, level=0, x=None, acquisition_par=.8):
    """
    Choose next sample points based on maximizing Upper Confidence Bound
    :model: Model from GP
    :acquisition_par: relative weighting of variance vs. mean
    """
    if x==None:
        x=workspace.x
    mean, sigma = get_moments(model, x)     
   
    mean=mean/mean.max()

    sigma=sigma/sigma.max()

    f_acqu = acquisition_par*sigma + (1-acquisition_par)*mean  #acquisition_par * (mean) +  sigma

    buffx=.1*(workspace.bounds[0][1]-workspace.bounds[0][0])
    buffy=.1*(workspace.bounds[1][1]-workspace.bounds[1][0])
    f_acqu[workspace.x[:,0]<workspace.bounds[0][0]+buffx]=f_acqu.mean()
    f_acqu[workspace.x[:,1]<workspace.bounds[1][0]+buffy]=f_acqu.mean()
    f_acqu[workspace.x[:,0]>workspace.bounds[0][1]-buffx]=f_acqu.mean()
    f_acqu[workspace.x[:,1]>workspace.bounds[1][1]-buffy]=f_acqu.mean()
    return x, f_acqu  # note: returns negative value for posterior 



def UCB_GPIS(model, workspace, level=0, x=None, acquisition_par=.1 ):
    """
    Upper Confidence Bound over the level set "level"
    :model: Model from GP
    :acquisition_par: relative weighting of variance vs. mean
    """
    x=workspace.x
    mean, sigma = get_moments(model, x)  
    bound = getLevelSet (workspace, mean, level)
    sigma=sigma/sigma.max()
    sdf=abs(mean-level)
    sdf=sdf/sdf.max()
    f_acqu =acquisition_par*sigma - (1-acquisition_par)* (sdf)
    f_acqu=f_acqu+abs(min(f_acqu)) 
    buffx=.01*(workspace.bounds[0][1]-workspace.bounds[0][0])

    buffy=.01*(workspace.bounds[1][1]-workspace.bounds[1][0])
    f_acqu[workspace.x[:,0]<workspace.bounds[0][0]+buffx]=f_acqu.mean()
    f_acqu[workspace.x[:,1]<workspace.bounds[1][0]+buffy]=f_acqu.mean()
    f_acqu[workspace.x[:,0]>workspace.bounds[0][1]-buffx]=f_acqu.mean()
    f_acqu[workspace.x[:,1]>workspace.bounds[1][1]-buffy]=f_acqu.mean()

    return x, f_acqu  # note: returns negative value for posterior minimization

def UCB_GPIS_implicitlevel(model, workspace, level=0, x=None, acquisition_par=[.1,.5]):
    """
    Upper Confidence Bound over the implicit level set
    :model: Model from GP
    :acquisition_par: implicit level, relative weighting of variance vs. mean
    """
    x=workspace.x
    
    mean, sigma = get_moments(model, x)  
    sigma=sigma/sigma.max()
    lev=(mean.max()-mean.min())+mean.min()
    implev=acquisition_par[0]*lev
    bound=getLevelSet (workspace, mean, implev)
    bound=bound.flatten()
    sdf=abs(mean-level)
    sdf=sdf/sdf.max()

    f_acqu =acquisition_par[1]*sigma - (1-acquisition_par[1])* (sdf)
    f_acqu=f_acqu+abs(min(f_acqu)) 

    buffx=.01*(workspace.bounds[0][1]-workspace.bounds[0][0])
    buffy=.01*(workspace.bounds[1][1]-workspace.bounds[1][0])
    f_acqu[workspace.x[:,0]<workspace.bounds[0][0]+buffx]=f_acqu.mean()
    f_acqu[workspace.x[:,1]<workspace.bounds[1][0]+buffy]=f_acqu.mean()
    f_acqu[workspace.x[:,0]>workspace.bounds[0][1]-buffx]=f_acqu.mean()
    f_acqu[workspace.x[:,1]>workspace.bounds[1][1]-buffy]=f_acqu.mean()

    return x, f_acqu  # note: returns negative value for posterior minimization

#######################################################################
# Utility functions for Planning
#######################################################################

def maxAcquisition(workspace, AcquisitionFunctionVals, numpoints=1):
    """
    Selects numpoints number of points that are maximal from the list of AcquisitionFunctionVals
    """
    ind = np.argpartition(AcquisitionFunctionVals.flatten(), -numpoints)[-numpoints:]
    newpts = workspace.x[ind]
    return newpts


def batch_optimization(model,workspace,aqfunction, n_inbatch, xinit, GP_params, level=0, acquisition_par=.1):   
    '''
    Computes batch optimization using the predictive mean to obtain new batch elements
    :param acquisition: acquisition function in which the batch selection is based
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


def length(x,y):
    # Calculate distance between points x and y
    val = np.sqrt((x-y).dot(x-y))
    return val


def solve_tsp_dynamic(points):
    '''
    Given a list of (x,y) locations to collect samples at, order points
    to minimize overall time travelled by solving a TSP
    '''
    # calc all  lengths between points
    all_distances = [[length(x,y) for y in points] for x in points]
    # initial value - just distance from 0 to every other point + keep the track of edges
    A = {(frozenset([0, idx+1]), idx+1): (dist, [0,idx+1]) for idx,dist in enumerate(all_distances[0][1:])}
    cnt = len(points)
    # approximately solve the TSP
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0} for C in combinations(range(1, cnt), m)]:
            for j in S - {0}:
                B[(S, j)] = min( [(A[(S-{j},k)][0] + all_distances[k][j], A[(S-{j},k)][1] + [j]) for k in S if k != 0 and k!=j])  #this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
        A = B
    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
    indices = res[1]
    return points[indices]


def randompoints(bounds, numpoints=1):
    # Select numpoints number of sample points within bounds
    rangeX = bounds[0]
    rangeY = bounds[1]
    return np.array([np.random.uniform(rangeX[0],
                                       rangeX[1],
                                       numpoints),
                     np.random.uniform(rangeY[0],
                                       rangeY[1],
                                       numpoints)]).T


# Other Acquisition Functions: Not verified

# def UCB_dGPIS2(model, workspace, level=0, x=None, acquisition_par=[0,0,0], numpoints=1):
#     """
#     Choose next sample points based on maximizing prior variance + gradient of the mean + mean
#     :model: Model from GP
#     :acquisition_par weights for acquisition function: level set, variance, mean
#     """
#     mean, sigma = get_moments(model, workspace.x)  
#     sigma=sigma/sigma.max()   
#     fd= gradfd(mean,workspace)
#     fd=fd/np.max(fd)
#     fd[np.isinf(fd)]=0

#     fd=np.array([fd.flatten()]).T
#     fd=fd/fd.max()
#     implev=acquisition_par[0]*(fd.max()-fd.min())+fd.min()

#     bound=getLevelSet (workspace, fd, implev)

#     sdf=abs(fd-implev)
#     sdf=sdf/sdf.max()

#     mean=mean/mean.max()
#     mean=np.abs(mean - np.mean(mean))

#     f_acqu =  acquisition_par[1]*sigma + acquisition_par[2]*mean - (1-acquisition_par[1]-acquisition_par[2])*sdf

#     buffx=.05*(workspace.bounds[0][1]-workspace.bounds[0][0])
#     buffy=.05*(workspace.bounds[1][1]-workspace.bounds[1][0])
#     f_acqu[workspace.x[:,0]<workspace.bounds[0][0]+buffx]=f_acqu.mean()
#     f_acqu[workspace.x[:,1]<workspace.bounds[1][0]+buffy]=f_acqu.mean()
#     f_acqu[workspace.x[:,0]>workspace.bounds[0][1]-buffx]=f_acqu.mean()
#     f_acqu[workspace.x[:,1]>workspace.bounds[1][1]-buffy]=f_acqu.mean()

#     return workspace.x, f_acqu  # note: returns negative value for posterior minimization

# def dmaxAcquisition(model, workspace, acfun, xinit=[.2,.3], numpoints=1, level=0):
#     """
#     Selects numpoints number of points that are maximal from the list of AcquisitionFunctionVals
#     """
#     dx=.1
#     x=[xinit]
#     for n in range(numpoints):
#         _, current_ac = acfun(model, workspace, x=x[n], level=level)
#         testpts=x[n]+np.array([[dx,0],[-dx,0],[-dx,-dx],[0,dx],[dx,dx],[dx,-dx],[-dx,dx],[0,-dx]])
#         allpts=np.vstack((x[n],testpts))
#         allpts, new_acqu = acfun(model, workspace, x= allpts,level=level)

#         grad=new_acqu-current_ac  
#         i=0
#         ind = np.argpartition(new_acqu.flatten(), -1)[-1-i]
#         newpt = allpts[ind]
#         while (newpt[0]>workspace.bounds[0][1] or newpt[0]<workspace.bounds[0][0] or 
#                 newpt[1]>workspace.bounds[1][1] or newpt[1]<workspace.bounds[1][0]):
#             i = i+1
#             ind = np.argpartition(new_acqu.flatten(), -1)[-1-i]
#             newpt = allpts[ind]
#         x.append(newpt)
#     return np.array(x[1:])
# def d_UCB_GP(model,x,acquisition_par=1):
#     """
#     Derivative of the Upper Confidence Band
#     """
#     dmdx, dsdx = get_d_moments(model, x)
#     df_acqu = acquisition_par * dmdx +  dsdx
#     return df_acqu

# def ergAcquisition(workspace, AcquisitionFunctionVals, LQ, xinit=[.2,.3], T=10, StepstoReplan=20):
#     """
#     Selects numpoints number of points that are maximal from the list of AcquisitionFunctionVals
#     inputs:
#         xinit: starting point for ergodic trajectory optimization
#         T : time horizon for ergodic trajectory optimization
#         dt : time discretization for ergodic trajectory optimization
#         StepstoReplan: number of steps to take/measurements to take before replanning (receding horizon)
#     """
#     pdf = AcquisitionFunctionVals.reshape(workspace.res,workspace.res)
#     newpts = ErgodicPlanner.ergoptimize(LQ,
#                                       pdf,
#                                       xinit,
#                                       maxsteps=25,plot=True)
#     return newpts #newpts[0::newlen]

    ## ---- Predictive batch optimization
# def d_UCB_GPIS(model,x,acquisition_par=1):
#     """
#     Derivative of the Upper Confidence Band
#     """
#     dmdx, dsdx = get_d_moments(model, x)
#     df_acqu = acquisition_par * dmdx +  dsdx
#     return df_acqu

# def EI_GP(model, workspace, level=0, x=None, acquisition_par = 0 ):
#     """
#     Expected Improvement
#     """
#     if x==None:
#         x=workspace.x
#     mean, sigma = get_moments(model, x)     
#     fmax = max(model.predict(model.X)[0])
#     phi, Phi, _ = get_quantiles(fmax, mean, sigma, acquisition_par=acquisition_par)    
#     f_acqu = (-fmax + mean - acquisition_par) * Phi + sigma * phi
#     return x, f_acqu  # note: returns negative value for posterior minimization 

# def EI_GPIS(model, workspace,  level=0, x=None, acquisition_par =0):
#     """
#     Expected Improvement
#     """
#     if x==None:
#         x=workspace.x
#     mean, sigma = get_moments(model, x)     
#     sdf=abs(mean-level)
#     fmin = min(abs(model.predict(model.X)[0]-level))
#     phi, Phi, _ = get_quantiles(fmin, sdf, sigma, acquisition_par=acquisition_par)    
#     f_acqu = (-sdf+fmin + acquisition_par) * Phi + sigma * phi
#     return x, f_acqu  # note: returns negative value for posterior minimization 

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
