import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.path as path
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm #colormap
from scipy.special import erfc
from scipy.spatial.distance import cdist
import GPy
import pickle

class Workspace(object):
    '''
    Class to store worksapce information for all calculations
    :bounds: 2 x 2 tuple specifying workspace dimensions
    :res: resolution along each dimension, in number of gridpoints (e.g. 200)
    '''
    def __init__(self, bounds, res):
        self.bounds=bounds
        self.res=res
        self.x = multigrid(bounds, res)
        self.xx=self.x.T[0].reshape(self.res,self.res).T
        self.yy=self.x.T[1].reshape(self.res,self.res).T
        self.xlin=np.linspace(bounds[0][0], bounds[0][1],res)
        self.ylin=np.linspace(bounds[1][0], bounds[1][1],res)


def stereo_pad(x,y,z,rangeX,rangeY):
        '''
        Function to pad the stereo measurements by a fixed amount. 
        Necessary to avoid variance issues at the edges of the region when using Gaussian Processes
        :x,y are measurement locations
        :z is list of measurements
        :rangeX and rangeY are the range of the stereo measurements
        '''
        percentpad=.1

        padbyX = percentpad*(rangeX[1]-rangeX[0])
        padbyY = percentpad*(rangeY[1]-rangeY[0])

        # how many extra points to add in each direction
        gridSize=40
        numpads=np.int(gridSize*percentpad)

        # pad grid arrays

        x=np.pad(x,(numpads,numpads),mode='linear_ramp',
                 end_values=(rangeX[0]-padbyX,rangeX[1]+padbyX))
        y=np.pad(y,(numpads,numpads),mode='linear_ramp',
                 end_values=(rangeY[0]-padbyY,rangeY[1]+padbyY))
        xx, yy = np.meshgrid(x, y)

        z=np.pad(z,numpads,mode='edge')
        # plt.scatter(xx, yy, c=z)
        # plt.show()
        return xx,yy,z


def get_quantiles(fmin, m, s, acquisition_par=0):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param acquisition_par: parameter of the acquisition function
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations. 
    '''
    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
    u = (m-fmin+acquisition_par)/s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return (phi, Phi, u)

def get_moments(model,x):
    '''
    Returns Moments (mean and sdev.) of a GP model at x
    :param model: GPy model.  
    :param x: location where the gradients are evaluated.
    '''
    input_dim = model.X.shape[1] 
    x = reshape(x,input_dim)    
    try:
        m, v = model.predict(x)
    except TypeError:
        # k1=GPy.kern.RBF(2)      
        # this is for Heteroscedastic-measurements in phase 1  
        # make prediction assuming the new variance is drawn from stereo only
        numPoints = np.shape(x)[0]        
        m,v = model.predict(x,Y_metadata={'output_index':np.zeros((numPoints,1))[:,None].astype(int)})

    s = np.sqrt(np.clip(v, 0, np.inf))
    return (m,s)

def get_d_moments(model,x):
    '''
    returns Gradients with respect to x of the moments (mean and sdev.) of the GP
    :param model: GPy model.  
    :param x: location where the gradients are evaluated.
    '''
    input_dim = model.input_dim
    x = reshape(x,input_dim)
    _, v = model.predict(x)
    dmdx, dvdx = model.predictive_gradients(x)
    dmdx = dmdx[:,:,0]
    dsdx = dvdx / (2*np.sqrt(v))
    return (dmdx, dsdx)

def multigrid(bounds, Ngrid):
    '''
    Generates a multidimensional lattice
    :param bounds: box constrains
    :param Ngrid: number of points per dimension.
    '''
    if len(bounds)==1:
        return np.linspace(bounds[0][0], bounds[0][1],
                           Ngrid).reshape(Ngrid, 1)
    xx = np.meshgrid(*[np.linspace(b[0], b[1], Ngrid) for b in bounds]) 
    return np.vstack([x.flatten(order='F') for x in xx]).T


def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns
    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x


def gridreshape(x,workspace):
    '''
    Reshapes x into a matrix with input_dim columns
    '''
    xl=x.reshape(workspace.res,
             workspace.res)
    xl=xl.T
    return xl


def distanceBetweenCurves(C1, C2):
    D = cdist(C1, C2, 'euclidean')
    # none symmetric Hausdorff distances
    H1 = np.max(np.min(D, axis=1))
    H2 = np.max(np.min(D, axis=0))

    return (H1 + H2) / 2.


def save_p2_data(dat, filename):
    try:
        pickle.dump(dat, open(filename, "wb"))
    except Exception as e:
        print "Exception: ", e


def gradfd(data,workspace):
    '''
    returns norm of the gradient (finite difference) of the data in x and y direction 
    :data: list of inputs, as (x,y) pairs
    '''
    # reshape data into grid
    meansq = data.reshape(workspace.res,workspace.res)

    # space between values
    dx=(workspace.bounds[0][1]-workspace.bounds[0][0])/float(workspace.res)
    dy=(workspace.bounds[1][1]-workspace.bounds[1][0])/float(workspace.res)

    grad = np.gradient(meansq,dy,dx)
    dMdx,dMdy = grad

    #normalize
    fd = np.sqrt((dMdx**2+dMdy**2))

    return fd

# def plot_acq():
#     plot=True
#     dirname='compareacquisition'
#     phantomname = rantumor
#     AcFunction=MaxVar_GP
#     control='Max' 
#     bounds=((-.04,.04),(-.04,.04))
#     gridres = 200
#     workspace = Workspace(bounds,gridres)
#     level = .5 #pick something between min/max deflection

#     plot_data = None

#     ###############
#     #Initializing
#     ###############

#     means, sigmas, acqvals, measures, healthyremoved, tumorleft, num_iters,gpmodel = run_single_phase2_simulation(
#         phantomname, dirname, AcFunction=UCB_GPIS, control='Max', plot=False, exp=False,iters=2)


#     AcFunctions=[UCB_GPIS_implicitlevel,MaxVar_plus_gradient,UCB_GP,UCB_GPIS,MaxVar_GP]
#     aqfunctionsnames=["UCB_GPIS_implicitlevel","MaxVar_plus_gradient","UCB_GP", "UCB_GPIS", "MaxVar_GP"]#, "random"]
#     for j in range (len(AcFunctions)): #(1,100,1)
#         AcFunction=AcFunctions[j]
        
#         # evaluate selected aqcuisition function over the grid
#         if AcFunction==UCB_GPIS:
#             acquisition_par=.05
#         elif AcFunction==UCB_GP:
#             acquisition_par=.7
#         elif AcFunction==MaxVar_plus_gradient:
#             acquisition_par=.6
#         elif AcFunction==UCB_GPIS_implicitlevel:
#             acquisition_par=[.6,.5]
#         else:
#             acquisition_par=0
#         xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, level=level, acquisition_par=acquisition_par)
#                 # Plot everything
#         directory=dirname+'_' + aqfunctionsnames[j]
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         if plot==True:
#             time.sleep(0.0001)
#             plt.pause(0.0001)  
#             plot_data = plot_beliefGPIS(phantomname, workspace, means[-1], sigmas[-1],
#                                       AqcuisFunction, measures[-1],
#                                       directory, [0,0],plot_data,level=level,
#                                       iternum=j, projection3D=False)

