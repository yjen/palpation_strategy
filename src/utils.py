import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.path as path
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm #colormap
from scipy.special import erfc
from scipy.spatial.distance import cdist
import GPy
import pickle

# def evalerror(tumor,workspace,mean,variance,level):
#     # see: http://toblerity.org/shapely/manual.html
#     boundaryestimate = getLevelSet (workspace, mean, level)
#     # boundaryestimateupper = getLevelSet (workspace, mean+variance, level)
#     # boundaryestimatelower = getLevelSet (workspace, mean-variance, level)
#     # boundaryestimate= boundaryestimateupper
#     GroundTruth = np.vstack((tumor,tumor[0]))
#     GroundTruth=Polygon(GroundTruth)
#     # print GroundTruth
#     if len(boundaryestimate)>0:

#         boundaryestimate=Polygon(boundaryestimate)
#         # print boundaryestimate

#         #boundaryestimate=polybuff(boundaryestimate, minus=True)
#         # print boundaryestimate
#         #healthyremoved=boundaryestimate.difference(GroundTruth) # mislabeled data ()
#         #boundaryestimate.difference(GroundTruth) #mislabeled as tumor--extra that would be removed
#         #
#         boundaryestimate=boundaryestimate.buffer(-offset)
#         err=GroundTruth.symmetric_difference(boundaryestimate)
#         #tumorleft=GroundTruth.difference(boundaryestimate) # mislbaled as not-tumor--would be missed and should be cut out
#         #correct=boundaryestimate.intersection(GroundTruth) #correctly labeled as tumor
#         #healthyremoved=healthyremoved.area
#         err=err.area

#     else:
#         err=.100
#         tumorleft=.100
#     return err, 0


def plotBelief (xx,yy,z):
	#xx,yy -- matrix obtained from meshgrid
	# z -- matrix of values		
	fig = plt.figure()	
	ax = fig.gca(projection='3d')

	ax.plot_surface(xx, yy, z, rstride=5, cstride=5, alpha=0.4)
	cset = ax.contourf(xx, yy, z, zdir='z', offset=-2, cmap=cm.coolwarm)
	cset = ax.contourf(xx, yy, z, zdir='x', offset=-5, cmap=cm.coolwarm)
	cset = ax.contourf(xx, yy, z, zdir='y', offset=5, cmap=cm.coolwarm)

	ax.set_xlabel('X')
	ax.set_xlim(-5, 5)
	ax.set_ylabel('Y')
	ax.set_ylim(-5, 5)
	ax.set_zlabel('Z')
	ax.set_zlim(-2, 1)

	plt.show(block=False)	
	# plt.draw()

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# # simulates probe & tumor. questionable
# def get_probe_data(self):
# 	pos = self.calculate_tip_pos()
# 	if pos[1] <= 0:
# 		if abs(0.5 - pos[0]) <= 0.125:
# 			#print(4096 -
# 			 abs(0.25 - pos[0]) * 4000)
# 			#return 4096 - abs(0.25 - pos[0]) * 4000
# 			return 4096 * gaussian(pos[0], 0.5, 0.1) + 64 * random.gauss(0, 6)  
# 	return 64 * random.gauss(0, 6)

class Params():
	def __init__(self):
		self.gridSize = [100,100]
		self.minHeight = -1
		self.maxHeight = 1
		print "Initializing Parameters"						

	def getParams(self):
		return self

	def setParams(self):
		pass

########################## New--added by Lauren

def stereo_pad(x,y,z,rangeX,rangeY):
        # pad the stereo measurements by a fixed amount. Necessary to avoid weird undertainty at the edges of the region when using Gaussian Processes
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

########################## Plot Scripts




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
    Moments (mean and sdev.) of a GP model at x
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

def get_moments_old(model,x):
    '''
    Moments (mean and sdev.) of a GP model at x
    '''
    input_dim = model.X.shape[1]
    x = reshape(x,input_dim)
    try:
        m, v = model.predict(x)
    except TypeError:
        k1=GPy.kern.RBF(2)        
        m, v = model._raw_predict(x)        
        v += model.likelihood.variance[0]
    s = np.sqrt(np.clip(v, 0, np.inf))
    return (m,s)

#need to get the old reshape function from gpyopt
def get_d_moments(model,x):
    '''
    Gradients with respect to x of the moments (mean and sdev.) of the GP
    :param model: GPy model.  :param x: location where the gradients are
    evaluated.
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

    #none symmetric Hausdorff distances
    H1 = np.max(np.min(D, axis=1))
    H2 = np.max(np.min(D, axis=0))

    return (H1 + H2) / 2.

def save_p2_data(dat, filename):
        try:
            pickle.dump(dat, open(filename, "wb"))
        except Exception as e:
            print "Exception: ", e


class Workspace(object):
    def __init__(self, bounds, res):
        self.bounds=bounds
        self.res=res
        self.x = multigrid(bounds, res)
        self.xx=self.x.T[0].reshape(self.res,self.res).T
        self.yy=self.x.T[1].reshape(self.res,self.res).T
        self.xlin=np.linspace(bounds[0][0], bounds[0][1],res)
        self.ylin=np.linspace(bounds[1][0], bounds[1][1],res)


def plot_acq():
    plot=True
    dirname='compareacquisition'
    phantomname = rantumor
    AcFunction=MaxVar_GP
    control='Max' 
    bounds=((-.04,.04),(-.04,.04))
    gridres = 200
    workspace = Workspace(bounds,gridres)
    level = .5 #pick something between min/max deflection

    plot_data = None

    ###############
    #Initializing
    ###############

    means, sigmas, acqvals, measures, healthyremoved, tumorleft, num_iters,gpmodel = run_single_phase2_simulation(
        phantomname, dirname, AcFunction=UCB_GPIS, control='Max', plot=False, exp=False,iters=2)


    AcFunctions=[UCB_GPIS_implicitlevel,MaxVar_plus_gradient,UCB_GP,UCB_GPIS,MaxVar_GP]
    aqfunctionsnames=["UCB_GPIS_implicitlevel","MaxVar_plus_gradient","UCB_GP", "UCB_GPIS", "MaxVar_GP"]#, "random"]
    for j in range (len(AcFunctions)): #(1,100,1)
        AcFunction=AcFunctions[j]
        
        # evaluate selected aqcuisition function over the grid
        if AcFunction==UCB_GPIS:
            acquisition_par=.05
        elif AcFunction==UCB_GP:
            acquisition_par=.7
        elif AcFunction==MaxVar_plus_gradient:
            acquisition_par=.6
        elif AcFunction==UCB_GPIS_implicitlevel:
            acquisition_par=[.6,.5]
        else:
            acquisition_par=0
        xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, level=level, acquisition_par=acquisition_par)
                # Plot everything
        directory=dirname+'_' + aqfunctionsnames[j]
        if not os.path.exists(directory):
            os.makedirs(directory)
        if plot==True:
            time.sleep(0.0001)
            plt.pause(0.0001)  
            plot_data = plot_beliefGPIS(phantomname, workspace, means[-1], sigmas[-1],
                                      AqcuisFunction, measures[-1],
                                      directory, [0,0],plot_data,level=level,
                                      iternum=j, projection3D=False)
def gradfd(mean,workspace):
    # reshape mean into grid
    meansq = mean.reshape(workspace.res,workspace.res)
    # space between values
    dx=(workspace.bounds[0][1]-workspace.bounds[0][0])/float(workspace.res)
    dy=(workspace.bounds[1][1]-workspace.bounds[1][0])/float(workspace.res)
    grad = np.gradient(meansq,dy,dx)
    dMdx,dMdy = grad
    fd = np.sqrt((dMdx**2+dMdy**2))

    # buffx=.02*workspace.bounds[0][1]
    # buffy=.02*workspace.bounds[1][1]
    # fd=fd/np.max(fd)
    # fd[np.isinf(fd)]=0
    # fd=np.array([fd.flatten()]).T
    # fd[workspace.x[:,0]<buffx]=0
    # fd[workspace.x[:,1]<buffy]=0
    # fd[workspace.x[:,0]>workspace.bounds[0][1]-buffx]=0
    # fd[workspace.x[:,1]>workspace.bounds[1][1]-buffy]=0
    return fd
