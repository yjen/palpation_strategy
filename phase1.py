import numpy as np
import GPy
import GPyOpt
from numpy.random import seed
import simulated_disparity
from matplotlib import pyplot as plt
from matplotlib import cm

# surfaces



def gaussian_pdf(X):
    mux = 2*.25
    muy = 2*.3
    mux2 =2* .24
    muy2=2*.4
    varx = .1
    vary = .1
    
    pd= np.exp(-((X[0] - mux)**2/( 2*varx**2)) -
               ((X[1] - muy)**2/(2*vary**2))) + \
               np.exp(-((X[0] - mux)**2/( 2*varx**2)) -
               ((X[1] - muy)**2/(2*vary**2))) + \
               np.exp(-((X[0] - mux2)**2/( 2*varx**2)) -
                      ((X[1] - muy2)**2/(2*vary**2)))
    return pd

# def plot_groundtruth(bounds=[(-2, 2), (-1, 1)]):
#     x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
#     x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
#     X1, X2 = np.meshgrid(x1, x2)
#     X = np.hstack((X1.reshape(100*100,1),X2.reshape(100*100,1)))
#     Y = sixhumpcamel(X)
        
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot_surface(X1, X2, Y.reshape((100,100)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#     ax.set_title("Ground Truth")    
            
#     plt.figure()    
#     plt.contourf(X1, X2, Y.reshape((100,100)),100)
#     # if (len(self.min)>1):    
#     #     plt.plot(np.array(self.min)[:,0], np.array(self.min)[:,1], 'w.', markersize=20, label=u'Observations')
#     # else:
#     #     plt.plot(self.min[0][0], self.min[0][1], 'w.', markersize=20, label=u'Observations')
#     plt.colorbar()
#     plt.xlabel('X1')
#     plt.ylabel('X2')
#     plt.title("df")
#     plt.show()

# def plot_GP(bounds=[(-2, 2), (-1, 1)],):
#     x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
#     x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
#     X1, X2 = np.meshgrid(x1, x2)
#     X = np.hstack((X1.reshape(100*100,1),X2.reshape(100*100,1)))
#     Y = sixhumpcamel(X)
        
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot_surface(X1, X2, Y.reshape((100,100)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     ax.set_title("GP Mean")    
            
#     plt.figure()    
#     plt.contourf(X1, X2, Y.reshape((100,100)),100)
#     # if (len(self.min)>1):    
#     #     plt.plot(np.array(self.min)[:,0], np.array(self.min)[:,1], 'w.', markersize=20, label=u'Observations')
#     # else:
#     #     plt.plot(self.min[0][0], self.min[0][1], 'w.', markersize=20, label=u'Observations')
#     plt.colorbar()
#     plt.xlabel('X1')
#     plt.ylabel('X2')
#     plt.title("df")
#     plt.show()
        
def update_GP(measurements,meas_locations,plot_GP=True):

    sensornoise=.00001

    # Instanciate and fit Gaussian Process Model
    X=meas_locations
    Y=measurements
   # Y=np.array([measurements]).T
    ker = GPy.kern.RBF(input_dim=2 #variance=1., lengthscale=.05
    ) +\
          GPy.kern.White(2)#GPy.kern.Bias(1)
    #print X.shape
    #print np.array(Y).T.shape
    m = GPy.models.GPRegression(X,Y,ker)
    #m = GPy.models.SparseGPClassification(X, Y, kernel=ker, num_inducing=measurements.shape[0])
    m.optimize(messages=True,max_f_eval = 10)
    print "dd"
    # evaluate the mean, variance on a grid
    x1, x2 = np.meshgrid(np.linspace(xmin+.2, xmax-.2, 100),
                     np.linspace(ymin+.2, ymax-.2, 100))
    xgrid = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T
    y_pred = m.predict(xgrid)[0]
    #y_pred=m.posterior.mean
    y_pred = y_pred.reshape(x1.shape)
    sigma = m.predict(xgrid)[1]
    #sigma=m.predict_quantiles(xgrid)

    sigma = sigma.reshape(x1.shape)
    print(m)
    m.plot()
    if plot_GP==True:
        #fig = plt.figure(figsize=(16, 4))
        ax = fig.add_subplot(132, projection='3d')
        #ax.plot_surface(xgrid.T[0], xgrid.T[1], measurements, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #ax = fig.add_subplot(132, projection='3d')
        ax.plot_surface(x1, x2, y_pred, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title("GP Mean")  
        #ax = fig.add_subplot(121, projection='3d')

        #ax = mlab.mesh(x1, x2, y_pred)
        #mlab.show()
        ax1 = fig.add_subplot(133, projection='3d')
        lim=1
        cs1=ax1.plot_surface(x1, x2,sigma, rstride=1, cstride=1, cmap=cm.Greys, linewidth=0, antialiased=False)
        #ax1.set_zlim3d(0, .35)
        ax1.set_title("GP Uncertainty")  

        plt.colorbar(cs1)
        plt.show()
    return [y_pred,sigma,m]


# setup the workspace
# boundaries: 
xmin=-2
xmax=2
ymin=-1
ymax=1
sample_res=20

x1, x2 = np.meshgrid(np.linspace(xmin, xmax, sample_res),
                     np.linspace(ymin, ymax, sample_res))
xx =np.array([x1,x2])
xgrid = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T

# simulated surface from GPyOpt
sixhumpcamel = GPyOpt.fmodels.experiments2d.sixhumpcamel().f
depth_init= sixhumpcamel(xgrid)+np.random.randn(xgrid.shape[0],1)*.6
print depth_init.shape

# Simulate  measurements from Stereo depth mapping
# in real experiment, we'll read these in:
# surface_model='image_pairs/exp'
# depth_init = simulated_disparity.getObservationModel(
#    surface_model).astype(float).flatten()
# surf_size=depth_init.shape[0]
# x1, x2 = np.meshgrid(np.linspace(xmin, xmax, surf_size),
#                      np.linspace(ymin, ymax, surf_size))
# xx =np.array([x1,x2])
# xgrid = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T
# depth_init=depth_init.flatten()

#Plot the initial data
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(x1, x2, depth_init.reshape(sample_res,sample_res), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_title("Depth from Disparity")


dat=update_GP(depth_init,xgrid)
mm=dat[2]
