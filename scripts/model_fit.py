import numpy as np
from scipy.optimize import curve_fit
import pylab
import pickle
# from probe_visualizations import single_row_stiffness_map
import matplotlib
import matplotlib.pyplot as plt

def sigmoidfit(dist, alpha=1000, a=0.0, b=1.0, c=-0.004):
    """  
    a, b: base and max readings of the probe with and without tumor
    dist = xProbe-xEdge
    xProbe: 
    xEdge: the center of sigmoid
    alpha: slope  
    Output:
    y = a + (b-a)/(1+ exp(-alpha*(xProbe-xEdge)))
    """
    y = a + np.divide((b-a),(1+ np.exp(-alpha*(dist-c))))  
    return y

def fit_measmodel(xdata,ydata,scale=True):
	#easnorm=ydata[xdata==xdata.max()]
	# print measnorm
	# if scale==True:
	# 	ydata=ydata/measnorm
    popt, pcov = curve_fit(sigmoidfit, xdata, ydata)
    return popt

# def single_row_stiffness_map(probe_data1, probe_data2):
# 	# import IPython; IPython.embed()
# 	index = len(probe_data1)
# 	# combine the data
# 	probe_data1.extend(probe_data2)

# 	probe_data = probe_data1
# 	positions_3d = np.array([np.array(a[1][0:3, 3].flatten())[0] for a in probe_data])
# 	positions_3d -= np.mean(positions_3d,0)
# 	U, s, V = np.linalg.svd(np.transpose(positions_3d), full_matrices = False)
# 	plane_normal = U[:, 2]
	
# 	# project points onto best fit plane
# 	projected_points = np.array([a - np.dot(plane_normal, a)*plane_normal for a in positions_3d])
	
# 	# axis align plane
# 	projected_points = np.transpose(np.dot(np.linalg.inv(U), np.transpose(projected_points)))

# 	# replace z-axis with stiffness
# 	stiffness = np.array([a[0] for a in probe_data])
# 	projected_points[:,2] = stiffness

# 	# display stiffness map
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)
# 	colors1 = ["red" for a in range(index)]
# 	colors2 = ["blue" for a in range(len(probe_data)-index)]
# 	colors1.extend(colors2)

def get_stiffness_data(probe_data):
	""" Displays a 2D scatter plot where x is postion along the x axis of the tissue and z is the probe measurement """
	index = len(probe_data)

	positions_3d = np.array([np.array(a[1][0:3, 3].flatten())[0] for a in probe_data])
	positions_3d -= np.mean(positions_3d,0)
	U, s, V = np.linalg.svd(np.transpose(positions_3d), full_matrices = False)
	plane_normal = U[:, 2]
	
	# project points onto best fit plane
	projected_points = np.array([a - np.dot(plane_normal, a)*plane_normal for a in positions_3d])
	
	# axis align plane
	projected_points = np.transpose(np.dot(np.linalg.inv(U), np.transpose(projected_points)))

	# replace z-axis with stiffness
	stiffness = np.array([a[0] for a in probe_data])
	projected_points[:,2] = stiffness
	xpoints=np.array(projected_points[:,0])
	zpoints=np.array(projected_points[:,2])
	zpoints=zpoints
	xpoints1=xpoints[xpoints>-.020]
	zpoints1=zpoints[xpoints>-.020] 
	return xpoints1[xpoints1<-.002],zpoints1[xpoints1<-.002] #xpoints[xpoints<-.01],zpoints[xpoints<-.01]

def plot_model(probe_data,model,scale=False):
	xp,zp=get_stiffness_data(probe_data)

	measnorm=zp[xp==xp.max()]
	if scale==True:
		zp=zp/measnorm
		xp=xp-model[3]
	# display stiffness map
	fig = plt.figure()
	ax = fig.add_subplot(111)
	matplotlib.rcParams.update({'font.size': 20,'font.family':'times'})

	# ax.set_title("Single Row Raster 100x on Baseline Tissue Phantom")
	ax.set_xlabel("Position (meters)")
	ax.set_ylabel("Probe Measurement")

	# ax.scatter(projected_points[:index,0], projected_points[:index,2], marker='.', edgecolors="none")
	ax.scatter(xp, zp, marker='.', edgecolors="none", color="blue",alpha=.2)
	if scale==True:
		model[3]=0
		model[1:3]=model[1:3]/measnorm
		x = np.linspace(xp.min(), xp.max(), 50)
	else:
		x = np.linspace(xp.min(), xp.max(), 50)

	z = sigmoidfit(x, *model)
	errb=200
	if scale==True:
		z=z#/measnorm
		errb=errb/measnorm
	ax.plot(x, z, color='red',linewidth=4)
	ax.plot(x, z+errb, color='red',alpha=.3)
	ax.plot(x, z-errb, color='red',alpha=.3)
	# ax.set_xlim([-.020,-.002])
	# ax.xaxis.set_ticks(np.arange(-.018,-.002, .006))
	# if scale==True:
	# 	ax.yaxis.set_ticks(np.arange(5000/,8, .5))
	# else:	
	# 	ax.yaxis.set_ticks(np.arange(5000,8000, 500))

	fig.savefig("measmodel.pdf" ,bbox_inches='tight')
	plt.show()
	return model


if __name__ == '__main__':
	data1 = pickle.load(open("saved_palpation_data/single_row_raster_100x.p", "rb"))
	# data2 = pickle.load(open("probe_data_L2R.p", "rb"))
	xdata,zdata=get_stiffness_data(data1)
	model=fit_measmodel(xdata,zdata)
	plot_model(data1,model)