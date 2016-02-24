import sys
sys.path.append("../src")
# import rospy
# import robot
# from std_msgs.msg import String, Float64
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# import PyKDL
from numpy.linalg import norm
# import tfx
import pickle
import matplotlib.pyplot as plt
import GaussianProcess
from utils import *


def stiffness_map(probe_data):
	""" Displays a 3D scatter plot where x,y are points on the tissue surface and z is the probe measurement """

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

	# display stiffness map
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2], c='r', marker='o')
	plt.show()

def GPstiffness(probe_data):
	""" Displays a 3D scatter plot where x,y are points on the tissue surface and z is the probe measurement """

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

        m=GaussianProcess.update_GP(projected_points)	# display stiffness map
        # set workspace boundary
        bounds = ((-.03,.03),(-.008,.008))

        # grid resolution: should be same for plots, ergodic stuff
        gridres = 200

        # initialize workspace object
        workspace = Workspace(bounds,gridres)
        mean, sigma = get_moments(m, workspace.x)
        m.plot_data()
        m.plot()

        GaussianProcess.plot_belief(mean,sigma,workspace)
	
        
def single_row_stiffness_map(probe_data):
	""" Displays a 2D scatter plot where x is postion along the x axis of the tissue and z is the probe measurement """
	
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

	# display stiffness map
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(projected_points[:,0], projected_points[:,2], marker='.', edgecolors="none",alpha=.25)
	ax.set_title("Single Row Raster 100x on Baseline Tissue Phantom")
	ax.set_xlabel("Position (meters)")
	ax.set_ylabel("Probe Measurement")
	plt.show()

if __name__ == '__main__':
	# load data
        fname="saved_palpation_data/point_probe_20x10x1.p"
        fname="saved_palpation_data/point_probe_40x20x1.p"
        # fname="saved_palpation_data/single_row_raster_100x.p"
        # fname="saved_palpation_data/single_row_raster_reverse_30x.p"
        #fname="probe_data.p"
	probe_data = pickle.load(open(fname, "rb"))
	single_row_stiffness_map(probe_data)
	GPstiffness(probe_data)