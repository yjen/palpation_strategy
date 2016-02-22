import rospy
import robot
from std_msgs.msg import String, Float64
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import PyKDL
from numpy.linalg import norm
import tfx
import pickle
import matplotlib.pyplot as plt


def stiffness_map(probe_data):
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

def single_row_stiffness_map(probe_data):
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
	ax.scatter(projected_points[:,0], projected_points[:,2], marker='.', edgecolors="none")
	ax.set_title("Single Row Raster 100x on Baseline Tissue Phantom")
	ax.set_xlabel("Position (meters)")
	ax.set_ylabel("Probe Measurement")
	plt.show()

if __name__ == '__main__':
	# load data
	probe_data = pickle.load(open("save_data/single_row_20x_raster.p", "rb"))
	single_row_stiffness_map(probe_data)
