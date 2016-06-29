import sys
sys.path.append("../src")
# import rospy
# import robot
# from std_msgs.msg import String, Float64
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# import PyKDL
from numpy.linalg import norm
import tfx
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
# import GaussianProcess
# from utils import *


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
    ax.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2], c='r', marker='.')
    plt.show()

def stiffness_map_combined(probe_data1, probe_data2):
    """ Displays a 3D scatter plot where x,y are points on the tissue surface and z is the probe measurement """

    probe_data1.extend(probe_data2)
    probe_data = probe_data1

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

    colors1 = ["red" for _ in range(len(probe_data1)-len(probe_data2))]
    colors2 = ["blue" for _ in range(len(probe_data2))]
    colors1.extend(colors2)
    print(colors1[0])
    print(colors1[-1])
    print(len(colors1))
    print(len(projected_points))

    # display stiffness map
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2], c=colors1, marker='o', edgecolors=colors1)
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

    m=GaussianProcess.update_GP(projected_points)   # display stiffness map
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


def single_row_stiffness_map(probe_data1, probe_data2):
    # import IPython; IPython.embed()
    index = len(probe_data1)
    # combine the data
    probe_data1.extend(probe_data2)

    probe_data = probe_data1
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
    colors1 = ["red" for a in range(index)]
    colors2 = ["blue" for a in range(len(probe_data)-index)]
    colors1.extend(colors2)



    # ax.scatter(projected_points[:index,0], projected_points[:index,2], marker='.', edgecolors="none")
    ax.scatter(projected_points[:,0], projected_points[:,2], marker='.', edgecolors="none", color=colors1)

    ax.set_title("Single Row Raster Both Directions on Baseline Tissue Phantom")
    ax.set_xlabel("Position (meters)")
    ax.set_ylabel("Probe Measurement")
    plt.show()

def single_row_stiffness_map(probe_data1, probe_data2, probe_data3):
    # splice probe data 3
    l = len(probe_data3)
    probe_data3 = probe_data3[3*l/10: 7*l/20]


    # import IPython; IPython.embed()
    index = len(probe_data1)
    index2 = len(probe_data2) + index
    # combine the data
    probe_data1.extend(probe_data2)
    probe_data1.extend(probe_data3)

    probe_data = probe_data1
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
    colors1 = ["red" for a in range(index)]
    colors2 = ["blue" for a in range(index2-index)]
    colors3 = ["green" for a in range(len(probe_data)-index2)]
    colors1.extend(colors2)
    colors1.extend(colors3)

    # ax.scatter(projected_points[:index,0], projected_points[:index,2], marker='.', edgecolors="none")
    ax.scatter(projected_points[:,0], projected_points[:,2], marker="o", edgecolors="none", color=colors1)

    ax.set_title("Single Row Raster Both Directions on Baseline Tissue Phantom")
    ax.set_xlabel("Position (meters)")
    ax.set_ylabel("Probe Measurement")
    plt.show()

def plot_rotations(probe_data):
    data = [tfx.pose(i[1]).tb_angles for i in probe_data]
    roll = [d.roll_deg for d in data]
    pitch = [d.pitch_deg for d in data]
    yaw = [d.yaw_deg for d in data]
    print("roll: max, min: " + str(max(roll)) + ", " + str(min(roll)))
    print("pitch: max, min: " + str(max(pitch)) + ", " + str(min(pitch)))
    print("yaw: max, min: " + str(max(yaw)) + ", " + str(min(yaw)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(roll, pitch, yaw, marker='o')
    plt.show()

def plot_rotations_combined(probe_data1, probe_data2):
    probe_data1.extend(probe_data2)
    probe_data = probe_data1

    colors1 = ["red" for _ in range(len(probe_data1)-len(probe_data2))]
    colors2 = ["blue" for _ in range(len(probe_data2))]
    colors1.extend(colors2)

    data = [tfx.pose(i[1]).tb_angles for i in probe_data]
    roll = [d.roll_deg for d in data]
    pitch = [d.pitch_deg for d in data]
    yaw = [d.yaw_deg for d in data]
    print("roll: max, min: " + str(max(roll)) + ", " + str(min(roll)))
    print("pitch: max, min: " + str(max(pitch)) + ", " + str(min(pitch)))
    print("yaw: max, min: " + str(max(yaw)) + ", " + str(min(yaw)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(roll, pitch, yaw, c=colors1, marker='o', edgecolors=colors1)
    plt.show()


def plot_interpolated_recorded_data(data_dict):
    data = np.array(data_dict['data'])

    x, y, z = data[:,0], data[:,1], data[:,2]

    from scipy.ndimage.filters import gaussian_filter
    z = gaussian_filter(z.reshape(21,41), sigma=1)
    z = z.reshape((21*41,))

    from scipy.interpolate import Rbf
    rbfi = Rbf(x, y, z)

    x_new = np.arange(np.min(x), np.max(x), 0.0005)
    y_new = np.arange(np.min(y), np.max(y), 0.0005)
    xx, yy = np.meshgrid(x_new, y_new)

    zz = rbfi(xx, yy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx, yy, zz, c='b', marker='o')
    plt.show()

def gen_figures(file_name):
    """ Displays a 3D scatter plot where x,y are points on the tissue surface and z is the probe measurement """

    data_dict = pickle.load(open(file_name+".p", "rb"))
    probe_data = data_dict[1]

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

    # save forward heat map

    fig = plt.figure()
    plt.scatter(projected_points[20:,0], projected_points[20:,1], c=projected_points[20:,2], marker='o', edgecolors='none', s=40, alpha=0.05)
    plt.xlim(-0.04, 0.04)
    plt.ylim(-0.04, 0.04)
    plt.savefig(file_name+"forward.png")
    plt.show()

    probe_data = data_dict[2]

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


    # save backward heat map
    fig = plt.figure()
    plt.scatter(projected_points[:,0], projected_points[:,1], c=projected_points[:,2], marker='o', edgecolors='none', s=40, alpha=0.1)
    plt.xlim(-0.04, 0.04)
    plt.ylim(-0.04, 0.04)
    plt.savefig(file_name+"backward.png")
    plt.show()


def gen_figures_single_row(file_name):
    data_dict = pickle.load(open(file_name+".p", "rb"))
    probe_data1 = data_dict[1]
    probe_data2 = data_dict[2]

    index = len(probe_data1)
    # combine the data
    probe_data1

    probe_data = probe_data1
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

    # ax.scatter(projected_points[:index,0], projected_points[:index,2], marker='.', edgecolors="none")
    ax.scatter(projected_points[:,0], projected_points[:,2], marker='.', edgecolors="none")
    plt.ylim(0, 3500)
    plt.savefig(file_name+"single_row.png")

if __name__ == '__main__':
    gen_figures("exp_data/no_skin_circle_raster")
