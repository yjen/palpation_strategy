import sys
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib import pyplot as pl

# import matplotlib.path as path
#import GPyOpt
#from utils import *
#from getMap import *
#from sensorModel import *
# import cplex, gurobipy
#import sys
#sys.path.append("..")
#import simulated_disparity
import rospy
# from simUtils import *
from utils import *
from scipy import stats

# def calculate_boundary(filename):
#     data_dict = None
#     try:
#         data_dict = pickle.load(open(filename, "rb"))
#     except Exception as e:
#         print "Exception: ", e
#         rospy.logerror("Error: %s", e)
#         rospy.logerror("Failed to load saved tissue registration.")

#     # compute tissue frame
#     nw = data_dict['nw']
#     ne = data_dict['ne']
#     sw = data_dict['sw']
#     nw_position = np.hstack(np.array(nw.position))
#     ne_position = np.hstack(np.array(ne.position))
#     sw_position = np.hstack(np.array(sw.position))
#     u = sw_position - nw_position
#     v = ne_position - nw_position
#     tissue_length = np.linalg.norm(u)
#     tissue_width = np.linalg.norm(v)
#     return ((0, tissue_length), (0, tissue_width))

# note we are manually defining the tissue size
def calculate_boundary(filename):
    tissue_length = 0.025
    tissue_width = 0.05
    return ((0, tissue_length), (0, tissue_width))


def getExperimentalStereoMeas(surface, workspace, plot = True):
    """
    needs to be written
    """


    # z needs to be read from robot
    # should return: np.array([xx, yy,
    #                 z]).T
    pass

def getExperimentalStiffnessMeas(sample_points,surface=None,noiselev=None):
    """
    needs to be written
    """
    print(len(sample_points))
    import rospy
    from palpation_strategy.msg import Points, FloatList
    rospy.init_node('gaussian_process', anonymous=True)
    
    global flagSTOPSPINNING
    flagSTOPSPINNING = False
    global measurementsNOC
    measurementsNOC = None


    def probe_callback(data):
        global flagSTOPSPINNING
        global measurementsNOC
        print('hi')
        print(data)
        measurementsNOC = data.data
        
        flagSTOPSPINNING = True
        print("CALLBACK: " + str(flagSTOPSPINNING))

    rospy.Subscriber("/palpation/measurements", FloatList, probe_callback)
    pts_publisher = rospy.Publisher("/gaussian_process/pts_to_probe", Points)
    x = sample_points[:, 0]
    y = sample_points[:, 1]

    """CHANGE THIS"""

    x = [0.001, 0.005, 0.02, 0.005, 0.02]
    y = [0.005, 0.025, 0.01, 0.03, 0.04]

    x2 = []
    y2 = []

    distance = 0.005
    for i in range(len(x)-1):
        x2.append(x[i])
        y2.append(y[i])
        dist = np.linalg.norm((x[i] - x[i+1], y[i] - y[i+1]))
        numInterpolatedPoints = int(dist/distance)-1
        deltaX = (x[i+1]-x[i])/(numInterpolatedPoints+1)
        deltaY = (y[i+1]-y[i])/(numInterpolatedPoints+1)
        for j in range(numInterpolatedPoints):
            x2.append(x[i] + (j+1)*deltaX)
            y2.append(y[i] + (j+1)*deltaY)
    x2.append(x[-1])
    y2.append(y[-1])
    x = np.array(x2)
    y = np.array(y2)


    p = Points()
    p.x, p.y = x, y
    rospy.sleep(0.2)
    pts_publisher.publish(p)
    # import IPython; IPython.embed()
    # print("bro")


    while not flagSTOPSPINNING:
        # print("spin")
        # print("WHILE: " + str(flagSTOPSPINNING))
        rospy.sleep(0.1)
    print('done')
    stiffness = []
    measurementsNOC = np.array(measurementsNOC)/1000.0
    for i in range(len(measurementsNOC)):
        stiffness.append([x[i], y[i], measurementsNOC[i]])
    return np.array(stiffness)


    # z needs to be read from robot
    # should return: np.array([xx, yy,
    #                 z]).T


# def getRecordedExperimentalStiffnessMeas(sample_points,surface=None):
#     filename = '../scripts/dense_grid.p'
#     data_dict = pickle.load(open(filename, "rb"))
#     data = np.array(data_dict['data'])

#     data = np.array(data_dict['data'])

#     x, y, z = data[:,0], data[:,1], data[:,2]

#     from scipy.ndimage.filters import gaussian_filter
#     z = gaussian_filter(z.reshape(21,41), sigma=1)
#     z = z.reshape((21*41,))

#     from scipy.interpolate import Rbf
#     rbfi = Rbf(x, y, z)

#     stiffnesses = np.array([rbfi(a[0], a[1]) for a in sample_points])

#     output = np.zeros((len(sample_points), 3))
#     output[:,:2] = sample_points
#     output[:,2] = stiffnesses
#     return output
