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
from simUtils import *
from utils import *
from scipy import stats

def calculate_boundary(filename):
    data_dict = None
    try:
        data_dict = pickle.load(open(filename, "rb"))
    except Exception as e:
        print "Exception: ", e
        rospy.logerror("Error: %s", e)
        rospy.logerror("Failed to load saved tissue registration.")

    # compute tissue frame
    nw = data_dict['nw']
    ne = data_dict['ne']
    sw = data_dict['sw']
    nw_position = np.hstack(np.array(nw.position))
    ne_position = np.hstack(np.array(ne.position))
    sw_position = np.hstack(np.array(sw.position))
    u = sw_position - nw_position
    v = ne_position - nw_position
    tissue_length = np.linalg.norm(u)
    tissue_width = np.linalg.norm(v)
    return ((0, tissue_length), (0, tissue_width))


def getExperimentalStereoMeas(surface, workspace, plot = True):
    """
    needs to be written
    """


    # z needs to be read from robot
    # should return: np.array([xx, yy,
    #                 z]).T
    pass

def getExperimentalStiffnessMeas(sample_points):
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
    p = Points()
    p.x, p.y = x, y
    rospy.sleep(0.2)
    pts_publisher.publish(p)
    # import IPython; IPython.embed()
    # print("bro")


    while not flagSTOPSPINNING:
        print("spin")
        print("WHILE: " + str(flagSTOPSPINNING))
        rospy.sleep(0.1)
    print('done')
    stiffness = []
    measurementsNOC/1000.0
    for i in range(len(measurementsNOC)):
        stiffness.append([x[i], y[i], measurementsNOC[i]])
    return np.array(stiffness).T


    # z needs to be read from robot
    # should return: np.array([xx, yy,
    #                 z]).T