import rospy
import robot
from std_msgs.msg import String
import numpy as np
import PyKDL
from numpy.linalg import norm
import tfx
import pickle
from palpation import Palpation

from probe_visualizations import stiffness_map

def test_raster_scan():
    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")

    raw_input("Will now execute raster test. Press any key to continue")
    palp.execute_raster()


def test_raster_single_row():
    """ Rasters a single row on the tissue 100 times """

    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")

    raw_input("Will now execute raster single row test. Press any key to continue")
    palp.execute_raster_single_row(100, False)

def test_raster_single_row_reverse():
    """ Rasters a single row on the tissue 30 times in the reverse direction"""

    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")

    raw_input("Will now execute raster single row reverse test. Press any key to continue")
    palp.execute_raster_single_row_reverse(30, False)

def test_point_probe_grid():
    """ Rasters a single row on the tissue 30 times in the reverse direction"""

    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")

    raw_input("Will now execute point probe grid test. Press any key to continue")
    palp.execute_point_probe_grid(20, 40, 1)

def test_point_probe():
    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")
    # palp.pick_up_tool()
    raw_input("Will now listen to runPhase2.py. Press Enter to continue")
    rospy.spin()    

if __name__ == '__main__':
    test_point_probe()