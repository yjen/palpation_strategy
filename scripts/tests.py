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

def test_raster_scan_tilted_L2R():
    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")

    raw_input("Will now execute raster test. Press any key to continue")
    palp.execute_raster_tilted(0, 1)

def test_raster_scan_tilted_R2L():
    opepalp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")

    raw_input("Will now execute raster test. Press any key to continue")
    palp.execute_raster_tilted(5, -1)


def test_raster_scan():
    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")

    raw_input("Will now execute raster test. Press any key to continue")
    palp.execute_raster()

def test_raster_scan_both_directions():
    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")

    raw_input("Will now execute raster test in both directions. Press any key to continue")
    palp.execute_raster_both_directions()


def test_raster_scan_reverse():
    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")

    raw_input("Will now execute raster test. Press any key to continue")
    palp.execute_raster_reverse()


def test_raster_single_row():
    """ Rasters a single row on the tissue 100 times """

    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")

    raw_input("Will now execute raster single row test. Press any key to continue")
    palp.execute_raster_single_row(30, False)

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
    palp.execute_point_probe_grid(10, 20, 1)

def test_point_probe():
    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")
    # palp.pick_up_tool()

    raw_input("Will now listen to runPhase2.py. Press Enter to continue")
    rospy.spin()    


def test_register_surface():
    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exeption as e:
        palp.register_environment("env_registration.p")
    raw_input("Will now register surface. Press enter to continue")
    palp.register_surface(5)

def test_scan_random_points():
    """Not done"""
    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exeption as e:
        palp.register_environment("env_registration.p")
    raw_input("Will now register surface. Press enter to continue")
    palp.execute_scan_points_continuous(8)

def test_register_environment():
    palp = Palpation()
    palp.register_environment("env_registration_shifted.p")

def execute_record_testing_grid_data():
    palp = Palpation()
    try:
        palp.load_environment_registration("env_registration.p")
    except Exception as e:
        palp.register_environment("env_registration.p")

    raw_input("Will now execute point probe grid test. Press any key to continue")
    palp.execute_record_testing_grid_data(20, 40, 1)


if __name__ == '__main__':
    # test_raster_scan()
    # test_raster_scan_both_directions()
    test_point_probe()