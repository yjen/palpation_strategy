import rospy
import robot
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Point
from palpation_strategy.msg import Points, FloatList
import numpy as np
import PyKDL
from numpy.linalg import norm
import tfx
import pickle
from probe_visualizations import stiffness_map
from tf_conversions import posemath
import json
import os
import sys
import signal

# define a signal handler to exit cleanly
def signal_handler(signal, frame):
    print "Exiting palpation script"
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class Palpation():
    def __init__(self):
        self.data_dict = None
        self.psm1 = robot.robot("PSM1")
        self.psm2 = robot.robot("PSM2")
        self.tissue_pose = None
        self.tissue_length = None
        self.tissue_width = None
        self.probe_offset = 0.030 #32 gives interesting readings? og 0.038
        # probe offset and pitch and speed should automatically be in filename
        self.probe_data = []
        self.probe_data_desired_pose = []
        self.record_data = False
        self.speed = 0.02
        self.curr_probe_value = None
        self.insert_probe_angle = 10
        self.lock_probe_angle = 36

        self.record_point_palpation_data = False
        self.locations = []
        self.baseline = float("-inf")
        self.num_points = 0
        self.cooldown = 0

        # subscribe to probe data
        rospy.Subscriber("/probe/measurement", Float64, self.probe_callback)
        rospy.Subscriber("/gaussian_process/pts_to_probe", Points, self.probe_points_callback)
        self.measurements_pub = rospy.Publisher("/palpation/measurements", FloatList, queue_size=1)



    ##################################################################################
    # PROBE METHODS
    ##################################################################################
    def probe_callback(self, msg):
        self.curr_probe_value = msg.data

        if self.record_point_palpation_data:
            if self.cooldown == 0:
                if self.baseline == float("-inf"):
                    self.baseline = msg.data
                elif msg.data - self.baseline > 100:
                    self.locations.append(self.psm1.get_current_cartesian_position().matrix)
                    self.num_points += 1
                    print("location recorded! points recorded: " + str(self.num_points))
                    self.cooldown = 100
                else:
                    self.baseline = 0.5*self.baseline + 0.5*msg.data
            else:
                self.cooldown -= 1

        if self.record_data:
            self.probe_data.append([msg.data, self.psm1.get_current_cartesian_position().matrix])
            self.probe_data_desired_pose.append([msg.data, self.psm1.get_desired_cartesian_position().matrix])

    def probe_start(self):
        self.record_data = True

    def probe_pause(self):
        self.record_data = False

    def probe_stop_reset(self):
        self.record_data = False
        self.probe_data = []
        self.probe_data_desired_pose = []

    def probe_single_point_record(self):
        self.probe_data.append([self.curr_probe_value, self.psm1.get_current_cartesian_position().matrix])
        self.probe_data_desired_pose.append([self.curr_probe_value, self.psm1.get_desired_cartesian_position().matrix])

    def probe_save(self, filename):
        try:
            pickle.dump(self.probe_data, open(filename, "wb"))
            pickle.dump(self.probe_data, open(filename + "desired.p", "wb"))
        except Exception as e:
            print "Exception: ", e
            rospy.logwarn('Failed to save probe data')

    def probe_save_locations(self, filename):
        try:
            pickle.dump(self.locations, open(filename, "wb"))
        except Exception as e:
            print("Exception: " + str(e))
            rospy.logwarn('Failed to save probe data')


    ##################################################################################
    # UTIL METHODS
    ##################################################################################
    def register_environment(self, filename):
        """ Saves the pose of important objects in the environment.
            This function asks the user to teleop to relevant points in the
            environment and records the pose of these points.
        """
        self.data_dict = {}  # erase old environment state info

        # record tissue location
        raw_input('Using PSM, teleop to the NW point on the tissue surface and hit any key to record: ')
        self.data_dict['nw'] = self.psm1.get_current_cartesian_position()

        raw_input('Using PSM, teleop to the NE point on the tissue surface and hit any key to record: ')
        self.data_dict['ne'] = self.psm1.get_current_cartesian_position()

        raw_input('Using PSM, teleop to the SW point on the tissue surface and hit any key to record: ')
        self.data_dict['sw'] = self.psm1.get_current_cartesian_position()

        raw_input('Using PSM, teleop to the SE point on the tissue surface and hit any key to record: ')
        self.data_dict['se'] = self.psm1.get_current_cartesian_position()

        # record tool locations
        raw_input('Using PSM, teleop to the picking location of the PALPATION tool and hit any key to record: ')
        self.data_dict['palpation_tool_pick_up_loc'] = self.psm1.get_current_cartesian_position()

        # compute tissue frame
        nw = self.data_dict['nw']
        ne = self.data_dict['ne']
        sw = self.data_dict['sw']

        # save registration to file
        try:
            pickle.dump(self.data_dict, open(filename, "wb"))
        except Exception as e:
            print "Exception: ", e
            rospy.logwarn('Failed to save registration')

        self.tissue_pose = self.compute_tissue_pose(nw,ne,sw)

    def load_environment_registration(self, filename):
        """ Loads a previously recorded environment state. """
        self.data_dict = None
        try:
            self.data_dict = pickle.load(open(filename, "rb"))
        except Exception as e:
            print "Exception: ", e
            rospy.logerror("Error: %s", e)
            rospy.logerror("Failed to load saved tissue registration.")

        # compute tissue frame
        nw = self.data_dict['nw']
        ne = self.data_dict['ne']
        sw = self.data_dict['sw']
        return self.compute_tissue_pose(nw,ne,sw)

    def load_tissue_pose_from_registration_brick_pose(self):
        # tissue pose offsets:
        x = -0.048
        y = 0.07
        z = 0.006
        rotation = 0.0

        tissue_frame = pickle.load(open("registration_brick_pose.p", "rb"))
        tissue_frame = tissue_frame.as_tf()*tfx.transform([x,y,z])*tfx.transform(tfx.rotation_tb(rotation, 0, 0))
        self.tissue_pose = tfx.pose(tissue_frame)
        self.tissue_width = 0.056
        self.tissue_length = 0.028

    def compute_tissue_pose(self, nw, ne, sw):
        nw_position = np.hstack(np.array(nw.position))
        ne_position = np.hstack(np.array(ne.position))
        sw_position = np.hstack(np.array(sw.position))
        u = sw_position - nw_position
        v = ne_position - nw_position
        self.tissue_length = norm(u)
        self.tissue_width = norm(v)
        u /= self.tissue_length
        v /= self.tissue_width
        origin = nw_position  # origin of palpation board is the NW corner
        w = np.cross(u, v)
        u = np.cross(v, w)  # to ensure that all axes are orthogonal
        rotation_matrix = np.array([u,v,w]).transpose()
        pose = tfx.pose(origin, rotation_matrix, frame=nw.frame)
        self.tissue_pose = pose
        return pose


    def pick_up_tool(self):
        tissue_pose = self.tissue_pose

        # get gripper orientation
        frame = np.array(tissue_pose.orientation.matrix)
        u, v, w = frame.T[0], frame.T[1], frame.T[2]
        rotation_matrix = np.array([v, u, -w]).transpose()

        # get gripper position
        origin = np.hstack(np.array(tissue_pose.position))
        offset = np.dot(frame, np.array([0.02, 0.0, 0.03]))

        pose = tfx.pose(origin + offset, rotation_matrix, frame=tissue_pose.frame)


        # send arm to appropriate pose
        self.psm1.move_cartesian_frame_linear_interpolation(pose, self.speed, False)
        self.psm1.set_gripper_angle(10)

        raw_input("Place tool on gripper and press any key to continue: ")
        self.psm1.set_gripper_angle(60)

    def drop_off_tool(self):
        tissue_pose = self.tissue_pose

        # get gripper orientation
        frame = np.array(tissue_pose.orientation.matrix)
        u, v, w = frame.T[0], frame.T[1], frame.T[2]
        rotation_matrix = np.array([v, u, -w]).transpose()

        # get gripper position
        origin = np.hstack(np.array(tissue_pose.position))
        offset = np.dot(frame, np.array([0.02, 0.09, 0.06]))
        pose = tfx.pose(origin + offset, rotation_matrix, frame=tissue_pose.frame)


        # send arm to appropriate pose
        # self.psm1.close_gripper()
        self.psm1.set_gripper_angle(10)
        rospy.sleep(2.0)
        self.psm1.move_cartesian_frame_linear_interpolation(pose, self.speed, False)



    ##################################################################################
    # PROBING METHODS
    ##################################################################################
    def deunicodify_hook(self, pairs):
        new_pairs = []
        for key, value in pairs:
            if isinstance(value, unicode):
                value = value.encode('utf-8')
            if isinstance(key, unicode):
                key = key.encode('utf-8')
            new_pairs.append((key, value))
        return dict(new_pairs)

    def execute_raster(self, config_file):
        # load exp config
        config = json.load(open(config_file), object_pairs_hook=self.deunicodify_hook)

        # create directory to save the data
        directory = "exp_data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # transform scan plane
        w, l = config["tissue_dimensions"]
        tissue_pose = self.load_environment_registration(config["surface_registration_file"])
        roll, pitch, yaw = config['rotation_offset']
        tissue_pose = tissue_pose.as_tf()*tfx.transform(config["position_offset"])*tfx.transform([w/2.0, l/2.0, 0.0])*tfx.transform(tfx.rotation_tb(0, 0, roll))*tfx.transform(tfx.rotation_tb(0, pitch, 0))*tfx.transform(tfx.rotation_tb(yaw, 0, 0))*tfx.transform([w/-2.0,  l/-2.0, 0.0])
        self.tissue_pose = tissue_pose

        def go_to_pose(x, y, z, speed):
            # construct tool rotation
            origin = np.hstack(np.array(self.tissue_pose.position))
            frame = np.array(tissue_pose.orientation.matrix)
            u, v, w = frame.T[0], frame.T[1], frame.T[2]
            rotation_matrix = np.array([v, u, -w]).transpose()

            offset = np.dot(frame, np.array([x, y, z + config['probe_depth']]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, speed, False)

        # pick up tool if necessary
        if config["pick_up_tool"]:
            self.pick_up_tool()

        # move to start point
        go_to_pose(config["x_start"]*w, 0.0, 0.02, self.speed)
        w, l = config["tissue_dimensions"]

        # scan rows
        self.probe_stop_reset()
        forward_probe_data = []
        backward_probe_data = []
        for i in range(config["number_rows"]):
            # are we scanning multiple rows?
            if config["number_rows"] > 1:
                step = 1.0/(config["number_rows"]-1)
            else:
                step = 0.0

            y_start = config['y_start']*l
            y_end = config['y_end']*l

            for _ in range(config["scans_per_row"]):
                x = (config["x_start"]*(1.0-i*step) + i*step*config["x_end"])*w
                
                # scan in forward direction
                go_to_pose(x, y_start, 0.0, config["raster_speed"])
                rospy.sleep(0.5)
                self.probe_start()
                go_to_pose(x, y_end, 0.0, config["raster_speed"])
                rospy.sleep(0.5)
                self.probe_pause()

                # add probe data to forward list
                forward_probe_data.extend(self.probe_data[:])
                self.probe_stop_reset()

                # scan in reverse direction
                self.probe_start()
                # go_to_pose(x, y_end, 0.01, config["raster_speed"])
                # go_to_pose(x, y_start, 0.01, config["raster_speed"])
                go_to_pose(x, y_start, 0.00, config["raster_speed"])
                rospy.sleep(0.5)
                self.probe_pause()

                # add probe data to backward list
                backward_probe_data.extend(self.probe_data[:])
                self.probe_stop_reset()

                # save recorded data to file after each scan
                data = [config, forward_probe_data, backward_probe_data]
                filename = directory+"/"+config["exp_name"]+".p"
                pickle.dump(data, open(filename, "wb"))


    def execute_raster_random(self, config_file):
        # load exp config
        config = json.load(open(config_file), object_pairs_hook=self.deunicodify_hook)

        # create directory to save the data
        directory = "exp_data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # transform scan plane
        tissue_pose = self.load_environment_registration(config["surface_registration_file"])
        roll, pitch, yaw = config['rotation_offset']
        tissue_pose = tissue_pose.as_tf()*tfx.transform(config["position_offset"])*tfx.transform(tfx.rotation_tb(0, 0, roll))*tfx.transform(tfx.rotation_tb(0, pitch, 0))*tfx.transform(tfx.rotation_tb(yaw, 0, 0))
        self.tissue_pose = tissue_pose

        def go_to_pose(x, y, z, speed):
            # construct tool rotation
            origin = np.hstack(np.array(tissue_pose.position))
            frame = np.array(tissue_pose.orientation.matrix)
            u, v, w = frame.T[0], frame.T[1], frame.T[2]
            rotation_matrix = np.array([v, u, -w]).transpose()

            offset = np.dot(frame, np.array([x, y, z + config['probe_depth']]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, speed, False)

        # pick up tool if necessary
        if config["pick_up_tool"]:
            self.pick_up_tool()

        # move to start point
        go_to_pose(config["x_start"], 0.0, 0.02, self.speed)
        go_to_pose(config["x_start"], 0.0, 0.00, self.speed)
        w, l = config["tissue_dimensions"]

        self.probe_stop_reset()
        self.probe_start()
        # move to random points on surface
        for _ in range(200):
            x = np.random.uniform(0.0, w)
            y = np.random.uniform(0.0, l)
            go_to_pose(x, y, 0.0, config["raster_speed"])
            rospy.sleep(0.2)

        data = [config, self.probe_data[:]]
        filename = directory+"/"+config["exp_name"]+".p"
        pickle.dump(data, open(filename, "wb"))

    def test_Points_publisher(self):
        publisher = rospy.Publisher("/palpation/probe_points", Points, queue_size=1)
        rospy.sleep(1.0)
        points = Points([0.1,0.2,0.3],[0.4,0.5, 0.6])
        publisher.publish(points)


    def test_callback(self, data):
        print "Received Points"
        print data

    def probe_points_callback(self, data):
        """ Listens for points to probe. Publishes probe measurements at each point """
        # load exp config
        config_file = "exp_config.json"
        config = json.load(open(config_file), object_pairs_hook=self.deunicodify_hook)

        # create directory to save the data
        directory = "exp_data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # transform scan plane
        tissue_pose = self.load_environment_registration(config["surface_registration_file"])
        roll, pitch, yaw = config['rotation_offset']
        tissue_pose = tissue_pose.as_tf()*tfx.transform(config["position_offset"])*tfx.transform(tfx.rotation_tb(0, 0, roll))*tfx.transform(tfx.rotation_tb(0, pitch, 0))*tfx.transform(tfx.rotation_tb(yaw, 0, 0))
        self.tissue_pose = tissue_pose

        def go_to_pose(x, y, z, speed):
            # construct tool rotation
            origin = np.hstack(np.array(tissue_pose.position))
            frame = np.array(tissue_pose.orientation.matrix)
            u, v, w = frame.T[0], frame.T[1], frame.T[2]
            rotation_matrix = np.array([v, u, -w]).transpose()

            offset = np.dot(frame, np.array([x, y, z + config['probe_depth']]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, speed, False)

        xs = data.x
        ys = data.y
        probe_measurements = []
        for i in range(len(data.x)):
            go_to_pose(xs[i], ys[i], 0.0, config["raster_speed"])
            rospy.sleep(0.1)
            probe_measurements.append(self.curr_probe_value)

        self.measurements_pub.publish(FloatList(probe_measurements))




    def load_config(self, config_file):
        self.config = json.load(open(config_file), object_pairs_hook=self.deunicodify_hook)
        self.speed = self.config["raster_speed"]

if __name__ == '__main__':
    palp = Palpation()
    # palp.execute_raster("exp_config.json")
    rospy.spin()
