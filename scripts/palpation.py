import rospy
import robot
from std_msgs.msg import String, Float64
import numpy as np
import PyKDL
from numpy.linalg import norm
import tfx
import pickle
from probe_visualizations import stiffness_map


class Palpation():
    def __init__(self):
        self.data_dict = None
        self.psm1 = robot.robot("PSM1")
        self.psm2 = robot.robot("PSM2")
        self.tissue_pose = None
        self.tissue_length = None
        self.tissue_width = None
        self.probe_offset = 0.034 #32 gives interesting readings? og 0.038
        self.probe_data = []
        self.record_data = False
        self.speed = 0.02
        self.curr_probe_value = None

        # subscribe to probe data
        rospy.Subscriber("/probe/measurement", Float64, self.probe_callback)


    ##################################################################################
    # PROBE METHODS
    ##################################################################################
    def probe_callback(self, msg):
        self.curr_probe_value = msg.data

        if self.record_data:
            self.probe_data.append([msg.data, self.psm1.get_current_cartesian_position().matrix])

    def probe_start(self):
        self.record_data = True

    def probe_pause(self):
        self.record_data = False

    def probe_stop_reset(self):
        self.record_data = False
        self.probe_data = []

    def probe_single_point_record(self):
        self.probe_data.append([self.curr_probe_value, self.psm1.get_current_cartesian_position().matrix])

    def probe_save(self, filename):
        try:
            pickle.dump(self.probe_data, open(filename, "wb"))
        except Exception as e:
            print "Exception: ", e
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
        raw_input('Using PSM, teleop to the NW point on the tissue \
                   surface and hit any key to record: ')
        self.data_dict['nw'] = self.psm1.get_current_cartesian_position()

        raw_input('Using PSM, teleop to the NE point on the tissue \
                   surface and hit any key to record: ')
        self.data_dict['ne'] = self.psm1.get_current_cartesian_position()

        raw_input('Using PSM, teleop to the SW point on the tissue \
                   surface and hit any key to record: ')
        self.data_dict['sw'] = self.psm1.get_current_cartesian_position()

        raw_input('Using PSM, teleop to the SE point on the tissue \
                   surface and hit any key to record: ')
        self.data_dict['se'] = self.psm1.get_current_cartesian_position()

        # record tool locations
        raw_input('Using PSM, teleop to the picking location of the \
                   PALPATION tool and hit any key to record: ')
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
        self.compute_tissue_pose(nw,ne,sw)

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
        offset = np.dot(frame, np.array([0.02, 0.09, 0.06]))
        pose = tfx.pose(origin + offset, rotation_matrix, frame=tissue_pose.frame)


        # send arm to appropriate pose
        
        self.psm1.move_cartesian_frame_linear_interpolation(pose, self.speed, False)
        self.psm1.close_gripper()
        rospy.sleep(1.0)

        raw_input("Place tool on gripper and press any key to continue: ")
        self.psm1.set_gripper_angle(70.0)
        rospy.sleep(1.0)

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
        self.psm1.close_gripper()
        rospy.sleep(2.0)
        self.psm1.move_cartesian_frame_linear_interpolation(pose, self.speed, False)
        


    ##################################################################################
    # PROBING METHODS
    ##################################################################################
    def execute_raster(self):
        """ Linearly interpolates through a series of palpation points """

        steps = 12
        poses = []

        origin = np.hstack(np.array(self.tissue_pose.position))
        frame = np.array(self.tissue_pose.orientation.matrix)

        u, v, w = frame.T[0], frame.T[1], frame.T[2]

        rotation_matrix = np.array([v, u, -w]).transpose()

        dy = self.tissue_length/(steps)
        z = self.probe_offset

        # pick up tool
        # self.pick_up_tool()

        self.probe_stop_reset()
        for i in range(steps-3):
            if i == 0:
                continue
            offset = np.dot(frame, np.array([i*dy, 0.0, z+0.02]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            offset = np.dot(frame, np.array([i*dy, 0.0, z]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            # start recording data
            rospy.sleep(0.2)
            self.probe_start()

            offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            # pause recording data
            rospy.sleep(0.2)
            self.probe_pause()

            offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z+0.02]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)


        self.probe_save("probe_data.p")
        # self.drop_off_tool()

    def execute_raster_rotated(self):
        """ Linearly interpolates through a series of palpation points """

        steps = 12
        poses = []

        origin = np.hstack(np.array(self.tissue_pose.position))
        frame = np.array(self.tissue_pose.orientation.matrix)

        u, v, w = frame.T[0], frame.T[1], frame.T[2]

        rotation_matrix = np.array([v, u, -w]).transpose()

        dx = self.tissue_width/(steps)
        z = self.probe_offset

        # pick up tool
        # self.pick_up_tool()

        self.probe_stop_reset()
        for i in range(steps-3):
            if i == 0:
                continue
            offset = np.dot(frame, np.array([0.0, i*dx, z+0.02]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            offset = np.dot(frame, np.array([0.0, i*dx, z]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            # start recording data
            rospy.sleep(0.2)
            self.probe_start()

            offset = np.dot(frame, np.array([self.tissue_length*0.95, i*dx, z]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            # pause recording data
            rospy.sleep(0.2)
            self.probe_pause()

            offset = np.dot(frame, np.array([self.tissue_length*0.95, i*dx, z+0.02]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)


        self.probe_save("probe_data.p")
        # self.drop_off_tool()

    def execute_raster_both_directions(self):
        """ Linearly interpolates through a series of palpation points """

        steps = 12
        poses = []

        origin = np.hstack(np.array(self.tissue_pose.position))
        frame = np.array(self.tissue_pose.orientation.matrix)

        u, v, w = frame.T[0], frame.T[1], frame.T[2]

        rotation_matrix = np.array([v, u, -w]).transpose()

        dy = self.tissue_length/(steps)
        z = self.probe_offset

        # pick up tool
        # self.pick_up_tool()

        self.probe_stop_reset()
        for i in range(steps-3):
            if i == 0:
                continue
            offset = np.dot(frame, np.array([i*dy, 0.0, z+0.02]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            offset = np.dot(frame, np.array([i*dy, 0.0, z]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            # start recording data
            rospy.sleep(0.2)
            self.probe_start()

            offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            # pause recording data
            rospy.sleep(0.2)
            self.probe_pause()

            offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z+0.02]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)


        for i in range(steps-3)[::-1]:
            if i == 0:
                continue
            offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z+0.02]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            # start recording data
            rospy.sleep(0.2)
            self.probe_start()

            offset = np.dot(frame, np.array([i*dy, 0.0, z]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

            # pause recording data
            rospy.sleep(0.2)
            self.probe_pause()

            offset = np.dot(frame, np.array([i*dy, 0.0, z+0.02]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)


        self.probe_save("probe_data.p")
        # self.drop_off_tool()

    def execute_raster_single_row(self, n, pick_up_tool=True):
        """ Repeatedly passes n times over a single row of a tissue brick."""

        steps = 12
        poses = []

        origin = np.hstack(np.array(self.tissue_pose.position))
        frame = np.array(self.tissue_pose.orientation.matrix)

        u, v, w = frame.T[0], frame.T[1], frame.T[2]

        rotation_matrix = np.array([v, u, -w]).transpose()

        dy = self.tissue_length/(steps)
        z = self.probe_offset

        # pick up tool
        if pick_up_tool:
            self.pick_up_tool()

        self.probe_stop_reset()
        for i in range(steps-3):
            if i == 0:
                continue
            if i == 3:
                for _ in range(n):
                    offset = np.dot(frame, np.array([i*dy, 0.0, z+0.02]))
                    pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                    self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

                    offset = np.dot(frame, np.array([i*dy, 0.0, z]))
                    pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                    self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

                    # start recording data
                    print "Executing row raster %s/%s" % (_+1, n)
                    rospy.sleep(0.2)
                    self.probe_start()

                    offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z]))
                    pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                    self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

                    # pause recording data
                    rospy.sleep(0.2)
                    self.probe_pause()

                    offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z+0.02]))
                    pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                    self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)


        self.probe_save("probe_data.p")
        # self.drop_off_tool()

    def execute_raster_single_row_reverse(self, n, pick_up_tool=True):
        """ Repeatedly passes n times over a single row of a tissue brick in the reverse direction"""

        steps = 12
        poses = []

        origin = np.hstack(np.array(self.tissue_pose.position))
        frame = np.array(self.tissue_pose.orientation.matrix)

        u, v, w = frame.T[0], frame.T[1], frame.T[2]

        rotation_matrix = np.array([v, u, -w]).transpose()

        dy = self.tissue_length/(steps)
        z = self.probe_offset

        # pick up tool
        if pick_up_tool:
            self.pick_up_tool()

        self.probe_stop_reset()
        for i in range(steps-3):
            if i == 0:
                continue
            if i == 3:
                for _ in range(n):
                    offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z+0.02]))
                    pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                    self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

                    # reverse pass over the row
                    offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z]))
                    pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                    self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

                    # start recording data
                    rospy.sleep(0.2)
                    self.probe_start()

                    offset = np.dot(frame, np.array([i*dy, 0.0, z]))
                    pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                    self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

                    # pause recording data
                    rospy.sleep(0.2)
                    self.probe_pause()

                    offset = np.dot(frame, np.array([i*dy, 0.0, z+0.02]))
                    pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                    self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)


        self.probe_save("probe_data.p")
        # self.drop_off_tool()

    def execute_point_probe_grid(self, m, n, k):
        """ Point probes on an m x n grid.
            Iterates over grid k times
        """
        speed = 0.05

        origin = np.hstack(np.array(self.tissue_pose.position))
        frame = np.array(self.tissue_pose.orientation.matrix)

        u, v, w = frame.T[0], frame.T[1], frame.T[2]

        rotation_matrix = np.array([v, u, -w]).transpose()

        dy = self.tissue_width/n
        dx = self.tissue_length/m
        z = self.probe_offset

        # pick up tool
        # self.pick_up_tool()

        self.probe_stop_reset()
        for _ in range(k):
            for i in range(m-1):
                for j in range(n):
                    offset = np.dot(frame, np.array([i*dx, j*dy, z+0.01]))
                    pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                    self.psm1.move_cartesian_frame_linear_interpolation(pose, self.speed, False)

                    offset = np.dot(frame, np.array([i*dx, j*dy, z]))
                    pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                    self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)
                    
                    rospy.sleep(0.4)
                    self.probe_single_point_record()
                    rospy.sleep(0.4)
                    
                    offset = np.dot(frame, np.array([i*dx, j*dy, z+0.01]))
                    pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                    self.psm1.move_cartesian_frame_linear_interpolation(pose, self.speed, False)

        self.probe_save("probe_data.p")
        # self.drop_off_tool()