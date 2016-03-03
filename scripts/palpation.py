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

        self.record_point_palpation_data = False
        self.locations = []
        self.baseline = float("-inf")
        self.num_points = 0
        self.cooldown = 0

        # subscribe to probe data
        rospy.Subscriber("/probe/measurement", Float64, self.probe_callback)

        rospy.Subscriber("/gaussian_process/pts_to_probe", Points, self.probe_points_callback)
        self.measurements_pub = rospy.Publisher("/palpation/measurements", FloatList)



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
    def execute_scan_points_continuous(self, n):
        poses = []

        frame = np.array(self.tissue_pose.orientation.matrix)

        u, v, w = frame.T[0], frame.T[1], frame.T[2]

        rotation_matrix = np.array([v, u, -w]).transpose()

        dy = self.tissue_length/(steps)
        z = self.probe_offset

        # pick up tool
        # self.pick_up_tool()

        randoms = np.random.random((n-1)*2)

        offset = np.dot(frame, np.array([np.random.random()*self.tissue_length, np.random.random()*self.tissue_width, z+0.02]))
        pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
        self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

        offset = np.dot(frame, np.array([np.random.random()*self.tissue_length, np.random.random()*self.tissue_width, z]))
        pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
        self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

        self.probe_stop_reset()
        rospy.sleep(0.2)
        self.probe_start()

        for i in range(n-1):
            offset = np.dot(frame, np.array([randoms[2*i]*self.tissue_length, randoms[2*i+1]*self.tissue_width, z]))
            pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)

        rospy.sleep(0.2)
        self.probe_pause()

        self.probe_save("probe_data_scan_points_continuous.p")

        offset = np.dot(frame, np.array([randoms[-2]*self.tissue_length, randoms[-1]*self.tissue_width, z+0.02]))
        pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
        self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)
        # self.drop_off_tool()

    def execute_raster_tilted(self, theta, direction):
        """ Direction should be 1 for L2R and -1 for R2L"""
        """ Linearly interpolates through a series of palpation points """
        
        print('hi')
        steps = 12
        poses = []

        origin = np.hstack(np.array(self.tissue_pose.position))
        frame = np.array(self.tissue_pose.orientation.matrix)

        u, v, w = frame.T[0], frame.T[1], frame.T[2]
        rotation_matrix = np.array([v, u, -w]).transpose()
        

        # frame = tfx.pose(tfx.pose(frame_og).as_tf()*tfx.transform(tfx.tb_angles(roll=10, pitch=0, yaw=0))).rotation.matrix
        # u, v, w = frame.T[0], frame.T[1], frame.T[2]
        # rot1 = np.array([v, u, -w]).transpose()
        # frame = tfx.pose(tfx.pose(frame_og).as_tf()*tfx.transform(tfx.tb_angles(roll=0, pitch=10, yaw=0))).rotation.matrix
        # u, v, w = frame.T[0], frame.T[1], frame.T[2]
        # rot2 = np.array([v, u, -w]).transpose()
        # frame = tfx.pose(tfx.pose(frame_og).as_tf()*tfx.transform(tfx.tb_angles(roll=0, pitch=0, yaw=10))).rotation.matrix
        # u, v, w = frame.T[0], frame.T[1], frame.T[2]
        # rot3 = np.array([v, u, -w]).transpose()

        # import IPython; IPython.embed()
        dy = self.tissue_length/(steps)
        z = self.probe_offset

        print('what"s')
        # pick up tool
        # self.pick_up_tool()
        print('up')

        self.probe_stop_reset()

        pose = tfx.pose(origin, rotation_matrix, frame=self.tissue_pose.frame)
        # # pose1 = tfx.pose(pose.as_tf()*tfx.transform(tfx.tb_angles(roll=15, pitch=0, yaw=0))) #apparently this tilts head of probe toward base of arms. wait no apparently this tilts head of probe toward monitors? (to the right) this is probably the direction that the probe constrains rotation in
        rotation_matrix = tfx.pose(pose.as_tf()*tfx.transform(tfx.tb_angles(roll=0, pitch=theta*direction, yaw=0))).rotation.matrix

        SPEED = 0.01

        for i in range(steps-3):
            if i == 0:
                continue
            # offset = np.dot(frame, np.array([i*dy, 0.0, z+0.02]))
            # pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
            # # # pose1 = tfx.pose(pose.as_tf()*tfx.transform(tfx.tb_angles(roll=15, pitch=0, yaw=0))) #apparently this tilts head of probe toward base of arms. wait no apparently this tilts head of probe toward monitors? (to the right) this is probably the direction that the probe constrains rotation in
            # rotation_matrix = tfx.pose(pose.as_tf()*tfx.transform(tfx.tb_angles(roll=0, pitch=theta*direction, yaw=0))).rotation.matrix #tilted even more toward monitors? what? also a bit towards base. why are roll & pitch the same o.o joint limits? moves more than pose1. 
            # should be done before for loop? lol
            # pose3 = tfx.pose(pose.as_tf()*tfx.transform(tfx.tb_angles(roll=0, pitch=0, yaw=20))) #away from arms and to the left
            # pose4 = tfx.pose(origin+offset, rot1) #so apparently these don't work
            # pose5 = tfx.pose(origin+offset, rot2)
            # pose6 = tfx.pose(origin+offset, rot3)


            if direction == 1:
                offset = np.dot(frame, np.array([i*dy, 0.0, z+0.02]))
                pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                self.psm1.move_cartesian_frame_linear_interpolation(pose, SPEED, False)
                # print('og pose')
                # self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)
                # import IPython; IPython.embed()
                # print('pose 1')
                # self.psm1.move_cartesian_frame_linear_interpolation(pose1, 0.005, False)
                # IPython.embed()
                # print('pose 2')
                # self.psm1.move_cartesian_frame_linear_interpolation(pose2, 0.005, False)
                # IPython.embed()
                # print('pose 3')
                # self.psm1.move_cartesian_frame_linear_interpolation(pose3, 0.005, False)
                # IPython.embed()
                # print('pose 4')
                # self.psm1.move_cartesian_frame_linear_interpolation(pose4, 0.005, False)
                # IPython.embed()
                # print('pose 5')
                # self.psm1.move_cartesian_frame_linear_interpolation(pose5, 0.005, False)
                # IPython.embed()
                # print('pose 6')
                # self.psm1.move_cartesian_frame_linear_interpolation(pose6, 0.005, False)
                # IPython.embed()


                offset = np.dot(frame, np.array([i*dy, 0.0, z]))
                pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                self.psm1.move_cartesian_frame_linear_interpolation(pose, SPEED, False)

                # start recording data
                rospy.sleep(0.2)
                self.probe_start()

                offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z]))
                pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                self.psm1.move_cartesian_frame_linear_interpolation(pose, SPEED, False)

                # pause recording data
                rospy.sleep(0.2)
                self.probe_pause()
                print(self.probe_data)

                offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z+0.02]))
                pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                self.psm1.move_cartesian_frame_linear_interpolation(pose, SPEED, False)
            else:
                offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z+0.02]))
                pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                self.psm1.move_cartesian_frame_linear_interpolation(pose, SPEED, False)

                offset = np.dot(frame, np.array([i*dy, self.tissue_width*0.95, z]))
                pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                self.psm1.move_cartesian_frame_linear_interpolation(pose, SPEED, False)

                # start recording data
                rospy.sleep(0.2)
                self.probe_start()

                offset = np.dot(frame, np.array([i*dy, 0.0, z]))
                pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                self.psm1.move_cartesian_frame_linear_interpolation(pose, SPEED, False)

                # pause recording data
                rospy.sleep(0.2)
                self.probe_pause()
                print(self.probe_data)

                offset = np.dot(frame, np.array([i*dy, 0.0, z+0.02]))
                pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
                self.psm1.move_cartesian_frame_linear_interpolation(pose, SPEED, False)

        print(self.probe_data)
        self.probe_save("probe_data.p")
        # self.drop_off_tool()




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


    def probe_points_callback(self, data):
        measurements = self.execute_point_probes(data.x, data.y)
        m = FloatList()
        m.data = measurements
        self.measurements_pub.publish(m)

    def execute_point_probes(self, points_x, points_y):
        print("a")
        print("numpoints: " + str(len(points_x)))
        measurements = []
        rospy.sleep(1)
        for i in range(len(points_x)):
            measurements.append(self.execute_point_probe(points_x[i], points_y[i]))
        return measurements


    def execute_point_probe(self, x, y):
        print("b")
        speed = 0.05

        print("x: " + str(x))
        print("y: " + str(y))
        print("tissue length: " + str(self.tissue_length))
        print("tissue width: " + str(self.tissue_width))

        if x < 0 or x > self.tissue_length or y < 0 or y > self.tissue_width:
            return None

        origin = np.hstack(np.array(self.tissue_pose.position))
        frame = np.array(self.tissue_pose.orientation.matrix)

        u, v, w = frame.T[0], frame.T[1], frame.T[2]

        rotation_matrix = np.array([v, u, -w]).transpose()

        z = self.probe_offset

        offset = np.dot(frame, np.array([x, y, z+0.01]))
        pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
        self.psm1.move_cartesian_frame_linear_interpolation(pose, self.speed, False)

        offset = np.dot(frame, np.array([x, y, z]))
        pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
        self.psm1.move_cartesian_frame_linear_interpolation(pose, 0.01, False)
        
        rospy.sleep(0.4)
        measurement = self.curr_probe_value
        rospy.sleep(0.4)
        
        offset = np.dot(frame, np.array([x, y, z+0.01]))
        pose = tfx.pose(origin+offset, rotation_matrix, frame=self.tissue_pose.frame)
        self.psm1.move_cartesian_frame_linear_interpolation(pose, self.speed, False)
        return measurement


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

    def register_surface(self, n):
        raw_input("Make sure probe is not touching anything. Press enter when ready")
        print("Palpate " + str(n) + " points")
        self.record_point_palpation_data = True
        while self.num_points < n:
            rospy.sleep(0.1)
        self.probe_save_locations("probe_locations.p")

