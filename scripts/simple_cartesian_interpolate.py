""" 
This example brings the robot's arm to the start_frame and interpolates to the end_frame 
Here we focus on how to use the function move_cartesian_frame_linear_interpolation under the robot class

Class: robot.py
Method: move_cartesian_frame_linear_interpolation(self, abs_frame, speed, False/True)
Dependencies: poseInterpolator.py / tfx

Input param:  
1st) abs_frame 		-> The frame of the end position expressed as a list type. 

2nd) speed 		-> (float) type if cubic interpolation is False
	  		-> (list) type of [velocity_Low, velocity_High] if cubic interpolation is True

3rd) False/True 	-> (Default) False which makes the interpolation linear
			-> (Optional) True which makes the interpolation a cubic function; Has yet to be implemented in future versions

Explaination:

- The function takes in an absolute frame in catresian space and applies SLERP interpolation to bring it from its current position to the position
that is specified by the user in abs_frame.

- Vector interpolation is done linearly in a straight line from the start to end points
- Rotation is done using quaternion SLERP function under the tfx.transformations.quaternion_slerp method

- We calculate the vector norm of the start and end points and divide into segments based on 1 interval per 0.1 mm of distance travelled

- The elements for abs_frame are a list type: [X,Y,Z,Roll,Pitch,Yaw] representing the vector and rotations of the end pose

- The 3rd parameter turns the cubic interpolator on / off. If cubic interpolator is False, then we are using a linear interpolator and the arm will sweep across
the coordinates at a uniform speed. As of now, no cubic interpolation function has been implemented yet so please help!!!!!!!!!!!!!!!!
"""

import time
import numpy as np
from robot import *

def Interpolate(robotname):
    R = robot(robotname)
    start_frame = [0.0447258528045, 0.0512065150661, -0.131469281192, -2.5396399475337668, -0.2903344173204437, 0.18226904308086755]      # Starting position in [X Y Z Roll Pitch Yaw]
    end_frame = [-0.0447258528045, 0.0512065150661, -0.131469281192, -2.5396399475337668, -0.2903344173204437, 0.18226904308086755]	  # Ending position in [X Y Z Roll Pitch Yaw]

    start_Vect = np.array([start_frame[0],start_frame[1],start_frame[2]])
    end_Vect = np.array([end_frame[0],end_frame[1],end_frame[2]])
    print "Norm: ", np.linalg.norm(end_Vect - start_Vect)				# Computes the norm of the vector between the start and end points

    R.move_cartersian_frame_linear_interpolation(start_frame, 0.01, False)              # (abs_frame, speed, cubic interpolation = False)
    time.sleep(1)
    R.move_cartersian_frame_linear_interpolation(end_frame, 0.01, False)               	# (abs_frame, speed, cubic interpolation = False)


if __name__ == '__main__':
    """Here will check if we are given the name of the robot arm or not, if we are we run Interpolate()"""
    """ Expected inputs are robot's arm, Cubic interpolator True / False"""

    if (len(sys.argv) == 2):
        Interpolate(sys.argv[1])
    elif (len(sys.argv) == 3):
    	Interpolate(sys.argv[1], sys.argv[2])
    else:
        print sys.argv[0] + ' requires one argument, i.e. name of dVRK arm and/or interpolate true or false'


