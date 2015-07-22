#!/usr/bin/python

import matplotlib.pyplot as plt
import rospy, math
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Header
import numpy as np
import random

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class ImaginaryArmSimulation:
	def __init__(self):
		print('hi')
		self.link1 = 1
		self.link2 = 1
		self.link3 = 1
		self.angles = [0, 0, 0]
		self.plan = None
		self.map = {} # log odds
		x = -1.0
		while x <= 1.5:
			self.map["%.2f" % x] = math.log(0.0001)
			x += 0.01

		#let's say we go from -1 to 1, with each cell being 0.1 wide

		# ROS stuff
		self.publisher = rospy.Publisher('arm_controls', JointState)
		rospy.init_node('arm_controller', anonymous=True)
		self.joint_state_msg = JointState()
		self.joint_state_msg.name = ['joint0', 'joint1', 'joint2']
		self.move(self.angles)
		print('bye')

	def calculate_tip_pos(self):
		angle1 = self.angles[0]
		angle2 = angle1 + self.angles[1]
		angle3 = angle2 + self.angles[2]
		x = self.link1 * math.cos(angle1) + self.link2 * math.cos(angle2) + self.link3 * math.cos(angle3)
		y = self.link1 * math.sin(angle1) + self.link2 * math.sin(angle2) + self.link3 * math.sin(angle3)
		return y, -1 * (x - 2.45)
		# apparently x-axis is up-down and y-axis is left-right...? and 0,0 is where the ball is, and
		# x-axis increases as you go down...
		# for ease of reading let's just reverse the order in which we return them
		# let's also reverse direction of x-axis and shift it by 3
		# oh it's because 0 deg is pointing down not to the right. I should just change it so 0 is right?
	def move(self, angles):
		# publish ROS
		self.angles = angles
		self.joint_state_msg.position = angles
		self.publisher.publish(self.joint_state_msg)
		

	# simulates probe & tumor. questionable
	def get_probe_data(self):
		pos = self.calculate_tip_pos()
		if pos[1] <= 0:
			if abs(0.5 - pos[0]) <= 0.125:
				#print(4096 - abs(0.25 - pos[0]) * 4000)
				#return 4096 - abs(0.25 - pos[0]) * 4000
				return 4096 * gaussian(pos[0], 0.5, 0.1) + 64 * random.gauss(0, 6)  
		return 64 * random.gauss(0, 6)

	# very questionable. idk if this log odds occupancy grid thing is working...
	def inv_sensor_model(self, data):
		prob = data / 4096.0
		# print(prob)
		a = 0
		try:
			a = math.log(prob / (1.0 - prob))
		except ValueError:
			if prob < 0.1:
				a = -12
			else:
				a = 12
		return a

	def update_map(self, data, x_coord):
		offset = range(-5, 6)
		offset = [i / 100.0 for i in offset]
		#x is 0, 0.01, 0.02...
		offset = [0]
		try:
			for x in offset:
				key = "%.2f" % (x_coord + x)
				if key == '-0.00':
					key = '0.00'
				#this is weird but might not be the problem
				# self.map[key] = self.map[key] + ((1-abs(x)) * (self.inv_sensor_model(data) - math.log(1)) + math.log(1)) - math.log(1)
				self.map[key] = self.map[key] + self.inv_sensor_model(data) - math.log(1)
		except KeyError as e:
			print(e)
			import IPython; IPython.embed()

	def optimize_plan(self):
		t = list(range(100000))
		t = [(i-100000.0)/100000 for i in t]
		plan = []
		for i in t:
			theta1 = i
			#print(str(theta1) + " theta1")
			#print(str(math.cos(theta1)) + " cos")
			theta2 = math.acos(1.45-math.cos(theta1)) - theta1
			theta3 = -theta1 - theta2
			plan.append([theta1, theta2, theta3])
		self.plan = plan

	def cost_function(self):
		return 0

	def simulate(self, max_time):
		
		"""
		print('sim')
		a = [(float(i) / 2000000) % 6 for i in list(range(1000000))]
		b = [(float(i) / 1000000) % 6 for i in list(range(1000000))]
		c = [(float(i) / 500000) % 6 for i in list(range(1000000))]
		print(self)
		for time in range(1000000):
			self.move([a[time], b[time], c[time]])
			if time % 10000 == 0:
				print(self.calculate_tip_pos())
			
			
		"""

		"""
		self.move(self.plan)
		data = self.get_probe_data()
		self.update_map(data) # particle filtering step
		self.plan = self.optimize_plan() #scp/optimization step
		"""

		self.optimize_plan()
		print('hi')
		WINDOW = 700
		x1 = np.r_[:WINDOW]
		y1 = np.array([0]*WINDOW)
		"""
		fig = plt.figure(figsize=(12,12))
		ax = fig.add_subplot(111)
		lx, = ax.plot(x1, y1)
		"""
		print(self.map)
		print('huh')
		refresh = 1
		i = 0

		y1 = []
		for t in self.plan:
			#print('y')
			self.move(t)
			data = self.get_probe_data()
			#y1 = np.append(y1, data)
			if i == 0:
				y1 = np.append(y1, data)
			i += 1
			i = i % refresh

			"""
			if i == 0:
				lx.set_ydata(y1[-WINDOW:])
				fig.canvas.draw()
			i += 1
			i = i % refresh
			"""
			self.update_map(data, self.calculate_tip_pos()[0])
			
		print(self.map)	
		x1 = np.r_[:len(y1)]
		fig = plt.figure(figsize=(12,12))
		ax = fig.add_subplot(111)
		lx, = ax.plot(x1, y1)
		fig.canvas.draw()
		plt.show(block=True)

		fig = plt.figure(figsize=(12,12))
		ax = fig.add_subplot(111)
		

		x = []
		y = []

		x = sorted(self.map.keys())
		for key in x:
			try:
				y.append(1.0 - (1.0 / (1.0 + math.exp(self.map[key]))))
			except OverflowError as e:
				if self.map[key] < 0:
					y.append(0)
				else:
					y.append(1)

		x = [float(i) for i in x]

		"""
		for key,value in self.map.iteritems():
			x.append(float(key))
			try:
				y.append(1.0 - (1.0 / (1.0 + math.exp(value))))
			except OverflowError as e:
				if value < 0:
					y.append(0)
				else:
					y.append(1)
		"""
		print('plot')
		print(x)
		print(y)
		# need to sort before plotting
		lx, = ax.plot(x, y)
		ax.set_xlim([0.25, 0.75])
		fig.canvas.draw()
		plt.show(block=True)
		print('where you go')
		#import IPython; IPython.embed()
		#rospy.spin()

	def display_result(self):
		return 0

if __name__ == '__main__':
	print('main')
	arm = ImaginaryArmSimulation()
	print('start sim')
	arm.simulate(10000)
	arm.display_result()