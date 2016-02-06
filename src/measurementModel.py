import numpy as np 
# from simUtils import getActualHeight

def measureHeight ():
	z = eta*exp (-pow(2*(sigSensor**2),-1)*((H_actual-H_probePose)**2))
	return z

def sensorHeight_atPoint(posToEstimate, probePos, spatialRes =1):
	"""
	Outputs a measurement for a single height value
	using a truncated gaussian
	Note: It uses the actual heights in calculation not the 
	current estimate of height
	"""
	# sigSensor controls the dropoff
	sigSensor = sqrt(2)*spatialRes
	#normalization constant for the palpation probe
	eta = 1
	eps = 1e-5 # results in using truncated gaussians

	#height at position to estimate
	H_actual = getActualHeight(posToEstimate)
	
	#height at probe tip position where measurement is made
	H_probePose = getActualHeight(probePos)
	
	# the simulated measurement returns a exponential dropoff
	z_act = eta*exp (-(1/sigSensor**2)*(H_actual-H_probePose)**2)
	z =  z_act if z_act > eps else 0

	mu, sigma = measurementNoise()
	z += mu + sigma*np.random.randn()

	return z	


# def palpationModel(tumor_location,probePos, spatialRes =1):	
# 	"""
# 	sensor model
# 	tumor_location = (x,y)
# 	probePos = position of probe for measurement (x,y)
# 	"""
# 	# sigma controls the dropoff
# 	sigma = sqrt(2)*spatialRes
# 	#normalization constant for the palpation probe
# 	eta = 1
# 	#baseline reading
# 	baseline = 0

# 	z = baseline + eta* exp (-(1/sigma**2)*np.linalg.norm(tumor_location-probePos)**2)
# 	return z


def palpate(tumor_location,probePos, spatialRes =1):
	"""
	measurement model
	tumor_location = (x,y)
	probePos = position of probe for measurement (x,y)
	"""
	# sigSensor controls the dropoff
	sigSensor = sqrt(2)*spatialRes
	#normalization constant for the palpation probe
	eta = 1
	eps = 1e-5 # results in using truncated gaussians

	z_act = eta*exp (-(1/sigSensor**2)*np.linalg.norm(tumor_location-probePos)**2)
	z =  z_act if z_act > eps else 0

	mu, sigma = measurementNoise()	
	z += mu + sigma*np.random.randn()
	
	return z


def sensorNoise():
	"""
	single sensor noise model 
	All changes go here
	"""
	mu = 0
	sigma = 1
	return mu, sigma