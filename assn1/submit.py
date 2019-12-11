import numpy as np
import random as rnd
import time as tm
from matplotlib import pyplot as plt
import math

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length


# def steplength(eta, t):
	# return eta/t

def grad(theta, C, X, y):
	(n, d) = X.shape
	# print(w)
	i = rnd.randint(0, n-1)
	x = X[i,:]
	# print(x)
	discriminant = (x.dot(theta)) * y[i]
	g = 0
	# print(discriminant)
	if discriminant < 1:
		g = 2*(1 - discriminant)
	# print(x)
	return theta - (C * n * x * g * y[i])

def batch_grad(theta, C, X, y, B):
	(n, d) = X.shape

	samples = rnd.sample( range(0, n), B )
	X_ = X[samples,:]
	y_ = y[samples]
	discriminant = np.multiply((X_.dot(theta)), y_)
	g = np.zeros( (B,) )
	g[discriminant < 1] = -1
	return theta + C * (n/B) * 2 * (g * X_.T).dot(np.multiply(y_, (1 - discriminant)))
	# if discriminant < 1:
		# g = 

	# return theta - (C * (n/B) * (X_.T * g).dot(y_))

def getObj( X, y, theta, C):
	w = theta[0:-1]
	hingeLoss = np.maximum( 1 - np.multiply( (X.dot( theta )), y ), 0 )
	return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )


def solver1( X, y, C, timeout, spacing ):
	
	(n, d) = X.shape
	X = np.c_[X, np.ones(n)]
	d = d + 1
	t = 0
	totTime = 0
	totalTime = 0

	# w is the normal vector and b is the bias
	# These are the variables that will get returned once timeout happens
	# w = np.zeros( (d-1,) )
	# b = 0
	tic = tm.perf_counter()

	theta = np.zeros( (d,) )
	cumulative = theta
	w = theta[0:-1]
	b = theta[-1]
	eta = 0.000003
	B = 500

	obj_SGD = np.array([getObj(X, y, theta, C)])
	time_SGD = np.array([0])

	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:

				# plt.plot( time_SGD, obj_SGD, color = 'r', linestyle = '-', label = "SGD" )
				# plt.xlabel( "Elapsed time (sec)" )
				# plt.ylabel( "C-SVM Objective value" )
				# plt.legend()
				# plt.ylim( 0, 20000 )
				# # plt.xlim( 0, timeout )
				# plt.show()

				# print(t)
				return (w, b, totTime, obj_SGD, time_SGD)
			else:
				tic = tm.perf_counter()
		
		tic1 = tm.perf_counter()

		# thetanew = theta - (grad(theta, C, X, y) * (eta/t))
		theta = theta - (batch_grad(theta, C, X, y, B) * (eta))
		# wnew = thetanew[0:-1]
		# bnew = thetanew[-1]
		# prev = getObj( X, y, theta, C )
		# new = getObj( X, y, thetanew, C )
		# print(new)
		# print(new, prev)
		# if new > prev:
		# 	eta = eta/2
		# 	continue
		# theta = thetanew

		toc1 = tm.perf_counter()

		totalTime = totalTime + toc1 - tic1
		obj_SGD = np.append(obj_SGD, getObj(X, y, theta, C))
		time_SGD = np.append(time_SGD, totalTime)
		# cumulative = cumulative + theta
		w = theta[0:-1]
		b = theta[-1]

	return (w, b, totTime) # This return statement will never be reached

# C = 1