import numpy as np
from sanghi import solver3
from two import solver2
from submit import solver1
from matplotlib import pyplot as plt


def getObj( X, y, w, b ):
	hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w ) + b), y ), 0 )
	return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )

Z = np.loadtxt( "data" )

y = Z[:,0]
X = Z[:,1:]
C = 1

avgTime = [0,0,0]
avgPerf = [0,0,0]

# To avoid unlucky outcomes try running the code several times
numTrials = 1
# 30 second timeout for each run
timeout = [0,4,0]
# Try checking for timeout every 100 iterations
spacing = 50

for t in range( numTrials ):
	(w, b, totTime, obj1, time1) = solver1( X, y, C, timeout[0], spacing )
	avgTime[0] = avgTime[0] + totTime
	tmp = getObj( X, y, w, b )
	avgPerf[0] = avgPerf[0] + tmp

	(w, b, totTime, obj2, time2) = solver2( X, y, C, timeout[1], spacing )
	avgTime[1] = avgTime[1] + totTime
	tmp = getObj( X, y, w, b )
	avgPerf[1] = avgPerf[1] + tmp

	(w, b, totTime, obj3, time3) = solver3( X, y, C, timeout[2], spacing )
	avgTime[2] = avgTime[2] + totTime
	tmp = getObj( X, y, w, b )
	avgPerf[2] = avgPerf[2] + tmp


plt.plot( time3, obj3, color = '#99FFFF', linestyle = '-', label = "Coordinate Maximization on D2" )
plt.plot( time2, obj2, color = '#00CC00', linestyle = '-', label = "Coordinate Descent on P1" )
plt.plot( time1, obj1, color = 'r', linestyle = '--', label = "Mini-Batch SGD on P1" )
plt.xlabel( "Elapsed time (sec)" )
plt.ylabel( "Value of Objective function(P1)" )
plt.legend()
plt.ylim( 0, 20000 )
# plt.xlim( 0, 5 )
# plt.show()


print( [x/numTrials for x in avgPerf], [x/numTrials for x in avgTime] )