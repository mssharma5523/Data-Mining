import numpy
import sys
import os
import cvxopt
import pylab as plotter
import random
import time

number_of_train_vectors = 100
number_of_test_vectors = 100

def generate_train_data():
	'''
	L1 = [random.normalvariate(10,1) for e in range(0,100)]
	R1 = numpy.ones(100)
	L2 = [random.normalvariate(-10,1) for e in range(0,100)]
	R2 = numpy.ones(100)*-1
	'''
	covariance = numpy.array([[2.5,2],[2,2.5]])
	L1 = numpy.random.multivariate_normal([0,3],covariance,number_of_train_vectors)
	R1 = numpy.ones(number_of_train_vectors)
	L2 = numpy.random.multivariate_normal([3,0],covariance,number_of_train_vectors)
	R2 = numpy.ones(number_of_train_vectors)*-1
	train_X = numpy.vstack((L1,L2)) # mixing the two separate data together so that they can be trained combinedly
	train_Y = numpy.hstack((R1,R2)) # mixing the two Y's together so that they can be trained combinedly
	#print train_Y
	return train_X,train_Y

if __name__ == '__main__':
	train_X,train_Y = generate_train_data()
	f = open('train_file_100','w')
	#l = train_X.shape()
	length = train_X.shape[0]
	for i in range(length):
		row_print = train_X[i]
		row_print = row_print.tolist()
		#row_print = row_print.translate(None,'[],')
		row_print = " ".join(map(str, row_print))
		row_print = row_print + " " + str(train_Y[i]) + "\n"
		f.write(row_print)
		print row_print
	f.close()