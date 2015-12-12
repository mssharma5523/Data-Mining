#!/usr/bin/env python

'''
This file is the implementation of the map part of the mapreduce portion in the SVM. The logic behind is to output
the weight and the intercept vector for  the small chunks of data. So basically it will be same as the function 
of svm_training in the serial implementation. These weights and intercept for the individual small chunks of training 
dataset will be passed to the reducer and the final weight will be the sum and average of the individual weights respectively

Now in the map part, the input needs to be taken from stdin, hence the input file will be (X,y) type where X is a vector
in a line. Just read this input and store it in a numpy array
'''

import numpy
import sys
import os
import cvxopt
import pylab as plotter
import random
import time

def kernel_function(data1,data2):
	return numpy.dot(data1,data2)

def svm_training(x_train,y_train,error_constant):
	'''
	The task here is to generate the matrix of the form described in the heading from the training dataset and then
	use cvxopt to get the values of lagrangian and then get W.X + b = 0 equation
	'''
	train_number,train_dim = x_train.shape
	kernel_matrix = numpy.zeros((train_number,train_number))
	#constructing the Kernel Matrix
	for i in range(train_number):
		for j in range(train_number): ##print x_train[i]
			##print x_train[i]
			kernel_matrix[i][j] = kernel_function(x_train[i],x_train[j])

	outer_matrix = cvxopt.matrix(numpy.outer(y_train,y_train)*kernel_matrix) #generating the matrix yy^T*K
	one_vector = cvxopt.matrix(numpy.ones(train_number)*-1) #generate the matrix for \sum a_i = 1^TA
	y_matrix = cvxopt.matrix(y_train,(1,train_number))
	b = cvxopt.matrix(0.0)

	#implementing the error part and generating the matrix for 0 <= a_i<= C
	alpha_matrix = cvxopt.matrix(numpy.vstack((numpy.diag(numpy.ones(train_number) * -1),numpy.identity(train_number)))) #LHS
	c_matrix = cvxopt.matrix(numpy.hstack((numpy.zeros(train_number),numpy.ones(train_number)*error_constant)))  #RHS

	#now solving the minimisation problem and getting the solution for the langrangian values
	optimisation_soln = cvxopt.solvers.qp(outer_matrix,one_vector,alpha_matrix,c_matrix,y_matrix,b)

	#Now we get the langrangians and discard those which are ~0, hence we get the remaining support vectors
	'''langrangians = numpy.ravel(optimisation_soln['x'])
	#print len(langrangians)
	support = langrangians > 0.000001
	left = numpy.arange(len(langrangians))[support]
	supportVectors_x = []
	supportVectors_y = []
	supportVectors_a = []

	count = 0
	for x in numpy.nditer(langrangians):
		##print x
		#raw_input()
		if x >0:
			supportVectors_x.append(x_train[count])
			supportVectors_y.append(y_train[count])
			supportVectors_a.append(x)
		count +=1
	supportVectors_a = numpy.asarray(supportVectors_a)
	supportVectors_x = numpy.asarray(supportVectors_x)
	supportVectors_y = numpy.asarray(supportVectors_y)'''
	langrangians = numpy.ravel(optimisation_soln['x'])
	support = langrangians>0.00001
	left = numpy.arange(len(langrangians))[support]
	supportVectors_a = langrangians[support]
	supportVectors_x = x_train[support]
	supportVectors_y = y_train[support]
	#finding the weight vectors for the matrix WX+b=0
	weight = numpy.zeros(train_dim)
	for i in range(len(supportVectors_a)):
		weight = weight + supportVectors_a[i]*supportVectors_y[i]*supportVectors_x[i]

	#finding the intercept
	intercept = 0
	count = 0
	#print supportVectors_a
	#print supportVectors_y.shape
	for count in range(len(supportVectors_x)):
		intercept = intercept + supportVectors_y[count]
		intercept = numpy.sum(supportVectors_a*supportVectors_y*kernel_matrix[left[count],support])
	#print len(supportVectors_a)
	intercept = intercept/len(supportVectors_a)
	print "The number of support vectors found were %d out of total %d data sets"%(len(supportVectors_a),train_number)
	return weight,intercept,supportVectors_x,supportVectors_y,supportVectors_a

##The input will be of the form of <Vector,y> per line
def read_train_input():
	# input comes from STDIN (standard input)
	#train_Y = numpy.empty([1])
	#train_X = numpy.empty([1,1])
	list_y = []
	list_x = []
	#with open('train_file') as fp:
   	    #for line in fp:
	for line in sys.stdin: #Commenting for time being testing pursposes
	    # remove leading and trailing whitespace
	    line = line.strip()
	    # split the line into words
	    words = line.split()
	    # increase counters
	    #print words[0]
	    #temp_array = numpy.empty([0])
	    temp_list = []
	    #print temp_array
	    for word in words:
	        # write the results to STDOUT (standard output);
	        # what we output here will be the input for the
	        # Reduce step, i.e. the input for reducer.py
	        #
	        # tab-delimited; the trivial word count is 1
	        #All words except the last are a row vector
	        #print float(word)
	        temp_list.append(float(word))
	    #print temp_list
	    #temp_array = numpy.asarray(temp_list)
	    list_y.append(temp_list[-1])
	    temp_list.pop()
	    list_x.append(temp_list)
	    #print list_x
	    #print list_y
	    #raw_input()
	    #numpy.append(train_Y,float(words[-1]))
	    #numpy.vstack([train_X,temp_array])
	train_Y = numpy.asarray(list_y)
	train_X = numpy.asarray(list_x)
	#print list_x
	#print train_Y
	#print train_X
	return train_X,train_Y

if __name__ == "__main__":
	train_X,train_Y=read_train_input()
	#print train_X
	#print train_Y
	weight,intercept,supportVectors_x,supportVectors_y,supportVectors_a = svm_training(train_X,train_Y,0.3)
	#output = []
	#output.append(weight)
	#output.append(intercept)
	weight = weight.tolist()
	weight = " ".join(map(str, weight))
	print '%s\t%s' % (weight,str(intercept))
	#f=open('weight_file','w')
	#f.write(weight+"\t"+str(intercept)+"\n")