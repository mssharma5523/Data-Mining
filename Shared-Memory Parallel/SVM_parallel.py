'''
Here is an implementation of a Simple SVM using the convex optimisation technique. The algorithm that we use 
is that we try to find the hyperplane which is at the maximum distance from the nearest set of points of the
two classes. If the data is linear, then we simply use the euclidean distance for the measurement, else the 
data is projected to the higher dimensions to make the properties linear. The distance is then calculated using 
the properties of Kernel matrix.
 The problem which we need to solve is -
 min 1/2x^TPx + q^Tx 
 s.t Gx<= h and Ax = b 

 As this is a convex optimisation problem, we use the cvxopt library of python which is solver for this type of eqn

 Also, during solving this problem, we have incorporated the noise component which does not affect much the eqn
 except changing the limits of 0 <= a_i<= C , where C is a constant

references - http://www.support-vector.net/icml-tutorial.pdf

'''
''''
In the parallel implementation of SVM, we have used the option of multiprocessing over multithreading. It is because the multithreading
option in python is not actually parallel as is the case with C/C++. Also as there are no I/O bound with our code, hence multiprocessing
was more viable for fastness of the code.
As we were using the cvxopt library for convex optimisation solver, we could not do much parallelism in the code for convex optimiser. 
This was a drawback for the code that we have written. We have performed the computation of funciton on parallel cores and hence, the code
performs better than the code of serial implementation.

'''
import numpy
import sys
import os
import cvxopt
import pylab as plotter
import random
import threading
from threading import Thread
import multiprocessing
from multiprocessing import Process, Manager
from multiprocessing.queues import Queue
import time

number_of_cores = 6 #This is the number of cores that you want to run your program on.. please set it for maximum benefit
number_of_train_vectors = 100
number_of_test_vectors = 100

def kernel_function(data1,data2):
	return numpy.dot(data1,data2)

'''
generate a two dimensional training dataset
'''
def generate_train_data(input_data):
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
	input_data['train_X'] = numpy.vstack((L1,L2)) # mixing the two separate data together so that they can be trained combinedly
	input_data['train_Y'] = numpy.hstack((R1,R2)) # mixing the two Y's together so that they can be trained combinedly
	#return train_X,train_Y

def generate_test_data(input_data):
	'''
	L1 = [random.normalvariate(5,3) for e in range(0,10)]
	R1 = numpy.ones(10)
	L2 = [random.normalvariate(-5,3) for e in range(0,10)]
	R2 = numpy.ones(10)*-1
	'''
	covariance = numpy.array([[2.5,2],[2,2.5]])
	L1 = numpy.random.multivariate_normal([0,3],covariance,number_of_test_vectors)
	R1 = numpy.ones(number_of_test_vectors)
	L2 = numpy.random.multivariate_normal([3,0],covariance,number_of_test_vectors)
	R2 = numpy.ones(number_of_test_vectors)*-1
	input_data['test_X'] = numpy.vstack((L1,L2)) # mixing the two separate data together so that they can be trained combinedly
	input_data['test_Y'] = numpy.hstack((R1,R2)) # mixing the two Y's together so that they can be trained combinedly
	#return test_X,test_Y

def test_data():
	train_X,train_Y = generate_train_data()
	test_X,test_Y = generate_test_data()

def compute_outer_matrix(y_train,kernel_matrix,dict_output):
	dict_output['outer_matrix'] = cvxopt.matrix(numpy.outer(y_train,y_train)*kernel_matrix) #generating the matrix yy^T*K

def compute_other_matrices(train_number,y_train,error_constant,dict_output):
	dict_output['one_vector'] = cvxopt.matrix(numpy.ones(train_number)*-1) #generate the matrix for \sum a_i = 1^TA
	dict_output['y_matrix'] = cvxopt.matrix(y_train,(1,train_number))
	dict_output['b'] = cvxopt.matrix(0.0)

	#implementing the error part and generating the matrix for 0 <= a_i<= C
	dict_output['alpha_matrix'] = cvxopt.matrix(numpy.vstack((numpy.diag(numpy.ones(train_number) * -1),numpy.identity(train_number)))) #LHS
	dict_output['c_matrix'] = cvxopt.matrix(numpy.hstack((numpy.zeros(train_number),numpy.ones(train_number)*error_constant)))  #RHS
'''
trains the dataset and generates the seperating hyperplane
'''
def svm_training(x_train,y_train,error_constant):
	'''
	The task here is to generate the matrix of the form described in the heading from the training dataset and then
	use cvxopt to get the values of lagrangian and then get W.X + b = 0 equation
	'''
	manager = Manager()
	dict_output = manager.dict()
	train_number,train_dim = x_train.shape
	kernel_matrix = numpy.zeros((train_number,train_number))
	#constructing the Kernel Matrix
	'''
	code can be parallelized here by using this function calling...
	'''
	for i in range(train_number):
		for j in range(train_number): ##print x_train[i]
			##print x_train[i]
			kernel_matrix[i][j] = kernel_function(x_train[i],x_train[j])
	#pool = multiprocessing.Pool(number_of_cores)

	################################
	process1 = Process(target=compute_outer_matrix,args=(y_train,kernel_matrix,dict_output))
	process1.start()
	process2 = Process(target=compute_other_matrices,args=(train_number,y_train,error_constant,dict_output))
	process2.start()
	process1.join()
	process2.join()
	outer_matrix = dict_output['outer_matrix']
	one_vector = dict_output['one_vector']
	y_matrix = dict_output['y_matrix']
	b = dict_output['b']
	alpha_matrix = dict_output['alpha_matrix']
	c_matrix = dict_output['c_matrix']

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
	'''
		weight and intercept evaluation can be done in parallel
	'''
	
	process1 = Process(target=calculate_weight,args=(train_dim,supportVectors_a,supportVectors_y,supportVectors_x,dict_output))
	process1.start()
	process2 = Process(target=calculate_intercept,args=(supportVectors_a,supportVectors_x,supportVectors_y,kernel_matrix,left,support,dict_output))
	process2.start()
	process1.join()
	process2.join()

	weight = dict_output['weight']
	intercept = dict_output['intercept']
	
	print "The number of support vectors found were %d out of total %d data sets"%(len(supportVectors_a),train_number)
	return weight,intercept,supportVectors_x,supportVectors_y,supportVectors_a


def calculate_weight(train_dim,supportVectors_a,supportVectors_y,supportVectors_x,dict_output):
	weight = numpy.zeros(train_dim)
	for i in range(len(supportVectors_a)):
		weight = weight + supportVectors_a[i]*supportVectors_y[i]*supportVectors_x[i]
	dict_output['weight'] = weight

def calculate_intercept(supportVectors_a,supportVectors_x,supportVectors_y,kernel_matrix,left,support,dict_output):
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
	dict_output['intercept'] = intercept

'''
returns a list of the result of the testing in the form of 1,-1
'''
def test_on_support_vectors(weight,intercept,test_data):
	output_matrix = []
	#print test_data
	'''
	can be parallelized here..:D
	'''
	for i in range(len(test_data)):	
		output = numpy.dot(weight,test_data[i])+intercept
		if output > 0:
			output_matrix.append(1)
		else :
			output_matrix.append(-1)
	return output_matrix

'''
This function is used to plot the points in a plotter using pylab- a variant of matplotlib
'''
def plot_data(train_X1, train_X2, supportVectors_x,supportVectors_y,supportVectors_a,weight,intercept):
        plotter.plot(train_X1[:,0], train_X1[:,1], "or")
        plotter.plot(train_X2[:,0], train_X2[:,1], "ok")
        plotter.scatter(supportVectors_x[:,0], supportVectors_x[:,1], s=100, c="g")

        X_first, X_second = numpy.meshgrid(numpy.linspace(-5,10), numpy.linspace(-5,10))
        test_vector = numpy.array([[first, second] for first, second in zip(numpy.ravel(X_first), numpy.ravel(X_second))])
        temp = numpy.asarray(test_on_support_vectors(weight,intercept,test_vector))
        final_output = temp.reshape(X_first.shape)
        plotter.contour(X_first, X_second, final_output, cvxopt.matrix(0.0), colors='k', origin='lower')
        plotter.contour(X_first, X_second, final_output + 1, cvxopt.matrix(0.0), colors='m',  origin='lower')
        plotter.contour(X_first, X_second, final_output - 1, cvxopt.matrix(0.0), colors='g',  origin='lower')

        plotter.show()

if __name__ == "__main__":
	##train_X,train_Y = generate_train_data()
	##test_X,test_Y = generate_test_data()
	start_time = time.time()
	'''
		generating the data parallely both the training and the test data
	'''
	manager = Manager()
	dict_output = manager.dict()
	process1 = Process(target=generate_train_data,args=(dict_output,))
	process1.start()
	process2 = Process(target=generate_test_data,args=(dict_output,))
	process2.start()
	process1.join()
	process2.join()
	#print dict_output.values()
	train_X = dict_output['train_X']
	train_Y = dict_output['train_Y']
	test_X = dict_output['test_X']
	test_Y = dict_output['test_Y']

	weight,intercept,supportVectors_x,supportVectors_y,supportVectors_a = svm_training(train_X,train_Y,0.3)
	test_data_predict = test_on_support_vectors(weight,intercept,test_X)
	count = 0
	for i in range(len(test_Y)):
		if test_data_predict[i] == test_Y[i]:
			count+=1
	print "%d predictions were correct out of %d test cases"%(count,len(test_Y))
	print("--- %s seconds ---" % (time.time() - start_time))
	plot_data(train_X[train_Y==1], train_X[train_Y==-1], supportVectors_x,supportVectors_y,supportVectors_a,weight,intercept)
