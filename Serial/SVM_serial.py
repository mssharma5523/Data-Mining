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

import numpy
import sys
import os
import cvxopt
import pylab as plotter
import random
import time

number_of_train_vectors = 100
number_of_test_vectors = 100

def kernel_function(data1,data2):
	return numpy.dot(data1,data2)

'''
generate a two dimensional training dataset
'''
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

def generate_test_data():
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
	test_X = numpy.vstack((L1,L2)) # mixing the two separate data together so that they can be trained combinedly
	test_Y = numpy.hstack((R1,R2)) # mixing the two Y's together so that they can be trained combinedly
	return test_X,test_Y

def test_data():
	train_X,train_Y = generate_train_data()
	test_X,test_Y = generate_test_data()

'''
trains the dataset and generates the seperating hyperplane
'''
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

'''
returns a list of the result of the testing in the form of 1,-1
'''
def test_on_support_vectors(weight,intercept,test_data):
	output_matrix = []
	#print test_data
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

def read_train_input():
	# input comes from STDIN (standard input)
	#train_Y = numpy.empty([1])
	#train_X = numpy.empty([1,1])
	list_y = []
	list_x = []
	with open('train_file') as fp:
   	    for line in fp:
		#for line in sys.stdin: #Commenting for time being testing pursposes
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
	start_time = time.time()
	#train_X,train_Y = generate_train_data()
	train_X,train_Y = read_train_input()
	#print train_X
	#print train_Y
	
	test_X,test_Y = generate_test_data()
	weight,intercept,supportVectors_x,supportVectors_y,supportVectors_a = svm_training(train_X,train_Y,0.3)
	print weight
	print intercept
	'''
	a = '%s\t%s' % (weight.tolist(),intercept)
	l,m = a.split('\t',1)
	print l
	l = l.translate(None,'[],')
	print l
	w = numpy.empty([1])
	l = l.strip()
	l = l.split()
	s = 0
	for element in l:
		element = float(element)
		s = s + element
	#	print element
	#	numpy.append(w,element)
	
	#weight = numpy.fromstring(l,dtype=float,sep=' ')
	#print weight
	'''
	test_data_predict = test_on_support_vectors(weight,intercept,test_X)
	count = 0
	for i in range(len(test_Y)):
		if test_data_predict[i] == test_Y[i]:
			count+=1
	print "%d predictions were correct out of %d test cases"%(count,len(test_Y))
	print("--- %s seconds ---" % (time.time() - start_time))
	plot_data(train_X[train_Y==1], train_X[train_Y==-1], supportVectors_x,supportVectors_y,supportVectors_a,weight,intercept)