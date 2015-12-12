Libraries to install to run this code
1. numpy
2. cvxopt
2. pylab

Points:-
1. We have discussed with Sir about using the library of cvxopt for convex optimisation solver. He had agreed to its use. We used it to solve the convex optimisation problem that occurs in the SVM case. For more details refer to this tutorial:- http://www.support-vector.net/icml-tutorial.pdf
2. As of now, we have implemented only a single kernel function, that is the very common, dot product kernel matrix
3. The data set is generated randomly in two dimenional in the code itself, for clarity of classification , we have used normal distribution which can be used in separating the data. We have kept the training set overlapping and the training data size is 100. Also we run the test on 100 data sets and the results are printed in the terminal.
4. We have shown the hyper-plane separating the two classes and the points both training and some testing for better visualisation. As it is not completely linearly separable, hence the hyperplane need not be a single straight line in two dimension.

To run the code:-
1. python SVM.py

for installing the repositories of python use pip install or install manually

Also This folder has the code for the parallel implementation for SVM namely SVM_parallel.py

Also the distributed implementation is also added. There are two files mapper_svm.py and reducer_svm.py. These two files need to be run on the Hadoop architecture and the output is the weight and the intercept values.
