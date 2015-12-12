#!/usr/bin/env python

import numpy
import sys
import os
import cvxopt
import pylab as plotter
import random
import time
from operator import itemgetter


def compute_weight_intercept():
	count = 0
	sum_intercept = 0
	sum_weight_1 = sum_weight_2 =0
	#with open('weight_file') as fp:
   	#for line in fp:
	for line in sys.stdin:
	    # remove leading and trailing whitespace
	    line = line.strip()
	    count = count + 1
	    # parse the input we got from mapper.py
	    weight, intercept = line.split('\t', 1)
	    weight = weight.translate(None,'[],')
	    weight = weight.strip()
	    weight = weight.split()
	    intercept = float(intercept)

	    # convert count (currently a string) to int
	    try:
	        intercept = float(intercept)
	        sum_intercept = sum_intercept + intercept
	        sum_weight_1 = sum_weight_1 + float(weight[0])
	        sum_weight_2 = sum_weight_2 + float(weight[1])

	    except ValueError:
	        # count was not a number, so silently
	        # ignore/discard this line
	        continue

	    # this IF-switch only works because Hadoop sorts map output
	    # by key (here: word) before it is passed to the reducer
	    '''
	    if current_word == word:
	        current_count += count
	    else:
	        if current_word:
	            # write result to STDOUT
	            print '%s\t%s' % (current_word, current_count)
	        current_count = count
	        current_word = word
	    '''
	output = []
	output = output + [sum_weight_1]
	output = output + [sum_weight_2]
	sum_intercept = float(sum_intercept)/count
	return output,sum_intercept
if __name__ == "__main__":
	global_weight,global_intercept = compute_weight_intercept()
	global_weight = " ".join(map(str, global_weight))
	print '%s\t%s' %(global_weight,global_intercept)