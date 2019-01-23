######
######  This file includes different test functions used 
######

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math

def svm_objective_function(w, features, labels, order):
    n=len(labels)
    if order==0:
        
        wx = features * w
        temp = np.multiply(labels, wx) 
        temp1 = np.maximum(np.ones(temp.shape) - temp, np.zeros(temp.shape))
        value = (1/n)*(np.sum(temp1))
        return value
    elif order==1:
        
        value = math.inf
        
        temp = np.multiply(labels, features * w) 
        labels_temp = (temp > 1 )
        new_labels=np.copy(labels)
        new_labels[labels_temp] = 0
        subgradient = - (1/n) * new_labels.T * features 

        return (value, subgradient.T)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")
    
def svm_objective_function_stochastic(w, features, labels, order, minibatch_size):
    n=len(labels)
    if order==0:
        
        wx = features * w
        temp = np.multiply(labels, wx) 
        temp1 = np.maximum(np.ones(temp.shape) - temp, np.zeros(temp.shape))
        value = (1/n)*(np.sum(temp1))
        return value
    elif order==1:
        
        value = math.inf
       
        #Random sampling for stochastic version
        random_num = np.random.randint(n, size = minibatch_size)
        labels_random = labels[random_num, :]
        features_random = features[random_num, :]
     
        #Updating Subgradient
        temp = np.multiply(labels_random, features_random * w) 
        labels_temp = (temp > 1 )
        labels_random[labels_temp] = 0   
        subgradient = - (1/minibatch_size) * labels_random.T * features_random
        return (value, subgradient.T)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")
