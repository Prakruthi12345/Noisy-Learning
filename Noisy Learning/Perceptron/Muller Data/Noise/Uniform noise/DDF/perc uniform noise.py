import random
import os
import csv
import numpy as np
from numpy.linalg import norm
import math
from data import func


def perceptron_pred(x, w):

    total = np.dot(w,x)

    if (total < 0):
        return -1
    else:
        return 1


def train_perceptron(X, Y, w, is_norm):

    i = 0
    
    for row in X:
        
        pred = perceptron_pred(row,w)
        
        if (pred != Y[i]):
            w = w + (Y[i]*row)
            
        if (is_norm == 0):
            w_norm = norm(w,2)
            if (w_norm != 0):
                w = w/w_norm
                
        i+=1
        
    return w


def test_perceptron_no_noise(X, Y, w):
    
    i = 0
    error = 0
    total = 0
    
    for row in X:
                
        pred = perceptron_pred(row,w)

            
        if (pred != Y[i]):
            error += 1
        total += 1
        
        i+= 1
        
    return (total-error)/total



def test_perceptron_noise(X, Y, w, bound,first,second):
     
    i = 0
    acc_error = 0
    acc_total_sum = 0
    accum =  0
    c = 0.1
 
    for row in X:
        
        total_val=0
        
        for j in range(first):
            
            for k in range(second):

                total_val += (w[0][second*j+k]*row[(second*j)+k])
            noise = np.random.uniform(-bound/first,bound/first)
            total_val += noise

              
                
            if (total_val >= 0):
                accum+= 1

            if (total_val < 0):
                accum-=1
                    
            total_val = 0

        if (accum > c*int((first**0.5))):
            pred = 1

        elif (accum < -c*int((first*(0.5)))):
            pred = -1

        else:
            result = np.random.binomial(1,0.5)

            if result == 1:
                pred = 1

            else:
                pred = -1
                    
        accum = 0

        if (pred != Y[i]):
            acc_error += 1

        acc_total_sum += 1
        i+= 1
        
    return (acc_total_sum-acc_error)/acc_total_sum



dim_list=  [30,100]
thirty = [(3,10),(5,6),(6,5),(10,3)]
hundred = [(2,50),(10,10),(20,5),(4,25)]


 
for dim in dim_list:

    train_data,train_labels,test_data,test_labels = func(0.7,dim)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    if dim == 30:
        pairlist = thirty
    if dim == 100:
        pairlist = hundred
     

    for first, second in pairlist:

        
        for bound in range(2,22,2):

            acc= 0

            for i in range(100):
                
                w = np.zeros((1,train_data.shape[1]))
                
                for j in range(10):

                    w = train_perceptron(train_data,train_labels,w,0)

                
                accuracy= test_perceptron_noise(XTest, yTest, w,bound,first,second)

                acc+=((accuracy)*100)

            acc = acc/100
            
        
