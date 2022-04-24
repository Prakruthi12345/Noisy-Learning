import math
import numpy as np
from NBdata import func


def test(data,labels,d,bound):
    
    correct = 0
    total = 0
    ones = np.ones((d))
    
    for i in range(len(data)):
        
        row = data[i]
       
        total_val = np.dot(ones,row)
        noise = np.random.uniform(-bound,bound)
        total_val += noise
        
        if total_val >= 0:
            pred = 1
        else:
            pred = -1

       
        if pred == labels[i][0]:
            correct+=1
            
        total+=1
        
    return correct/total



dimensions = [30,100]



for d in dimensions:

    data,labels = func(0.7,d)

    for bound in range(10,110,10):

        accuracy = 100*test(data,labels,d,bound)

         
