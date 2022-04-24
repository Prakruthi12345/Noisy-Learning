import math
import numpy as np
from NBdata import func


def test(data,labels,d,first,second,std):
    
    correct = 0
    total = 0
    c= 0.1
    accum=0
    ones = np.ones((d))
    
    for i in range(len(data)):

        row = data[i]
       
        for j in range(first):
            
            total_val=0
            
            for k in range(second):
                
                total_val += np.dot(ones[(second*j)+k],row[(second*j)+k])

            noise = np.random.normal(0,std/first,1)
            total_val+=noise[0]

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
        
        if pred == labels[i][0]:
            correct+=1
        total+=1

    return correct/total



dimensions = [30,100]
thirty = [(2,15),(3,10),(5,6),(6,5),(10,3),(15,2)]
hundred = [(2,50),(4,25),(5,20),(10,10),(20,5),(25,4),(50,2)]



for d in dimensions:



    data,labels = func(0.7,d)

    if d == 30:
        pairlist = thirty
    if d == 100:
        pairlist = hundred
    


    for elem in pairlist:

        (first,second)=elem


        for std in range(0,110,10):

            accuracy = 100*test(data,labels,d,first,second,std)
            print(accuracy)

   
