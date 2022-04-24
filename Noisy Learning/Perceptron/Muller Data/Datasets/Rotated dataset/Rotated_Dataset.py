import random
import numpy as np
import math

def rotate(w, a):
    v1 = np.array(np.ones(30))
    v2 = np.array( w )

    # Gram-Schmidt orthogonalization
    n1 = v1 / np.linalg.norm(v1)
    v2 = v2 - np.dot(n1,v2) * n1
    n2 = v2 / np.linalg.norm(v2)
    #creating rotation matrix for convenience
    I = np.identity(30)
    R = I + ( np.outer(n2,n1) - np.outer(n1,n2) ) * np.sin(a) + ( np.outer(n1,n1) + np.outer(n2,n2) ) * (np.cos(a)-1)
    return np.matmul( R, n1 )


def func(radius,dim):
    
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    temp_labels = []
    temp_data = []
    
    num_points = 1000
    
    for i in range (num_points):

        #Muller's method: randomly sampling from d-ball
     
      
        data_arr = np.random.rand(dim)
        norm = np.sum(data_arr**2)**(0.5)
        r = random.random()**(1.0/dim)
        data_point = r*arr/norm
        
        # rotating by pi/2
        data_point = rotate(data_point,np.pi/2)
        data_elem = []
        

        #positive or negative point with prob 0.5
        
        prob = 0.5
        rand_label = np.random.binomial(1,prob)

        #centre for positive ball is (1,1,..)
        #centre for negative ball is (-1,-1,..)
        
        for x in data_point:

            if (rand_label == 1):
                x = (x*radius) + 1

            else:
                x = (x*radius) - 1
                rand_label = -1

            data_elem.append(x)

        temp_labels.append(rand_label)
        temp_data.append(data_elem)

    train_num_points = 700

    #Training data
    
    for k in range(train_num_points):
        train_data.append(temp_data[k])
        train_labels.append(temp_labels[k])

    #Testing data
        
    for j in range(k,num_points,1):
        test_data.append(temp_data[j])
        test_labels.append(temp_labels[j])

    return (train_data,train_labels,test_data,test_labels)






