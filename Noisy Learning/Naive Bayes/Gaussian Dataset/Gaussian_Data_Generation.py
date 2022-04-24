from scipy import stats
import numpy as np
def func(rad,d):
    Xtrain = []
    ytrain = []
    Xtest = []
    ytest = []
    XTrain = []
    yTrain = []

    for i in range(700):
        mylist = []

        prob = 0.5

        result = np.random.binomial(1,prob)

        mat = np.identity((d))


        if result == 1:
            vec = np.ones((d))
            mylist = np.random.multivariate_normal(vec,mat,1)
            mylist = mylist[0]
        else:
            vec = np.ones((d))
            #vec[vec==1]=-1
            vec = -vec
            mylist = np.random.multivariate_normal(vec,mat,1)
            mylist = mylist[0]
            result = -1


        ytrain.append([result])
        Xtrain.append((mylist))


    return (Xtrain,ytrain)

