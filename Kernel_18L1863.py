import math
import pandas as pd
import numpy as np

def dot_kf(u, v):
    result = 1
    for i in range(0,len(u)):
            result = u[i]*v[i]+result
    
    return result

def poly_kernel(d):
    def kf(u, v):
        s = dot_kf(u,v)
        s = s ** d
        return s
    return kf

def exp_kernel(s):
    def kf(u, v):
        result = 0
        for i in range(0,len(u)):
            result = u[i]-v[i]+result
        result = abs(result)
        result = -float(result)/(2*s*s)
        result = math.exp(result)
        return result
    return kf

class Perceptron(object):

    def __init__(self, kf,data,label,a):
        self.kf = kf
        self.data = data
        self.label = label
        self.weight = a

    def update(self, point, label):
        # TODO
        if (self.predict(point)==label):
            return False
        else:
            return True

    def predict(self, point):
        s = 0
        for i in range(0,len(self.data)):
            s = s + self.weight[i]*self.kf(point,self.data[i])
        if s > 0 :
            return 1
        else:
            return -1



if __name__ == '__main__':
    train = pd.read_csv('D:\Projects\MLProjects\mnist_train.csv')
    test = pd.read_csv('D:\Projects\MLProjects\mnist_test.csv')
    val_data = train.iloc[:,1:].values
    val_labs = train.iloc[:,0].values
    test_data = train.iloc[:,1:].values
    test_labs = train.iloc[:,0].values
    def kernel1():
         a = [0]*len(val_data)
         train = Perceptron(dot_kf,val_data,val_labs,a)
         L = [0]*(len(val_data)/100)
         count = 0
         loss = 0
         for t in range(0,1):
             for i in range(0,len(val_data)):
                 if train.update(val_data[i],val_labs[i]):
                     train.weight[i] = train.weight[i] + val_labs[i]
                     loss = loss + 1
                 if ((i+1)%100 ==0):
                     L[count] = float(loss)/(i+1)
                     count = count + 1
         print(L)
         
    def polykernel():
        a = [0]*len(val_data)
        count = 0
        loss = 0
        d = [1,3,5,7,10,15,20]
        L = [0]*len(d)
        for j in range(0,len(d)):
            train = Perceptron(poly_kernel(d[j]),val_data,val_labs,a)
            for i in range(0,len(val_data)):
                if train.update(val_data[i],val_labs[i]):
                    train.weight[i] = train.weight[i] + val_labs[i]
                    loss = loss + 1
            L[count] = float(loss)/1000
            count = count + 1
            loss = 0
        print(L)


    
    #implement Exponential kernel
    def expkernel(d):
        a = [0]*len(val_data)
        L = [0]*(len(val_data)/100)
        count = 0
        loss = 0
        train = Perceptron(exp_kernel(d),val_data,val_labs,a)
        for t in range(0,1):
            for i in range(0,len(val_data)):
                if train.update(val_data[i],val_labs[i]):
                    train.weight[i] = train.weight[i] + val_labs[i]
                    loss = loss + 1
                if ((i+1)%100 ==0):
                    L[count] = float(loss)/(i+1)
                    count = count + 1
        print(L)
    

    #run the test on three different kinds of kernels
    kernel1()
    polykernel()
    expkernel(5)
    expkernel(10)
