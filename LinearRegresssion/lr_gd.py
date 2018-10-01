# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 07:05:19 2018

@author: Lucifer
"""

import numpy as np
import matplotlib.pyplot as plt


def costFunction(points,m,b):
    cost=0
    for i in range(len(points)):
        x=points[i,0]
        y=points[i,1]
        cost+=(y-((m*x)+b))**2
    return cost/((len(points)))    
    


def stepGradientDescent(m,b,learning_rate,points):
    b_gradient=0
    m_gradient=0
    N=len(points)
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        b_gradient+=-(2/N)*(y-((m*x)+b))
        m_gradient+=-(2/N)*(x*(y-((m*x)+b)))
    new_b=b-(learning_rate*b_gradient) 
    new_m=m-(learning_rate*m_gradient)
    return [new_b,new_m]
        
        
    

def gradientRunner(starting_m,starting_b,learning_rate,points,iters):
    m=starting_m
    b=starting_b
    for i in range(iters):
        b,m=stepGradientDescent(m,b,learning_rate,np.array(points))        
    return[b,m]
    

def run():
    data=np.loadtxt('data.csv', delimiter=',')
    print ('cost is {0}'.format(costFunction(np.array(data),0,0)))
    [b,m]=gradientRunner(0,0,0.0001,data,1000)
    print('Our b is {0} and m is {1}'.format(b,m))
    print ('reduced cost is {0}'.format(costFunction(np.array(data),m,b)))
    plt.scatter(data[:,0],data[:,1])    
    x = np.linspace(data.min(), data.max(), data.max())  
    y=(m*x)+b
    plt.plot(x,y,color='red')
    plt.show()
    print((m*120)+b)
    






if __name__ == '__main__':
    run()