# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 06:50:49 2018

@author: Lucifer
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


df = pd.read_csv('iris-data.csv')

df = df.dropna(subset=['petal_width_cm'])


#Sinple Logistic regression

final_df = df[df['class'] != 'Iris-virginica']


final_df['class'].replace(["Iris-setosa","Iris-versicolor","Iris-setossa","versicolor"], [1,0,1,0], inplace=True)

inp_df = final_df.drop(final_df.columns[[4]], axis=1)
out_df = final_df.drop(final_df.columns[[0,1,2,3]], axis=1)

scaler =StandardScaler()
inp_df = scaler.fit_transform(inp_df)


X_train, X_test, y_train, y_test = train_test_split(inp_df, out_df, test_size=0.2, random_state=42)


X_tr_arr = X_train
X_ts_arr = X_test
y_tr_arr = y_train.as_matrix()
y_ts_arr = y_test.as_matrix()


#sigmoid function
def sigmoid_calculation(x):    
   final_result = 1/(1+np.exp(-x))
   return final_result

#weight initialization
def weight_initialization(n_features):
    w=np.zeros((1,n_features))
    b=0
    return w,b

def model_optimize(w,b,X,Y):
    m=X.shape[0]
    result=sigmoid_calculation(np.dot(w,X.T)+b)
    Y_T=Y.T    
    cost=(-1/m)*(np.sum((Y_T*np.log(result))+((1-Y_T)*np.log(1-result))))    
    dw=(1/m)*(np.dot(X.T,(result-Y.T).T))
    db=(1/m)*np.sum(result-Y.T)
    
    return dw,db,cost

def model_predict(w,b,X,Y,learning_rate,no_of_iterations):
    costs=[]
    for i in range(no_of_iterations):
        dw,db,cost=model_optimize(w,b,X,Y)
        w=w-(learning_rate*(dw.T))
        b=b-(learning_rate*db)
        costs=cost
    return w,b,costs      

def indic(data):
    #alternatively you can calulate any other indicators
    max = np.max(data, axis=1)
    min = np.min(data, axis=1)
    return max, min  

w,b=weight_initialization(X_tr_arr.shape[1])

w,b,cost=model_predict(w,b,X_tr_arr,y_tr_arr,learning_rate=0.0001,no_of_iterations=4500)
 
print(sigmoid_calculation(np.dot(w,X_ts_arr.T)+b))
print(y_ts_arr)


        
    
    



    
    
    
    
    




    