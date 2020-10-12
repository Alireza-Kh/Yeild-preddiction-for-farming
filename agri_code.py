# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:53:05 2020

@author: Alireza Kheradmand

Subject: AltaML Coding Interview dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model
from scipy.stats import zscore


address = 'G:\My Drive\Personal\LeetCode\AltaML Interview\dataset.csv'

# Import dataset as Pandas function
def readdata(address):
    
    file = pd.read_csv (address)

    X = file.drop(['id','yield'], axis = 1)

    Y = file['yield']

    return X,Y

# Categorical variable conversion function
def CategoricalVar(U,category):

   A = np.zeros (shape=(len(U), len(category)))
   Ap = pd.DataFrame(A, columns=category)
    
   for i in range(0,len(U)):

      for k in range(0,len(category)):
         
          if category[k] in U[i]:

            Ap.iloc[i,k] = 1

   return(Ap) 

# preprocessing function for missing variable treating, mean substitution and missing removal
def missing(X):

    for i in range(0,len(X)):
        
        for j in range(0, len(X.columns)):
            
            if pd.isnull(X.iloc[i,j]):
                
                if i == 0:
                    
                    X.iloc[i,j] = X.iloc[i+1,j]
                    
                if i == (len(X)-1):
                    
                    X.iloc[i,j] = X.iloc[i-1,j]
                    
                else:
                    
                    X.iloc[i,j] = 0.5*( X.iloc[i-1,j] + X.iloc[i+1,j] )
                    
    return (X)

# Outlier detection and replacement function

def Outlier(X):

  Z_scores = np.abs(zscore(X))

  for i in range(0,len(X)):
    
      for j in range(0,len(X.columns)):
        
          if Z_scores[i,j] > 3:
            
              X.iloc[i,j] =  0.5*( X.iloc[i-1,j] + X.iloc[i+1,j] )
            
  return(X) 

# Input ata normalization function
def Norm(X):
  
     for col in ['water','uv', 'area','pesticides']:
    
        X[col] = (X[col] - X[col].mean()) / X[col].std()
    
     return(X)

# Single partition model function
def Modelfit(X,Y,test_fraction):
    
     [X_train, X_test, Y_train, Y_test] = model_selection.train_test_split(X, Y, test_size=test_fraction, random_state=41)
     Lin_model = linear_model.LinearRegression().fit(X_train, Y_train)
     Y_train_pred = Lin_model.predict(X_train)
     R2_train = Lin_model.score(X_train, Y_train)

     Y_test_pred = Lin_model.predict(X_test)
     R2_test = Lin_model.score(X_test, Y_test)
     R2 = [R2_train,R2_test]

     return(Y_train_pred, Y_train, Y_test_pred,Y_test, R2)

[X,Y] = readdata(address)

U = X['categories']
category = ['a','b','c','d']

Ap =CategoricalVar(U,category)

X = X.drop(['categories'], axis=1)    
X = pd.concat([X,Ap], axis=1)  

X = missing(X)
T = pd.concat([X,Y], axis=1)
T = T.dropna(axis=0)

X = T.iloc[:,0:len(T.columns)-1]
Y = T.iloc[:,len(T.columns)-1]

X = Outlier(X)

Correlation_X = X.corr()

Correlation_Y = T.corr()          

X = Norm(X)

test_fraction = 0.35


[Y_train_pred, Y_train, Y_test_pred,Y_test, R2S] = Modelfit(X,Y,test_fraction)

plt.figure()
plt.scatter(Y_train, Y_train_pred, label='Training')
plt.scatter(Y_test, Y_test_pred, label='Testing')
plt.legend()
plt.xlabel('Yield (Real Value)')
plt.ylabel('Yield (Prediction)')
plt.text(100,15, f"Training R2 = {str(np.round(R2S[0], decimals =4))}", fontsize=10)
plt.text(100,5, f"Testing R2 = {str(np.round(R2S[1], decimals =4))}", fontsize =10)
plt.show()

