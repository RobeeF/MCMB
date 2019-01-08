# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 20:39:49 2019

@author: Martin
"""

import os
os.chdir("C:/Users/robin/Documents/GitHub/MCMB")

from functions import *
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning


### Paper He & Hu 2000 setting

conf_lvl = 0.10 # coverage for a 90 % confidence interval averaged over the three slope parameters
Kn = 1000 # chain length
tau = 0.5   # percentile
#tau = 0.2
#tau = 0.8

# MCMB interval coverage and interval length
    
mean_coverage = [] # store proportion of "insiders" by iteration
mean_length = [] # store mean length by iteration
exploding = 0

samples = 500   # choose number of samples
#samples = 12
beta_exploding = []
IC_exploding = []
chain_exploding = []
iteration = 1
fails=0

while iteration<=samples+fails:
    print(iteration)
    np.random.seed(iteration)
    
    with warnings.catch_warnings(record=True) as w: # If Quantrreg does not converge we redraw a sample
        model = simul_originmod(n=50,df=3,seed=iteration)
        #model = simul_originmod(n=50,df=8, seed=iteration)
        Y = model[0]
        X = model[1]

        beta, IC = MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=conf_lvl, parallelize_mode='seq')
        if any([issubclass(warn.category, IterationLimitWarning)for warn  in w]):
            print('Fail')
            fails+=1
            iteration+=1
            continue # go on to the next iteration
   
    tag=0
    length=[]
    check_convergence = 0
    
    if any([any(np.abs(elem)>3) for elem  in IC]): # If the borns of one interval is too big then we treats the intervals as exploding
        chain_exploding.append(MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=conf_lvl, parallelize_mode='seq', return_chain=True))
        beta_exploding.append(beta)
        exploding +=1
        IC_exploding.append(IC)
        
    else: # Else we count the number of intervals among the p intervals that includes the right values 
        for i in IC:
            length.append(i[1] - i[0])
            if 0 >= i[0] and i[1] >= 0 :
                tag += 1 
        mean_coverage.append(np.divide(tag,len(beta)))
        mean_length.append(np.divide(length,len(beta)))
        
    iteration+=1
    
print('Mean coverage', np.mean(mean_coverage)*100)
print('Mean length ', np.mean(mean_length))
print('Percentage of exploding bounds', np.divide(exploding,samples)*100)


### Model with heteroskedasticity

np.random.seed(1)
b_0 = 6      # true intercept
b_1 = 0.1    # true slope
coefs = np.array([b_0,b_1])
  
x = np.arange(0,100)        # independent variable
sig = 0.1 + 0.05*x          # non-constant variance                           
e = np.random.normal(loc = 0, scale = sig, size=100) # normal random error with non-constant variance
X = np.transpose(np.ndarray((2,100), buffer=np.array([np.ones(100), x]),dtype=float))   
Y = np.dot(X,coefs) + e

plt.scatter(x,Y)

# MCMB one try
Kn = 1000
seed = 2043
tau = 0.5
p = X.shape[1]

beta, IC = MCMB(Y=Y, X=X, tau=tau, size=Kn,seed=seed, alpha=0.05, parallelize_mode='seq', extension='A')
IC

# Coverage
samples = 200
#samples = 500

store = []
for seed in range(1,samples+1):
    np.random.seed(seed)
    x = np.arange(0,100)        # independent variable
    sig = 0.1 + 0.05*x           # non-constant variance                           
    e = np.random.normal(loc = 0, scale = sig, size=100) # normal random error with non-constant variance
    X = np.transpose(np.ndarray((2,100), buffer=np.array([np.ones(100), x]),dtype=float))   
    Y = np.dot(X,coefs) + e
    
    beta, IC = MCMB(Y=Y, X=X, tau=tau, size=Kn,seed=seed, alpha=conf_lvl, parallelize_mode='seq', extension='A')
    
    tag=0
    for i in IC:
        if i==IC[0]:
            if 6 >= i[0] and i[1] >= 6 :
                tag += 1 
        else:
            if 0.1 >= i[0] and i[1] >= 0.1 :
                tag += 1
    store.append(np.divide(tag,len(beta)))
    
print('Mean coverage %s ' % np.mean(store))


### Model with clustered heteroskedasticity and some integer variables
n = 100

# Population parameter
beta = np.array([1,0.5,-2,3])

# Variables
X_1 = np.random.randint(-10, 10, n)
X_2 = np.random.randint(-100, 100, n) / 100
X_3 = np.random.normal(1, size=n)
X = np.transpose(np.ndarray((4,n), buffer=np.array([np.ones(n),X_1, X_2, X_3]),dtype=float))

epsilon = np.random.normal(1, scale = 1 + np.concatenate([np.repeat(0.05,n/4),np.repeat(0.1,n/4),np.repeat(0.2,n/4),np.repeat(0.8,n/4)]), size=n)

plt.plot(epsilon)

Y = np.dot(X,beta) + epsilon

plt.plot(Y)
plt.scatter(X_1, Y)
plt.scatter(X_2, Y)
plt.scatter(X_3, Y)

# MCMB one try
Kn = 1000
seed = 2043
tau = 0.5
p = X.shape[1]

beta, IC = MCMB(Y=Y, X=X, tau=tau, size=Kn,seed=seed, alpha=0.05, parallelize_mode='p')
IC

# Coverage
samples = 50
#samples = 500

store = []
for seed in range(1,samples+1):
    np.random.seed(seed)
    X_1 = np.random.randint(-10, 10, n)
    X_2 = np.random.randint(-100, 100, n) / 100
    X_3 = np.random.normal(1, size=n)
    X = np.transpose(np.ndarray((4,n), buffer=np.array([np.ones(n),X_1, X_2, X_3]),dtype=float))

    epsilon = np.random.normal(1, scale = 1 + np.concatenate([np.repeat(0.05,n/4),np.repeat(0.1,n/4),np.repeat(0.2,n/4),np.repeat(0.8,n/4)]), size=n)
    Y = np.dot(X,beta) + epsilon
    
    beta, IC = MCMB(Y=Y, X=X, tau=tau, size=Kn,seed=seed, alpha=conf_lvl, parallelize_mode='seq', extension='A')
    
    tag=0
    for i in IC:
        if i==IC[0]:
            if 1 >= i[0] and i[1] >= 1 :
                tag += 1
        elif i==IC[1]:
            if 0.5 >= i[0] and i[1] >= 0.5 :
                tag += 1
        elif i==IC[2]:
            if -2 >= i[0] and i[1] >= -2 :
                tag += 1
        else:
            if 3 >= i[0] and i[1] >= 3 :
                tag += 1
    store.append(np.divide(tag,len(beta)))
    
print('Mean coverage %s ' % np.mean(store))


### Model with varying parameters

n = 1000
# Population parameter
beta = np.arange([0,3])


### Minimum Lq estimator ?


## Varie n, Kn et p