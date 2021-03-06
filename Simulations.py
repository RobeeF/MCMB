# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 20:39:49 2019

@author: Martin
"""

import os
os.chdir("G:/Documents/ENSAE/3A ENSAE/Semi and non parametric econometrics")

from functions import *
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning


#====================================================================================
# Reproducing the results of He & Hu (2002)
#====================================================================================

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


#====================================================================================
# Model with heteroskedasticity 
#====================================================================================

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
quant_fail = 0

while iteration<=samples+fails:
    print(iteration)
    np.random.seed(iteration)
    
    with warnings.catch_warnings(record=True) as w: # If Quantrreg does not converge we redraw a sample
        #model = simul_originmod_het(n=50,df=3,seed=iteration)
        model = simul_originmod_het(n=50,df=8,seed=iteration)
        Y = model[0]
        X = model[1]

        try:
            beta, IC = MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=conf_lvl, parallelize_mode='seq')
        except ValueError:
            print("Quantreg fail")
            quant_fail +=1
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



#====================================================================================
# Model with clustered heteroskedasticity and some integer variables (1)
#====================================================================================

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


#====================================================================================
# Model with clustered heteroskedasticity and some integer variables (2)
#====================================================================================

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

#====================================================================================
# Comparing the intervals of the regular and block-parallel versions of the algorithm
#====================================================================================

samples = 500
conf_lvl=0.10

mean_coverage_seq = [] # store proportion of "insiders" by iteration
mean_length_seq = [] # store mean length by iteration

mean_coverage_bp = [] # store proportion of "insiders" by iteration
mean_length_bp = [] # store mean length by iteration



for seed in range(1,samples+1):
    print(seed)
    Kn = 100
    covariances= 0.6
    p = 10
    n = 500
    mu = 0
    sigma = 1
    
    cov = np.full(p*p,covariances).reshape(p,p)
    np.fill_diagonal(cov, np.full(sigma,1))
    
    sigma = np.ones(p)
    sigma_e = 1
    coefs = np.zeros(p).reshape(-1,1)
    
    model = simul_model_multi_gaussian(n, p, mu, cov, sigma_e ,coefs, seed=None)
    Y = model[0]
    X = model[1]
    tau = 0.5

    beta_seq, IC_seq = MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=conf_lvl, parallelize_mode='seq', extension='A')
    beta_bp, IC_bp = MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=conf_lvl, parallelize_mode='bp', extension='A')
        
    tag_seq=0
    tag_bp=0
    
    length_seq=[]
    length_bp=[]

    for i in range(p):
        length_seq.append(IC_seq[i][1] - IC_seq[i][0])
        length_bp.append(IC_bp[i][1] - IC_bp[i][0])
        
        if 0 >= IC_seq[i][0] and IC_seq[i][1] >= 0:
            tag_seq+=1 
        if 0 >= IC_bp[i][0] and IC_bp[i][1] >= 0 :
            tag_bp+=1
                    
    mean_coverage_seq.append(tag_seq/p)
    mean_length_seq.append(np.mean(length_seq))
    mean_coverage_bp.append(tag_bp/p)
    mean_length_bp.append(np.mean(length_bp))
            
print('Mean coverage seq %s ' % np.mean(mean_coverage_seq))
print('Mean length seq %s ' % np.mean(mean_length_seq))
print('Variance length seq %s ' % np.std(mean_length_seq))

print('Mean coverage bp %s ' % np.mean(mean_coverage_bp))
print('Mean length bp %s ' % np.mean(mean_length_bp))
print('Variance length bp %s ' % np.std(mean_length_bp))

