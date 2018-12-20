# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:09:39 2018

@author: Bruno
"""

#==============================================================================
# Semi and Non Parametrics Econometrics - Project
#==============================================================================

# Initialisation
#import os
#os.chdir(r"C:\Users\Bruno\Documents\Cours\ENSAE\Semi and Non Parametric Econometrics\Projet")

# Packages
import numpy as np
import numpy.random as rd
#import matplotlib.pyplot as plt
from wquantiles import quantile_1D
from sklearn.utils import resample
from statsmodels.regression.quantile_regression import QuantReg
import scipy.linalg as sci
import scipy.stats
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import copy
import multiprocessing

#==============================================================================
# Simulation of a dataset
#==============================================================================

### Linear model

# Residuals
mu = 0
sigma = 1
n = 1000
epsilon = rd.normal(mu, sigma, n)

#plt.hist(epsilon, bins=100)

# Covariates
mean = [5, -20, 3, 4]
cov = [[10, 0, 0, 0],
       [0, 100, 0, 0],
       [0, 0, 10, 0],
       [0, 0, 0, 5]]  # diagonal covariance
X = rd.multivariate_normal(mean, cov, n)


# Model y = 2 + beta*x + epsilon
beta = [1, 2, 3, -2]
b0 = 0
y = b0 + np.dot(X, beta) + epsilon
Y = np.reshape(y, (-1,1))


### Model 1
def simul_model1(n, with_cst=False ,seed=None): # Model 1 of Kocherginsky (2002)
    """ Simulate y= 1 + b1*x1 + b2*x2 + b3*x3 + e, with x1 following a standard normal, 
    x3 is a U[0,1], x2=x1 + x3 + z, with z and e following a standard normal"""
    rnd = np.random.RandomState(seed)
    x1,e,z = rnd.normal(size=n).reshape((-1,1)), rnd.normal(size=n).reshape((-1,1)), rnd.normal(size=n).reshape((-1,1)) # generate 3 n-sample of std normal 
    x3 = rnd.rand(n).reshape((-1,1))
    x2 = x1+x3+z
    X = np.hstack([np.ones((n,1)), x1, x2, x3]) if with_cst else np.hstack([x1, x2, x3])
    coefs_ = np.ones((X.shape[1],1)).reshape(-1,1)
    return (np.dot(X,coefs_) + e,X)


def simul_indep_multi_gaussian(size, sigma, mean ,seed=None):
    """ Simulate a multivariate gaussian matrix of size "size" 
    independant gaussian vector of variance sigma (and null covariance by construction)"""
    mean = (1, 2)
    cov = [[1, 0], [0, 1]]
    rnd = np.random.RandomState(seed)
    x = rnd.multivariate_normal(mean, cov, (3, 3))
    return x

#==============================================================================
# MCMB-A Algorithm
#==============================================================================

def compute_A(X): # Anciennement "standardisation"
    '''
    X: array-like dataset
    Compute A = (X'X)^-(1/2) given in Kocherginsky & al. (2007) p1261 eq. (6)
    '''
    # Standardization of input array
    X_t = np.ndarray.transpose(X)
    XX = np.dot(X_t,X) #X'X
    XX = sci.sqrtm(XX) # (X'X)^(1/2)
    XX = np.linalg.matrix_power(XX, -1) # (X'X)^-(1/2)
        
    return XX


def psi(x, tau):
    '''
    Derivative of the check function
    
    x: scalar
    tau: real between 0 and 1
    '''
    return tau*int(x >= 0) + (tau-1)*int(x < 0)


def residuals(Y, X, beta):
    '''
    Computes residuals for a linear model
    Y: dependent variable, array
    X: regressors, array
    beta: parameter, list
    '''
    Y = Y.reshape(-1,1)
    return Y-np.dot(X,beta).reshape(-1,1)
    
    
def X_to_Z(X, Y, beta, tau):
    '''
    Transforms a series of x in a series of z following the formula
    z_i = psi(residual_i)*x_i - z_hat
    
    X: array-like object
    Y: array-like object
    beta: list
    tau: scalar between 0 and 1
    '''
    
    # Computation of residuals
    R = residuals(Y, X, beta)
    
    # Computation of z_hat
    vfunct = np.vectorize(psi, excluded=['tau'])
    psi_r = vfunct(R, tau)
    X_psi = np.multiply(X, np.reshape(psi_r,(-1,1)))
    z_hat = np.mean(X_psi, 0)
    
    #Computation of each z_i
    Z = X_psi - z_hat
    
    return Z


def weighted_quantile(X, Y, Z, beta, j, tau, seed):
    '''
    Weighted quantile of Z, as solution of (3.4)
    '''
    # Draw a bootstrapped sample
    Z_boot = resample(Z, random_state=seed)
            
    #Take the j-th columns
    Z_j = Z_boot[:,j]
    c_star = Z_j.sum()
    
    # Defining Z

    beta_star = np.concatenate((beta[:j],beta[j+1:]))
    X_star = np.hstack([X[:,:j], X[:,j+1:]])
    X_j = X[:,j]
    Y_star = residuals(Y, X_star, beta_star)    
    # Adding the n+1th row to Y_star and X_j
    Y_star = np.append(Y_star, 10**15)
    X_j = np.append(X_j, -c_star/tau)
    
    Z_star = np.divide(Y_star, X_j)
    
    # Tau_star
    abs_X_j = abs(X_j)
    tau_star = 0.5 + (tau-0.5)*sum(X_j)/sum(abs_X_j)
    
    # Normalization of weights (sum up to 1)
    S = sum(abs_X_j)
    abs_X_j = abs_X_j/S
    
    # Sorting Z in ascending order
    abs_X_j = np.reshape(abs_X_j, (-1,1))
    Z_star = np.reshape(Z_star, (-1,1))

    
    return quantile_1D(np.reshape(Z_star, -1), np.reshape(abs_X_j, -1), tau_star)
    

def MCMB(Y, X, tau, size=50, extension=None, alpha=0.05 ,seed=None, verbose=False, return_chain=False, sample_spacing=1, parallelize_mode='seq'):
    '''
    MCMB algorithm
    Y: dependant variable 1-d numpy.ndarray
    X: Covariates (n,p) numpy.ndarray
    max-iter: length of the Markov Chain to generate
    extension: Which extension of the MCMB algorithm to use: A, B or AB
    alpha: degree of confidence for which the intervals are returned
    seed: Seed used to have reproductible results
    verbose: Set to True to display the computation details. Only one level of verbose
    sample_spacing: the frequency at which the betas are sampled: a large sample_spacing prevents from autocorrelations
    parallelize_mode: Type of parallelization the computation: p for parallel (all the betas are updated in parallel), bp for block parallel
        (n_jobs parallel betas are updated simultaneously), seq: as in Kocherginsky & al. the betas are updated sequentially.
    -----------------------------------
    returns: the initial estimate of the Betas and the CIs computed if return_chain==False the beta chain otherwise
    '''
    n_cores = multiprocessing.cpu_count()

    if extension=='A':
        A = compute_A(X)
        X = np.dot(X, A) # Normalisation
    
    # Estimation of beta_hat
    mod = QuantReg(Y, X)
    res = mod.fit(q=tau)
    beta_hat = res.params
    
    
    #Initialisation of parameters
    p = len(beta_hat)
    beta = beta_hat.copy()
    Beta = []
    i = 0

    Z = X_to_Z(X, Y, beta_hat, tau)
    vec_wq = np.vectorize(weighted_quantile, excluded=['X','Y','Z','beta','tau','seed'])

    remaining_iter = size*sample_spacing

    while remaining_iter>0:
        if parallelize_mode=='seq': # Same updating than in Kocherginsky & al.
            for j in range(p):            
                beta_j =  weighted_quantile(X, Y, Z, beta, j, tau, seed)
                beta = np.concatenate((beta[:j],[beta_j],beta[j+1:]))
                
        elif parallelize_mode=='p': # All the betas_j are updated at the sime time.
            beta = vec_wq(j=np.arange(p),beta=beta, X=X, Y=Y, Z=Z, tau=tau, seed=seed)
            
        else: # n_cores betas_j are updated at each iteration of the loop 
            for k in range(1,int(np.ceil(p/n_cores)+1)):
                min_index = (k-1)*n_cores
                max_index = min(k*n_cores,p)
                beta = np.concatenate((beta[0:min_index],vec_wq(j=np.arange(min_index,max_index),
                                       beta=beta, X=X, Y=Y, Z=Z, tau=tau, seed=seed),beta[max_index:]))
        
        # Each sample_spacing iterations, we sample the betas
        if remaining_iter%sample_spacing == 0:
            Beta.append(copy.deepcopy(beta))
        
        i +=1
        remaining_iter-=1
        if verbose:
            print('Iteration ' + str(i) + ' reussie !')


    Beta = [np.dot(np.array(Beta[i]),A).tolist() for i in range(len(Beta))] if extension=='A' else Beta
    beta_hat= np.dot(beta_hat,A) if extension=='A' else beta_hat
    
    # Covariance matrix
    Sigma = np.cov(np.array(Beta), rowvar=False)
    # Compute the Confidence Intervals
    CI =[]
    CI = [[beta_hat[i]-scipy.stats.norm.ppf(1-(alpha/2))*np.sqrt(Sigma[i,i]), 
       beta_hat[i]+scipy.stats.norm.ppf(1-(alpha/2))*np.sqrt(Sigma[i,i])] for i in range(p)]
    
    return Beta if return_chain else (beta_hat, CI) 

#==============================================================================
# Test
#==============================================================================
Kn = 50
seed = 132

model1 = simul_model1(n, seed=seed)
Y = model1[0]
X = model1[1]
tau = 0.5

# MCMB
beta, IC = MCMB(Y=Y, X=X, tau=tau, size=Kn,seed=seed, alpha=0.05, parallelize_mode='p')
print(beta)
IC

#==============================================================================
# Evaluation de la qualité
#==============================================================================

####### Autocorrélations
p = X.shape[1]
# For the classic MCMB
chain = MCMB(Y=Y, X=X, tau=tau, size=Kn,seed=seed, alpha=0.05, return_chain=True)
betas_chains = [[chain[i][j] for i in range(Kn)] for j in range(p)] # Get the p betas chains generated by the MCMB algorithm 

plot_acf(betas_chains[2], lags=20)
plot_pacf(betas_chains[2], lags=20)

# For the  MCMB-A
chain_A = MCMB(Y=Y, X=X, tau=tau, size=Kn,seed=seed, alpha=0.05, return_chain=True, extension='A')
betas_chains_A = [[chain_A[i][j] for i in range(Kn)] for j in range(p)] # Same with A extension

plot_acf(betas_chains_A[2], lags=20) # Decrease without suppressing the autocorrelations
plot_pacf(betas_chains_A[2], lags=20)

# For the classic MCMB with the sample-spacing = 7
chain_7s = MCMB(Y=Y, X=X, tau=tau, size=Kn,seed=seed, alpha=0.05, return_chain=True, sample_spacing=7)
betas_chains_7s = [[chain_7s[i][j] for i in range(Kn)] for j in range(p)] # Get the p betas chains generated by the MCMB algorithm with sample_spacing 

plot_acf(betas_chains_7s[2], lags=20) # Kill the autocorrelation ! (with a huge increase in computation time)
plot_pacf(betas_chains_7s[0], lags=20)

# For the parallelized version of MCMB
chain_p = MCMB(Y=Y, X=X, tau=tau, size=Kn,seed=seed, alpha=0.05, return_chain=True, parallelize_mode='p')
betas_chains_p = [[chain_p[i][j] for i in range(Kn)] for j in range(p)] 

plot_acf(betas_chains_p[2], lags=20) # The betas are more correlated than in the MCMB classic version
plot_pacf(betas_chains_p[2], lags=20)

# For the parallelized version of MCMB-A
chain_A_p = MCMB(Y=Y, X=X, tau=tau, size=Kn,seed=seed, alpha=0.05, return_chain=True, extension='A',parallelize_mode='p')
betas_chains_A_p = [[chain_A_p[i][j] for i in range(Kn)] for j in range(p)] # Get the p betas chains generated by the MCMB algorithm 

plot_acf(betas_chains_A_p[0], lags=20) # The betas are more correlated than in the MCMB-A classic version
plot_pacf(betas_chains_A_p[0], lags=20)

###### Need a burn in ? Look a the first 40 iterations to identify at which iteration the betas converge to a stable value
# Classic MCMB
plot_chain = pd.Series(betas_chains[1]).iloc[0:40] 
plot_chain.plot()

#MCMB-A
plot_chain_A = pd.Series(betas_chains_A[1]).iloc[0:40] 
plot_chain_A.plot() # No real change on the burn-in period

# MCMB-sp7
plot_chain_sp7 = pd.Series(betas_chains_7s[1]).iloc[0:40] 
plot_chain_sp7.plot() # Logically a lower burn in period (= burn_in/sample_spacing = burn_in/7)

# MCMB-p
plot_chain_p = pd.Series(betas_chains_p[1]).iloc[0:40] 
plot_chain_p.plot() # Need a longer burn-in due to the higher autocorrelation than the classic MCMB

# MCMB-pa
plot_chain_pa = pd.Series(betas_chains_A_p[1]).iloc[0:40] 
plot_chain_pa.plot()


