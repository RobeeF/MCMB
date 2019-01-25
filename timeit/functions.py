# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:09:39 2018
@author: Bruno
"""

#==============================================================================
# Semi and Non Parametrics Econometrics - Project
#==============================================================================

# Initialisation

# Packages
import numpy as np
import matplotlib.pyplot as plt
from wquantiles import quantile_1D
from sklearn.utils import resample
from statsmodels.regression.quantile_regression import QuantReg
import scipy.linalg as sci
import scipy.stats
import copy
import multiprocessing
import pandas as pd
from numba import jit, prange

#==============================================================================
# MCMB-A Algorithm
#==============================================================================

def psi(x, tau):
    '''
    Derivative of the check function
    
    X (ndarray): Covariates (n,p) numpy.ndarray
    tau (float): The quantile for the quantile regression
    -------------------------------------------------------------
    returns (array-like): The derivative of the check function evaluated in X and tau
    '''
    return tau*int(x >= 0) + (tau-1)*int(x < 0)


def residuals(Y, X, beta):
    '''
    Computes residuals for a linear model
    
    Y (ndarray): dependant variable 1-d numpy.ndarray
    X (ndarray): Covariates (n,p) numpy.ndarray
    beta (array-like): The estimates of the coefficients
    --------------------------------------------------------
    returns e (array-like): The residuals of the estimated model
    '''
    Y = Y.reshape(-1,1)
    return Y-np.dot(X,beta).reshape(-1,1)
    
    
def X_to_Z(X, Y, beta, tau):
    '''
    Transforms a series of x in a series of z following the formula
    z_i = psi(residual_i)*x_i - z_hat
    
    Y (ndarray): dependant variable 1-d numpy.ndarray
    X (ndarray): Covariates (n,p) numpy.ndarray
    beta (array-like): The estimates of the coefficients
    tau (float): The quantile for the quantile regression
    --------------------------------------------------------
    returns Z (ndarray): The empirical counterpart of the first order condition of the quantile regression
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


def weighted_quantile(X, Y, Z, beta, j, tau):
    '''
    Update n_cores components of beta at each loop iteration
    
    Y (ndarray): dependant variable 1-d numpy.ndarray
    X (ndarray): Covariates (n,p) numpy.ndarray
    Z (ndarray): Empirical counterpart of the first order condition
    beta (array-like): The estimates of the coefficients
    j (integer): The coordinate of the coefficient to update
    tau (float): The quantile for the quantile regression
    ----------------------------------------------------------
    returns (array-like): The weighted quantile of Z to update beta_j, as solution of (3.4)
    '''
    # Draw a bootstrapped sample
    Z_boot = resample(Z)
            
    #Take the j-th columns
    Z_j = Z_boot[:,j]
    c_star = Z_j.sum()
    
    # Defining Z

    beta_star = np.concatenate((beta[:j],beta[j+1:]))
    X_star = np.hstack([X[:,:j], X[:,j+1:]])
    X_j = X[:,j]
    Y_star = residuals(Y, X_star, beta_star)    
    # Adding the n+1th row to Y_star and X_j
    Y_star = np.append(Y_star, 3000)
    X_j = np.append(X_j, -c_star/tau)
    Z_star = np.divide(Y_star, X_j)
    
    # Tau_star
    abs_X_j = abs(X_j)
    tau_star = 0.5 + (tau-0.5)*sum(X_j)/sum(abs_X_j)
    
    # Normalization of weights (sum up to 1)
    S = sum(abs_X_j)
    abs_X_j = abs_X_j/S

    return quantile_1D(np.reshape(Z_star, -1), np.reshape(abs_X_j, -1), tau_star)

@jit(parallel=True, nogil=True)
def beta_update_numba(p, beta, X, Y, Z, tau, n_cores):
    ''' Update n_cores components of beta at each loop iteration
    Y (ndarray): dependant variable 1-d numpy.ndarray
    X (ndarray): Covariates (n,p) numpy.ndarray
    Z (ndarray): Empirical counterpart of the first order condition
    n_cores (integer): The number of cores of the computer
    tau (float): The quantile for the quantile regression

    --------------------------------------------------------------------------
    returns beta (array-like): the updated betas
    '''
    for k in np.arange(1,int(np.ceil(p/n_cores)+1)):
        new_beta = [] 
        min_index = (k-1)*n_cores
        max_index = min(k*n_cores,p)
        
        for idx in prange((k-1)*n_cores, min(k*n_cores,p)):
            new_beta.append(weighted_quantile(X, Y, Z, beta, idx, tau))
        
        beta = np.concatenate((beta[0:min_index],np.array(new_beta),beta[max_index:]))
    return np.array(beta)



def seq_update(p, beta, X, Y, Z, tau):
    ''' Update sequentially the components of beta at each loop iteration
    Y (ndarray): dependant variable 1-d numpy.ndarray
    X (ndarray): Covariates (n,p) numpy.ndarray
    Z (ndarray): Empirical counterpart of the first order condition
    tau (float): The quantile for the quantile regression

    --------------------------------------------------------------------------
    returns beta (array-like): the updated betas
    '''
    for j in range(p):            
        beta_j =  weighted_quantile(X, Y, Z, beta, j, tau)
        beta = np.concatenate((beta[:j],[beta_j],beta[j+1:]))
    return beta


#=================================================================================
# Simulation functions
#================================================================================

def simul_model_multi_gaussian(n, p, mu, cov, sigma_e, coefs, seed=None):
    """ Simulate a multivariate gaussian X and and a residual vector e from a centred normal 
    and Y = X*coefs + e. 
    n (integer): The size of the samples to generate
    p (integer): The dimension of X
    sigma (integer): The variance of each X
    sigma_e (integer): The variance of the residuals 
    coefs (array-like): The coefficients of the linear model
    seed (integer): A seed to stuck the generator to a precise point and obtain reproducible results
    ------------------------------------------------------------------------------------------
    returns Y,X (array-like, ndarray): The independant variables and the dependant variable
    """
    mean = np.full(p, mu)
    rnd = np.random.RandomState(seed)
    
    e = rnd.normal(size=n, loc=0, scale=sigma_e).reshape((-1,1))
    #e = np.transpose(e)
    
    X = rnd.multivariate_normal(mean=mean, cov=cov, check_valid='raise', size=n)
    #X = rnd.multivariate_normal(mean=mean, cov=cov, size=n)
    return (np.dot(X,coefs) + e,X)