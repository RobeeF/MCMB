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


def weighted_quantile(X, Y, Z, beta, j, tau):
    '''
    Weighted quantile of Z, as solution of (3.4)
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
    

def MCMB(Y, X, tau, size=50, extension=None, alpha=0.05, verbose=False, return_chain=False, sample_spacing=1, burn_in=0, parallelize_mode='seq'):
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
    res = mod.fit(q=tau, max_iter=7000)
    beta_hat = res.params
    
    
    #Initialisation of parameters
    p = len(beta_hat)
    beta = beta_hat.copy()
    Beta = []
    i = 0

    Z = X_to_Z(X, Y, beta_hat, tau)
    vec_wq = np.vectorize(weighted_quantile, excluded=['X','Y','Z','beta','tau'])

    remaining_iter = size*sample_spacing + burn_in

    while remaining_iter>0:
        if parallelize_mode=='seq': # Same updating than in Kocherginsky & al.
            for j in range(p):            
                beta_j =  weighted_quantile(X, Y, Z, beta, j, tau)
                beta = np.concatenate((beta[:j],[beta_j],beta[j+1:]))
                
        elif parallelize_mode=='p': # All the betas_j are updated at the sime time.
            beta = vec_wq(j=np.arange(p),beta=beta, X=X, Y=Y, Z=Z, tau=tau)
            
        elif parallelize_mode=='bp': # n_cores betas_j are updated at each iteration of the loop 
            for k in range(1,int(np.ceil(p/n_cores)+1)):
                min_index = (k-1)*n_cores
                max_index = min(k*n_cores,p)
                beta = np.concatenate((beta[0:min_index],vec_wq(j=np.arange(min_index,max_index),
                                       beta=beta, X=X, Y=Y, Z=Z, tau=tau),beta[max_index:]))
        else:
            raise ValueError('Please enter a valid parallezing mode')
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
# Ploting the betas
#=============================================================================
    
def plot_same_graph(betas_chains, autocorr=True, title=''):
    ''' Plot the betas series or the betas_autocorrelations series
    betas_chains: (ndarray) the series
    autocorr: (bool) display the serie or the autocorrelation serie
    title: (str) a string to add to the title to identify the graph
    ------------------------------------------------------------------
    returns: The requested graph
    '''
    
    p = len(betas_chains)
    Kn = len(betas_chains[0])
    clrs = {}
    for i in range(p):
        clrs[i] = [np.random.rand() for i in range(0,3)] # couleur aléatoire pour la trajectoire
        clrs[i].append(1) 
        
    if autocorr:
        df = pd.DataFrame([[betas_chains[j].autocorr(i) for i in range(Kn)] for j in range(p)]).dropna(how='all', axis=1).transpose()
    else:
        df = pd.DataFrame(betas_chains).transpose()
    
    plt.figure(figsize=(12,5))
    if autocorr:
        plt.xlabel('Autocorrelations of the betas'+title)
    else:
        plt.xlabel('Betas values at each iteration'+title)

    
    axs = {}
    for i in range(p):
        axs[i] = df.iloc[:,i].plot(color=(clrs[i][0],clrs[i][1],clrs[i][2],clrs[i][3]), grid=True, label='Beta'+str(i))
        #axs[i] = auto_corr.iloc[:,i].plot(color=(0,0.5,0.2,0.1), grid=True, label='Beta'+str(i))
       
    h = {}
    l = ['Beta'+str(i) for i in range(1,p+1)]
    for i in range(p):
        h[i] = axs[i].get_legend_handles_labels()[0]
    
    
    plt.legend([h[i] for i in range(p)][0], [l for i in range(p)][0], loc=2)
    plt.show()


#==============================================================================
# Simulations
#=============================================================================
    
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


def simul_model_multi_gaussian(n, p, mu, cov, sigma_e, coefs, seed=None):
    """ Simulate a multivariate gaussian matrix of size "size" 
    independant gaussian vector of variance sigma (and null covariance by construction)"""
    mean = np.full(p, mu)
    rnd = np.random.RandomState(seed)
    
    e = rnd.normal(size=n, loc=0, scale=sigma_e).reshape((-1,1))
    e = np.transpose(e)
    
#    X = rnd.multivariate_normal(mean=mean, cov=cov, check_valid='raise', size=n)
    X = rnd.multivariate_normal(mean=mean, cov=cov, size=n)
    return (np.dot(X,coefs) + e,X)

### Paper He & Hu 2000 setting

def simul_originmod(n,df=3,seed=None):
    """ Simulate y= b0 + b1*x1 + b2*x2 + b3*x3 + e, with x1,x2,x3 and e following a standard t-distribution (df = v), 
    and b0=b1=b2=b3=0.
  """
    np.random.RandomState(seed)
    X_1 = np.random.standard_t(df,n).reshape((-1,1))
    X_2 = np.random.standard_t(df,n).reshape((-1,1))
    X_3 = np.random.standard_t(df,n).reshape((-1,1))
    e = np.random.standard_t(df,n).reshape((-1,1))
    X = np.hstack([np.ones((n,1)), X_1, X_2, X_3])
    coefs_ = np.zeros((X.shape[1],1)).reshape(-1,1)
    return (np.dot(X,coefs_) + e,X)