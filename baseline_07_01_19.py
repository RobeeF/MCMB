# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:09:39 2018

@author: Bruno
"""

#==============================================================================
# Semi and Non Parametrics Econometrics - Project
#==============================================================================

# Initialisation
import os
os.chdir(r"C:\Users\Bruno\Documents\Cours\ENSAE\Semi and Non Parametric Econometrics\Projet")

# Packages
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from wquantiles import quantile_1D

#==============================================================================
# Simulation of a dataset
#==============================================================================

### Linear model

# Residuals
mu = 0
sigma = 1
n = 5000
epsilon = rd.normal(mu, sigma, n)

plt.hist(epsilon, bins=100)

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


#==============================================================================
# MCMB-A Algorithm
#==============================================================================
# To compute the square root of a matrix
import scipy.linalg as sci

from sklearn import preprocessing
X_scaled = preprocessing.scale(X)


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

def standardization(X):
    '''
    X: array-like dataset
    '''
    # Standardization of input array
    X_t = np.ndarray.transpose(X)
    XX = np.dot(X_t,X) #X'X
    XX = sci.sqrtm(XX) # (X'X)^(1/2)
    XX = np.linalg.matrix_power(XX, -1) # (X'X)^-(1/2)
    X_tilde = np.dot(X, XX)
        
    return X_tilde


def rho(x, tau):
    '''
    Check function
    
    x: scalar
    tau: real between 0 and 1
    '''
    return x*(tau-int(x<0))


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


def weighted_quantile(Y, X, beta, j, c_star, tau):
    '''
    Weighted quantile of Z, as solution of (3.4)
    '''
    # Defining Z
    beta_star = beta[:j] + beta[j+1:]
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
    ZX = np.hstack([Z_star, abs_X_j])
    
    #Cumulative weights
    ZX_sort = np.sort(ZX, 0)
    cum = np.cumsum(ZX_sort[:,1])
    ZX_sort = np.hstack([ZX_sort, np.reshape(cum, (-1,1))])
    
#    # Weighted quantile
#    i = 0
#    w = 0
#    while w < tau_star:
#        w = ZX_sort[i,2]
#        i+=1
#    return ZX_sort[i,0]
    
    return quantile_1D(np.reshape(Z_star, -1), np.reshape(abs_X_j, -1), tau_star)
    



def MCMB(Y, X, beta_hat, tau, maxiter=50):
    '''
    MCMB algorithm
    '''
    #Initialisation of parameters
    from sklearn.utils import resample
    p = len(beta_hat)
    beta = beta_hat.copy()
    Beta = []
    i = 0
    Z = X_to_Z(X, Y, beta_hat, tau)

    while i<maxiter:
        #for each element of the beta vector
        for j in range(p):
            # Draw a bootstrapped sample
            Z_boot = resample(Z)
            
            #Take the j-th columns
            Z_j = Z_boot[:,j]
            c_star = Z_j.sum()
            
            #Solves the minimiztion problem
            beta_j = weighted_quantile(Y, X, beta, j, c_star, tau)
            
            #New beta
            beta = beta[:j] + [beta_j] + beta[j+1:]
        
        Beta.append(beta)
        i +=1
        print('Iteration ' + str(i) + ' reussie !')

    return Beta



#==============================================================================
# Simulations 
#==============================================================================

tau = 0.5
transform = 'A'
# Compte le nombre de fois où l'intervalle de confiance inclut la vraie valeur (ici 1)
inside = 0

iter_max = 300
for iteration in range(iter_max):
    np.random.seed(iteration)
    model1 = simul_model1(n=100, seed=iteration)
    Y = model1[0]
    X = model1[1]
    X_tilde = np.dot(X, compute_A(X))

    # Estimation of beta_hat
    from statsmodels.regression.quantile_regression import QuantReg
    
    data = np.hstack([Y,X])
    mod = QuantReg(Y, X)
    res = mod.fit(q=tau)
    print(res.summary())
    
    beta_hat = list(res.params)


    # MCMB
    if transform == 'A':
        chain = MCMB(Y=Y, X=X_tilde, beta_hat=beta_hat, tau=tau, maxiter=50)
        chain = np.dot(compute_A(X), np.transpose(np.array(chain)) )
        chain = np.transpose(chain)
    else:
        chain = MCMB(Y=Y, X=X_tilde, beta_hat=beta_hat, tau=tau, maxiter=50)
        chain = np.array(chain)

    # Covariance matrix
    Sigma = np.cov(chain, rowvar=False)
    
    # Confidence interval
    p = len(beta_hat)
    IC = []
    
    for i in range(p):
        IC.append([beta_hat[i]-1.64*np.sqrt(Sigma[i,i]),
                   beta_hat[i]+1.64*np.sqrt(Sigma[i,i])])
    
    IC = np.array(IC)
    
    # Ajoute 1 si la vraie valeur est incluse
    if (1>IC[1][0]) & (1<IC[1][1]):
        inside += 1

coverage = inside/iter_max*100
print('The coverage is ' + str(coverage)+' %.')
    

### Model 2 (original He Hu 2002) LAD regression f(0) ? ###

def simul_originmod(n,df=3,seed=None): # Model 1 of Kocherginsky (2002)
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

# Interval coverage
    
mean_coverage = [] # store proportion of "insiders" by iteration
mean_length = [] # store mean length by iteration

samples = 500   # choose number of samples
for iteration in range(samples):
    np.random.seed(iteration)
    model2 = simul_originmod(n=50,df=3, seed=iteration)
    #model2 = simul_originmod(n=50,df=8, seed=iteration)
    Y = model2[0]
    X = model2[1]

    # Estimation of beta_hat
    from statsmodels.regression.quantile_regression import QuantReg
    
    data = np.hstack([Y,X])
    mod = QuantReg(Y, X)
    res = mod.fit(q=tau)
    
    beta_hat = list(res.params)

    # MCMB
    chain = MCMB(Y=Y, X=X, beta_hat=beta_hat, tau=tau, maxiter=1000)
    
    # Covariance matrix
    Sigma = np.cov(np.array(chain), rowvar=False)
    
    # Confidence interval
    p = len(beta_hat)
    IC = []
    
    for i in range(p):
        IC.append([beta_hat[i]-1.64*np.sqrt(Sigma[i,i]),
                   beta_hat[i]+1.64*np.sqrt(Sigma[i,i])])
    
    tag=0
    length=[]
    for i in IC:
        length.append(i[1] - i[0])
        if 0 >= i[0] and i[1] >= 0 :
            tag += 1 
    mean_coverage.append(np.divide(tag,len(beta)))
    mean_length.append(np.divide(length,len(beta)))
    print(IC)
    
    print(iteration)
    
print('Mean coverage %s ' % np.mean(mean_coverage))
print('Mean length %s ' % np.mean(mean_length))
    


