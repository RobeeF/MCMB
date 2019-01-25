# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 18:51:51 2018

@author: robin
"""
import os 
os.chdir('C:/Users/robin/Documents/GitHub/MCMB')

from functions import *
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf
#from statsmodels.graphics.tsaplots import plot_acf
import numpy as np


#==============================================================================
# Simulate the chain according to a multinomial gaussian
#==============================================================================
Kn = 300
#seed = 2042
covariances= 0.6
p = 5
n = 1000
mu = 0
sigma = 1

cov = np.full(p*p,covariances).reshape(p,p)
np.fill_diagonal(cov, np.full(sigma,1))

sigma = np.ones(p)
sigma_e = 1
coefs = np.ones(p).reshape(-1,1)

model = simul_model_multi_gaussian(n, p, mu, cov, sigma_e ,coefs, seed=None)
Y = model[0]
X = model[1]
tau = 0.5


# MCMB
beta, IC = MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=0.05, parallelize_mode='seq', extension='A')
IC

#==============================================================================
# Burn-in and autocorrelation computation and plot
#==============================================================================
####### Chain reformating (might be done with a stack, will be cleaner)
# For the classic MCMB
chain = MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=0.05, return_chain=True)
betas_chains = [pd.Series([chain[i][j] for i in range(Kn)]) for j in range(p)] # Get the p betas chains generated by the MCMB algorithm 

# For the  MCMB-A
chain_A = MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=0.05, return_chain=True, extension='A')
betas_chains_A = [pd.Series([chain_A[i][j] for i in range(Kn)]) for j in range(p)] # Get the p betas chains generated by the MCMB algorithm 
betas_chains_A_50_first_iterations = [pd.Series([chain_A[i][j] for i in range(Kn)]).iloc[0:50] for j in range(p)] # Get the p betas chains generated by the MCMB algorithm 


# For the classic MCMB with the sample-spacing = 7
chain_7s = MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=0.05, return_chain=True, sample_spacing=7)
betas_chains_7s = [pd.Series([chain_7s[i][j] for i in range(Kn)]) for j in range(p)] # Get the p betas chains generated by the MCMB algorithm 
# kill the autocorrelations

# For the classic MCMB with the sample-spacing = 7
chain_3s_A = MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=0.05, return_chain=True, extension='A' ,sample_spacing=3)
betas_chains_3s_A = [pd.Series([chain_3s_A[i][j] for i in range(Kn)]) for j in range(p)] # Get the p betas chains generated by the MCMB algorithm 


# For the parallelized version of MCMB
chain_p = MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=0.05, return_chain=True, parallelize_mode='p')
betas_chains_p = [pd.Series([chain_p[i][j] for i in range(Kn)]) for j in range(p)] # Get the p betas chains generated by the MCMB algorithm 
# The betas are more correlated than in the MCMB classic version

# For the parallelized version of MCMB-A
chain_A_p_3s = MCMB(Y=Y, X=X, tau=tau, size=Kn, alpha=0.05, return_chain=True, extension='A',parallelize_mode='bp', sample_spacing=3)
betas_chains_A_p_3s = [pd.Series([chain_A_p_3s[i][j] for i in range(Kn)]) for j in range(p)] # Get the p betas chains generated by the MCMB algorithm 

## Autocorrelations plot

plot_same_graph(betas_chains, title=' - classic method')
plot_same_graph(betas_chains_A, title=' - A method') 
plot_same_graph(betas_chains_7s, title=' - 7 sample spacing method') 
plot_same_graph(betas_chains_p, title=' - parallelized method') 
plot_same_graph(betas_chains_A_p, title=' - A and parallelized method') 

# Burn-in evaluation: no burn-in needed
plot_same_graph(betas_chains, autocorr=False, title=' - classic method') # 10-18 iterations
plot_same_graph(betas_chains_A, autocorr=False, title=' - A method') # 3 iterations
plot_same_graph(betas_chains_A_50_first_iterations, autocorr=False, title=' - A method') # 3 iterations
plot_same_graph(betas_chains_7s, autocorr=False, title=' - 7 sample spacing method') # 1-3 iterations
plot_same_graph(betas_chains_p, autocorr=False, title=' - parallelized method') # 10-15 iterations with more variations
plot_same_graph(betas_chains_A_p, autocorr=False, title=' - A and parallelized method') # 3 iterations


# Evaluate the sample-spacing: 3 is enough.
plot_pacf(betas_chains[0], lags=50)
plot_pacf(betas_chains_A[0], lags=50)
plot_pacf(betas_chains_3s_A[0], lags=50)
plot_pacf(betas_chains_A_p_3s[1], lags=50)

