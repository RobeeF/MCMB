# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:59:33 2019

@author: robin
"""

import os 
os.chdir('C:/Users/robin/Documents/GitHub/MCMB/timeit')

from functions import *
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import numpy as np
import timeit
import pickle 


#=================================================================================
# Sequential
#=================================================================================
# Make p vary from 5 to 80

time_seq = []

SETUP_CODE = ''' 
from functions import seq_update 
from functions import simul_model_multi_gaussian 
from functions import X_to_Z 
import numpy as np

Kn = 100
covariances= 0.6
p = 80
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

beta = np.full(p,1)
Z = X_to_Z(X,Y, beta,tau)
'''
  
TEST_CODE = ''' 
beta = seq_update(p, beta, X, Y, Z, tau)'''
  
# timeit.repeat statement 
time = timeit.timeit(setup = SETUP_CODE, 
                      stmt = TEST_CODE,
                      number = 1000) 

time_seq.append(time/1000)

#=================================================================================
# Numba parallel
#=================================================================================
time_numba_parallel = []

SETUP_CODE = ''' 
from functions import beta_update_numba 
from functions import simul_model_multi_gaussian 
from functions import X_to_Z 
import numpy as np

Kn = 100
covariances= 0.6
p = 80
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

beta = np.full(p,1)
Z = X_to_Z(X,Y, beta,tau)
n_cores = 4
'''
  
TEST_CODE = ''' 
beta = beta_update_numba(p, beta, X, Y, Z, tau, n_cores)'''
  
# timeit.repeat statement 
time = timeit.timeit(setup = SETUP_CODE, 
                      stmt = TEST_CODE,
                      number = 1000) 

time_numba_parallel.append(time/1000)

#=================================================================================
# Storing the results
#=================================================================================

df = pd.DataFrame(np.stack([time_seq, time_numba_parallel]).T, 
                  index=np.arange(5,81,5), columns=['sequential','block parallel'])

df.plot()

with open('timeit_comp_True', 'wb') as fichier:
    mon_pickler = pickle.Pickler(fichier)
    mon_pickler.dump(df)
