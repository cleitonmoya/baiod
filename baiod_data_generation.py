# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:56:12 2024
Outliers inseridos manualmente
@author: cleiton
"""

import numpy as np
from scipy.stats import poisson, bernoulli
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
plt.close('all')

rng = np.random.default_rng(42)

# PoINAR Model
alpha = 0.85
lamb = 1
T = 500
mu = lamb/(1-alpha)
print("mu:", mu)

Xt = poisson.rvs(lamb/(1-alpha), size=1, random_state=rng)[0]
X = [Xt]

t0 = [100, 200, 300, 400]    # localização dos outliers
Eta_t0 = [9]*len(t0)        # tamanho dos outliers

# prepara os vetores Eta e Delta
Delta = np.zeros(T)
Eta = np.zeros(T)
Delta[t0] = 1
Eta[t0] = Eta_t0

#%% Simula o processo
for t in range(1,T):
    
    et = poisson.rvs(lamb, random_state=rng)
    
    # binomial thinning
    Qsi_t = bernoulli.rvs(alpha, size=Xt, random_state=rng)
    Rt = sum(Qsi_t)
    Xt = Rt + et
    X.append(Xt)
    
#%% Insere os outliers
Y = X.copy();
for j,t in enumerate(t0):
    Y[t] = Y[t] + Eta_t0[j]
Y = np.array(Y)
# salva o vetor simulado
np.savetxt("data/Y2.txt", Y, fmt="%d")

#%% Gráficos
t_ = np.arange(T)
Eta = np.array(Eta)
fig,ax = plt.subplots(nrows=2, sharex=True, layout='constrained')

ax[0].plot(t_, Y, linewidth=1)
delta_ = Delta.astype(bool)
ax[0].scatter(t_[delta_], Y[delta_], label='outlier', s=10, color='r')
ax[0].legend()
ax[1].plot(t_, Eta*Delta, linewidth=1)
ax[0].legend()

print("Mean X:", np.mean(X))