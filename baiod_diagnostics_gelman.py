# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 00:11:13 2024

@author: cleiton
"""

import numpy as np
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector, BoolVector
import rpy2.robjects as robjects
import warnings

warnings.filterwarnings("default")
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
plt.close('all')

# Importa o pacode Coda e funções de diagnóstico
coda = importr('coda')
rMcmc = robjects.r['mcmc']
rGelman = robjects.r['gelman.diag']
rMcmcList = robjects.r['mcmc.list']

def gelman_rubin(x):
    lista_mcmc = rMcmcList(x)
    g = rGelman(lista_mcmc, autoburnin=BoolVector([False]), multivariate=BoolVector([False]))
    return g[0][0]

def gelman_rubin_multivar(x):
    lista_mcmc = rMcmcList(x)
    g = rGelman(lista_mcmc, autoburnin=BoolVector([False]), multivariate=BoolVector([True]))
    return g[0][0]


# Carrega as linhas de simulações
burnin = 1000   # número de amostras para descartar

Alphas = []
Mus = []
Deltas = []
Etas = []
Ps = []
Betas = []

for j in [1,2,3]:
    results_path = f"results/5000_{j}/"
    
    Alpha = np.loadtxt(results_path + "Alpha.txt")[burnin:] 
    Alphas.append(Alpha)
    
    Mu = np.loadtxt(results_path + "Mu.txt")[burnin:] 
    Mus.append(Mu) 
    
    Delta = np.loadtxt(results_path + "Prob_delta.txt")[burnin:] 
    Deltas.append(Delta)
    
    Eta = np.loadtxt(results_path + "Eta.txt")[burnin:] 
    Etas.append(Eta)
    
    P = np.loadtxt(results_path + "P.txt")[burnin:] 
    Ps.append(P)
    
    Beta = np.loadtxt(results_path + "Beta.txt")[burnin:] 
    Betas.append(Beta)
    
    
# Prepara as lista de simulações para o diagnóstico
Alphas_mcmc = [rMcmc(FloatVector(alpha)) for alpha in Alphas]
Mus_mcmc = [rMcmc(FloatVector(mu)) for mu in Mus]
Deltas_mcmc = [rMcmc(FloatVector(Delta)) for Delta in Deltas]
Etas_mcmc = [rMcmc(FloatVector(Eta)) for Eta in Etas]
Ps_mcmc = [rMcmc(FloatVector(P)) for P in Ps]
Betas_mcmc = [rMcmc(FloatVector(Beta)) for Beta in Betas]

print("Diagnóstico de Gelman-Rubin:")
R_alpha = gelman_rubin(Alphas_mcmc)
if R_alpha < 1.01:
    print(f"\tAlpha: convergiu, R={R_alpha:.3f}")
else:
    print(f"\tAlpha: não convergiu, R={R_alpha:.3f}")

R_mu = gelman_rubin(Mus_mcmc)
if R_mu < 1.01:
    print(f"\tMu: convergiu, R={R_mu:.3f}")
else:
    print(f"\tMu: não convergiu, R={R_mu:.3f}")
   
R_Delta = gelman_rubin_multivar(Deltas_mcmc)
if R_Delta < 1.01:
    print(f"\tDelta: convergiu, R={R_Delta:.3f}")
else:
    print(f"\tDelta: não convergiu, R={R_Delta:.3f}")
    
R_Eta = gelman_rubin_multivar(Etas_mcmc)
if R_Eta < 1.01:
    print(f"\tEta: convergiu, R={R_Eta:.3f}")
else:
    print(f"\tEta: não convergiu, R={R_Eta:.3f},")
    
R_P = gelman_rubin_multivar(Ps_mcmc)
if R_P < 1.01:
    print(f"\tP: convergiu, R={R_P:.3f}")
else:
    print(f"\tP: não convergiu, R={R_P:.3f}")
    
R_Beta = gelman_rubin_multivar(Betas_mcmc)
if R_Beta < 1.01:
    print(f"\tBeta: convergiu, R={R_Beta:.3f}")
else:
    print(f"\tBeta: não convergiu, R={R_Beta:.3f}")