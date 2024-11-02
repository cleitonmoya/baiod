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
    return g[0][0], g[0][1]

def gelman_rubin_multivar(x):
    lista_mcmc = rMcmcList(x)
    g = rGelman(lista_mcmc, autoburnin=BoolVector([False]), multivariate=BoolVector([True]))
    return g[0][0], g[0][1]


# Carrega as linhas de simulações
burnin = 300   # número de amostras para descartar

Alphas = []
Mus = []
Prob_deltas = []
Deltas = []
Etas = []
Ps = []
Betas = []

for j in range(1,4):
    results_path = f"results/Y1_5300_{j}/"
    
    Alpha = np.loadtxt(results_path + "Alpha.txt")[burnin:] 
    Alphas.append(Alpha)
    
    Mu = np.loadtxt(results_path + "Mu.txt")[burnin:] 
    Mus.append(Mu) 

    Prob_delta = np.loadtxt(results_path + "Prob_delta.txt")[burnin:] 
    Prob_deltas.append(Prob_delta)
    
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
Prob_deltas_mcmc = [rMcmc(FloatVector(Prob_delta)) for Prob_delta in Prob_deltas]
Deltas_mcmc = [rMcmc(FloatVector(Delta)) for Delta in Deltas]
Etas_mcmc = [rMcmc(FloatVector(Eta)) for Eta in Etas]
Ps_mcmc = [rMcmc(FloatVector(P)) for P in Ps]
Betas_mcmc = [rMcmc(FloatVector(Beta)) for Beta in Betas]

print("Diagnóstico de Gelman-Rubin:")
R_alpha, CI_alpha = gelman_rubin(Alphas_mcmc)
if CI_alpha < 1.01:
    print(f"\tAlpha: convergiu, R={R_alpha:.3f}, upper CI={CI_alpha:.3f}")
else:
    print(f"\tAlpha: não convergiu, R={R_alpha:.3f}, upper CI={CI_alpha:.3f}")

R_mu, CI_mu = gelman_rubin(Mus_mcmc)
if CI_mu < 1.01:
    print(f"\tMu: convergiu, R={R_mu:.3f}, upper CI={CI_mu:.3f}")
else:
    print(f"\tMu: não convergiu, R={R_mu:.3f}, upper CI={CI_mu:.3f}")

R_Prob_delta, CI_Prob_delta = gelman_rubin_multivar(Prob_deltas_mcmc)
if CI_mu < 1.01:
    print(f"\tProb_delta: convergiu, R={R_Prob_delta:.3f}, upper CI={CI_Prob_delta:.3f}")
else:
    print(f"\tProb_delta: não convergiu, R={R_Prob_delta:.3f}, upper CI={CI_Prob_delta:.3f}")
    
R_Delta, CI_Delta = gelman_rubin_multivar(Deltas_mcmc)
if CI_mu < 1.01:
    print(f"\tDelta: convergiu, R={R_Delta:.3f}, upper CI={CI_Delta:.3f}")
else:
    print(f"\tDelta: não convergiu, R={R_Delta:.3f}, upper CI={CI_Delta:.3f}")
    
R_Eta, CI_Eta = gelman_rubin_multivar(Etas_mcmc)
if CI_mu < 1.01:
    print(f"\tEta: convergiu, R={R_Eta:.3f}, upper CI={CI_Eta:.3f}")
else:
    print(f"\tEta: não convergiu, R={R_Eta:.3f}, upper CI={CI_Eta:.3f}")
    
R_P, CI_P = gelman_rubin_multivar(Ps_mcmc)
if CI_mu < 1.01:
    print(f"\tP: convergiu, R={R_P:.3f}, upper CI={CI_P:.3f}")
else:
    print(f"\tP: não convergiu, R={R_P:.3f}, upper CI={CI_P:.3f}")
    
R_Beta, CI_Beta = gelman_rubin_multivar(Betas_mcmc)
if CI_mu < 1.01:
    print(f"\tBeta: convergiu, R={R_Beta:.3f}, upper CI={CI_Beta:.3f}")
else:
    print(f"\tBeta: não convergiu, R={R_Beta:.3f}, upper CI={CI_Beta:.3f}")