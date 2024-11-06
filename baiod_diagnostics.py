# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:43:39 2024

@author: cleiton
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
import rpy2.robjects as robjects


plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
plt.close('all')

# Importa o pacode Coda e funções de diagnóstico
coda = importr('coda')
rMcmc = robjects.r['mcmc']
rGeweke = robjects.r['geweke.diag']
rEffectiveSize = robjects.r['effectiveSize']
rGelman = robjects.r['gelman.diag']


# Lê a série temporal
Y = np.loadtxt("data/Y.txt")
T = len(Y)
results_path = "results/1000/"

# Parâmetros para descarte de amostras
burnin = 200  # número de amostras para descartar
salto = 10    # salto para redução de correlações

# Chamada Python das funções de diagnóstico (Pacote Coda)
# Diagnóstico de Geweke
# x: amostra
# p1: porcentagem de amostras do início do vetor
# p2: porcentagem de amostras do final do vetor
def geweke(x, p1, p2):
    g = rGeweke(rMcmc(FloatVector(x)), frac1=p1, frac2=p2)[0][0]
    return g

# Effective sample size
def ess(x):
    ess = rEffectiveSize(rMcmc(FloatVector(x)))[0]
    return ess

# Carrega os dados simulados
Alpha = np.loadtxt(results_path + "Alpha.txt")
Mu = np.loadtxt(results_path + "Mu.txt")
Delta = np.loadtxt(results_path + "Delta.txt")
Prob_delta = np.loadtxt(results_path + "Prob_delta.txt")
Eta = np.loadtxt(results_path + "Eta.txt")
P = np.loadtxt(results_path + "P.txt")
Beta = np.loadtxt(results_path + "Beta.txt")

# Aplica burn-in
Alpha_burn = Alpha[burnin:]
Mu_burn = Mu[burnin:]
Delta_burn = Delta[burnin:]
Prob_delta_burn = Prob_delta[burnin:]
Eta_burn = Eta[burnin:]
P_burn = P[burnin:]
Beta_burn = Beta[burnin:]

# Aplica o salto
Alpha_final = Alpha_burn[::salto]
Mu_final = Mu_burn[::salto]
Delta_final = Delta_burn[::salto]
Prob_delta_final = Prob_delta_burn[::salto]
Eta_final = Eta_burn[::salto]
P_final = P_burn[::salto]
Beta_final = Beta_burn[::salto]

# Variáveis auxiliares
tam_amostra_burn = len(Alpha_burn)
tam_amostra_final = len(Alpha_final)
_, num_t = Delta.shape
Delta_est = np.percentile(Delta_final, q=50, method='closest_observation', axis=0)
Prob_delta_est = Prob_delta_final.mean(axis=0)

Nomes_esc = ["Alpha", "Mu"] # nome dos parâmetros que são escalares
Nomes_vet = ["Delta", "Eta", "P", "Beta"] # nome dos parâmetros que são vetores

Param_esc_burn = [Alpha_burn, Mu_burn]
Param_esc_final = [Alpha_final, Mu_final]
Param_vet_burn = [Delta_burn, Eta_burn, P_burn, Beta_burn]
Param_vet_final = [Delta_final, Eta_final, P_final, Beta_final]

# Detecção de outliers através de threshold
thr = 0.9 # threshold para detecção dos outliers
Out = np.argwhere(Prob_delta_est >= thr).reshape(-1)


#%% 1. Gráficos de convergência (traceplot)
fig,ax = plt.subplots(nrows=3, sharex=True, figsize=(6,4), layout='constrained')
ax[0].set_title(r"$\hat{\alpha}$")
ax[0].plot(Alpha, linewidth=0.5)

ax[1].set_title(r"$\hat{\mu}$")
ax[1].plot(Mu, linewidth=0.5)

ax[2].set_title(r"$\hat{P}[\delta_t=1]t$")
for j in Out:
    ax[2].plot(Prob_delta[:,j], label=f"t={j}", linewidth=0.5)
_ = ax[2].legend()
_ = ax[2].set_xlabel('Iteração')

#%%
fig,ax = plt.subplots(nrows=3, sharex=True, figsize=(6,4), layout='constrained')
ax[0].set_title(r"$\hat{\eta}_t$")
for j in Out:
#for j in [26,60]:
    ax[0].plot(Eta[:,j], label=f"t={j}", linewidth=0.5)
_ = ax[0].legend(loc='upper right')

ax[1].set_title(r"$\hat{\beta}_t$")
for j in Out:
#for j in [26,60]: 
    ax[1].plot(Beta[:,j], linewidth=0.5)

ax[2].set_title(r"$\hat{p}_t$")
#for j in [26,60]: 
for j in Out:
    ax[2].plot(P[:,j], linewidth=0.5)
_ = ax[2].set_xlabel('Iteração')


#%% 2. Diagnóstico de convergência Geweke
p1 = 0.1
p2 = 0.5

# 2.1. Após burn-in e antes dos saltos
print("\nDiagnóstico de Gewek")
for i, par in enumerate(Param_esc_burn):
    nome = Nomes_esc[i]
    g = geweke(par, p1, p2)
    if g > 1.96 or g < -1.96:
        print(f"\t{nome}: não convergiu (g={g:.2f})")
    else:
        print(f"\t{nome}: convergiu (g={g:.2f})")

# 2.1.1 considerando todos os t
_, num_t = Prob_delta.shape
print("\n Considerando todos os t:")
for i, vet in enumerate(Param_vet_burn):
    vet = vet.round(decimals=2) # arredondamento para evitar erros
    G = np.array([geweke(vet[:,t], p1, p2) for t in range(num_t)])
    t_nc = np.where( (G>1.96) | (G<-1.96))[0]
    if len(t_nc)>0:
        print(f"\t{Nomes_vet[i]}: não convergiu para: {t_nc}")
    else:
        print(f"\t{Nomes_vet[i]}: convergiu")

# considerando somente t | delta_t = 1
t_delta1 = np.where(Delta_est)[0]
print("\nConsiderando somente t | delta_t=1:")
for i, vet in enumerate(Param_vet_burn):
    vet = vet.round(decimals=2) # arredondamento para evitar erros
    G = np.array([geweke(vet[:,t], p1, p2) for t in t_delta1])
    t_nc = np.where( (G>1.96) | (G<-1.96))[0]
    if len(t_nc)>0:
        print(f"\t{Nomes_vet[i]}: não convergiu para t={t_delta1[t_nc]}")
    else:
        print(f"\t{Nomes_vet[i]}: convergiu")


#%% 3. Gráficos de auto-correlação das amostras antes e após salto
fig,ax = plt.subplots(figsize=(5,4), nrows=2, layout='constrained')
#fig.suptitle('Antes da aplicação dos saltos (após burn-in)')
vlines_kwargs = {'linewidth':0.7}
_ = plot_acf(Alpha_burn, ax=ax[0], bartlett_confint=False, 
             marker='.', vlines_kwargs=vlines_kwargs,
             title = r"Autocorrelação - amostra de $\alpha$")
ax[0].set_ylim(0,1.1)
_ = plot_acf(Mu_burn, ax=ax[1], bartlett_confint=False, 
             marker='.', vlines_kwargs=vlines_kwargs,
             title = r"Autocorrelação - amostra de $\mu$")
ax[1].set_ylim(0,1.1)

fig,ax = plt.subplots(figsize=(5,4), nrows=2, layout='constrained')
#fig.suptitle('Após a aplicação dos saltos (e após burn-in)')
vlines_kwargs = {'linewidth':0.7}
_ = plot_acf(Alpha_final, ax=ax[0], bartlett_confint=False, 
             marker='.', vlines_kwargs=vlines_kwargs,
             title = r"Autocorrelação - amostra de $\alpha$")
ax[0].set_ylim(0,1.1)
_ = plot_acf(Mu_final, ax=ax[1], bartlett_confint=False, 
             marker='.', vlines_kwargs=vlines_kwargs,
             title = r"Autocorrelação - amostra de $\mu$")
ax[1].set_ylim(0,1.1)


#%% 4. Tamanho da amostra efetiva
print("\nTamanho da amostra efetiva (após burin-in")
print(f"Antes dos saltos (total = {tam_amostra_burn} observações):")
for i, par in enumerate(Param_esc_burn):
    nome = Nomes_esc[i]
    tam = ess(par)
    print(f"\t{Nomes_esc[i]}: {tam:.0f}")
for i, vet in enumerate(Param_vet_burn):
    vet = vet.round(decimals=2) # arredondamento para evitar erros
    Tam = np.mean([ess(vet[:,t]) for t in t_delta1])
    print(f"\t{Nomes_vet[i]} (média para t|delta_t=1): {Tam:.0f}")

print(f"\nApós os saltos (total = {tam_amostra_final} observações):")
for i, par in enumerate(Param_esc_final):
    nome = Nomes_esc[i]
    tam = ess(par)
    print(f"\t{Nomes_esc[i]}: {tam:.0f}")
for i, vet in enumerate(Param_vet_final):
    vet = vet.round(decimals=2) # arredondamento para evitar erros
    Tam = np.mean([ess(vet[:,t]) for t in t_delta1])
    print(f"\t{Nomes_vet[i]} (média para t|delta_t=1): {Tam:.0f}")