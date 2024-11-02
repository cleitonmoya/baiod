# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 20:45:15 2024

@author: cleiton
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
plt.close('all')

# Lê a série temporal
Y = np.loadtxt("data/Y1.txt")
T = len(Y)
results_path = "results/Y1_5300_1/"

# Parâmetros para descarte de amostras
burnin = 300    # número de amostras para descartar
salto = 20      # salto para redução de correlações

# Carrega os dados simulados
Nomes = ["Alpha", "Mu", "Delta", "Prob_delta", "Eta", "P", "Beta"]
Alpha = np.loadtxt(results_path + "Alpha.txt")
Mu = np.loadtxt(results_path + "Mu.txt")
Delta = np.loadtxt(results_path + "Delta.txt")
Prob_delta = np.loadtxt(results_path + "Prob_delta.txt")
Eta = np.loadtxt(results_path + "Eta.txt")
P = np.loadtxt(results_path + "P.txt")
Beta = np.loadtxt(results_path + "Beta.txt")

#%% Aplica burn-in e salto
Alpha_burn = Alpha[burnin:]
Mu_burn = Mu[burnin:]
Delta_burn = Delta[burnin:]
Prob_delta_burn = Prob_delta[burnin:]
Eta_burn = Eta[burnin:]
P_burn = P[burnin:]
Beta_burn = Beta[burnin:]

Alpha_final = Alpha_burn[::salto]
Mu_final = Mu_burn[::salto]
Delta_final = Delta_burn[::salto]
Prob_delta_final = Prob_delta_burn[::salto]
Eta_final = Eta_burn[::salto]
P_final = P_burn[::salto]
Beta_final = Beta_burn[::salto]

tam_amostra_final = len(Alpha[burnin::salto])
print("Tamanho da amostra final:", tam_amostra_final)


#%% Parâmestros estimados do modelo
alpha_est = np.mean(Alpha_final)
mu_est = np.mean(Mu_final)
lambda_est = mu_est*(1-alpha_est)
Delta_est = np.percentile(Delta_final, q=50, method='closest_observation', axis=0)
Prob_delta_est = Prob_delta_final.mean(axis=0)
Eta_est = np.percentile(Eta_final, q=50, method='closest_observation', axis=0)

# Detecção de outliers através de threshold
thr = 0.9 # threshold para detecção dos outliers
Out = np.argwhere(Prob_delta_est >= thr).reshape(-1)

print(f"Alpha: {alpha_est:.2f}")
print(f"Mu: {mu_est:.2f}")
print(f"Lambda: {lambda_est:.2f}")
print("t | Delta_t=1:", np.where(Delta_est)[0])
print("Outliers detectados (threshold):", Out)
print(f"Prob[delta_t=1] (p/ outliers): {Prob_delta_est[Out]}")
print(f"Eta: {Eta_est[Out]}")


#%% Gráfico principal - Detecção de outliers
t_ = np.arange(T)
fig,ax = plt.subplots(nrows=3, sharex=True, constrained_layout=True)
ax[0].set_title("Poisson INAR(1) e outliers dectados")
ax[0].plot(Y, linewidth=0.7)
ax[0].scatter(t_[Out], Y[Out], label='outlier', s=10, color='r')
_ = ax[0].legend()

ax[1].set_title(r"Probabilidade a posteriori do outlier - $P[\delta_t=1]$")
_, stemlines, _ = ax[1].stem(Prob_delta_est, markerfmt=" ", basefmt=" ")
plt.setp(stemlines, 'linewidth', 0.7)
ax[1].axhline(thr, c='r', linewidth=0.7, label='threshold')
_ = ax[1].legend()

ax[2].set_title(r"Magnitude estimada do outlier - $\hat{\eta}_t$")
_, stemlines, _ = ax[2].stem(t_, Eta_est, markerfmt=" ", basefmt=" ")
plt.setp(stemlines, 'linewidth', 0.7)
_ = ax[2].set_xlabel('t')


#%% Densidades a posteriori estimadas de Alpha e Mu
fig,ax = plt.subplots(ncols=2, constrained_layout=True, figsize=(5,2.5))
fig.suptitle("Densidades estimadas")
ax[0].set_xlabel(r"$\alpha$")
sns.kdeplot(Alpha[burnin:], bw_adjust=2, ax=ax[0])

ax[1].set_xlabel(r"$\mu$")
sns.kdeplot(Mu[burnin:], bw_adjust=2, ax=ax[1])


