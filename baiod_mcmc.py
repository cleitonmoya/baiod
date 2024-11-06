# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:15:51 2024

@author: cleiton
"""

import numpy as np
from numpy import log, exp
from scipy.stats import norm, uniform, beta, bernoulli
from scipy.special import comb, factorial, logsumexp
from time import time
import warnings

# Considera warning (ex., underflow) como erro
warnings.filterwarnings("error")

# Configura o gerador de números aleatórios
rng = np.random.default_rng(seed=10)

# Lê a série temporal
Y = np.loadtxt("data/Y.txt")
T = len(Y)

# Diretório para salvar os resultados
results_path = "results/1000"

#%% Condicionais completas

# Função auxiliar para as condicionais completas de alpha e mu
def logfunT(t, i, alpha, mu, Delta, Eta):
    c1 = Y[t] - Eta[t]*Delta[t]
    c2 = Y[t-1] - Eta[t-1]*Delta[t-1]
    
    p1 = (c1-i)*log(mu) - log(factorial(c1-i))
    p2 = log(comb(c2, i))
    p3 = i*log(alpha) + (c2-i)*log(1-alpha)
    p4 = (c1-i)*log(1-alpha)
    
    p = p1+p2+p3+p4
    return p

# Condicional completa de  alpha
def logpost_alpha(alpha, mu, Delta, Eta, Y, a, b):
    T = len(Y)
    logp1 = (a-1)*log(alpha) + (b-1)*log(1-alpha)
    logp2 = alpha*mu*(T-1)
    
    soma1 = 0
    for t in range(1,T):
        Mt = min([Y[t-1]-Eta[t-1]*Delta[t-1], Y[t]-Eta[t]*Delta[t]]).astype(int)
        #soma2 = sum([funT(t, i, alpha, mu, Delta, Eta) for i in range(Mt+1)])
        if Mt >= 0:
            soma2 = logsumexp([logfunT(t, i, alpha, mu, Delta, Eta) for i in range(Mt+1)]) 
        else:
            soma2 = 0
        soma1 = soma1 + soma2

    p = logp1 + logp2 + soma1
    
    return p

# Condicional completa de mu
def logpost_mu(mu, alpha, Delta, Eta, Y, c, d):
    T = len(Y)
    logp1 = (c-1)*log(mu) - (d+(T-1)*(1-alpha))*mu
    soma1 = 0
    
    soma1 = 0
    for t in range(1,T):
        Mt = min([Y[t-1]-Eta[t-1]*Delta[t-1], Y[t]-Eta[t]*Delta[t]]).astype(int)
        #soma2 = sum([funT(t, i, alpha, mu, Delta, Eta) for i in range(Mt+1)])
        if Mt >= 0:
            soma2 = logsumexp([logfunT(t, i, alpha, mu, Delta, Eta) for i in range(Mt+1)])
        else:
            soma2 = 0
        soma1 = soma1 + soma2

    p = logp1 + soma1
    return p

# Funções auxiliar para a condicional completa de delta_t
# ys:  Y*[t]
# ys1: Y*[t-1]
def logauxf(ys, ys1, alpha, mu, Delta):
    if not (ys.is_integer() and ys1.is_integer()):
        raise Exception(f"Erro: ys={ys} ou ys1={ys1} não são inteiros")
    
    logp1 = -mu*(1-alpha)

    Mt = min(ys1, ys).astype(int)
    
    if Mt > 0:
        logp2 = logsumexp([log(comb(ys1,i)) + i*log(alpha) + 
                          (ys1-i)*log(1-alpha) + (ys-i)*log(mu*(1-alpha)) - 
                          log(factorial(ys-i)) for i in range(Mt+1)])
    else:
        logp2 = log(10**-20)
    y = logp1 + logp2
    return y

# Utilizada também na condicional completa de eta_t
def f1(Y, mu, Delta, Eta, P, t):
    fa = exp(logauxf(Y[t]-Eta[t], Y[t-1]-Eta[t-1], alpha, mu, Delta))
    fb = exp(logauxf(Y[t]-Eta[t], Y[t-1], alpha, mu, Delta))
    fc = exp(logauxf(Y[t+1]-Eta[t+1], Y[t]-Eta[t], alpha, mu, Delta))
    fd = exp(logauxf(Y[t+1], Y[t]-Eta[t], alpha, mu, Delta))
    
    if Delta[t-1] == 1:
        y = fa*(fc*P[t+1] + fd*(1-P[t+1]))
    elif Delta[t-1] == 0:
        y = fb*(fc*P[t+1] + fd*(1-P[t+1]))
    else:
        raise Exception("Erro: valor inválido de delta")
    return y

def f0(Y, mu, Delta, Eta, P, t):
    fa = exp(logauxf(Y[t], Y[t-1]-Eta[t-1], alpha, mu, Delta))
    fb = exp(logauxf(Y[t], Y[t-1], alpha, mu, Delta))
    fc = exp(logauxf(Y[t+1]-Eta[t+1], Y[t], alpha, mu, Delta))
    fd = exp(logauxf(Y[t+1], Y[t], alpha, mu, Delta))
    
    if Delta[t-1] == 1:
        y = fa*(fc*P[t+1] + fd*(1-P[t+1]))
    elif Delta[t-1] == 0:
        y = fb*(fc*P[t+1] + fd*(1-P[t+1]))
    else:
        raise Exception("Erro: valor inválido de delta")
    return y

# Condicional completa de delta_t = 1
# P[delta_j = 1 | Gamma_delta, Y]
def post_delta_j(j, P, alpha, mu, Delta, Eta, Y):
    Delta_ = Delta.copy()
    Delta_[j] = 1
    f0y = f0(Y, mu, Delta_, Eta, P, j)
    f1y = f1(Y, mu, Delta_, Eta, P, j)
    num = P[j]*f1y
    den = P[j]*f1y + (1-P[j])*f0y
    if den == 0:
        print(f"n={n}: post_delta_j: Alerta: num={num}, den={den}, j={j}, f0y={f0y}, f1y={f1y}")
        pi_j = 1
    else:
        pi_j = num/den
    return pi_j

# Condicional completa de eta_t dado que delta_t = 1
def logpost_eta_j(j, eta_j, mu, Eta, Delta, P, Beta, Y):
    logprior = -Beta[j] + eta_j*log(Beta[j]) - log(factorial(eta_j))
    Eta_ = Eta.copy()
    Eta_[j] = eta_j
    
    Delta_ = Delta.copy()
    Delta_[j] = 1
    f1y = f1(Y, mu, Delta_, Eta_, P, j)
    if f1y == 0:
        print(f"n={n}: logpost_eta_j: Alerta: f1y={f1y}")
        y = logprior + log(10**-15)
    else:
        y = logprior + log(f1y)
    return y

# Condicional completa de beta_t
def logpost_beta_j(j, beta_j, Eta, l, m):
    y = -(m+1)*beta_j + (Eta[j]+l-1)*log(beta_j)
    return y

#%% Funções auxiliares para o algoritmo de Metropolis
def simula_xprop(mu, sigma):
    x = norm.rvs(loc=mu, scale=sigma, random_state=rng)
    return x

def log_prob_aceit_alpha(x, x_prop, mu, Delta, Eta, Y, a, b):
    p1 = logpost_alpha(x_prop, mu, Delta, Eta, Y, a, b)
    p2 = logpost_alpha(x, mu, Delta, Eta, Y, a, b)
    if p1 < p2:
        prob_aceit = min([1, exp(p1-p2)])
    else:
        prob_aceit = 1
    return prob_aceit

def log_prob_aceit_mu(x, x_prop, alpha, Delta, Eta, Y, c, d):
    p1 = logpost_mu(x_prop, alpha, Delta, Eta, Y, c, d)
    p2 = logpost_mu(x,  alpha, Delta, Eta, Y, c, d)
    if p1 < p2:
        prob_aceit = min([1, exp(p1-p2)])
    else:
        prob_aceit = 1
    return prob_aceit

def log_prob_aceit_beta_j(j, x, x_prop, Eta, l, m):
    p1 = logpost_beta_j(j, x_prop, Eta, l, m)
    p2 = logpost_beta_j(j, x,  Eta, l, m)
    if p1 < p2:
        prob_aceit = min([1, exp(p1-p2)])
    else:
        prob_aceit = 1
    return prob_aceit


#%% Gerações de amostras
# Gera uma amostra de alpha pelo método de Metropolis
def amostra_alpha(sigma, x0, mu, Delta, Eta, Y, a, b):
     
    aceito = 0
    x = x0

    # simula valor proposto
    x_prop = simula_xprop(x, sigma)
    if x_prop > 0 and x_prop < 1:
    
        # probabilidade de aceitação
        prob_aceit = log_prob_aceit_alpha(x, x_prop, mu, Delta, Eta, Y, a, b)
        
        # critério de aceição
        u = uniform.rvs(0,1, random_state=rng)
        if  u < prob_aceit:
            x = x_prop
            aceito = 1
    
    return x,aceito


# Gera uma amostra de mu pelo método de Metropolis
def amostra_mu(sigma, x0, alpha, Delta, Eta, Y, c, d):
    aceito = 0
    x = x0

    # simula valor proposto
    x_prop = simula_xprop(x, sigma)
    if x_prop > 0:
    
        # probabilidade de aceitação
        prob_aceit = log_prob_aceit_mu(x, x_prop, alpha, Delta, Eta, Y, c, d)
        
        # critério de aceição
        u = uniform.rvs(0,1, random_state=rng)
        if  u < prob_aceit:
            x = x_prop
            aceito = 1
    
    return x,aceito


# Gera uma amostra de Delta (distribuição Bernoulli)
def amostra_Delta(alpha, mu, Delta, Eta, P, Y):
    Prob_delta = np.array([post_delta_j(j, P, alpha, mu, Delta, Eta, Y) for j in range(1,T-1)])
    
    Delta_pos = bernoulli.rvs(p=Prob_delta, random_state=rng)
    
    Delta_pos = np.insert(Delta_pos, 0, 0)
    Delta_pos = np.append(Delta_pos, 0)
    Prob_delta = np.insert(Prob_delta, 0, P[0])
    Prob_delta = np.append(Prob_delta, P[0])
    return Delta_pos, Prob_delta


# Gera uma amostra de eta_j|delta_j=1 pelo método de Metropolis Discreto
def amostra_eta_j(j, x0, Eta, Delta, P, Beta, Y, Q):
    
    ns,_ = Q.shape
    x = x0
    aceito = 0
    
    # define um valor proposto
    x_prop = rng.choice(ns, p=Q[x])
    
    # calcula a probabilidade de aceitação
    num = logpost_eta_j(j, x_prop, mu, Eta, Delta, P, Beta, Y)
    den = logpost_eta_j(j, x, mu, Eta, Delta, P, Beta, Y)
    if num < den:
        prob_aceit = min([1, exp(num-den)])
    else:
        prob_aceit = 1
    
    # aceita ou rejeita x_prop
    u = rng.uniform()
    if u < prob_aceit:
        x = x_prop
        aceito = 1

    return x, aceito


# Gera uma amostra de eta (somente para delta_j=1)
def amostra_Eta(x0, Delta, P, Beta, Y, Q):
    J1 = np.argwhere(Delta).reshape(-1) # índices j nos quais delta_j= 1
    nJ1 = len(J1)
    Eta_ = x0.copy()
    taxa_aceit = 0
    for j in J1:
        if j > 0 and j< T-2:
           eta_j, taxa_aceit_j = amostra_eta_j(j, x0[j], Eta_, Delta, P, Beta, Y, Q)
           Eta_[j] = eta_j
           taxa_aceit = (nJ1-1)*(taxa_aceit)/nJ1 + taxa_aceit_j/nJ1
    return Eta_, taxa_aceit

# Gera uma amosta de p_j (distribuição Beta)
def amostra_p_j(j, Delta, g, h):
    y = beta.rvs(a=Delta[j]+g, b=h-Delta[j]+1, random_state=rng)
    return y


def amostra_P(Delta, g, h):
    P = [amostra_p_j(j, Delta, g, h) for j in range(T)]
    return P


# Gera uma amostra de beta_j pelo método de Metropolis
def amostra_beta_j(j, sigma, x0, Eta, l, m):
    aceito = 0
    x = x0

    # simula valor proposto
    x_prop = simula_xprop(x, sigma)
    if x_prop > 0:
    
        # probabilidade de aceitação
        prob_aceit = log_prob_aceit_beta_j(j, x, x_prop, Eta, l, m)
        
        # critério de aceição
        u = uniform.rvs(0,1, random_state=rng)
        if  u < prob_aceit:
            x = x_prop
            aceito = 1
    
    return x,aceito


# Gera uma amostra de Beta pelo método de Metropolis
def amostra_Beta(sigma, x0, Eta, l, m):
    # Beta a posteriori
    Beta_pos = x0.copy()
    
    # índices j nos quais delta_j= 1
    #J1 = np.argwhere(Delta).reshape(-1)
    J1 = np.arange(T)
    nJ = len(J1)
    na_beta = 0
    
    for j in J1:
        beta_j, aceito_j = amostra_beta_j(j, sigma, x0[j], Eta, l, m)
        Beta_pos[j] = beta_j
        na_beta = na_beta + aceito_j
    
    # taxa média de aceitação dos eta_j
    taxa_aceit_beta = na_beta/nJ
    return Beta_pos, taxa_aceit_beta


#%% Parâmetros da simulação
N = 1000 # número de passos

# Hiperparâmetros das distribuições a priori
a = b = 0.01
c = d = 0.1
g = 5
h = 95
l = 10
m = 1

# Parâmetros do modelo (inicialização)
alpha = 0.5
mu = 0.5
Delta_n = np.zeros(T, dtype='int32')
Eta_n = np.ones(T, dtype='int32')
P_n = np.ones(T)*0.01
Beta_n = np.ones(T)

# Parâmetros dos métodos MCMC
sigma_alpha = 0.07  # desvio-padrão do passeio aleatório de alpha
sigma_mu = 1.7      # desvio-padrão do passeio aleatório de mu
sigma_beta = 6      # desvio-padrão do passeio aleatório de beta_t

# matriz de transição Q (simétrica) para simulação de eta_j
ns = 15     # número de estados
p1 = 0.3   # probabilidade sj = si
Q = np.ones((ns,ns))*(1-p1)/(ns-1)
np.fill_diagonal(Q, p1)
#%%
# Variáveis auxiliares para cálculo da taxa de aceitação
aceit_alpha = 0
aceit_mu = 0
aceit_Eta = 0
aceit_Beta = 0

# Listas com os parâmetros ao longo das iterações
Alpha = [alpha]
Mu = [mu]
Delta = [Delta_n]
Eta = [Eta_n]
P = [P_n]
Beta = [Beta_n]
Prob_delta = [np.zeros(T)]

# Salva os hiperparâmetros utilizados na simulação
dict_hyperparams = {
    'a': a,
    'b': b,
    'c': c,
    'd': d,
    'g': g,
    'h': h,
    'l': l,
    'm': m}

with open(results_path+'hiperparametros.txt', 'w') as f:
    print(dict_hyperparams, file=f)

# Salva os parâmetros utilizados na simulação
dict_params = {
    'alpha': alpha,
    'mu': mu,
    'Delta_n': Delta_n[0],
    'Eta_n': Eta_n[0],
    'P_n': P_n[0],
    'Beta_n': Beta_n[0],
    'sigma_alpha': sigma_alpha,
    'sigma_mu': sigma_mu,
    'sigma_beta': sigma_beta,
    'ns': ns,
    'p1': p1}

with open(results_path+'parametros.txt', 'w') as f:
    print(dict_params, file=f)

#%% Algoritmo principal - Gibbs Sampling com passos de Metroplis
startTime = time()
for n in range(1,N):
    
    if n%100==0:
        print(f"Passo {n}/{N}: aceitações: alpha={aceit_alpha:.2f}, "
              f"mu={aceit_mu:.2f}, eta={aceit_Eta:.2f}, " 
              f"beta={aceit_Beta:.2f}")
    # gera uma amostra de P_n (Distribuição Beta)
    P_n = amostra_P(Delta_n, g, h)
    P.append(P_n)
    
    # gera uma amostra de alpha (Metrpolis contínuo)
    alpha, aceit_alpha_n = amostra_alpha(sigma_alpha, alpha, mu, Delta_n, Eta_n, Y, a, b)
    aceit_alpha = (n-1)*(aceit_alpha)/n + aceit_alpha_n/n
    Alpha.append(alpha)
    
    # gera uma amostra de mu (Metropolis contínuo)
    mu, aceit_mu_n = amostra_mu(sigma_mu, mu, alpha, Delta_n, Eta_n, Y, c, d)
    aceit_mu = (n-1)*(aceit_mu)/n + aceit_mu_n/n
    Mu.append(mu)
        
    # gera uma amostra de Delta_n (distribuição Bernoulli)
    Delta_n, Prob_delta_n = amostra_Delta(alpha, mu, Delta_n, Eta_n, P_n, Y)
    Delta.append(Delta_n)
    Prob_delta.append(Prob_delta_n) 
    
    # gera uma amostra de Eta_n (Metropolis discreto - sim. em bloco)
    Eta_n, aceit_eta_n = amostra_Eta(Eta_n, Delta_n, P_n, Beta_n, Y, Q)
    aceit_Eta = (n-1)*(aceit_Eta)/n + aceit_eta_n/n  # atualiza a taxa de aceitação média
    Eta.append(Eta_n)
    
    # gera uma amostra de Beta_n (Metropolis contínuo)
    Beta_n, aceit_beta_n = amostra_Beta(sigma_beta, Beta_n, Eta_n, l, m)
    aceit_Beta = (n-1)*(aceit_Beta)/n + aceit_beta_n/n  # atualiza a taxa de aceitação média
    Beta.append(Beta_n)
    
endTime = time()
elapsedTime = endTime-startTime
print(f'\nTempo gasto: {elapsedTime:.2f}s')

# Conversões auxiliares para Numpy
Alpha = np.array(Alpha)
Mu = np.array(Mu)
Delta = np.array(Delta)
Prob_delta = np.array(Prob_delta)
Eta = np.array(Eta)
P = np.array(P)
Beta = np.array(Beta)

# Imprime as taxas de aceitações dos passos de Metropolis
print("\nTaxas de aceitações:")
print(f"Alpha: {aceit_alpha:.2f}")
print(f"Mu: {aceit_mu:.2f}")
print(f"Eta: {aceit_Eta:.2f}")
print(f"Beta: {aceit_Beta:.2f}")

with open(results_path+'taxas_aceitacao.txt', 'w') as f:
    print("\nTaxas de aceitações:", file=f)
    print(f"Alpha: {aceit_alpha:.2f}", file=f)
    print(f"Mu: {aceit_mu:.2f}", file=f)
    print(f"Eta: {aceit_Eta:.2f}", file=f)
    print(f"Beta: {aceit_Beta:.2f}", file=f)

#%% Salva os vetores simulados
np.savetxt(results_path + "Alpha.txt", Alpha)
np.savetxt(results_path + "Mu.txt", Mu)
np.savetxt(results_path + "Delta.txt", Delta, fmt='%d')
np.savetxt(results_path + "Prob_delta.txt", Prob_delta)
np.savetxt(results_path + "Eta.txt", Eta, fmt='%d')
np.savetxt(results_path + "P.txt", P)
np.savetxt(results_path + "Beta.txt", Beta)