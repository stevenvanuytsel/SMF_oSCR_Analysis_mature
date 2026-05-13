# %%
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize

%matplotlib qt
matplotlib.style.reload_library()
plt.style.use('2025VanuytselNRJ')

# %%
framerate = 485.44 # Hz

# %%
# capture rate
folder = 'D:/KCL_PhD_THESIS/R9.4.1_simulmeas/32bp_Cy5_32bp_BHQ2_0.1uM/oscr_kinetics_analysis'
fname = '32bp_Cy5_32bp_BHQ2_0.1uM_closed_frames.csv'

data = pd.read_csv(os.path.join(folder, fname))
print(data)

# %%
# Fit the one-step pdf 
t0 = 1/framerate # We have a truncated exponential as we don't measure up to 0
closed_times = data['length_frames']*t0
t = np.asarray(closed_times)
t = t[t>=t0] # impose larger than t0 (always the case)

t_shift = t-t0 # we will fit ke^-k(t-t0) due to the shift
tau_on = t_shift.mean() # characteristic time
k_on = 1.0/tau_on

# %%
# Plot the fitted pdf vs histogram
x = np.linspace(t0, t.max())
pdf = k_on*np.exp(-k_on*(x-t0))

fig, ax = plt.subplots()
ax.hist(t, ec='black', bins='auto', density=True)
ax.plot(x, pdf, lw=1)
ax.set_xlabel('t (s)')
ax.set_ylabel('density')
plt.show()

# %%
# Plot the empirical cdf vs the fitted one
t_sorted = np.sort(t)
CDF_empirical = np.arange(1, len(t_sorted)+1)/len(t_sorted)

# Shift the theoretical CDF as we have a truncated distribution
def CDF_theory(x, x0, k):
    return 1-np.exp(-k*(x-x0))

# Plot the survival probability (1-CDF) on semilog to assess how good it is
fig, ax = plt.subplots()
ax.plot(t_sorted, 1-CDF_empirical, '.', label='data')
ax.plot(t_sorted, 1-CDF_theory(t_sorted, t0, k_on), '-', label='theory')
ax.set_yscale('log')
ax.set_xlabel('t (s)')
ax.set_ylabel(r'$P(\mathrm{open} \geq t)$')
plt.show()

# %%
# Clearly not a one-step proces
# Let's try a two-step process first

def pdf_2step(t, k1, k2):
    t = np.asarray(t)
    coeff = (k1*k2)/(k2-k1)
    f = coeff*(np.exp(-k1*t)-np.exp(-k2*t))
    return np.clip(f, 1e-300, np.inf) #ensure we don't return 0

def S_2step(t, k1, k2):
    t = np.asarray(t)
    numerator = k2*np.exp(-k1*t)-k1*np.exp(-k2*t)
    denominator = (k2-k1)
    S = numerator/denominator
    return np.clip(S, 1e-300, 1.0)

def neg_loglik_2step(params, t, t0):
    log_k1, log_k2 = params
    k1 = np.exp(log_k1)
    k2 = np.exp(log_k2)

    t = np.asarray(t)
    t = t[t>=t0]

    pdf = pdf_2step(t, k1, k2)
    S_t0 = S_2step(t0, k1, k2)

    loglik = np.sum(np.log(pdf))-len(t)*np.log(S_t0)
    return -loglik

from scipy.optimize import minimize
def fit_2step(t, t0, k1_init=None, k2_init=None):
    t = np.asarray(t)
    t = t[t >= t0]

    # crude initial guesses from shifted mean (like your 1-step estimate)
    if k1_init is None or k2_init is None:
        tau_mean = t.mean() - t0
        k_mean = 1.0 / tau_mean
        k1_init = 3.0 * k_mean
        k2_init = 0.3 * k_mean

    params0 = np.log([k1_init, k2_init])

    res = minimize(
        neg_loglik_2step,
        params0,
        args=(t, t0),
        method='Nelder-Mead'
    )

    log_k1, log_k2 = res.x
    k1 = np.exp(log_k1)
    k2 = np.exp(log_k2)

    return k1, k2, res

data = pd.read_csv(os.path.join(folder, fname))
print(data)

# Fit the 2-step pdf 
t0 = 1/framerate # We have a truncated exponential as we don't measure up to 0
closed_times = data['length_frames']*t0
t = np.asarray(closed_times)
t = t[t>=t0] # impose larger than t0 (always the case)

k1, k2, res=  fit_2step(t, t0)
print(k1, k2)

# Plot
t_sorted = np.sort(t)
CDF_empirical = np.arange(1, len(t_sorted)+1)/len(t_sorted)

S_untr = S_2step(t_sorted, k1, k2)
S_t0 = S_2step(t0, k1, k2)
S_model = S_untr/S_t0

fig, ax = plt.subplots()
ax.plot(t_sorted, 1-CDF_empirical, '.', label='data')
ax.plot(t_sorted, S_model, '-')
ax.set_yscale('log')
plt.show()

# %%
# Sum of two exponentials
def pdf_2exp(t, k1, k2, a):
    t = np.asarray(t)
    pdf = a * k1 * np.exp(-k1 * t) + (1.0 - a) * k2 * np.exp(-k2 * t)
    return np.clip(pdf, 1e-300, np.inf)

def S_2exp(t, k1, k2, a):
    t = np.asarray(t)
    S = a * np.exp(-k1 * t) + (1.0 - a) * np.exp(-k2 * t)
    return np.clip(S, 1e-300, 1.0)

def neg_loglik_2exp(params, t, t0):
    log_k1, log_k2, logit_a = params
    k1 = np.exp(log_k1)
    k2 = np.exp(log_k2)
    a  = 1.0 / (1.0 + np.exp(-logit_a))  # sigmoid to keep a in (0,1)

    t = np.asarray(t)
    t = t[t >= t0]

    # untruncated pdf at data points
    pdf = pdf_2exp(t, k1, k2, a)
    # survival at t0
    S_t0 = S_2exp(t0, k1, k2, a)

    # same logic: log L = sum log f_T(t_i) - N log S_T(t0)
    loglik = np.sum(np.log(pdf)) - len(t) * np.log(S_t0)
    return -loglik

from scipy.optimize import minimize
def fit_2exp(t, t0, k1_init=None, k2_init=None, a_init=None):
    t = np.asarray(t)
    t = t[t >= t0]

    # crude initial guesses from shifted mean (like your 1-step estimate)
    if k1_init is None or k2_init is None or a_init is None:
        tau_mean = t.mean() - t0
        k_mean = 1.0 / tau_mean
        if k1_init is None: 
            k1_init = 3.0 * k_mean
        if k2_init is None: 
            k2_init = 0.3 * k_mean
        if a_init  is None: 
            a_init  = 0.5
       
    params0 = np.log([k1_init, k2_init, a_init/(1-a_init)])

    res = minimize(
        neg_loglik_2exp,
        params0,
        args=(t, t0),
        method='Nelder-Mead'
    )

    log_k1, log_k2, logit_a = res.x
    k1 = np.exp(log_k1)
    k2 = np.exp(log_k2)
    a = 1.0/(1.0+np.exp(-logit_a))

    return k1, k2, a, res

data = pd.read_csv(os.path.join(folder, fname))
print(data)

# Fit the 2-step pdf 
t0 = 1/framerate # We have a truncated exponential as we don't measure up to 0
closed_times = data['length_frames']*t0
t = np.asarray(closed_times)
t = t[t>=t0] # impose larger than t0 (always the case)

k1, k2, a, res=  fit_2exp(t, t0)
print(k1, k2, a)
print(k1, k2)

# Plot
t_sorted = np.sort(t)
CDF_empirical = np.arange(1, len(t_sorted)+1)/len(t_sorted)

S_untr = S_2exp(t_sorted, k1, k2, a)
S_t0 = S_2exp(t0, k1, k2, a)
S_model = S_untr/S_t0

fig, ax = plt.subplots()
ax.plot(t_sorted, 1-CDF_empirical, '.', label='data')
ax.plot(t_sorted, S_model, '-')
ax.set_yscale('log')
ax.set_xlabel('t (s)')
ax.set_ylabel(r'$P(\mathrm{closed} \geq t)$')
fig.savefig('32bp_Cy5_32bp_BHQ2_all_unzipping_events_survival.svg', dpi=300)
plt.close(fig)

# %%
p95 = np.percentile(t, 95)
mask = t <=p95
t_hist = t[mask]

x = np.linspace(t0, t_hist.max(), 200)
pdf = pdf_2exp(x, k1, k2, a)
pdf_trunc = pdf/S_t0

fig, ax = plt.subplots()
ax.hist(t_hist, bins='auto', ec='black', density=True)
ax.plot(x, pdf)
ax.set_xlabel('t (s)')
ax.set_ylabel('density')
fig.savefig('32bp_Cy5_32bp_BHQ2_all_unzipping_events_perc95_histogram.svg', dpi=300)
plt.close(fig)


# %%
# Assign probabilities to each of the closed states to see where they belong to
def posterior_probs_2exp(t, k1, k2, a):
    t = np.asarray(t)
    num1 = a * k1 * np.exp(-k1 * t)
    num2 = (1.0 - a) * k2 * np.exp(-k2 * t)
    denom = num1 + num2
    p1 = num1 / denom          # P(regime 1 | t)
    p2 = num2 / denom          # P(regime 2 | t)
    return p1, p2

p_fast, p_slow = posterior_probs_2exp(t, k1, k2, a)

print(p_fast, p_slow)
# %%
# Add them to the dataframe
data['p_k1'] = p_fast
data['p_k2'] = p_slow

# %%
# Save the dataframe back to where it came from 
data.to_csv(os.path.join(folder, fname), index=False)
# %%
