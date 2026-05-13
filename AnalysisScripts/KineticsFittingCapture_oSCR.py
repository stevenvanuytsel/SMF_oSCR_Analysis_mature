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
fname = '32bp_Cy5_32bp_BHQ2_0.1uM_open_frames.csv'

data = pd.read_csv(os.path.join(folder, fname))
print(data)

# %%
# Fit the one-step pdf 
t0 = 1/framerate # We have a truncated exponential as we don't measure up to 0
open_times = data['length_frames']*t0
t = np.asarray(open_times)
t = t[t>=t0] # impose larger than t0 (always the case)

t_shift = t-t0 # we will fit ke^-k(t-t0) due to the shift
tau_off = t_shift.mean() # characteristic time
k_off = 1.0/tau_off

# %%
# Plot the fitted pdf vs histogram
x = np.linspace(t0, t.max())
pdf = k_off*np.exp(-k_off*(x-t0))

fig, ax = plt.subplots()
ax.hist(t, ec='black', bins='auto', density=True)
ax.plot(x, pdf, lw=1)
ax.set_xlabel('t (s)')
ax.set_ylabel('density')
fig.savefig('32bp_Cy5_32bp_BHQ2_all_capture_events_histogram.svg', dpi=300)
plt.close(fig)

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
ax.plot(t_sorted, 1-CDF_theory(t_sorted, t0, k_off), '-', label='theory')
ax.set_yscale('log')
ax.set_xlabel('t (s)')
ax.set_ylabel(r'$P(\mathrm{open} \geq t)$')
fig.savefig('32bp_Cy5_32bp_BHQ2_all_capture_events_survival.svg', dpi=300)
plt.close(fig)

# %%
