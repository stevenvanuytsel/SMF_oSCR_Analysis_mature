# %%
import os
import numpy as np
import pandas as pd
import scipy
from skimage import io


import matplotlib.pyplot as plt
import matplotlib
import napari
try:
    get_ipython().run_line_magic('matplotlib', 'qt')
except Exception:
    pass

matplotlib.style.reload_library()
plt.style.use('2025VanuytselNRJ')

# %%
# First look at the median intensity per frame to see if the average fluorescence intensity changes over time
datasets = ['001']
framerate = 485.44 # Hz

median_bg_intensity = []
times = []
voltage = []

for d in datasets:
    data = pd.read_csv(os.path.join(f'D:/KCL_PhD_THESIS/R9.4.1_simulmeas/32bp_Cy5_32bp_BHQ2_0.1uM/20211010/data_cleaning+tracking/droplet_1_{d}_-100mV/cleaned_data/20211010_droplet_1_{d}_-100mV_ephys.csv'))
    data = data[data['median_green'].notna()]
    data_grouped = data.groupby('cleaned_frame').agg({
        'Vm1 (mV)': 'mean',
        'median_green': 'mean',
        'median_red': 'mean',
        'Im1 (pA)': 'mean'
    }).reset_index()

    median_bg_intensity.append(data_grouped['median_green'].to_list())
    times.append((data_grouped['cleaned_frame']/framerate).to_list())
    voltage.append(data_grouped['Vm1 (mV)'].to_list())

# %%
# Calculate the Fourier spectrum
fourier_bg_intensity = []
freq = []

dt = 1./framerate

for idx, i in enumerate(median_bg_intensity):
    fourier_bg_intensity.append(np.fft.fftshift(np.fft.fft(i)))
    freq.append(np.fft.fftshift(np.fft.fftfreq(len(i), d = dt)))

fig, ax = plt.subplots()
ax.plot(freq[0], np.log(np.abs(fourier_bg_intensity[0])))
ax.axvline(14.13, c='black', ls='--')
ax.set_xlim(-50, 50)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'log$_{10}(|\mathrm{FFT}(\tilde{F})|)$')
fig.savefig('FT_median_background_intensity.svg')
plt.close(fig)


# %%
fig, ax = plt.subplots()
for idx, t in enumerate(times):
    ax.plot(t, median_bg_intensity[idx]/np.mean(median_bg_intensity[idx]))
    ax.set_xlim(0, 6)
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\tilde{\mathrm{F}}$ (a.u.)')
fig.savefig('Median_fluorescence_intensity.svg')
plt.close(fig)

fig, ax = plt.subplots()
for idx, t in enumerate(times):
    ax.plot(t, voltage[idx])
    ax.set_xlim(0, 6)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Applied potential (mV)')
fig.savefig('Voltage_protocol.svg')
plt.close(fig)


# %%
