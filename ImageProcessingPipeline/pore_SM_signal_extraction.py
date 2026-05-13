# %%
# File handling
from glob import glob
import os

# I/O
from skimage import io

# Processing
import pandas as pd
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN, KMeans
from scipy.stats import pearsonr
from hmmlearn.hmm import GaussianHMM
from scipy import sparse
from scipy.sparse.linalg import spsolve


# Visualization
from matplotlib import rcParams
import matplotlib.pyplot as plt
try:
    get_ipython().run_line_magic('matplotlib', 'qt')
except Exception:
    pass
plt.style.reload_library()
plt.style.use('2025VanuytselNRJ')

# %%
def subtract_pore_signal(sm_signal, pore_full, q_quiet=90):
    sm_signal = np.asarray(sm_signal, float)
    pore_full = np.asarray(pore_full, float)

    quiet_mask = sm_signal < np.percentile(sm_signal, q_quiet)

    pore_q = pore_full[quiet_mask]
    sm_q = sm_signal[quiet_mask]

    X = np.vstack([pore_q, np.ones_like(pore_q)]).T
    alpha, beta = np.linalg.lstsq(X, sm_q, rcond=None)[0]

    sm_resid = sm_signal - (alpha * pore_full + beta)
    return sm_resid, alpha, beta

def subtract_baseline(sm_resid, w=101, q=0.5):
    baseline = (
        pd.Series(sm_resid)
        .rolling(window=w, center=True, min_periods=1)
        .quantile(q)
        .to_numpy()
    )
    sm_corr = sm_resid - baseline
    return sm_corr, baseline

def estimate_sigma(z):
    z = np.asarray(z, float)
    central = z[np.abs(z) <= np.percentile(np.abs(z), 50)]
    if len(central) < 10:
        return 0.0
    return central.std()    

def run_hmm_trace(z):
    z = np.asarray(z, float)
    X = z.reshape(-1, 1)

    low_init = np.percentile(z, 25)
    high_init = np.percentile(z, 90)
    var_init = max(np.var(z), 1e-6)

    model = GaussianHMM(
        n_components=2,
        covariance_type='diag',
        n_iter=200,
        tol=1e-4,
        init_params='',
        params='stmc'
    )

    model.startprob_ = np.array([0.95, 0.05])
    model.transmat_ = np.array([
        [0.995, 0.005],
        [0.05,  0.95 ]
    ])
    model.means_ = np.array([[low_init], [high_init]])
    model.covars_ = np.array([[var_init], [var_init]])

    model.fit(X)
    states = model.predict(X)

    state_means = model.means_.ravel()
    low_state = np.argmin(state_means)
    high_state = np.argmax(state_means)

    event_mask = (states == high_state).astype(int)

    return states, event_mask, model

# %%
framerate = 485.44 # Hz

# %%
# Load idealized traces
datafolder = 'pore_idealization'
droplet = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
expname = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
dataname = f'{expname}_{droplet}_idealization.csv'
data = pd.read_csv(os.path.join(datafolder, dataname))

# Make folder
savefolder = 'pore+SM'
os.makedirs(savefolder, exist_ok=True)
savename = f'{expname}_{droplet}'

# %%
# Load closed segments
segmentfolder = 'D:/KCL_PhD_THESIS/R9.4.1_simulmeas/32bp_Cy5_32bp_BHQ2_0.1uM/oscr_kinetics_analysis'
fname = '32bp_Cy5_32bp_BHQ2_0.1uM_closed_frames.csv'
segments = pd.read_csv(os.path.join(segmentfolder, fname))

# Find all rows belonging to this recording
droplet2 = droplet.split('_-100')[0]
local_segments = segments.loc[segments['savename']==f'{expname}_{droplet2}'].copy()

if local_segments.empty:
    print(f'No segments found for {expname}_{droplet2}, exiting')
    import sys
    sys.exit(0)

# %%
# Load single molecule trace, registration and pore trace
stackfolder = 'cleaned_data'
smname = glob(os.path.join(stackfolder, '*_red.tif'))[0]
porename = glob(os.path.join(stackfolder, '*_green.tif'))[0]
sm_stack = io.imread(smname)
pore_stack = io.imread(porename)

# Remember, shift has to be applied to red stack
registration_params = pd.read_csv(os.path.join('..', 'registry', 'registration_params.csv'), index_col=0)

window_big = 5
window_small = 3

# %%
# Loop over particles and extract interesting data
subdfs = []
for p in local_segments['particle'].unique():
    print(f'particle {p}')

    subdf = data.loc[data['track_id']==p].copy()
    sm_signal_big = []
    sm_signal_small = []
    for idx, row in subdf.iterrows():
        # print(f'{idx} out of {len(subdf)}')

        if row['r2_moffat']>0.8:
            px, py = row['x_fit'], row['y_fit']
        else:
            px, py = row['x_centr'], row['y_centr']
        
        # Transform locations
        smx = px + registration_params['shift_x'].iloc[0]
        smy = py + registration_params['shift_y'].iloc[0]

        # Turn into pixel location
        smx, smy = int(round(smx)), int(round(smy))

        frame = int(row['frame'])

        smpatch_big = sm_stack[frame, smy-window_big//2:smy+window_big//2+1, smx-window_big//2:smx+window_big//2+1]
        sm_signal_big.append(smpatch_big.sum())

        smpatch_small = sm_stack[frame, smy-window_small//2:smy+window_small//2+1, smx-window_small//2:smx+window_small//2+1]
        sm_signal_small.append(smpatch_small.sum())

    # Assume that sm signal is scaled_pore+sm signal+offset
    # Don't use all frames to estimate, only the ones where we don't have peaks
    pore_full = subdf['bg_corr_raw_sum'].to_numpy(dtype=float)

    sm_resid_big, alpha_big, beta_big = subtract_pore_signal(sm_signal_big, pore_full, q_quiet=90)
    sm_resid_small, alpha_small, beta_small = subtract_pore_signal(sm_signal_small, pore_full, q_quiet=90)

    sm_corr_big, baseline_big = subtract_baseline(sm_resid_big, w=101, q=0.5)
    sm_corr_small, baseline_small = subtract_baseline(sm_resid_small, w=101, q=0.5)

    sigma = estimate_sigma(sm_corr_small)
    sigma_thresh = 3
    if sigma>0 and np.any(sm_corr_small>=sigma_thresh*sigma):
        states_big, events_big, model_big = run_hmm_trace(sm_corr_big)
        states_small, events_small, model_small = run_hmm_trace(sm_corr_small)

        event_mask_confirmed =(events_big & events_small).astype(int)
    else:
        event_mask_confirmed = np.zeros(len(sm_signal_small))

    subdf['sm_raw_sum_small'] = sm_signal_small
    subdf['sm_raw_sum_big'] = sm_signal_big
    subdf['sm_pore_baseline_corrected_big'] = sm_corr_big
    subdf['sm_pore_baseline_corrected_small'] = sm_corr_small
    subdf['sm_events'] = event_mask_confirmed

    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(subdf['frame'], subdf['bg_corr_raw_sum'])
    ax[1].plot(subdf['frame'], subdf['sm_pore_baseline_corrected_small'])
    ax[2].plot(subdf['frame'], subdf['sm_events'])
    plt.tight_layout()
    fig.savefig(os.path.join(savefolder, f'{savename}_particle_{p}_pore_sm.png'))
    plt.close(fig)

    subdfs.append(subdf)
# %%
data_with_sm = pd.concat(subdfs, ignore_index=True)

# %%
# Save
data_with_sm.to_csv(os.path.join(savefolder, f'{savename}_pore_sm.csv'), index=False)


# %%
