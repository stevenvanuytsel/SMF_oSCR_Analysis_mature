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
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN, KMeans

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
framerate = 485.44 # Hz

# %%
ephysfolder = 'cleaned_data'
trackingfolder = 'pore_tracks'

droplet = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
expname = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

savename = f'{expname}_{droplet}'
outdir = 'pore_idealization'
os.makedirs(outdir, exist_ok=True)

ephys = glob(os.path.join(f'{ephysfolder}', '*.csv'))[0]
ephys = pd.read_csv(ephys, index_col=0)

tracks = glob(os.path.join(f'{trackingfolder}', '*.csv'))[0]
tracks = pd.read_csv(tracks, index_col=0)

print(f'Working on {droplet}')

# %%
all_pfs = []
for p in tracks.track_id.unique():
    print(f'Particle {p}')
    particle_df = tracks.loc[tracks['track_id']==p]
    print(len(particle_df))
    # From ephys we can find potentials, etcetera
    pot_tgt = -95
    ephys_pf = (ephys.query("cleaned_frame >= 0")
                .groupby('cleaned_frame', as_index=False)
                .agg(current_median=('Im1 (pA)', 'median'),
                    potential_median=('Vm1 (mV)','median'),
                     potential_min=('Vm1 (mV)','min'),
                     potential_max = ('Vm1 (mV)', 'max'),
                     median_green = ('median_green', 'median'), # median green is one value already in the df
                     median_red = ('median_red', 'median'))
                .rename(columns={'cleaned_frame':'frame'}))

    ephys_pf['is_ctrl_any'] = ephys_pf['potential_max']>pot_tgt
    # Merge electrical data with optical data
    pf = (particle_df.merge(ephys_pf, on='frame', how='left')
                     .sort_values('frame')
                     .reset_index(drop=True))
    
    # Make a column with background corrected intensity
    pf['bg_corr_raw_sum'] = pf['raw_sum']-pf['N_pix_patch']*pf['bkg_fix']

    # Filter 1: skip if trace is mostly control potential
    frac_ctrl = pf['is_ctrl_any'].mean()
    if frac_ctrl > 0.85:
        continue

    # Filter 2: Normalize intensity on non-control potentials and look at intensity and r2 
    non_ctrl = pf[~pf['is_ctrl_any']].copy()
    if len(non_ctrl) == 0:
        continue

    non_ctrl_intensity_max = non_ctrl['bg_corr_raw_sum'].max()
    pf['norm_bg_corr_raw_sum'] = pf['bg_corr_raw_sum']/non_ctrl_intensity_max
    
    # Filter 2.1: if there are normalized intensities >1 in the trace after normalizing on non-control
    # it means that the non-control frames are lower intensity than the control -> red flag, skip
    if (pf['norm_bg_corr_raw_sum']>1.0+1e-2).any():
        continue

    # Filter 2.2: if cluster with highest intensity is centred lower than 0.5, it's probably bad
    # Cluster the non-control ones
    X = np.column_stack([pf['norm_bg_corr_raw_sum'].values, pf['r2_moffat'].values])
    labels = DBSCAN(eps=0.05, min_samples=5).fit_predict(X)
    pf['cluster'] = labels
    valid = pf[pf['cluster']!=-1] #reject noise

    # Find cluster with highest median intensity
    main_cid = (valid.groupby("cluster")["norm_bg_corr_raw_sum"].median().idxmax())
    main = valid[valid["cluster"]==main_cid]
    centre_i = main["norm_bg_corr_raw_sum"].mean()
    centre_r2 = main["r2_moffat"].mean()

    # Ignore traces where highest intensity cluster has very poor fitting, indicative of no open pore in trace
    if centre_r2<0.5:
        continue

    # Perform changepoint detection on the traces
    # Calculate the derivative

    dI = np.diff(pf['norm_bg_corr_raw_sum'])
    abs_d_I = np.abs(dI)
    # Find noise
    med_dI = np.median(dI)
    mad_dI = np.median(np.abs(dI-med_dI))
    noise_std = 1.4826*mad_dI
    k=5.0
    thresh = k*noise_std

    peaks_up, _ = find_peaks(dI, height=thresh)
    peaks_down, _ = find_peaks(-dI, height=thresh)

    candidate_cps = np.sort(np.concatenate([peaks_up, peaks_down]))
    # Build segments in the pf
    N = len(pf)
    segments = []
    prev=0
    for k in candidate_cps:
        if k <= prev:
            continue
        segments.append((prev, k))
        prev = k
    segments.append((prev, N))

    pf['segment_id'] = -1
    for j, (s, e) in enumerate(segments):
        pf.loc[pf.index[s:e], 'segment_id'] = j
    
    all_pfs.append(pf)

pf_all = pd.concat(all_pfs, ignore_index=True)

# %%
# Remove the per-trace cluster column because we can't use it anymore
if "cluster" in pf_all.columns:
    pf_all = pf_all.drop(columns=["cluster"])
print(pf_all)

# %%
# Per track normalization
pf_all['I_norm_track'] = np.nan
for tid, dt_t in pf_all.groupby('track_id'):
    non_ctrl = dt_t[~dt_t['is_ctrl_any'].astype(bool)]
    I_max = non_ctrl['bg_corr_raw_sum'].max()
    idx = dt_t.index
    pf_all.loc[idx, 'I_norm_track'] = pf_all.loc[idx, 'bg_corr_raw_sum']/I_max

mask = (~pf_all['is_ctrl_any'].astype(bool)) & pf_all['I_norm_track'].notna()
I_norm = pf_all.loc[mask, 'I_norm_track'].to_numpy().reshape(-1, 1)
R2 = pf_all.loc[mask, ['r2_moffat']].to_numpy().reshape(-1,1)

X = np.column_stack([I_norm, R2])
km = KMeans(n_clusters=2, n_init=50, random_state=0).fit(X)
labels = km.labels_
centers = km.cluster_centers_

pf_all.loc[mask, "cluster_global"] = labels

# Assign clusters to closed and open based on mean intensity
means = [I_norm[labels == k].mean() for k in (0, 1)]
closed_label = int(np.argmin(means))
open_label   = 1 - closed_label

# %%
plt.figure(figsize=(6, 5))
unique_labels = np.unique(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for col, lab in zip(colors, unique_labels):
    sel = labels == lab
    name = "closed" if lab == closed_label else "open"
    plt.scatter(I_norm[sel], R2[sel],
                c=[col], s=4, alpha=0.6, label=f"{name} (cl {lab})")

# plot cluster centers
plt.scatter(centers[:, 0], centers[:, 1],
            c="k", s=60, marker="x", label="centers")

plt.xlabel("I_norm_track (per track norm)")
plt.ylabel("r2_moffat")
plt.legend(markerscale=3, fontsize=8)
plt.tight_layout()
plt.show()

# %%

# point states
state_map = {closed_label: "closed", open_label: "open"}
pf_all.loc[mask, "state_point"] = pf_all.loc[mask, "cluster_global"].map(state_map)
pf_all.loc[pf_all["is_ctrl_any"].astype(bool), "state_point"] = "control"

# 2) segment state only from non-control frames
seg_states = (
    pf_all.loc[mask]
          .groupby(["track_id", "segment_id"])["state_point"]
          .value_counts(normalize=True)
          .unstack(fill_value=0)
          .reset_index()
)
seg_states["state_seg"] = np.where(
    seg_states["open"] > seg_states["closed"], "open", "closed"
)
pf_all = pf_all.merge(
    seg_states[["track_id", "segment_id", "state_seg"]],
    on=["track_id", "segment_id"],
    how="left",
)

pf_all.loc[pf_all["state_point"] == "control", "state_seg"] = "control"

# --- 4) build a refined segment id that follows state_seg changes ---
# We stay fully procedural, no custom functions.

# start with zeros
pf_all["segment_id_refined"] = 0

# process track by track
for tid, idx in pf_all.groupby("track_id").groups.items():
    idx = idx.sort_values()
    states = pf_all.loc[idx, "state_seg"].to_numpy()

    n = len(states)
    new_seg_ids = np.zeros(n, dtype=int)

    # first pass: define initial segments from raw state changes
    seg_start = [0]
    seg_state = [states[0]]

    for i in range(1, n):
        if states[i] != states[i-1]:
            seg_start.append(i)
            seg_state.append(states[i])

    seg_start.append(n)  # sentinel end

    # now seg_start[k]..seg_start[k+1]-1 is segment k with state seg_state[k]

    # 1) merge adjacent segments with identical labels (can occur after later edits)
    merged_start = [seg_start[0]]
    merged_state = [seg_state[0]]

    for k in range(1, len(seg_state)):
        if seg_state[k] == merged_state[-1]:
            # same label as previous, just extend last segment
            continue
        else:
            merged_start.append(seg_start[k])
            merged_state.append(seg_state[k])
    merged_start.append(n)

    # 2) remove 1-frame segments between different neighbors
    #    e.g. control | closed(1 frame) | open  -> control | open
    cleaned_start = [merged_start[0]]
    cleaned_state = [merged_state[0]]

    for k in range(1, len(merged_state) - 1):
        start_k = merged_start[k]
        end_k = merged_start[k+1]
        length_k = end_k - start_k

        prev_state = cleaned_state[-1]
        next_state = merged_state[k+1]

        if length_k == 1 and prev_state != merged_state[k] and next_state != merged_state[k]:
            # micro segment with different neighbors on both sides
            # decide how to merge: here we bias towards the right if prev is control and next is open/closed
            if prev_state == "control" and next_state in ("open", "closed"):
                # absorb into next segment
                pass  # do nothing, the one frame will get next_state later
            elif next_state == "control" and prev_state in ("open", "closed"):
                # absorb into previous
                # we simply do not start a new segment; frame will take prev_state
                pass
            else:
                # symmetric case open-closed-open etc., merge into next by default
                pass
        else:
            cleaned_start.append(start_k)
            cleaned_state.append(merged_state[k])

    # always keep last segment
    cleaned_start.append(merged_start[-2])  # start of last segment
    cleaned_state.append(merged_state[-1])
    cleaned_start.append(n)

    # 3) assign new refined ids from cleaned segments
    curr_seg = 0
    for k in range(len(cleaned_state)):
        s0 = cleaned_start[k]
        s1 = cleaned_start[k+1]
        new_seg_ids[s0:s1] = curr_seg
        curr_seg += 1

    # write back
    pf_all.loc[idx, "segment_id_refined"] = new_seg_ids

pf_all.to_csv(os.path.join(outdir, f"{savename}_idealization.csv"), index=False)
# %%
# simple color map for segment state
seg_color = {"open": "tab:red", "closed": "tab:green", "control": "tab:blue", None: "lightgray"}

for p in pf_all.track_id.unique():
    df_p = pf_all[pf_all["track_id"] == p].sort_values("frame")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(
        df_p["frame"],
        df_p["bg_corr_raw_sum"],
        color="0.7",
        lw=0.8,
        label="intensity",
    )

    # shading: refined segments (open/closed/control)
    for seg_ref, seg_df in df_p.groupby("segment_id_refined"):
        state = seg_df["state_seg"].iloc[0]
        color = seg_color.get(state, "lightgray")
        f0, f1 = seg_df["frame"].min(), seg_df["frame"].max()
        ax.axvspan(f0, f1, color=color, alpha=0.15)

    # changepoints: original segment_id boundaries
    for seg, seg_df in df_p.groupby("segment_id"):
        f0 = seg_df["frame"].min()
        ax.axvline(f0, color="k", alpha=0.4, lw=0.7)

    # control frames markers (optional, if you still want them)
    ctrl = df_p["is_ctrl_any"].astype(bool)
    ax.scatter(
        df_p.loc[ctrl, "frame"],
        df_p.loc[ctrl, "bg_corr_raw_sum"],
        s=10,
        c="tab:blue",
        marker="x",
        label="control frames",
    )

    ax.set_title(f"track_id {p}")
    ax.set_xlabel("frame")
    ax.set_ylabel("bg_corr_raw_sum")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"{savename}_particle_{p}.png"))
    plt.close(fig)

# %%
