# %%
import os
import time
import subprocess
import glob
import pandas as pd
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'qt')
except Exception:
    pass

# %%
# Get list of subfolders (excluding registry)
subfolders = [
    entry.path
    for entry in os.scandir(".")
    if entry.is_dir() and "registry" not in entry.name.lower()
]

# %%
# Path to the data_cleaning script
script_name = "pore_localization.py"

# Get all subdirectories in the current directory (or change root as needed)
base_dir = os.path.abspath(os.path.dirname(__file__))
subfolders = [
    entry.path
    for entry in os.scandir(base_dir)
    if entry.is_dir() and "registry" not in entry.name.lower()
]
print(subfolders)
# %%
r2 = []
intensity = []

start = time.perf_counter()
for folder in subfolders:
    print(folder)

    # Ephys
    efolder = 'cleaned_data'
    ephys_data = pd.read_csv(glob.glob(os.path.join(folder, efolder, '*.csv'))[0], index_col=0)
    
    # Tracking
    track_folder = 'pore_tracks'
    track_data = pd.read_csv(glob.glob(os.path.join(folder, track_folder, '*.csv'))[0])
    
    # Create mask based on potential
    valid_frames = (ephys_data["cleaned_frame"] != -1) & (ephys_data["Vm1 (mV)"] > -105) & (ephys_data["Vm1 (mV)"]<-95)
    valid_frames = ephys_data.loc[valid_frames, "cleaned_frame"].unique()
    
    # Extract intensity and r2 from track
    filtered_track_data = track_data[track_data['frame'].isin(valid_frames)]
    filtered_track_data["bg_corr_raw_sum"] = filtered_track_data['raw_sum']-filtered_track_data['N_pix_patch']*filtered_track_data['bkg_fix']
    
    r2.extend(filtered_track_data['r2_moffat'].to_list())
    intensity.extend(filtered_track_data['bg_corr_raw_sum'].to_list())

# %%
plt.scatter(intensity, r2, s=1)
plt.show()
    
# %%

# %%
