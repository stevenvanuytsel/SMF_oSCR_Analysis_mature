# %%
import os
import numpy as np
import pandas as pd
import scipy
from skimage import io
import pyabf

import matplotlib.pyplot as plt
import napari
try:
    get_ipython().run_line_magic('matplotlib', 'qt')
except Exception:
    pass

# %%
blankname = 'blank_270em.tif'
droplet = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
fpath = os.path.join('..', '..', f'{droplet}.tif')
outdir = 'cleaned_data'
os.makedirs(outdir, exist_ok=True)
expname = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
savename = f"{expname}_{droplet}"
print(savename)

# Set up starting potential 
first_applied_potential = int(droplet.split('_')[-1].split('mV')[0]) +5 # mV
last_applied_potential = 95 # mV, always the same

# %%
# Registration values
regpath = os.path.join('..', 'registry', 'registration_params.csv')
regparams = pd.read_csv(regpath, index_col=0)
shift_y, shift_x, rows = regparams['shift_y'].iloc[0], regparams['shift_x'].iloc[0], regparams['rows'].iloc[0]

framerate = 485.44

# Load in stack
stack = io.imread(fpath)

# %%
# Load in electrical data
epath = os.path.join('..', '..', f'{droplet}.abf')

def load_abf_to_df(path):
    abf = pyabf.ABF(path)
    dfs = []
    for s in range(abf.sweepCount):
        # time for this sweep
        abf.setSweep(s, channel=0)
        t = abf.sweepX  # seconds
        df = pd.DataFrame({"time (s)": t + s*abf.sweepLengthSec})
        # add every recorded ADC channel
        for ch in range(abf.channelCount):
            abf.setSweep(s, channel=ch)
            name = (abf.adcNames[ch] or f"ch{ch}").strip()
            unit = (abf.adcUnits[ch] or "").strip()
            df[f"{name} ({unit})".strip()] = abf.sweepY.copy()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), abf.dataRate

ephys, datarate = load_abf_to_df(epath)

# Sometimes shutter is inf and then we can swap it in from a different recording as it's always the same
if np.isinf(ephys['Shutter (mV)']).any():
    shutter = pd.read_csv(os.path.join('..', 'shutter_fallback.csv'), index_col=0)
    ephys['Shutter (mV)'] = shutter

# Somtimes voltage isn't transcribed correctly so we do proper fallback there too
if np.isinf(ephys['Vm1 (mV)']).any():
    voltage = pd.read_csv(os.path.join('..', 'voltage_fallback.csv'), index_col=0)
    ephys['Vm1 (mV)'] = voltage

# Shutter is open at start because that triggered the recording
# Assign frame indices
ephys['original_frame'] = (ephys['time (s)']/(1/framerate)).astype(int)

# Ensure that there are no gaps in the original_frame column
frames_in = np.sort(ephys['original_frame'].unique())
expected = np.arange(ephys['original_frame'].min(), ephys['original_frame'].max() + 1, dtype=int)
assert np.array_equal(frames_in, expected), \
    f"Original_frame gap detected. Missing={set(expected) - set(frames_in)}, Extra={set(frames_in) - set(expected)}"

# Need to find the first frame that is entirely first applied potential (as the ephys isn't synced with the camera)
frame_max_voltage = ephys.groupby("original_frame")['Vm1 (mV)'].max() # group by frame and take max (first potential is negative so max would be 0 mV)
valid_frames = frame_max_voltage[frame_max_voltage < first_applied_potential].index

buffer = 3 # Add frames to be in the clear

first_valid_frame = valid_frames.min()+buffer # Throw out the first frames to be in the clear

# Find where shutter closes (that's the end of the stack)
shutter_close_index = ephys.index[ephys['Shutter (mV)']<4000].min()
last_valid_frame = ephys.loc[shutter_close_index-1, 'original_frame']
ephys.loc[shutter_close_index:, 'original_frame'] = -1

# Define a new column that shows which frames will end up in the cleaned stack, and which frame they are
ephys['cleaned_frame'] = -1  # start with all -1
mask = (ephys['original_frame'] >= first_valid_frame) & (ephys['original_frame'] <= last_valid_frame)
ephys.loc[mask, 'cleaned_frame'] = ephys.loc[mask, 'original_frame'] - first_valid_frame

# %%
# Pick the frames in the cleaned stack that correspond to the ones we're interested in
cleaned_stack = np.copy(stack[first_valid_frame:, :, :]).astype(np.float32)

# Subtract median dark frame from stack before shifting
dark_stack = io.imread(os.path.join('..', '..', f'{blankname}'))
median_dark = np.median(dark_stack, axis=0).astype(np.float32)
cleaned_stack-=median_dark

# %%
# Split the stack into the two channels
cleaned_stack_green = cleaned_stack[:, 1:rows, :]
cleaned_stack_red = cleaned_stack[:, -rows+1:, :]

# %%
# Need to find the first frame where the control voltage is applied
frame_min_voltage = ephys.groupby("cleaned_frame")['Vm1 (mV)'].min()
valid_frames = frame_min_voltage[frame_min_voltage > last_applied_potential].index

first_control_frame = valid_frames.min()+buffer 

# %%
# GREEN
# Divide away the emission profile for both stacks
emission_profile_green = np.median(cleaned_stack_green[first_control_frame:], axis=0).astype(np.float32)
# Rescale to 1 so we don't change overall brightness
emission_profile_green /= emission_profile_green.mean()
# Ensure it doesn't contain any zeros
emission_profile_green = np.clip(emission_profile_green, 1e-6, None)

# Divide
cleaned_stack_green_flattened = cleaned_stack_green/emission_profile_green

# Throw away control frames because we don't need them and adjust ephys
cleaned_stack_green_flattened = cleaned_stack_green_flattened[:first_control_frame-buffer]

# %%
# RED (repeat of the oSCR channel but with less blur)
emission_profile_red = np.median(cleaned_stack_red[first_control_frame:], axis=0).astype(np.float32)
emission_profile_red /= emission_profile_red.mean()
emission_profile_red = np.clip(emission_profile_red, 1e-6, None)
cleaned_stack_red_flattened = cleaned_stack_red/emission_profile_red
cleaned_stack_red_flattened = cleaned_stack_red_flattened[:first_control_frame-buffer]

# %%
# We don't filter, we will just shift the ROI in the red channel!!

# %%
# Extract the median values for every channel and save in the ephys dataframe
num_frames = cleaned_stack_green_flattened.shape[0]
idx = np.arange(num_frames, dtype=int)

med_green = np.median(cleaned_stack_green_flattened, axis=(1,2))
med_red = np.median(cleaned_stack_red_flattened, axis=(1,2))

ephys['median_green'] = ephys['cleaned_frame'].map(pd.Series(med_green, index=idx))
ephys['median_red']   = ephys['cleaned_frame'].map(pd.Series(med_red,   index=idx))

# mark post-control frames as -1
mask = ephys['cleaned_frame'] >= first_control_frame-buffer
ephys.loc[mask, 'cleaned_frame'] = -1


# %%
# Save stacks and dataframe, cast the values to integers
ephys.to_csv(os.path.join(outdir, f'{savename}_ephys.csv'), index=True)
io.imsave(os.path.join(outdir, f'{savename}_green.tif'), np.clip(cleaned_stack_green_flattened, 0, np.iinfo(stack.dtype).max).astype(stack.dtype))
io.imsave(os.path.join(outdir, f'{savename}_red.tif'), np.clip(cleaned_stack_red_flattened, 0, np.iinfo(stack.dtype).max).astype(stack.dtype))


# %%
# # TESTING STUFF

# # Let's look at the intensities (using the tracking we did before)
# tracks = pd.read_csv(os.path.join('pore_tracking', '20211010_droplet_1_001_-100mV.csv'), index_col=0)

# # Look at the intensities
# window = 3
# particles = tracks.particle.unique()

# for p in particles:
#     p_df = tracks.loc[tracks['particle']==p]

#     intensities = []
#     positions = np.zeros((len(p_df), 3))
#     for i, (idx, row) in enumerate(p_df.iterrows()):
#         frame, px, py = row['frame'], int(np.round(row['x_centroid'])), int(np.round(row['y_centroid']))
#         intensities.append(np.sum(cleaned_stack_green_flattened[frame, py-window//2:py+window//2+1, px-window//2:px+window//2+1]))
#         positions[i] = [frame, py, px]
    
#     points = positions.copy()
#     points[:, 0] -= positions[0,0]
#     viewer = napari.Viewer()
#     viewer.add_image(cleaned_stack_green_flattened[positions[:, 0].astype(int)])
#     viewer.add_points(points, name='particle positions', size=3, border_color='red', face_color='None', opacity=0.5)

#     plt.plot(intensities)
#     plt.show(block=True)
    