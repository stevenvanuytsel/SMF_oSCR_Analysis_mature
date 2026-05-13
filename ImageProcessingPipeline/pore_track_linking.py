# %%
# File handling
from glob import glob
import os

# Processing
import pandas as pd
import time
import numpy as np
from scipy.spatial import cKDTree

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
start_global = time.perf_counter()

# %%
mpp = 10/29 #um

# %%
# Load localization dataframe
datafolder = 'pore_localization'
droplet = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
expname = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
savename = f'{expname}_{droplet}'
outdir = 'pore_tracks'
os.makedirs(outdir, exist_ok=True)

# %%
print(savename)

# %%
data = pd.read_csv(os.path.join(f'{datafolder}', f'{savename}_pore_localization.csv'))

# %%
print(data.head(50))

def link_pores_ckdtree_no_gaps(df, max_disp_pix):
    """
    Link per-frame localizations into tracks, using cKDTree.
    No gaps allowed: tracks must have a detection every frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns required: 'frame', 'x_fit', 'y_fit'.
    max_disp_pix : float
        Maximum allowed displacement between consecutive frames (pixels).

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of df with added column 'track_id'.
    """
    df = df.sort_values('frame').copy()
    df['track_id'] = -1

    frames = np.sort(df['frame'].unique())
    frame_to_idx = {f: df.index[df['frame'] == f].to_numpy() for f in frames}

    active_tracks = {}          # track_id -> (x, y)
    current_track_id = 0

    for i, f in enumerate(frames):
        print(f'Frame {i}')
        idx_this = frame_to_idx[f]
        coords_this = df.loc[idx_this, ['x_fit', 'y_fit']].to_numpy()

        if i == 0:
            # First frame: each detection starts a new track
            for j, idx_row in enumerate(idx_this):
                df.at[idx_row, 'track_id'] = current_track_id
                active_tracks[current_track_id] = coords_this[j]
                current_track_id += 1
            continue

        # Build KDTree for detections in this frame
        if coords_this.shape[0] > 0:
            tree = cKDTree(coords_this)
        else:
            tree = None

        assigned_tracks = set()
        assigned_detections = set()

        # Try to extend each active track with nearest neighbor in this frame
        for tid, prev_coord in list(active_tracks.items()):
            if tree is None:
                # No detections this frame: track must terminate
                del active_tracks[tid]
                continue

            dist, idx_near = tree.query(prev_coord, k=1, distance_upper_bound=max_disp_pix)
            if dist == np.inf:
                # No neighbor within max_disp_pix: terminate track
                del active_tracks[tid]
                continue

            if idx_near in assigned_detections:
                # That detection already claimed by another track; terminate this one
                del active_tracks[tid]
                continue

            # Assign detection to this track
            idx_row = idx_this[idx_near]
            df.at[idx_row, 'track_id'] = tid
            active_tracks[tid] = coords_this[idx_near]
            assigned_tracks.add(tid)
            assigned_detections.add(idx_near)

        # Any detection not assigned starts a new track
        for j, idx_row in enumerate(idx_this):
            if j in assigned_detections:
                continue
            df.at[idx_row, 'track_id'] = current_track_id
            active_tracks[current_track_id] = coords_this[j]
            current_track_id += 1

    return df
# %%
linked_df = link_pores_ckdtree_no_gaps(data, 7)

# %%
# print(linked_df.track_id.unique())

# # %%
# # Ensure sorted by time within each track
# d = linked_df.sort_values(['track_id', 'frame']).copy()

# # Compute per-step displacements within each track
# d['dx'] = d.groupby('track_id')['x_fit'].diff()
# d['dy'] = d.groupby('track_id')['y_fit'].diff()
# d['step'] = np.sqrt(d['dx']**2 + d['dy']**2)

# # Drop first step per track (NaN)
# steps = d.dropna(subset=['step'])

# %%

# # Drop first step per track (NaN)
# all_steps = d['step'].dropna().to_numpy()

# plt.figure()
# plt.hist(all_steps, bins=50, density=False)
# plt.xlabel('Step size (pixels)')
# plt.ylabel('Probability density')
# plt.tight_layout()
# plt.show()

# %%
# Filter out tracks that are too short
min_len = 300
track_lengths = linked_df.groupby('track_id').size()
keep_ids = track_lengths[track_lengths>=min_len].index
linked_df = linked_df[linked_df['track_id'].isin(keep_ids)].copy()

# %%
# Save the dataset
linked_df.to_csv(os.path.join(outdir, f"{savename}_pore_tracks.csv"), index=False)

# %%
# for p in linked_df.track_id.unique():
#     subdf = linked_df.loc[linked_df['track_id']==p]
#     fig, ax = plt.subplots()
#     ax.plot(subdf['frame'], subdf['raw_sum'])
#     plt.show(block=True)

# # %%
# plt.scatter(linked_df['raw_sum'], linked_df['r2_moffat'])
# plt.show()

# %%
# # Visualize
# from skimage import io
# stack = io.imread(glob(os.path.join(f'cleaned_data', '*green.tif'))[0])

# tracks = linked_df[['frame', 'y_fit', 'x_fit']].to_numpy()

# import napari
# viewer = napari.view_image(stack, name='data')
# features = {'track_id': linked_df['track_id'].to_numpy()}

# points = viewer.add_points(
#     tracks,
#     name='pores',
#     features=features,
#     size=4,                 # circle radius in pixels (tune)
#     border_color='track_id',  # color by track_id (categorical)  # use default cycle, or give your own list
#     border_width=0.1,
#     face_color='transparent',
#     symbol='o',             # 'o' is the default circle marker
# )

# points.border_color_mode = 'cycle'
# points.border_color_cycle = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
# %%
