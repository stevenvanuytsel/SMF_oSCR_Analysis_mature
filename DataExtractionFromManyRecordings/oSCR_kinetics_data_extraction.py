# %%
import glob
import os

import pandas as pd

# %%
folders = [f.path for f in os.scandir(path='..') if f.is_dir() and f.name.startswith("20")]

# %%
all_open_segments = []
all_closed_segments = []

for folder in folders:
    print(folder)
    common_path = "data_cleaning+tracking"
    base = os.path.join(folder, common_path)

    subfolders = [
        entry.path
        for entry in os.scandir(base)
        if entry.is_dir() and entry.name.startswith("drop")
    ]

    for subf in subfolders:
        print(subf)
        data_folder = "pore_idealization"
        pattern = os.path.join(subf, data_folder, "*.csv")
        matches = glob.glob(pattern)
        if not matches:
            continue  # or raise

        data = matches[0]
        
        # get part before '-100'
        basename = os.path.basename(data)
        expname = basename.split("_-100")[0]

        # Load dataframe
        df = pd.read_csv(data)

        # Loop over particles
        for particle in df.track_id.unique():
            print(particle)
            subdf = df.loc[df['track_id']==particle].copy()

            # Per particle segment table
            seg = subdf.groupby("segment_id_refined", as_index=False).agg(
                state=("state_seg", "first"), # first value of the state segment because they're all the same
                start_frame = ("frame", "min"), # first frame is the start
                end_frame = ("frame", "max"), # end frame is largest frame
                length_frames = ("frame", "count")) 
            
            seg = seg.sort_values("start_frame").reset_index(drop=True) # groupby doesn't guarantee order
            
            # Look at previous and next segment states
            seg["prev_state"] = seg["state"].shift(1)
            seg["next_state"] = seg["state"].shift(-1)
            
            # Find valid states
            is_valid_open = ((seg["state"]=='open') & (seg["next_state"]=="closed"))
            is_valid_closed = ((seg["state"] == "closed")
                                        & (seg["prev_state"] == "open")
                                        & (seg["next_state"] == "open"))

            # Add rows to global lists
            open_seg = seg.loc[is_valid_open, ["segment_id_refined", "start_frame", "end_frame", "length_frames"]]
            open_seg["savename"] = expname
            open_seg["particle"] = particle

            closed_seg = seg.loc[is_valid_closed, ["segment_id_refined", "start_frame", "end_frame", "length_frames"]]
            closed_seg["savename"] = expname 
            closed_seg["particle"] = particle

            # Save for later
            all_open_segments.append(open_seg)
            all_closed_segments.append(closed_seg)      

# Turn lists into dataframes
open_df = pd.concat(all_open_segments, ignore_index=True)
closed_df = pd.concat(all_closed_segments, ignore_index=True)

# Save
open_df.to_csv("32bp_Cy5_32bp_BHQ2_0.1uM_open_frames.csv", index=False)
closed_df.to_csv("32bp_Cy5_32bp_BHQ2_0.1uM_closed_frames.csv", index=False)

# %%
